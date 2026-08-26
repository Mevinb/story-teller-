"""Small, localhost-only rotating HTTP proxy for OpenCode and other clients.

The listener accepts normal HTTP proxy requests and HTTPS CONNECT requests,
then forwards each new client connection through the next configured upstream
proxy. It does not provide anonymity by itself; use trusted upstream proxies.
"""

from __future__ import annotations

import argparse
import os
import select
import socket
import socketserver
import threading
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urlsplit


def parse_proxy_list(value: str) -> list[str]:
    """Parse comma/newline-separated proxy URLs and reject unsupported values."""
    proxies = [item.strip() for item in value.replace("\n", ",").split(",") if item.strip()]
    for proxy in proxies:
        parsed = urlsplit(proxy)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError(f"Unsupported proxy URL: {proxy!r}; use http[s]://host:port")
        if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
            raise ValueError(f"Proxy URL must not contain a path/query: {proxy!r}")
    return proxies


class ProxyPool:
    """Thread-safe round-robin upstream selection."""

    def __init__(self, proxies: Iterable[str]):
        self.proxies = list(proxies)
        if not self.proxies:
            raise ValueError("At least one upstream proxy is required")
        self._index = 0
        self._lock = threading.Lock()

    def next(self) -> str:
        with self._lock:
            proxy = self.proxies[self._index % len(self.proxies)]
            self._index += 1
            return proxy


def _authority(value: str) -> tuple[str, int]:
    parsed = urlsplit(f"//{value}")
    if not parsed.hostname:
        raise ValueError(f"Invalid destination: {value!r}")
    return parsed.hostname, parsed.port or 443


@dataclass(frozen=True)
class Upstream:
    host: str
    port: int


def upstream_address(proxy_url: str) -> Upstream:
    parsed = urlsplit(proxy_url)
    return Upstream(parsed.hostname or "", parsed.port or 80)


class RotatingProxyHandler(socketserver.StreamRequestHandler):
    pool: ProxyPool
    timeout = 30

    def handle(self) -> None:
        self.connection.settimeout(self.timeout)
        line = self.rfile.readline(65536)
        if not line:
            return
        try:
            method, target, version = line.decode("latin-1").strip().split(" ", 2)
            headers = self._read_headers()
            upstream = upstream_address(self.pool.next())
            if method.upper() == "CONNECT":
                self._connect(upstream, target)
            else:
                self._forward_http(upstream, method, target, version, headers)
        except (OSError, ValueError) as exc:
            self._error(502, str(exc))

    def _read_headers(self) -> list[tuple[str, str]]:
        headers = []
        while True:
            line = self.rfile.readline(65536)
            if not line or line in {b"\r\n", b"\n"}:
                return headers
            name, _, value = line.decode("latin-1").partition(":")
            if not _:
                raise ValueError("Malformed HTTP header")
            if name.lower() not in {"proxy-connection", "connection", "keep-alive"}:
                headers.append((name, value.strip()))

    def _connect(self, upstream: Upstream, target: str) -> None:
        host, port = _authority(target)
        with socket.create_connection((upstream.host, upstream.port), self.timeout) as remote:
            remote.sendall(f"CONNECT {host}:{port} HTTP/1.1\r\nHost: {host}:{port}\r\n\r\n".encode())
            response = self._read_response(remote)
            if not response.startswith(b"HTTP/") or b" 2" not in response[:16]:
                self.wfile.write(response)
                return
            self.wfile.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            self._tunnel(remote)

    def _forward_http(self, upstream: Upstream, method: str, target: str, version: str, headers: list[tuple[str, str]]) -> None:
        parsed = urlsplit(target)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("HTTP proxy requests must use an absolute URL")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        path = parsed.path or "/"
        if parsed.query:
            path += f"?{parsed.query}"
        request = [f"{method} {target if parsed.scheme == 'http' else path} {version}\r\n"]
        request.extend(f"{name}: {value}\r\n" for name, value in headers if name.lower() != "host")
        request.append(f"Host: {parsed.hostname}:{port}\r\n\r\n")
        body = self.rfile.read(int(dict(headers).get("Content-Length", "0")))
        with socket.create_connection((upstream.host, upstream.port), self.timeout) as remote:
            remote.sendall("".join(request).encode("latin-1") + body)
            self._copy(remote, self.connection)

    def _read_response(self, remote: socket.socket) -> bytes:
        data = b""
        while b"\r\n\r\n" not in data and len(data) < 65536:
            chunk = remote.recv(4096)
            if not chunk:
                break
            data += chunk
        return data

    def _tunnel(self, remote: socket.socket) -> None:
        self._copy(remote, self.connection)

    def _copy(self, left: socket.socket, right: socket.socket) -> None:
        sockets = [left, right]
        while True:
            readable, _, _ = select.select(sockets, [], [], self.timeout)
            if not readable:
                return
            for source in readable:
                data = source.recv(65536)
                if not data:
                    return
                (right if source is left else left).sendall(data)

    def _error(self, code: int, message: str) -> None:
        self.wfile.write(f"HTTP/1.1 {code} Bad Gateway\r\nContent-Length: 0\r\n\r\n".encode())


class RotatingProxyServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a localhost rotating HTTP proxy")
    parser.add_argument("--listen-host", default=os.getenv("ROTATING_PROXY_HOST", "127.0.0.1"))
    parser.add_argument("--listen-port", type=int, default=int(os.getenv("ROTATING_PROXY_PORT", "8080")))
    parser.add_argument("--proxies", default=os.getenv("ROTATING_PROXY_URLS", ""))
    args = parser.parse_args()
    pool = ProxyPool(parse_proxy_list(args.proxies))
    RotatingProxyHandler.pool = pool
    with RotatingProxyServer((args.listen_host, args.listen_port), RotatingProxyHandler) as server:
        print(f"Rotating proxy listening on {args.listen_host}:{args.listen_port} ({len(pool.proxies)} upstreams)")
        server.serve_forever()


if __name__ == "__main__":
    main()
