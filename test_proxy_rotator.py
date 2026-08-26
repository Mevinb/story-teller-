import unittest

from proxy_rotator import ProxyPool, parse_proxy_list, upstream_address


class ProxyRotatorTests(unittest.TestCase):
    def test_parse_proxy_list(self):
        self.assertEqual(parse_proxy_list("http://a:1,\nhttp://b:2"), ["http://a:1", "http://b:2"])

    def test_rejects_invalid_proxy(self):
        with self.assertRaises(ValueError):
            parse_proxy_list("socks5://localhost:1080")

    def test_round_robin(self):
        pool = ProxyPool(["http://a:1", "http://b:2"])
        self.assertEqual([pool.next(), pool.next(), pool.next()], ["http://a:1", "http://b:2", "http://a:1"])

    def test_upstream_address(self):
        self.assertEqual(upstream_address("http://proxy.example:3128").host, "proxy.example")
        self.assertEqual(upstream_address("http://proxy.example:3128").port, 3128)


if __name__ == "__main__":
    unittest.main()
