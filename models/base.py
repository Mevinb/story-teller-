"""
Abstract base class for all LLM backends.
Provides a unified interface so agents don't care about the provider.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional
import threading
import time
import json
import logging
import re
import ast

import config
from pipeline.errors import PipelineCancelledError

logger = logging.getLogger(__name__)


class ContentBlockedError(Exception):
    """Raised when any provider's content moderation blocks a request.

    Defined once here so callers (e.g. the writer's local-model fallback)
    can catch a single exception type regardless of provider. Backends
    re-export this name for backward compatibility.
    """


# ─── Reasoning-tag filtering (shared by OpenAI-compatible backends) ──────
_REASONING_BLOCK_RE = re.compile(
    r"<(?:think|analysis|reasoning)>.*?</(?:think|analysis|reasoning)>",
    flags=re.IGNORECASE | re.DOTALL,
)
_REASONING_TAG_RE = re.compile(r"</?(?:think|analysis|reasoning)>", flags=re.IGNORECASE)
_REASONING_OPEN_TAGS = ("<think>", "<analysis>", "<reasoning>")
_REASONING_CLOSE_TAGS = {
    "<think>": "</think>",
    "<analysis>": "</analysis>",
    "<reasoning>": "</reasoning>",
}
_MAX_REASONING_TAG_LEN = max(len(tag) for tag in _REASONING_OPEN_TAGS)


class ReasoningStreamFilter:
    """Mixin that strips <think>/<analysis>/<reasoning> blocks from output.

    Works both on complete strings (`_strip_reasoning`) and incrementally on
    token streams (`_filter_reasoning_stream`) so partial tags held across
    chunk boundaries are never leaked to the UI.
    """

    @staticmethod
    def _strip_reasoning(text: str) -> str:
        cleaned = _REASONING_BLOCK_RE.sub("", text or "")
        cleaned = _REASONING_TAG_RE.sub("", cleaned)
        return cleaned.strip()

    @staticmethod
    def _remove_reasoning_tags(text: str) -> str:
        return _REASONING_TAG_RE.sub("", text or "")

    @staticmethod
    def _first_reasoning_tag(text: str):
        lower = text.lower()
        best = None
        for tag in _REASONING_OPEN_TAGS:
            idx = lower.find(tag)
            if idx != -1 and (best is None or idx < best[0]):
                best = (idx, tag)
        return best

    @classmethod
    def _filter_reasoning_stream(cls, chunks: Iterable[str]):
        pending = ""
        hidden_close_tag = None

        for chunk in chunks:
            pending += chunk
            while pending:
                lower = pending.lower()

                if hidden_close_tag:
                    end = lower.find(hidden_close_tag)
                    if end == -1:
                        pending = pending[-len(hidden_close_tag):]
                        break
                    pending = pending[end + len(hidden_close_tag):]
                    hidden_close_tag = None
                    continue

                found = cls._first_reasoning_tag(pending)
                if found:
                    idx, open_tag = found
                    visible = cls._remove_reasoning_tags(pending[:idx])
                    if visible:
                        yield visible
                    pending = pending[idx + len(open_tag):]
                    hidden_close_tag = _REASONING_CLOSE_TAGS[open_tag]
                    continue

                if len(pending) <= _MAX_REASONING_TAG_LEN:
                    break
                visible = pending[:-_MAX_REASONING_TAG_LEN]
                pending = pending[-_MAX_REASONING_TAG_LEN:]
                if visible:
                    yield visible

        if pending and not hidden_close_tag:
            visible = cls._remove_reasoning_tags(pending)
            if visible:
                yield visible


def parse_retry_after_seconds(error: Exception, default: float = 5.0) -> float:
    """Best-effort extraction of a Retry-After hint from a rate-limit error.

    Checks the Retry-After header first, then common "try again in X"
    error-message formats (minutes, seconds, bare numbers).
    """
    retry_after = None
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        raw_retry_after = headers.get("retry-after") or headers.get("Retry-After")
        if raw_retry_after:
            try:
                retry_after = float(raw_retry_after)
            except (TypeError, ValueError):
                retry_after = None

    error_str = str(error)
    if retry_after is None:
        match = re.search(r"try again in\s+(\d+)m", error_str, re.IGNORECASE)
        if match:
            retry_after = int(match.group(1)) * 60
    if retry_after is None:
        match = re.search(r"try again in\s+(\d+\.?\d*)\s*s", error_str, re.IGNORECASE)
        if match:
            retry_after = float(match.group(1))
    if retry_after is None:
        match = re.search(r"try again in\s+(\d+\.?\d*)", error_str, re.IGNORECASE)
        if match:
            retry_after = float(match.group(1))

    return max(float(retry_after if retry_after is not None else default), 0.2)


class RateLimitTracker:
    """Thread-safe cooldown/request-interval bookkeeping for cloud backends.

    One instance per backend class; entries are keyed by an arbitrary
    track id (e.g. "(model, key)") so multi-key rotation stays correct.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_request_by_key: dict = {}
        self._key_cooldowns: dict = {}

    def throttle_wait(self, track_id: str, now: float, min_interval: float) -> float:
        """Seconds the caller must sleep before the next request."""
        with self._lock:
            cooldown_wait = max(0.0, self._key_cooldowns.get(track_id, 0.0) - now)
            elapsed = now - self._last_request_by_key.get(track_id, 0.0)
            interval_wait = max(0.0, min_interval - elapsed)
            return max(cooldown_wait, interval_wait)

    def record_request(self, track_id: str, now: float) -> None:
        with self._lock:
            self._last_request_by_key[track_id] = now

    def mark_cooldown(self, track_id: str, until: float) -> None:
        with self._lock:
            self._key_cooldowns[track_id] = until

    def cooldown_expiry(self, track_id: str) -> float:
        with self._lock:
            return self._key_cooldowns.get(track_id, 0.0)


@dataclass
class LLMResponse:
    """Standardized response from any LLM backend."""
    content: str
    model: str
    provider: str              # "llama.cpp" | "groq"
    usage: dict = field(default_factory=dict)
    raw: Any = None

    @staticmethod
    def _strip_reasoning(text: str) -> str:
        cleaned = text
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r"<analysis>.*?</analysis>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r"<reasoning>.*?</reasoning>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
        return cleaned.strip()

    @staticmethod
    def _extract_fenced(text: str) -> list[str]:
        blocks = []
        for pattern in (r"```json\s*(.*?)```", r"```\s*(.*?)```"):
            for match in re.finditer(pattern, text, flags=re.DOTALL | re.IGNORECASE):
                block = match.group(1).strip()
                if block:
                    blocks.append(block)
        return blocks

    @staticmethod
    def _extract_balanced_json(text: str) -> list[str]:
        candidates = []
        starts = [i for i, ch in enumerate(text) if ch in "{["]
        for start in starts:
            stack = []
            in_str = False
            escaped = False
            for idx in range(start, len(text)):
                ch = text[idx]
                if in_str:
                    if escaped:
                        escaped = False
                    elif ch == "\\":
                        escaped = True
                    elif ch == "\"":
                        in_str = False
                    continue
                if ch == "\"":
                    in_str = True
                    continue
                if ch in "{[":
                    stack.append(ch)
                    continue
                if ch == "}":
                    if not stack or stack[-1] != "{":
                        break
                    stack.pop()
                elif ch == "]":
                    if not stack or stack[-1] != "[":
                        break
                    stack.pop()
                if not stack:
                    fragment = text[start:idx + 1].strip()
                    if fragment:
                        candidates.append(fragment)
                    break
        return candidates

    @staticmethod
    def _normalize_json_candidate(text: str) -> str:
        out = text.strip()
        out = out.replace("“", "\"").replace("”", "\"").replace("’", "'")
        out = re.sub(r",(\s*[}\]])", r"\1", out)
        return out

    @staticmethod
    def _try_parse(candidate: str):
        try:
            return json.loads(candidate)
        except Exception:
            pass
        normalized = LLMResponse._normalize_json_candidate(candidate)
        try:
            return json.loads(normalized)
        except Exception:
            pass
        try:
            parsed = ast.literal_eval(normalized)
            if isinstance(parsed, (dict, list)):
                return parsed
        except Exception:
            pass
        return None

    def as_json(self) -> Optional[dict]:
        """Attempt to parse the content as JSON."""
        raw = self.content or ""
        if not raw.strip():
            return None

        cleaned = self._strip_reasoning(raw)

        candidates = [raw.strip(), cleaned]
        candidates.extend(self._extract_fenced(cleaned))
        candidates.extend(self._extract_balanced_json(cleaned))

        # Last-resort raw decoder from every possible JSON start.
        for idx, ch in enumerate(cleaned):
            if ch not in "{[":
                continue
            try:
                parsed, _ = json.JSONDecoder().raw_decode(cleaned[idx:])
                candidates.append(json.dumps(parsed))
            except Exception:
                continue

        for candidate in candidates:
            if not candidate:
                continue
            parsed = self._try_parse(candidate)
            if isinstance(parsed, (dict, list)):
                return parsed

        return None


class LLMInterface(ABC):
    """Abstract LLM backend interface."""

    # ─── Cancellation support ───────────────────────────────────────────────
    # The orchestrator installs a `_should_cancel` callable on each model so
    # that blocking (non-streaming) generation, throttle waits, and retry
    # backoff can abort promptly when the user presses a Stop/Cancel button.

    _should_cancel: Optional[Callable[[], bool]] = None

    def _cancel_requested(self) -> bool:
        check = getattr(self, "_should_cancel", None)
        return bool(callable(check) and check())

    def _raise_if_cancelled(self, message: str = "Generation cancelled by user") -> None:
        if self._cancel_requested():
            raise PipelineCancelledError(message)

    def _sleep_interruptible(self, seconds: float) -> None:
        """Sleep in small increments, aborting immediately on cancellation."""
        if not isinstance(seconds, (int, float)) or seconds <= 0:
            return
        end = time.time() + seconds
        while True:
            self._raise_if_cancelled()
            remaining = end - time.time()
            if remaining <= 0:
                return
            time.sleep(min(0.25, max(0.05, remaining)))

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system: str = "",
        schema: Optional[dict] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> LLMResponse:
        """
        Generate a response from the model.

        Args:
            prompt: User message / instruction
            system: System prompt for context/persona
            schema: Optional JSON schema to constrain output format
            temperature: Override default temperature
            max_tokens: Override default max tokens
            stream: Whether to request streaming generation mode

        Returns:
            LLMResponse with the generated content
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is reachable and ready."""
        ...

    @abstractmethod
    def get_name(self) -> str:
        """Return a human-readable name for this backend."""
        ...

    def generate_with_retry(
        self,
        prompt: str,
        system: str = "",
        schema: Optional[dict] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        max_retries: int = None,
    ) -> LLMResponse:
        """
        Generate with exponential backoff retry logic.
        Retries on transient failures, not on content issues.
        """
        if max_retries is None:
            max_retries = config.MAX_GENERATION_RETRIES

        delay = config.RETRY_BASE_DELAY
        last_error = None

        for attempt in range(max_retries):
            self._raise_if_cancelled()
            try:
                return self.generate(
                    prompt=prompt,
                    system=system,
                    schema=schema,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=stream,
                )
            except PipelineCancelledError:
                raise
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                
                # Auto-adjust max_tokens if the model rejects the requested amount
                if "max_tokens" in err_str and "less than or equal to" in err_str:
                    match = re.search(r"less than or equal to `?(\d+)`?", str(e))
                    if match:
                        new_max = int(match.group(1))
                        max_tokens = new_max
                        logger.warning(
                            f"[{self.get_name()}] Auto-adjusting max_tokens to {new_max} based on model limits."
                        )
                        continue  # Retry immediately with new limit
                        
                # Handle 413 Request Entity Too Large (context window / token request too big)
                if "request entity too large" in err_str or "request_too_large" in err_str:
                    if max_tokens is None or max_tokens > 1024:
                        new_max = 1024 if (max_tokens is None or max_tokens > 2048) else (max_tokens // 2)
                        logger.warning(f"[{self.get_name()}] 413 Entity Too Large. Reducing max_tokens to {new_max}.")
                        max_tokens = new_max
                        continue # Retry immediately with fewer requested tokens
                        
                # For rate limits, use longer delay or exact requested delay
                if 'rate' in err_str and 'limit' in err_str:
                    match = re.search(r"try again in ([\d\.]+)s", err_str)
                    if match:
                        wait = float(match.group(1)) + 1.5
                    else:
                        wait = max(delay, 15)  # At least 15s for rate limits
                else:
                    wait = delay
                if attempt < max_retries - 1:
                    logger.warning(
                        f"[{self.get_name()}] Attempt {attempt + 1}/{max_retries} "
                        f"failed: {e}. Retrying in {wait:.1f}s..."
                    )
                    self._sleep_interruptible(wait)
                    delay = min(delay * config.RETRY_BACKOFF_FACTOR, config.RETRY_MAX_DELAY)

        raise RuntimeError(
            f"[{self.get_name()}] All {max_retries} attempts failed. "
            f"Last error: {last_error}"
        )
