"""
OpenRouter wrapper using OpenAI-compatible SDK.
Routes requests to any model available on OpenRouter.
"""
import json
import logging
import re
from typing import Optional

from openai import OpenAI, APIError, RateLimitError

import config
from .base import LLMInterface, LLMResponse

logger = logging.getLogger(__name__)

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


class OpenRouterModel(LLMInterface):
    """Cloud LLM via OpenRouter API. Access to hundreds of models through one API."""

    def __init__(self, model: str = None, api_key: str = None):
        self.model = model or config.OPENROUTER_MODEL
        self._custom_api_key = api_key
        self._client = None
        self._content_blocked = False

    _last_request_time = 0.0
    _last_request_by_key = {}
    _key_cooldowns = {}

    def _prepare_messages(self, prompt: str, system: str = "") -> list:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

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
    def _filter_reasoning_stream(cls, chunks):
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

    @property
    def api_key(self) -> str:
        return self._custom_api_key or config.OPENROUTER_API_KEY

    @property
    def client(self) -> OpenAI:
        current_key = self.api_key
        if self._client is not None:
            if getattr(self, '_client_key', None) != current_key:
                self._client = None

        if self._client is None:
            if not current_key:
                raise RuntimeError(
                    "OPENROUTER_API_KEY not set. Get a key at https://openrouter.ai/keys "
                    "and add it to your .env file."
                )
            self._client = OpenAI(
                base_url=config.OPENROUTER_BASE_URL,
                api_key=current_key,
                timeout=120.0,
                max_retries=0,
                default_headers={
                    "HTTP-Referer": "http://localhost:5000",
                    "X-Title": "Story Teller",
                },
            )
            self._client_key = current_key
        return self._client

    @property
    def was_content_blocked(self) -> bool:
        """Returns True if the last request was blocked by content moderation."""
        return self._content_blocked

    @staticmethod
    def _retry_after_seconds(error: RateLimitError) -> float:
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
            match = re.search(r'try again in\s+(\d+)m', error_str, re.IGNORECASE)
            if match:
                retry_after = int(match.group(1)) * 60
        if retry_after is None:
            match = re.search(r'try again in\s+(\d+\.?\d*)\s*s', error_str, re.IGNORECASE)
            if match:
                retry_after = float(match.group(1))
        if retry_after is None:
            match = re.search(r'try again in\s+(\d+\.?\d*)', error_str, re.IGNORECASE)
            if match:
                retry_after = float(match.group(1))

        return max(float(retry_after or 5.0), 0.2) + 2.0  # 2s buffer

    def _throttle_before_request(self) -> None:
        import time as _time

        key = self.api_key or "default"
        now = _time.time()
        cooldown_wait = max(0.0, OpenRouterModel._key_cooldowns.get(key, 0.0) - now)
        elapsed = now - OpenRouterModel._last_request_by_key.get(key, 0.0)
        # OpenRouter is generally more generous with rate limits
        min_interval = float(getattr(config, 'OPENROUTER_MIN_REQUEST_INTERVAL', 1.0))
        interval_wait = max(0.0, min_interval - elapsed)
        wait = max(cooldown_wait, interval_wait)
        if wait > 0:
            logger.info("[OpenRouter] Throttling %.1fs before next request.", wait)
            _time.sleep(wait)
        request_time = _time.time()
        OpenRouterModel._last_request_time = request_time
        OpenRouterModel._last_request_by_key[key] = request_time

    def _mark_rate_limited(self, error: RateLimitError) -> float:
        import time as _time

        retry_after = self._retry_after_seconds(error)
        key = self.api_key or "default"
        OpenRouterModel._key_cooldowns[key] = _time.time() + retry_after
        logger.warning(
            "[OpenRouter] Rate limited; cooling down for %.0fs.",
            retry_after,
        )
        return retry_after

    def generate(
        self,
        prompt: str,
        system: str = "",
        schema: Optional[dict] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        _retry_count: int = 0,
    ) -> LLMResponse:
        if stream:
            content = "".join(
                self.generate_streaming(
                    prompt=prompt,
                    system=system,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    _retry_count=_retry_count,
                )
            )
            return LLMResponse(
                content=content,
                model=self.model,
                provider="openrouter",
                usage={},
                raw=None,
            )

        self._content_blocked = False
        messages = self._prepare_messages(prompt=prompt, system=system)

        # If schema requested, instruct the model to output JSON
        if schema:
            schema_instruction = (
                "\n\nReturn ONLY minified valid JSON matching this schema. "
                "No markdown, no prose, no reasoning, no <think> tags.\n"
                f"Schema: {json.dumps(schema, separators=(',', ':'))}"
            )
            if messages and messages[0]["role"] == "system":
                messages[0]["content"] += schema_instruction
            else:
                messages.insert(0, {"role": "system", "content": schema_instruction.strip()})

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature or config.CLOUD_MODEL_PARAMS["temperature"],
            "top_p": config.CLOUD_MODEL_PARAMS["top_p"],
            "max_tokens": max_tokens or config.CLOUD_MODEL_PARAMS["max_tokens"],
        }

        if schema:
            kwargs["response_format"] = {"type": "json_object"}

        self._throttle_before_request()

        try:
            logger.debug(f"[OpenRouter] Generating with model={self.model}")
            response = self.client.chat.completions.create(**kwargs)

            choice = response.choices[0]

            # Check for content filter
            if choice.finish_reason == "content_filter":
                self._content_blocked = True
                raise ContentBlockedError(
                    "OpenRouter content moderation blocked this request."
                )
            if choice.finish_reason == "length":
                logger.warning(
                    "[OpenRouter] Response hit max_tokens. Partial content returned."
                )

            return LLMResponse(
                content=self._strip_reasoning(choice.message.content or ""),
                model=self.model,
                provider="openrouter",
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                },
                raw=response,
            )

        except RateLimitError as e:
            import time as _time

            retry_after = self._mark_rate_limited(e)

            if _retry_count < 5:
                logger.warning(
                    f"[OpenRouter] Rate limited — waiting {retry_after:.2f}s then retrying "
                    f"(attempt {_retry_count + 1}/5)..."
                )
                _time.sleep(retry_after)
                return self.generate(
                    prompt, system, schema, temperature, max_tokens, stream,
                    _retry_count=_retry_count + 1,
                )

            logger.error("[OpenRouter] Rate limit persisted after 5 retries. Raising.")
            raise
        except APIError as e:
            if hasattr(e, 'status_code') and e.status_code == 400:
                error_msg = str(e).lower()
                if "json_validate_failed" in error_msg or "failed to generate json" in error_msg:
                    if kwargs.pop("response_format", None):
                        logger.warning(
                            "[OpenRouter] Strict JSON mode failed; retrying with prompt-only JSON instruction."
                        )
                        response = self.client.chat.completions.create(**kwargs)
                        choice = response.choices[0]
                        if choice.finish_reason == "length":
                            raise RuntimeError(
                                "OpenRouter response hit max_tokens before finishing."
                            )
                        return LLMResponse(
                            content=self._strip_reasoning(choice.message.content or ""),
                            model=self.model,
                            provider="openrouter",
                            usage={
                                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                            },
                            raw=response,
                        )
                if any(kw in error_msg for kw in ["safety", "content", "moderation", "policy"]):
                    self._content_blocked = True
                    raise ContentBlockedError(f"Content blocked by OpenRouter: {e}")
            raise

    def is_available(self) -> bool:
        """Check if OpenRouter API key is valid using the lightweight /auth/key endpoint."""
        if not self.api_key:
            logger.warning("[OpenRouter] No API key configured.")
            return False
        try:
            import httpx
            resp = httpx.get(
                "https://openrouter.ai/api/v1/auth/key",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10.0,
            )
            if resp.status_code == 200:
                data = resp.json().get("data", {})
                logger.info(
                    "[OpenRouter] API key valid. Label: %s, Limit: %s",
                    data.get("label", "N/A"),
                    data.get("limit") or "unlimited",
                )
                return True
            logger.warning("[OpenRouter] Auth check returned status %d", resp.status_code)
            return False
        except ImportError:
            # httpx not installed — fall back to a quick completion test
            try:
                self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": "hi"}],
                    max_tokens=1,
                )
                return True
            except Exception as e:
                logger.warning(f"[OpenRouter] Availability check failed: {e}")
                return False
        except Exception as e:
            logger.warning(f"[OpenRouter] Auth check failed: {e}")
            return False

    def get_name(self) -> str:
        return f"OpenRouter ({self.model})"

    def generate_streaming(
        self,
        prompt: str,
        system: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        _retry_count: int = 0,
    ):
        """
        Streaming generator — yields content chunks.
        Used by the web UI for real-time output.
        """
        self._content_blocked = False
        messages = self._prepare_messages(prompt=prompt, system=system)

        self._throttle_before_request()

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or config.CLOUD_MODEL_PARAMS["temperature"],
                top_p=config.CLOUD_MODEL_PARAMS["top_p"],
                max_tokens=max_tokens or config.CLOUD_MODEL_PARAMS["max_tokens"],
                stream=True,
            )
        except RateLimitError as e:
            import time as _time

            retry_after = self._mark_rate_limited(e)

            if _retry_count < 5:
                logger.warning(
                    f"[OpenRouter] Stream rate limited — waiting {retry_after:.2f}s "
                    f"then retrying (attempt {_retry_count + 1}/5)..."
                )
                _time.sleep(retry_after)
                yield from self.generate_streaming(
                    prompt, system, temperature, max_tokens,
                    _retry_count=_retry_count + 1,
                )
                return

            logger.error("[OpenRouter] Rate limit persisted after 5 retries on stream. Raising.")
            raise

        finish_reason = None

        def content_chunks():
            nonlocal finish_reason
            for chunk in stream:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
                delta = choice.delta
                if delta and delta.content:
                    yield delta.content

        try:
            for content in self._filter_reasoning_stream(content_chunks()):
                if content:
                    yield content
            if finish_reason == "length":
                logger.warning(
                    "[OpenRouter] Streaming response hit max_tokens. Partial content returned."
                )

        except APIError as e:
            if hasattr(e, 'status_code') and e.status_code == 400:
                self._content_blocked = True
            raise


class ContentBlockedError(Exception):
    """Raised when OpenRouter's content moderation blocks a request."""
    pass
