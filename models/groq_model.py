"""
Groq Cloud wrapper using OpenAI-compatible SDK.
Primary backend for cloud prose generation.
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

class GroqKeyManager:
    """Manages rotation of multiple Groq API keys."""
    _current_idx = 0

    @classmethod
    def get_keys(cls):
        return config.GROQ_API_KEYS

    @classmethod
    def get_key(cls) -> str:
        keys = cls.get_keys()
        if not keys:
            return config.GROQ_API_KEY
        return keys[cls._current_idx % len(keys)]

    @classmethod
    def rotate(cls):
        keys = cls.get_keys()
        if not keys or len(keys) < 2:
            return False
        cls._current_idx = (cls._current_idx + 1) % len(keys)
        logger.info(f"Rotating to Groq API key #{cls._current_idx + 1} ({cls.get_key()[:8]}...)")
        return True


class GroqModel(LLMInterface):
    """Cloud LLM via Groq API. Fast inference on large models."""

    def __init__(self, model: str = None, api_key: str = None):
        self.model = model or config.GROQ_MODEL
        self._custom_api_key = api_key
        self._client = None
        self._content_blocked = False
    
    _last_request_time = 0.0
    _last_request_by_key = {}
    _key_cooldowns = {}

    def _use_strict_json_mode(self) -> bool:
        """Some Groq-hosted models, notably Qwen, can fail server-side JSON validation."""
        return not self._is_qwen_model()

    def _is_qwen_model(self) -> bool:
        return "qwen" in (self.model or "").lower()

    def _qwen_reasoning_params(self) -> dict:
        if not self._is_qwen_model():
            return {}
        return {
            # Groq-specific Qwen control: prevents spending completion tokens on reasoning.
            "reasoning_effort": "none",
            # Backup: if reasoning is ever enabled, do not return it in content.
            "reasoning_format": "hidden",
        }

    def _prepare_messages(self, prompt: str, system: str = "") -> list:
        if self._is_qwen_model():
            qwen_instruction = (
                "Reasoning mode is disabled. Do not output hidden reasoning, analysis, "
                "<think> tags, or planning. Start directly with the final requested content."
            )
            system = f"{system.rstrip()}\n\n{qwen_instruction}".strip()
            prompt = f"{prompt.rstrip()}\n\n/no_think"

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
        return self._custom_api_key or GroqKeyManager.get_key()

    @property
    def client(self) -> OpenAI:
        # Check if the current manager key differs from what the client was initialized with
        current_key = self.api_key
        if self._client is not None:
            # We must re-initialize the client if the key changed globally
            if getattr(self, '_client_key', None) != current_key:
                self._client = None

        if self._client is None:
            if not current_key:
                raise RuntimeError(
                    "GROQ_API_KEY not set. Get a free key at https://console.groq.com "
                    "and add it to your .env file."
                )
            self._client = OpenAI(
                base_url=config.GROQ_BASE_URL,
                api_key=current_key,
                timeout=30.0,
                max_retries=0,  # Disable internal retries to allow our key rotation to take over
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

        return max(float(retry_after or 5.0), 0.2) + config.GROQ_RATE_LIMIT_BUFFER

    def _throttle_before_request(self) -> None:
        import time as _time

        key = self.api_key or "default"
        now = _time.time()
        cooldown_wait = max(0.0, GroqModel._key_cooldowns.get(key, 0.0) - now)
        elapsed = now - GroqModel._last_request_by_key.get(key, 0.0)
        interval_wait = max(0.0, config.GROQ_MIN_REQUEST_INTERVAL - elapsed)
        wait = max(cooldown_wait, interval_wait)
        if wait > 0:
            logger.info("[Groq] Throttling %.1fs before next request.", wait)
            _time.sleep(wait)
        request_time = _time.time()
        GroqModel._last_request_time = request_time
        GroqModel._last_request_by_key[key] = request_time

    def _mark_rate_limited(self, error: RateLimitError) -> float:
        import time as _time

        retry_after = self._retry_after_seconds(error)
        key = self.api_key or "default"
        GroqModel._key_cooldowns[key] = _time.time() + retry_after
        logger.warning(
            "[Groq] Rate limited on current key; cooling it down for %.0fs.",
            retry_after,
        )
        return retry_after

    def _rotate_to_available_key(self) -> bool:
        if not config.GROQ_ROTATE_ON_RATE_LIMIT:
            return False
        keys = GroqKeyManager.get_keys()
        if len(keys) < 2:
            return False

        import time as _time

        for _ in range(len(keys) - 1):
            if not GroqKeyManager.rotate():
                return False
            key = self.api_key or "default"
            if GroqModel._key_cooldowns.get(key, 0.0) <= _time.time():
                return True
        return False

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
                provider="groq",
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
        qwen_reasoning_params = self._qwen_reasoning_params()
        if qwen_reasoning_params:
            kwargs["extra_body"] = qwen_reasoning_params

        if schema and self._use_strict_json_mode():
            kwargs["response_format"] = {"type": "json_object"}

        self._throttle_before_request()

        try:
            logger.debug(f"[Groq] Generating with model={self.model}")
            response = self.client.chat.completions.create(**kwargs)

            choice = response.choices[0]

            # Check for content filter
            if choice.finish_reason == "content_filter":
                self._content_blocked = True
                raise ContentBlockedError(
                    "Groq content moderation blocked this request."
                )
            if choice.finish_reason == "length":
                logger.warning(
                    "[Groq] Response hit max_tokens. Partial content returned."
                )

            return LLMResponse(
                content=self._strip_reasoning(choice.message.content or ""),
                model=self.model,
                provider="groq",
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                },
                raw=response,
            )

        except RateLimitError as e:
            import time as _time
            error_str = str(e)

            # Check if this is a DAILY token limit (waiting won't help for long)
            is_daily_limit = 'tokens per day' in error_str.lower() or 'TPD' in error_str
            if is_daily_limit:
                logger.error("[Groq] Daily token limit reached. Falling back to local model.")
                raise

            # Mark this key as cooled down and calculate wait
            retry_after = self._mark_rate_limited(e)

            # Try rotating to a fresh key first (free, fast)
            if _retry_count < max(len(GroqKeyManager.get_keys()), 1) * 3 and self._rotate_to_available_key():
                logger.warning(
                    "[Groq] Rate limited — rotated to available key (attempt %s).",
                    _retry_count + 1,
                )
                return self.generate(
                    prompt, system, schema, temperature, max_tokens, stream,
                    _retry_count=_retry_count + 1,
                )

            # No fresh key available — wait and retry (up to 5 waits total)
            if _retry_count < 5:
                logger.warning(f"[Groq] Rate limited — waiting {retry_after:.2f}s then retrying (attempt {_retry_count + 1}/5)...")
                _time.sleep(retry_after)
                return self.generate(
                    prompt, system, schema, temperature, max_tokens, stream,
                    _retry_count=_retry_count + 1,
                )

            logger.error("[Groq] Rate limit persisted after 5 retries. Raising to trigger local fallback.")
            raise
        except APIError as e:
            # Check if this is a content moderation error
            if hasattr(e, 'status_code') and e.status_code == 400:
                error_msg = str(e).lower()
                if "json_validate_failed" in error_msg or "failed to generate json" in error_msg:
                    if kwargs.pop("response_format", None):
                        logger.warning(
                            "[Groq] Strict JSON mode failed; retrying once with prompt-only JSON instruction."
                        )
                        response = self.client.chat.completions.create(**kwargs)
                        choice = response.choices[0]
                        if choice.finish_reason == "length":
                            raise RuntimeError(
                                "Groq response hit max_tokens before finishing. "
                                "Treating this as incomplete generation."
                            )
                        return LLMResponse(
                            content=self._strip_reasoning(choice.message.content or ""),
                            model=self.model,
                            provider="groq",
                            usage={
                                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                            },
                            raw=response,
                        )
                if any(kw in error_msg for kw in ["safety", "content", "moderation", "policy"]):
                    self._content_blocked = True
                    raise ContentBlockedError(f"Content blocked by Groq: {e}")
            raise

    def is_available(self) -> bool:
        """Check if Groq API is reachable with a minimal request."""
        if not self.api_key:
            logger.warning("[Groq] No API key configured.")
            return False
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            return bool(response.choices)
        except Exception as e:
            logger.warning(f"[Groq] Availability check failed: {e}")
            return False

    def get_name(self) -> str:
        return f"Groq ({self.model})"

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
                extra_body=self._qwen_reasoning_params() or None,
            )
        except RateLimitError as e:
            import time as _time

            error_str = str(e)
            is_daily_limit = 'tokens per day' in error_str.lower() or 'TPD' in error_str
            if is_daily_limit:
                logger.error("[Groq] Daily token limit reached on stream. Raising.")
                raise

            retry_after = self._mark_rate_limited(e)

            if _retry_count < max(len(GroqKeyManager.get_keys()), 1) * 3 and self._rotate_to_available_key():
                logger.warning(
                    "[Groq] Rotated to an available key after stream rate limit (attempt %s).",
                    _retry_count + 1,
                )
                yield from self.generate_streaming(
                    prompt, system, temperature, max_tokens,
                    _retry_count=_retry_count + 1,
                )
                return

            if _retry_count < 5:
                logger.warning(f"[Groq] Stream rate limited — waiting {retry_after:.2f}s then retrying (attempt {_retry_count + 1}/5)...")
                _time.sleep(retry_after)
                yield from self.generate_streaming(
                    prompt, system, temperature, max_tokens,
                    _retry_count=_retry_count + 1,
                )
                return

            logger.error("[Groq] Rate limit persisted after 5 retries on stream. Raising.")
            raise

        self._throttle_before_request()
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature or config.CLOUD_MODEL_PARAMS["temperature"],
            top_p=config.CLOUD_MODEL_PARAMS["top_p"],
            max_tokens=max_tokens or config.CLOUD_MODEL_PARAMS["max_tokens"],
            stream=True,
            extra_body=self._qwen_reasoning_params() or None,
        )

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
                    "[Groq] Streaming response hit max_tokens. Partial content returned."
                )

        except APIError as e:
            if hasattr(e, 'status_code') and e.status_code == 400:
                self._content_blocked = True
            raise


class ContentBlockedError(Exception):
    """Raised when Groq's content moderation blocks a request."""
    pass
