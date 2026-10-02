"""
OpenRouter wrapper using OpenAI-compatible SDK.
Routes requests to any model available on OpenRouter.
"""
import json
import logging
import time
from typing import Optional

from openai import OpenAI, APIError, RateLimitError

import config
from .base import (
    ContentBlockedError,
    LLMInterface,
    LLMResponse,
    RateLimitTracker,
    ReasoningStreamFilter,
    parse_retry_after_seconds,
)
from .key_rotator import KeyRotator, AccountKey, KeyStatus
from logger import get_logger, log_llm_call

logger = get_logger("openrouter")

# OpenRouter is generally more generous with rate limits than Groq.
_RATE_LIMIT_BUFFER = 2.0

_openrouter_rotator = KeyRotator(
    provider="openrouter",
    keys=config.OPENROUTER_API_KEYS,
    strategy=getattr(config, "KEY_ROTATION_STRATEGY", "least_loaded"),
)


class OpenRouterKeyManager:
    """Manages multi-account rotation and health tracking of OpenRouter API keys."""
    rotator = _openrouter_rotator

    @classmethod
    def sync_keys(cls, keys=None):
        if keys is None:
            keys = config.OPENROUTER_API_KEYS
        else:
            config.OPENROUTER_API_KEYS = [k.strip() for k in keys if k.strip()]
            if config.OPENROUTER_API_KEYS:
                config.OPENROUTER_API_KEY = config.OPENROUTER_API_KEYS[0]
        cls.rotator.set_keys(keys)

    @classmethod
    def get_keys(cls) -> list:
        keys = cls.rotator.get_keys()
        if not keys and config.OPENROUTER_API_KEY:
            return [config.OPENROUTER_API_KEY]
        return keys

    @classmethod
    def get_key(cls, proactive_rotate: bool = False, model: Optional[str] = None, estimated_tokens: int = 0) -> str:
        key = cls.rotator.get_key(proactive_rotate=proactive_rotate, model=model, estimated_tokens=estimated_tokens)
        if not key:
            return config.OPENROUTER_API_KEY
        return key

    @classmethod
    def get_active_account(cls, proactive_rotate: bool = False, model: Optional[str] = None, estimated_tokens: int = 0):
        return cls.rotator.get_active_account(proactive_rotate=proactive_rotate, model=model, estimated_tokens=estimated_tokens)

    @classmethod
    def rotate(cls) -> bool:
        return cls.rotator.rotate()

    @classmethod
    def mark_rate_limited(cls, key: str, retry_after: float, is_daily: bool = False, header_limit: Optional[int] = None, model: Optional[str] = None):
        return cls.rotator.mark_rate_limited(key, retry_after, is_daily, header_limit, model)

    @classmethod
    def mark_success(cls, key: str, tokens: int = 0):
        cls.rotator.mark_success(key, tokens)

    @classmethod
    def mark_error(cls, key: str, error: Exception):
        cls.rotator.mark_error(key, error)

    @classmethod
    def are_all_daily_exhausted(cls) -> bool:
        return cls.rotator.are_all_daily_exhausted()

    @classmethod
    def status_summary(cls):
        return cls.rotator.status_summary()


class OpenRouterModel(ReasoningStreamFilter, LLMInterface):
    """Cloud LLM via OpenRouter API. Access to hundreds of models through one API."""

    def __init__(self, model: str = None, api_key: str = None):
        self.model = model or config.OPENROUTER_MODEL
        self._custom_api_key = api_key
        self._client = None
        self._content_blocked = False

    # Shared, thread-safe cooldown / min-interval bookkeeping.
    _rl = RateLimitTracker()

    def _prepare_messages(self, prompt: str, system: str = "") -> list:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    @property
    def api_key(self) -> str:
        return self._custom_api_key or OpenRouterKeyManager.get_key()

    @property
    def client(self) -> OpenAI:
        current_key = self.api_key
        if self._client is not None:
            if getattr(self, '_client_key', None) is not None and self._client_key != current_key:
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
        return parse_retry_after_seconds(error) + _RATE_LIMIT_BUFFER

    def _track_id(self) -> str:
        return self.api_key or "default"

    def _throttle_before_request(self) -> None:
        track_id = self._track_id()
        now = time.time()
        wait = self._rl.throttle_wait(track_id, now, config.OPENROUTER_MIN_REQUEST_INTERVAL)
        if wait > 0:
            logger.info("[OpenRouter] Throttling %.1fs before next request.", wait)
            self._sleep_interruptible(wait)
        self._rl.record_request(track_id, time.time())

    def _mark_rate_limited(self, error: RateLimitError) -> float:
        retry_after = self._retry_after_seconds(error)
        self._rl.mark_cooldown(self._track_id(), time.time() + retry_after)
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

        self._raise_if_cancelled()
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
            "temperature": config.CLOUD_MODEL_PARAMS["temperature"] if temperature is None else temperature,
            "top_p": config.CLOUD_MODEL_PARAMS["top_p"],
            "max_tokens": max_tokens or config.CLOUD_MODEL_PARAMS["max_tokens"],
        }

        if schema:
            kwargs["response_format"] = {"type": "json_object"}

        if _retry_count == 0 and getattr(config, "OPENROUTER_PROACTIVE_ROTATION", False) and not self._custom_api_key and len(OpenRouterKeyManager.get_keys()) > 1:
            OpenRouterKeyManager.rotate()
            self._client = None

        current_key = self.api_key
        self._throttle_before_request()
        start_time = time.monotonic()

        try:
            logger.debug(f"[OpenRouter] Generating with model={self.model} using key={current_key[:8]}...")
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

            tokens = 0
            usage_dict = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            }
            if response.usage:
                tokens = (response.usage.prompt_tokens or 0) + (response.usage.completion_tokens or 0)
            OpenRouterKeyManager.mark_success(current_key, tokens=tokens)

            result_text = self._strip_reasoning(choice.message.content or "")
            latency = time.monotonic() - start_time
            log_llm_call(
                provider="openrouter",
                model=self.model,
                prompt=prompt,
                system=system,
                response=result_text,
                latency=latency,
                usage=usage_dict,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            return LLMResponse(
                content=result_text,
                model=self.model,
                provider="openrouter",
                usage=usage_dict,
                raw=response,
            )

        except Exception as e:
            latency = time.monotonic() - start_time
            log_llm_call(
                provider="openrouter",
                model=self.model,
                prompt=prompt,
                system=system,
                error=e,
                latency=latency,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            if isinstance(e, RateLimitError):
                current_key = self.api_key
                retry_after = self._mark_rate_limited(e)

                all_keys = OpenRouterKeyManager.get_keys()
                if (getattr(config, "OPENROUTER_ROTATE_ON_RATE_LIMIT", True) or len(all_keys) > 1) and not self._custom_api_key:
                    has_alt, min_wait, next_acc = OpenRouterKeyManager.mark_rate_limited(
                        key=current_key,
                        retry_after=retry_after,
                        model=self.model,
                    )
                    if has_alt and _retry_count < max(len(all_keys), 1) * 3:
                        logger.warning(
                            "[OpenRouter] Rate limited on key %s. Immediately rotating to %s (attempt %d).",
                            current_key[:8] + "...", next_acc.account_id if next_acc else "next account", _retry_count + 1,
                        )
                        self._client = None
                        return self.generate(
                            prompt, system, schema, temperature, max_tokens, stream,
                            _retry_count=_retry_count + 1,
                        )
                    if _retry_count < 3:
                        self._raise_if_cancelled()
                        logger.warning(
                            "[OpenRouter] All %d accounts in cooldown. Waiting %.1fs (attempt %d/3).",
                            len(all_keys), min_wait, _retry_count + 1,
                        )
                        self._sleep_interruptible(min_wait)
                        self._client = None
                        return self.generate(
                            prompt, system, schema, temperature, max_tokens, stream,
                            _retry_count=_retry_count + 1,
                        )

            if _retry_count < 5:
                self._raise_if_cancelled()
                logger.warning(
                    f"[OpenRouter] Rate limited — waiting {retry_after:.2f}s then retrying "
                    f"(attempt {_retry_count + 1}/5)..."
                )
                self._sleep_interruptible(retry_after)
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
        self._raise_if_cancelled()
        self._content_blocked = False
        messages = self._prepare_messages(prompt=prompt, system=system)

        self._throttle_before_request()

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=config.CLOUD_MODEL_PARAMS["temperature"] if temperature is None else temperature,
                top_p=config.CLOUD_MODEL_PARAMS["top_p"],
                max_tokens=max_tokens or config.CLOUD_MODEL_PARAMS["max_tokens"],
                stream=True,
            )
        except RateLimitError as e:
            current_key = self.api_key
            retry_after = self._mark_rate_limited(e)

            all_keys = OpenRouterKeyManager.get_keys()
            if (getattr(config, "OPENROUTER_ROTATE_ON_RATE_LIMIT", True) or len(all_keys) > 1) and not self._custom_api_key:
                has_alt, min_wait, next_acc = OpenRouterKeyManager.mark_rate_limited(
                    key=current_key,
                    retry_after=retry_after,
                    model=self.model,
                )
                if has_alt and _retry_count < max(len(all_keys), 1) * 3:
                    logger.warning(
                        "[OpenRouter] Streaming rate limited on key %s. Immediately rotating to %s (attempt %d).",
                        current_key[:8] + "...", next_acc.account_id if next_acc else "next account", _retry_count + 1,
                    )
                    self._client = None
                    yield from self.generate_streaming(
                        prompt, system, temperature, max_tokens,
                        _retry_count=_retry_count + 1,
                    )
                    return

            if _retry_count < 5:
                self._raise_if_cancelled()
                logger.warning(
                    f"[OpenRouter] Stream rate limited — waiting {retry_after:.2f}s "
                    f"then retrying (attempt {_retry_count + 1}/5)..."
                )
                self._sleep_interruptible(retry_after)
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
                self._raise_if_cancelled()
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
