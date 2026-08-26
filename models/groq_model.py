"""
Groq Cloud wrapper using OpenAI-compatible SDK.
Primary backend for cloud prose generation.
"""
import collections
import json
import logging
import threading
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
from pipeline.errors import PipelineCancelledError

logger = logging.getLogger(__name__)


class GroqKeyManager:
    """Manages rotation of multiple Groq API keys."""
    _current_idx = 0
    _rotate_lock = threading.Lock()

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
        with cls._rotate_lock:
            keys = cls.get_keys()
            if not keys or len(keys) < 2:
                return False
            cls._current_idx = (cls._current_idx + 1) % len(keys)
            logger.info(f"Rotating to Groq API key #{cls._current_idx + 1} ({cls.get_key()[:8]}...)")
            return True


class GroqModel(ReasoningStreamFilter, LLMInterface):
    """Cloud LLM via Groq API. Fast inference on large models."""

    def __init__(self, model: str = None, api_key: str = None):
        self.model = model or config.GROQ_MODEL
        self._custom_api_key = api_key
        self._client = None
        self._content_blocked = False

    # Shared, thread-safe cooldown / min-interval bookkeeping.
    _rl = RateLimitTracker()

    # ─── Adaptive TPM budget ─────────────────────────────────────────────────
    # Groq quotas are PER (model, key): llama-3.3-70b and qwen3 each have their
    # own TPM window on the same key. Every tracker below is keyed by
    # (model, key) so switching models never inherits another model's budget.
    _tpm_lock = threading.Lock()
    _tpm_usage = {}  # (model, key) -> deque[(timestamp, tokens)]

    @classmethod
    def _tpm_now(cls) -> float:
        return time.time()

    def _track_id(self, key: Optional[str] = None) -> str:
        """Composite identity for rate-limit trackers: (model, key)."""
        return f"{self.model}::{key or self.api_key or 'default'}"

    @classmethod
    def _prune_tpm(cls, track_id: str, now: float) -> None:
        cutoff = now - config.GROQ_TPM_WINDOW_SECONDS
        dq = cls._tpm_usage.get(track_id)
        if not dq:
            return
        while dq and dq[0][0] < cutoff:
            dq.popleft()

    @classmethod
    def _record_tpm_tokens(cls, track_id: str, tokens: float) -> None:
        if not tokens:
            return
        now = cls._tpm_now()
        with cls._tpm_lock:
            dq = cls._tpm_usage.setdefault(track_id, collections.deque())
            cls._prune_tpm(track_id, now)
            dq.append((now, tokens))

    @classmethod
    def _tpm_usage_in_window(cls, track_id: str) -> float:
        with cls._tpm_lock:
            cls._prune_tpm(track_id, cls._tpm_now())
            return sum(tokens for _, tokens in cls._tpm_usage.get(track_id, ()))

    @staticmethod
    def _estimate_prompt_tokens(prompt: str, system: str = "") -> int:
        text = f"{system}\n{prompt}" if system else (prompt or "")
        return max(1, int(len(text) / 4.0))

    @staticmethod
    def _estimate_completion_tokens(text: str) -> int:
        return max(0, int(len(text or "") / 4.0))

    def _wait_for_tpm_budget(self, track_id: str, estimated: int) -> None:
        """Sleep until the rolling window + this request stays under the TPM limit."""
        if not config.GROQ_TPM_PACING:
            return
        limit = config.GROQ_TPM_LIMIT
        budget = limit * (1.0 - config.GROQ_TPM_SAFETY_MARGIN)
        tokens_per_sec = limit / max(1.0, config.GROQ_TPM_WINDOW_SECONDS)
        start = time.time()
        while time.time() - start < config.GROQ_TPM_WINDOW_SECONDS:
            self._raise_if_cancelled()
            usage = self._tpm_usage_in_window(track_id)
            if usage + estimated <= budget:
                break
            excess = usage + estimated - budget
            wait = excess / tokens_per_sec
            logger.info(
                "[Groq] TPM pacing: waiting %.1fs (usage %.0f + %d > %.0f TPM budget).",
                wait, usage, estimated, budget,
            )
            self._sleep_interruptible(min(max(wait, 0.5), 10.0))
        # Reserve the estimate for the duration of the in-flight request.
        self._record_tpm_tokens(track_id, estimated)

    def _settle_tpm(self, track_id: str, reservation: float, actual: float) -> None:
        """Correct the budget after a request: replace the estimate with real usage."""
        if not config.GROQ_TPM_PACING:
            return
        self._record_tpm_tokens(track_id, (actual or 0.0) - reservation)

    def _rotate_to_most_available_key(self) -> None:
        """Preemptively pick the key with the most remaining TPM budget for this model."""
        if not config.GROQ_PROACTIVE_ROTATION:
            return
        keys = GroqKeyManager.get_keys()
        if len(keys) < 2:
            return
        now = time.time()
        candidates = [
            k for k in keys
            if self._rl.cooldown_expiry(self._track_id(k)) <= now
        ]
        if not candidates:
            return
        best = min(candidates, key=lambda k: GroqModel._tpm_usage_in_window(self._track_id(k)))
        best_idx = keys.index(best)
        if best_idx != GroqKeyManager._current_idx:
            GroqKeyManager._current_idx = best_idx
            logger.info(
                "[Groq] Proactively rotated to key #%d (most TPM budget available).",
                best_idx + 1,
            )

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

    def _retry_after_with_buffer(self, error: RateLimitError) -> float:
        """Parsed Retry-After hint plus the configured safety buffer."""
        return parse_retry_after_seconds(error) + config.GROQ_RATE_LIMIT_BUFFER

    def _throttle_before_request(
        self, prompt: str = "", system: str = "", max_tokens: Optional[int] = None,
    ) -> float:
        """Apply RPM/cooldown + adaptive TPM pacing. Returns the reserved estimate."""
        self._rotate_to_most_available_key()
        track_id = self._track_id()
        now = time.time()
        wait = self._rl.throttle_wait(track_id, now, config.GROQ_MIN_REQUEST_INTERVAL)
        if wait > 0:
            logger.info("[Groq] Throttling %.1fs before next request.", wait)
            self._sleep_interruptible(wait)
        self._rl.record_request(track_id, time.time())

        if config.GROQ_TPM_PACING:
            estimated = (
                self._estimate_prompt_tokens(prompt, system)
                + (max_tokens or config.GROQ_TPM_RESERVE_DEFAULT)
            )
            self._wait_for_tpm_budget(track_id, estimated)
            return float(estimated)
        return 0.0

    def _mark_rate_limited(self, error: RateLimitError) -> float:
        retry_after = self._retry_after_with_buffer(error)
        track_id = self._track_id()
        self._rl.mark_cooldown(track_id, time.time() + retry_after)
        logger.warning(
            "[Groq] Rate limited on %s; cooling it down for %.0fs.",
            track_id,
            retry_after,
        )
        return retry_after

    def _rotate_to_available_key(self) -> bool:
        if not config.GROQ_ROTATE_ON_RATE_LIMIT:
            return False
        keys = GroqKeyManager.get_keys()
        if len(keys) < 2:
            return False

        for _ in range(len(keys) - 1):
            if not GroqKeyManager.rotate():
                return False
            if self._rl.cooldown_expiry(self._track_id()) <= time.time():
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
            "temperature": temperature or config.CLOUD_MODEL_PARAMS["temperature"],
            "top_p": config.CLOUD_MODEL_PARAMS["top_p"],
            "max_tokens": max_tokens or config.CLOUD_MODEL_PARAMS["max_tokens"],
        }
        qwen_reasoning_params = self._qwen_reasoning_params()
        if qwen_reasoning_params:
            kwargs["extra_body"] = qwen_reasoning_params

        if schema and self._use_strict_json_mode():
            kwargs["response_format"] = {"type": "json_object"}

        track_id = self._track_id()
        reservation = self._throttle_before_request(prompt, system, kwargs.get("max_tokens"))

        try:
            logger.debug(f"[Groq] Generating with model={self.model}")
            response = self.client.chat.completions.create(**kwargs)

            choice = response.choices[0]

            # Check for content filter
            if choice.finish_reason == "content_filter":
                self._content_blocked = True
                self._settle_tpm(track_id, reservation, 0.0)
                raise ContentBlockedError(
                    "Groq content moderation blocked this request."
                )
            if choice.finish_reason == "length":
                logger.warning(
                    "[Groq] Response hit max_tokens. Partial content returned."
                )

            actual = 0.0
            if response.usage:
                actual = (response.usage.prompt_tokens or 0) + (response.usage.completion_tokens or 0)
            self._settle_tpm(track_id, reservation, actual)

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
            self._settle_tpm(track_id, reservation, 0.0)
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
                self._raise_if_cancelled()
                logger.warning(f"[Groq] Rate limited — waiting {retry_after:.2f}s then retrying (attempt {_retry_count + 1}/5)...")
                self._sleep_interruptible(retry_after)
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
        self._raise_if_cancelled()
        self._content_blocked = False

        stream, track_id, reservation = self._open_stream(
            prompt, system, temperature, max_tokens, _retry_count,
        )

        finish_reason = None
        streamed_text = []

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
                    streamed_text.append(content)
                    yield content
            if finish_reason == "length":
                logger.warning(
                    "[Groq] Streaming response hit max_tokens. Partial content returned."
                )
            self._settle_tpm(
                track_id,
                reservation,
                self._estimate_prompt_tokens(prompt, system)
                + self._estimate_completion_tokens("".join(streamed_text)),
            )

        except APIError as e:
            if hasattr(e, 'status_code') and e.status_code == 400:
                self._content_blocked = True
            self._settle_tpm(track_id, reservation, 0.0)
            raise

    def _open_stream(self, prompt, system, temperature, max_tokens, _retry_count):
        """Throttle once and open a single streaming request (with 429 retry)."""
        track_id = self._track_id()
        reservation = self._throttle_before_request(prompt, system, max_tokens)

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=self._prepare_messages(prompt=prompt, system=system),
                temperature=temperature or config.CLOUD_MODEL_PARAMS["temperature"],
                top_p=config.CLOUD_MODEL_PARAMS["top_p"],
                max_tokens=max_tokens or config.CLOUD_MODEL_PARAMS["max_tokens"],
                stream=True,
                extra_body=self._qwen_reasoning_params() or None,
            )
            return stream, track_id, reservation
        except RateLimitError as e:
            self._settle_tpm(track_id, reservation, 0.0)
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
                return self._open_stream(
                    prompt, system, temperature, max_tokens, _retry_count + 1,
                )

            if _retry_count < 5:
                self._raise_if_cancelled()
                logger.warning(f"[Groq] Stream rate limited — waiting {retry_after:.2f}s then retrying (attempt {_retry_count + 1}/5)...")
                self._sleep_interruptible(retry_after)
                return self._open_stream(
                    prompt, system, temperature, max_tokens, _retry_count + 1,
                )

            logger.error("[Groq] Rate limit persisted after 5 retries on stream. Raising.")
            raise

