"""
Google Gemini wrapper implementing LLMInterface.
Allows using Gemini as a primary story-generation backend,
just like Groq or llama.cpp.
"""
import json
import logging
import re
import time
from typing import Optional

from google import genai
from google.genai import types

import config
from .base import ContentBlockedError, LLMInterface, LLMResponse, ReasoningStreamFilter
from pipeline.errors import PipelineCancelledError

from .key_rotator import KeyRotator, AccountKey, KeyStatus
from logger import get_logger, log_llm_call

logger = get_logger("gemini")

_gemini_rotator = KeyRotator(
    provider="gemini",
    keys=config.GEMINI_API_KEYS,
    strategy=getattr(config, "KEY_ROTATION_STRATEGY", "least_loaded"),
)


class GeminiKeyManager:
    """Manages multi-account rotation and health tracking of Gemini API keys."""
    rotator = _gemini_rotator

    @classmethod
    def sync_keys(cls, keys=None):
        if keys is None:
            keys = config.GEMINI_API_KEYS
        else:
            config.GEMINI_API_KEYS = [k.strip() for k in keys if k.strip()]
            if config.GEMINI_API_KEYS:
                config.GEMINI_API_KEY = config.GEMINI_API_KEYS[0]
        cls.rotator.set_keys(keys)

    @classmethod
    def get_keys(cls) -> list:
        keys = cls.rotator.get_keys()
        if not keys and config.GEMINI_API_KEY:
            return [config.GEMINI_API_KEY]
        return keys

    @classmethod
    def get_key(cls, proactive_rotate: bool = False, model: Optional[str] = None, estimated_tokens: int = 0) -> str:
        key = cls.rotator.get_key(proactive_rotate=proactive_rotate, model=model, estimated_tokens=estimated_tokens)
        if not key:
            return config.GEMINI_API_KEY
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


_MODELS_WITHOUT_SAMPLING_PARAMS = ("gemini-3.6-flash", "gemini-3.5-flash-lite")

_UNCENSORED_SAFETY_SETTINGS = [
    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_CIVIC_INTEGRITY", threshold="BLOCK_NONE"),
]


def _model_supports_sampling_params(model: str) -> bool:
    """Gemini 3.6 Flash and 3.5 Flash-Lite ignore temperature/top_p/top_k."""
    if not model:
        return True
    return not any(model.startswith(prefix) for prefix in _MODELS_WITHOUT_SAMPLING_PARAMS)


def _model_supports_thinking_budget(model: str) -> bool:
    """Check if model supports thinking_config with thinking_budget."""
    if not model:
        return False
    if "lite" in model.lower():
        return False
    return any(p in model.lower() for p in ("flash", "pro"))


class GeminiModel(ReasoningStreamFilter, LLMInterface):
    """Cloud LLM via Google Gemini API."""

    _last_request_time = 0.0
    _MIN_REQUEST_INTERVAL = 2.0  # Gemini has generous limits but still throttle

    def __init__(self, model: str = None, api_key: str = None):
        self.model = model or config.GEMINI_MODEL
        self._custom_api_key = api_key
        self._client = None
        self._client_key = None
        self._content_blocked = False

    @property
    def api_key(self) -> str:
        return self._custom_api_key or GeminiKeyManager.get_key()

    @property
    def client(self):
        current_key = self.api_key
        if self._client is not None and getattr(self, '_client_key', None) is not None and self._client_key != current_key:
            self._client = None

        if self._client is None:
            if not current_key:
                raise RuntimeError(
                    "GEMINI_API_KEY not set. Get a key at https://aistudio.google.com/app/apikey "
                    "and add it to your .env file."
                )
            if getattr(self, "_fast_support", False):
                self._client = genai.Client(api_key=current_key, http_options=types.HttpOptions(
                    timeout=15000, retry_options=types.HttpRetryOptions(attempts=1)))
            else:
                self._client = genai.Client(api_key=current_key)
            self._client_key = current_key
        return self._client

    @property
    def was_content_blocked(self) -> bool:
        return self._content_blocked

    def _throttle(self) -> None:
        now = time.time()
        elapsed = now - GeminiModel._last_request_time
        wait = max(0.0, self._MIN_REQUEST_INTERVAL - elapsed)
        if wait > 0:
            logger.debug("[Gemini] Throttling %.1fs before next request.", wait)
            self._sleep_interruptible(wait)
        GeminiModel._last_request_time = time.time()

    def _build_contents(self, prompt: str, system: str = "") -> tuple:
        """Build Gemini-compatible contents and config."""
        gen_config = {}
        system_instruction = None

        if system:
            system_instruction = system

        return prompt, system_instruction, gen_config

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
                )
            )
            return LLMResponse(
                content=content,
                model=self.model,
                provider="gemini",
                usage={},
                raw=None,
            )

        self._raise_if_cancelled()
        self._content_blocked = False
        self._throttle()

        # Build config
        gen_config_kwargs = {
            "safety_settings": _UNCENSORED_SAFETY_SETTINGS,
        }
        if _model_supports_thinking_budget(self.model):
            gen_config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        if _model_supports_sampling_params(self.model):
            if temperature is not None:
                gen_config_kwargs["temperature"] = temperature
            else:
                gen_config_kwargs["temperature"] = config.CLOUD_MODEL_PARAMS.get("temperature", 0.8)

        if max_tokens:
            gen_config_kwargs["max_output_tokens"] = max_tokens
        else:
            gen_config_kwargs["max_output_tokens"] = config.CLOUD_MODEL_PARAMS.get("max_tokens", 4096)

        # Handle JSON schema mode
        if schema:
            schema_instruction = (
                "\n\nReturn ONLY minified valid JSON matching this schema. "
                "No markdown, no prose, no reasoning.\n"
                f"Schema: {json.dumps(schema, separators=(',', ':'))}"
            )
            system = f"{system.rstrip()}\n{schema_instruction}" if system else schema_instruction.strip()
            gen_config_kwargs["response_mime_type"] = "application/json"

        # Build the request
        config_obj = types.GenerateContentConfig(
            system_instruction=system if system else None,
            **gen_config_kwargs,
        )

        start_time = time.monotonic()
        try:
            logger.debug(f"[Gemini] Generating with model={self.model}")
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config_obj,
            )

            text = response.text or ""
            # If response.text is empty (e.g. thought flag was attached), extract parts directly
            if not text and hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content and hasattr(candidate.content, 'parts'):
                    parts_text = []
                    for part in candidate.content.parts:
                        if hasattr(part, 'text') and part.text:
                            parts_text.append(part.text)
                    if parts_text:
                        text = "".join(parts_text)

            text = self._strip_reasoning(text)

            finish_reason = None
            if hasattr(response, 'candidates') and response.candidates:
                finish_reason = getattr(response.candidates[0], 'finish_reason', None)
            
            block_reason = None
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                block_reason = getattr(response.prompt_feedback, 'block_reason', None)

            # Check for blocked or empty content
            if not text.strip():
                logger.warning(
                    "[Gemini:%s] Model returned empty text! finish_reason=%s, block_reason=%s, candidates_count=%d",
                    self.model, finish_reason, block_reason, len(response.candidates or [])
                )
                if str(finish_reason).upper() == "SAFETY" or str(block_reason).upper() == "SAFETY":
                    self._content_blocked = True
                    raise ContentBlockedError("Gemini content safety blocked this request.")

                # If empty, attempt retry with gemini-2.5-flash
                if _retry_count < 2:
                    fallback_model = "gemini-2.5-flash" if self.model != "gemini-2.5-flash" else "gemini-3.5-flash"
                    logger.info("[Gemini] Retrying empty output with fallback model %s...", fallback_model)
                    orig_model = self.model
                    try:
                        self.model = fallback_model
                        return self.generate(
                            prompt=prompt,
                            system=system,
                            schema=schema,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            stream=stream,
                            _retry_count=_retry_count + 1,
                        )
                    finally:
                        self.model = orig_model

                raise RuntimeError(
                    f"Gemini model '{self.model}' returned an empty response (finish_reason={finish_reason}, block_reason={block_reason})."
                )

            current_key = self.api_key
            if _retry_count == 0 and getattr(config, "GEMINI_PROACTIVE_ROTATION", False) and not self._custom_api_key and len(GeminiKeyManager.get_keys()) > 1:
                GeminiKeyManager.rotate()
                self._client = None

            usage = {}
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage = {
                    "prompt_tokens": getattr(response.usage_metadata, 'prompt_token_count', 0) or 0,
                    "completion_tokens": getattr(response.usage_metadata, 'candidates_token_count', 0) or 0,
                }

            completion_tokens = usage.get("completion_tokens") or 0
            GeminiKeyManager.mark_success(current_key, tokens=completion_tokens)

            latency = time.monotonic() - start_time
            log_llm_call(
                provider="gemini",
                model=self.model,
                prompt=prompt,
                system=system,
                response=text,
                latency=latency,
                usage=usage,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            return LLMResponse(
                content=text,
                model=self.model,
                provider="gemini",
                usage=usage,
                raw=response,
            )

        except ContentBlockedError:
            raise
        except Exception as e:
            latency = time.monotonic() - start_time
            current_key = self.api_key
            error_str = str(e).lower()

            log_llm_call(
                provider="gemini",
                model=self.model,
                prompt=prompt,
                system=system,
                error=e,
                latency=latency,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            if "safety" in error_str or "blocked" in error_str:
                self._content_blocked = True
                raise ContentBlockedError(f"Content blocked by Gemini: {e}")

            # 1. Handle transient spikes: 503 Unavailable / high demand / timeouts
            is_transient = any(phrase in error_str for phrase in (
                "503", "unavailable", "high demand", "spikes in demand", "overloaded",
                "500", "internal", "504", "deadline_exceeded", "timeout",
            ))
            if is_transient and _retry_count < 3:
                all_keys = GeminiKeyManager.get_keys()
                if len(all_keys) > 1 and not self._custom_api_key:
                    GeminiKeyManager.rotate()
                    self._client = None
                wait = 2.0 * (_retry_count + 1)
                logger.warning(
                    "[Gemini] Server transient spike (%s). Retrying in %.1fs (attempt %d/3)...",
                    error_str[:60], wait, _retry_count + 1,
                )
                self._sleep_interruptible(wait)
                self._client = None
                return self.generate(
                    prompt, system, schema, temperature, max_tokens, stream,
                    _retry_count=_retry_count + 1,
                )

            # 2. Handle 429 / Quota / Rate limit
            if "quota" in error_str or "rate" in error_str or "resource_exhausted" in error_str or "429" in error_str:
                if getattr(self, "_fast_support", False):
                    raise
                match = re.search(r"retry after (\d+)", error_str)
                wait = float(match.group(1)) if match else 30.0
                all_keys = GeminiKeyManager.get_keys()
                if (getattr(config, "GEMINI_ROTATE_ON_RATE_LIMIT", True) or len(all_keys) > 1) and not self._custom_api_key:
                    has_alt, min_wait, next_acc = GeminiKeyManager.mark_rate_limited(
                        key=current_key,
                        retry_after=wait,
                        model=self.model,
                    )
                    if has_alt and _retry_count < max(len(all_keys), 1) * 3:
                        logger.warning(
                            "[Gemini] Rate limited on key %s. Immediately rotating to %s (attempt %d).",
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
                            "[Gemini] All %d Gemini accounts in cooldown. Waiting %.1fs (attempt %d/3).",
                            len(all_keys), min_wait, _retry_count + 1,
                        )
                        self._sleep_interruptible(min_wait)
                        self._client = None
                        return self.generate(
                            prompt, system, schema, temperature, max_tokens, stream,
                            _retry_count=_retry_count + 1,
                        )
                logger.warning(f"[Gemini] Rate limited — waiting {wait}s before propagating error...")
                self._sleep_interruptible(wait)
                raise
            raise

    def generate_streaming(
        self,
        prompt: str,
        system: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        """Streaming generator — yields content chunks."""
        self._raise_if_cancelled()
        self._content_blocked = False
        self._throttle()

        gen_config_kwargs = {}
        if getattr(self, "_fast_support", False) and self.model.startswith("gemini-2.5-flash"):
            gen_config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        if _model_supports_sampling_params(self.model):
            if temperature is not None:
                gen_config_kwargs["temperature"] = temperature
            else:
                gen_config_kwargs["temperature"] = config.CLOUD_MODEL_PARAMS.get("temperature", 0.8)

        if max_tokens:
            gen_config_kwargs["max_output_tokens"] = max_tokens
        else:
            gen_config_kwargs["max_output_tokens"] = config.CLOUD_MODEL_PARAMS.get("max_tokens", 4096)

        config_obj = types.GenerateContentConfig(
            system_instruction=system if system else None,
            **gen_config_kwargs,
        )

        max_retries = 1 if getattr(self, "_fast_support", False) else 5
        delay = config.RETRY_BASE_DELAY
        self.last_usage = {}
        emitted = False
        
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content_stream(
                    model=self.model,
                    contents=prompt,
                    config=config_obj,
                )

                def raw_chunks():
                    nonlocal emitted
                    for chunk in response:
                        self._raise_if_cancelled()
                        usage = getattr(chunk, "usage_metadata", None)
                        if usage:
                            self.last_usage = {
                                "prompt_tokens": getattr(usage, "prompt_token_count", 0),
                                "completion_tokens": getattr(usage, "candidates_token_count", 0),
                            }
                        if hasattr(chunk, "text") and chunk.text:
                            emitted = True
                            yield chunk.text

                # Reasoning tags can be split across provider chunks.  Filter
                # the stream as one logical sequence and preserve whitespace.
                yield from self._filter_reasoning_stream(raw_chunks())
                # Break on success
                break

            except Exception as e:
                if isinstance(e, PipelineCancelledError) or emitted:
                    raise
                error_str = str(e).lower()
                if "safety" in error_str or "blocked" in error_str:
                    self._content_blocked = True
                    raise
                
                is_rate_limit = "503" in error_str or "unavailable" in error_str or "quota" in error_str or "rate" in error_str or "resource_exhausted" in error_str
                if is_rate_limit and not self._custom_api_key and len(GeminiKeyManager.get_keys()) > 1:
                    match = re.search(r"retry after (\d+)", error_str)
                    wait = float(match.group(1)) if match else 15.0
                    has_alt, min_wait, next_acc = GeminiKeyManager.mark_rate_limited(
                        key=self.api_key, retry_after=wait, model=self.model
                    )
                    if has_alt:
                        logger.warning(
                            "[Gemini] Streaming rate limited on %s. Intelligently swapping to %s.",
                            self.api_key[:8] + "...", next_acc.account_id if next_acc else "next account"
                        )
                        self._client = None
                        yield from self.generate_streaming(prompt, system, temperature, max_tokens)
                        return

                is_transient = is_rate_limit
                if is_transient and attempt < max_retries - 1:
                    wait = delay
                    if "rate" in error_str or "quota" in error_str:
                        match = re.search(r"retry after (\d+)", error_str)
                        wait = float(match.group(1)) if match else 15.0
                    self._raise_if_cancelled()
                    logger.warning(
                        f"[Gemini] Streaming attempt {attempt + 1}/{max_retries} failed: {e}. "
                        f"Retrying in {wait:.1f}s..."
                    )
                    self._sleep_interruptible(wait)
                    delay = min(delay * config.RETRY_BACKOFF_FACTOR, config.RETRY_MAX_DELAY)
                else:
                    raise

    def is_available(self) -> bool:
        """Check if Gemini API is reachable with a minimal request."""
        if not self.api_key:
            logger.warning("[Gemini] No API key configured.")
            return False
            
        max_retries = 5
        delay = 2.0
        
        for attempt in range(max_retries):
            try:
                config_obj = types.GenerateContentConfig(
                    max_output_tokens=40,
                )
                response = self.client.models.generate_content(
                    model=self.model,
                    contents="Say hello.",
                    config=config_obj,
                )
                return response is not None and len(response.candidates) > 0
            except Exception as e:
                error_str = str(e).lower()
                is_transient = "503" in error_str or "unavailable" in error_str or "quota" in error_str or "rate" in error_str
                if is_transient and attempt < max_retries - 1:
                    logger.warning(
                        f"[Gemini] Availability check attempt {attempt + 1}/{max_retries} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    # Interruptible so a cancel during startup isn't stuck sleeping.
                    self._sleep_interruptible(delay)
                    delay *= 2.0
                else:
                    logger.warning(f"[Gemini] Availability check failed: {e}")
                    return False
        return False

    def get_name(self) -> str:
        return f"Gemini ({self.model})"
