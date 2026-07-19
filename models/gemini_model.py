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
from .base import LLMInterface, LLMResponse

logger = logging.getLogger(__name__)

_REASONING_BLOCK_RE = re.compile(
    r"<(?:think|analysis|reasoning)>.*?</(?:think|analysis|reasoning)>",
    flags=re.IGNORECASE | re.DOTALL,
)
_REASONING_TAG_RE = re.compile(r"</?(?:think|analysis|reasoning)>", flags=re.IGNORECASE)


class GeminiModel(LLMInterface):
    """Cloud LLM via Google Gemini API."""

    _last_request_time = 0.0
    _MIN_REQUEST_INTERVAL = 2.0  # Gemini has generous limits but still throttle

    def __init__(self, model: str = None, api_key: str = None):
        self.model = model or config.GEMINI_MODEL
        self._api_key = api_key or config.GEMINI_API_KEY
        self._client = None
        self._content_blocked = False

    @property
    def client(self):
        if self._client is None:
            if not self._api_key:
                raise RuntimeError(
                    "GEMINI_API_KEY not set. Get a key at https://aistudio.google.com/app/apikey "
                    "and add it to your .env file."
                )
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    @staticmethod
    def _strip_reasoning(text: str) -> str:
        cleaned = _REASONING_BLOCK_RE.sub("", text or "")
        cleaned = _REASONING_TAG_RE.sub("", cleaned)
        return cleaned.strip()

    @property
    def was_content_blocked(self) -> bool:
        return self._content_blocked

    def _throttle(self) -> None:
        now = time.time()
        elapsed = now - GeminiModel._last_request_time
        wait = max(0.0, self._MIN_REQUEST_INTERVAL - elapsed)
        if wait > 0:
            logger.debug("[Gemini] Throttling %.1fs before next request.", wait)
            time.sleep(wait)
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

        self._content_blocked = False
        self._throttle()

        # Build config
        gen_config_kwargs = {}
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

        try:
            logger.debug(f"[Gemini] Generating with model={self.model}")
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config_obj,
            )

            text = response.text or ""
            text = self._strip_reasoning(text)

            # Check for blocked content
            if not text and hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason') and candidate.finish_reason == 'SAFETY':
                    self._content_blocked = True
                    raise ContentBlockedError("Gemini content safety blocked this request.")

            usage = {}
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage = {
                    "prompt_tokens": getattr(response.usage_metadata, 'prompt_token_count', 0),
                    "completion_tokens": getattr(response.usage_metadata, 'candidates_token_count', 0),
                }

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
            error_str = str(e).lower()
            if "safety" in error_str or "blocked" in error_str:
                self._content_blocked = True
                raise ContentBlockedError(f"Content blocked by Gemini: {e}")
            if "quota" in error_str or "rate" in error_str:
                # Extract retry-after if available
                match = re.search(r"retry after (\d+)", error_str)
                wait = float(match.group(1)) if match else 30.0
                logger.warning(f"[Gemini] Rate limited — waiting {wait}s before propagating error...")
                time.sleep(wait)
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
        self._content_blocked = False
        self._throttle()

        gen_config_kwargs = {}
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

        max_retries = 5
        delay = config.RETRY_BASE_DELAY
        
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content_stream(
                    model=self.model,
                    contents=prompt,
                    config=config_obj,
                )

                for chunk in response:
                    if hasattr(chunk, 'text') and chunk.text:
                        cleaned = self._strip_reasoning(chunk.text)
                        if cleaned:
                            yield cleaned
                # Break on success
                break

            except Exception as e:
                error_str = str(e).lower()
                if "safety" in error_str or "blocked" in error_str:
                    self._content_blocked = True
                    raise
                
                is_transient = "503" in error_str or "unavailable" in error_str or "quota" in error_str or "rate" in error_str
                if is_transient and attempt < max_retries - 1:
                    wait = delay
                    if "rate" in error_str or "quota" in error_str:
                        match = re.search(r"retry after (\d+)", error_str)
                        wait = float(match.group(1)) if match else 15.0
                    logger.warning(
                        f"[Gemini] Streaming attempt {attempt + 1}/{max_retries} failed: {e}. "
                        f"Retrying in {wait:.1f}s..."
                    )
                    time.sleep(wait)
                    delay = min(delay * config.RETRY_BACKOFF_FACTOR, config.RETRY_MAX_DELAY)
                else:
                    raise

    def is_available(self) -> bool:
        """Check if Gemini API is reachable with a minimal request."""
        if not self._api_key:
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
                    time.sleep(delay)
                    delay *= 2.0
                else:
                    logger.warning(f"[Gemini] Availability check failed: {e}")
                    return False
        return False

    def get_name(self) -> str:
        return f"Gemini ({self.model})"


class ContentBlockedError(Exception):
    """Raised when Gemini's content safety blocks a request."""
    pass
