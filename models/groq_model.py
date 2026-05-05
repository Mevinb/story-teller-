"""
Groq Cloud wrapper using OpenAI-compatible SDK.
Primary backend for heavy prose generation (llama-3.3-70b-versatile).
"""
import json
import logging
from typing import Optional

from openai import OpenAI, APIError, RateLimitError, APIConnectionError

import config
from .base import LLMInterface, LLMResponse

logger = logging.getLogger(__name__)


class GroqModel(LLMInterface):
    """Cloud LLM via Groq API. Fast inference on large models."""

    def __init__(self, model: str = None, api_key: str = None):
        self.model = model or config.GROQ_MODEL
        self.api_key = api_key or config.GROQ_API_KEY
        self._client = None
        self._content_blocked = False

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            if not self.api_key:
                raise RuntimeError(
                    "GROQ_API_KEY not set. Get a free key at https://console.groq.com "
                    "and add it to your .env file."
                )
            self._client = OpenAI(
                base_url=config.GROQ_BASE_URL,
                api_key=self.api_key,
                timeout=30.0,  # 30s timeout to prevent hanging on content blocks
            )
        return self._client

    @property
    def was_content_blocked(self) -> bool:
        """Returns True if the last request was blocked by content moderation."""
        return self._content_blocked

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
                provider="groq",
                usage={},
                raw=None,
            )

        self._content_blocked = False
        messages = []
        if system:
            messages.append({"role": "system", "content": system})

        # If schema requested, instruct the model to output JSON
        if schema:
            schema_instruction = (
                "\n\nYou MUST respond with ONLY valid JSON matching this schema. "
                "No additional text, no markdown fences.\n"
                f"Schema: {json.dumps(schema, indent=2)}"
            )
            if system:
                messages[0]["content"] += schema_instruction
            else:
                messages.append({"role": "system", "content": schema_instruction.strip()})

        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature or config.CLOUD_MODEL_PARAMS["temperature"],
            "top_p": config.CLOUD_MODEL_PARAMS["top_p"],
            "max_tokens": max_tokens or config.CLOUD_MODEL_PARAMS["max_tokens"],
        }

        if schema:
            kwargs["response_format"] = {"type": "json_object"}

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

            return LLMResponse(
                content=choice.message.content or "",
                model=self.model,
                provider="groq",
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                },
                raw=response,
            )

        except RateLimitError as e:
            import re, time as _time
            error_str = str(e)

            # Check if this is a DAILY token limit (waiting won't help)
            if 'tokens per day' in error_str.lower() or 'TPD' in error_str:
                logger.error(f"[Groq] Daily token limit reached. Falling back to local model.")
                raise  # Immediately trigger local fallback

            # Per-minute/request rate limit — wait and retry
            retry_after = 60
            match = re.search(r'try again in\s+(\d+)m', error_str, re.IGNORECASE)
            if match:
                retry_after = int(match.group(1)) * 60 + 10  # minutes to seconds + buffer
            else:
                match = re.search(r'try again in\s+(\d+\.?\d*)', error_str, re.IGNORECASE)
                if match:
                    retry_after = max(float(match.group(1)) + 5, 30)

            logger.warning(f"[Groq] Rate limited — waiting {retry_after:.0f}s then retrying...")
            _time.sleep(retry_after)
            try:
                response = self.client.chat.completions.create(**kwargs)
                choice = response.choices[0]
                if choice.finish_reason == "content_filter":
                    self._content_blocked = True
                    raise ContentBlockedError("Groq content moderation blocked this request.")
                return LLMResponse(
                    content=choice.message.content or "",
                    model=self.model, provider="groq",
                    usage={"prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                           "completion_tokens": response.usage.completion_tokens if response.usage else 0},
                    raw=response,
                )
            except RateLimitError:
                logger.error("[Groq] Still rate limited after waiting. Raising to trigger local fallback.")
                raise
        except APIError as e:
            # Check if this is a content moderation error
            if hasattr(e, 'status_code') and e.status_code == 400:
                error_msg = str(e).lower()
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
    ):
        """
        Streaming generator — yields content chunks.
        Used by the web UI for real-time output.
        """
        self._content_blocked = False
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or config.CLOUD_MODEL_PARAMS["temperature"],
                top_p=config.CLOUD_MODEL_PARAMS["top_p"],
                max_tokens=max_tokens or config.CLOUD_MODEL_PARAMS["max_tokens"],
                stream=True,
            )

            for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    yield delta.content

        except APIError as e:
            if hasattr(e, 'status_code') and e.status_code == 400:
                self._content_blocked = True
            raise


class ContentBlockedError(Exception):
    """Raised when Groq's content moderation blocks a request."""
    pass
