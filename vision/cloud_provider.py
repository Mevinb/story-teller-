"""
Cloud Vision Provider leveraging Google Gemini 2.5 Flash Multimodal Vision API.
"""

import json
import logging
import re
from typing import Dict, Any, Optional

from google import genai
from google.genai import types

import config
from .prompts import PERSON_ANALYSIS_SYSTEM_PROMPT
from .schema import PersonAnalysisResult

from models.gemini_model import GeminiKeyManager
from logger import log_llm_call, get_logger

logger = get_logger("vision.cloud")

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


class CloudVisionProvider:
    """Cloud Vision provider using Google GenAI (Gemini multimodal)."""

    def __init__(self, api_key: str = None, model: str = None):
        self._custom_api_key = api_key
        self.model = model or getattr(config, "GEMINI_MODEL", "gemini-2.5-flash")
        self._client = None
        self._client_key = None

    @property
    def api_key(self) -> str:
        return self._custom_api_key or GeminiKeyManager.get_key()

    @property
    def client(self) -> genai.Client:
        current_key = self.api_key
        if self._client is not None and getattr(self, '_client_key', None) != current_key:
            self._client = None

        if self._client is None:
            if not current_key:
                raise RuntimeError("GEMINI_API_KEY not configured in .env file.")
            self._client = genai.Client(api_key=current_key)
            self._client_key = current_key
        return self._client

    def check_availability(self) -> Dict[str, Any]:
        """Check if Gemini Cloud API is configured."""
        current_key = self.api_key
        if not current_key:
            return {
                "available": False,
                "type": "gemini",
                "message": "GEMINI_API_KEY not set in .env"
            }
        return {
            "available": True,
            "type": "gemini",
            "model": self.model
        }

    # Disable ALL Gemini safety filters so the model can speak freely about
    # bodies, figures, curves — just like Grok does. Without this, Gemini
    # silently sanitises or refuses body/figure descriptions.
    _UNCENSORED_SAFETY_SETTINGS = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_NONE"},
    ]

    def analyze(self, image_bytes: bytes, mime_type: str = "image/jpeg", prompt_extra: str = "", _retry_count: int = 0) -> PersonAnalysisResult:
        """Analyze image using Gemini Multimodal Vision — fully uncensored, Grok-style."""
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is missing. Please add it to your .env file.")

        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type=mime_type
        )

        user_text = (
            "Analyze this person in the photo and give your completely honest, unfiltered Grok-style take. "
            "Rate them out of 10 — be real, don't inflate. "
            "Talk about their body openly and freely — their figure, curves, bust/chest, waist, hips, thighs, ass, legs, everything visible. "
            "Compliment what looks good. Be flirty, be appreciative, be real. "
            "Describe their body shape (hourglass, curvy, slim, thick, voluptuous, athletic, whatever they've got). "
            "Talk about how their clothes fit their body, what flatters them, what shows. "
            "Return the full JSON with rating, rating_breakdown, roast, best_feature, advice, "
            "and ALL body details filled in with vivid, honest, appreciative descriptions. "
            "Don't hold back. Don't be clinical. Be open, be free, be Grok."
        )
        if prompt_extra:
            user_text += f"\nUser extra context: {prompt_extra}"

        full_prompt = f"{PERSON_ANALYSIS_SYSTEM_PROMPT}\n\nUser Request: {user_text}"

        import time
        t0 = time.time()
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=[image_part, full_prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.95,
                    safety_settings=self._UNCENSORED_SAFETY_SETTINGS,
                )
            )
            dur = time.time() - t0
            text_content = response.text or ""
            usage = {}
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage = {
                    "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", 0),
                    "completion_tokens": getattr(response.usage_metadata, "candidates_token_count", 0),
                }
            log_llm_call(
                provider="gemini-vision",
                model=self.model,
                prompt=user_text,
                system=PERSON_ANALYSIS_SYSTEM_PROMPT[:200],
                response=text_content,
                latency=dur,
                usage=usage,
            )

            parsed = self._extract_json(text_content)

            if parsed:
                parsed["backend_used"] = f"gemini ({self.model})"
                return PersonAnalysisResult.from_dict(parsed)
            else:
                return PersonAnalysisResult(
                    person_detected=True,
                    summary="Gemini vision returned raw narrative.",
                    narrative_description=text_content,
                    backend_used=f"gemini ({self.model})"
                )
        except Exception as e:
            dur = time.time() - t0
            log_llm_call(
                provider="gemini-vision",
                model=self.model,
                prompt=user_text,
                error=e,
                latency=dur,
            )
            error_str = str(e).lower()
            if ("quota" in error_str or "rate" in error_str or "resource_exhausted" in error_str or "429" in error_str) and not self._custom_api_key:
                all_keys = GeminiKeyManager.get_keys()
                if len(all_keys) > 1 and _retry_count < len(all_keys) * 2:
                    match = re.search(r"retry after (\d+)", error_str)
                    wait = float(match.group(1)) if match else 15.0
                    has_alt, min_wait, next_acc = GeminiKeyManager.mark_rate_limited(
                        key=self.api_key, retry_after=wait, model=self.model
                    )
                    if has_alt:
                        logger.warning(
                            "[Gemini Vision] Rate limited. Intelligently swapping to %s.",
                            next_acc.account_id if next_acc else "next account"
                        )
                        self._client = None
                        return self.analyze(image_bytes, mime_type, prompt_extra, _retry_count=_retry_count + 1)
            logger.error("Gemini Vision API error: %s", e)
            raise RuntimeError(f"Cloud vision analysis failed: {str(e)}")

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None

        match = _JSON_BLOCK_RE.search(text)
        if match:
            text = match.group(1)

        try:
            return json.loads(text.strip())
        except Exception:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start:end+1])
                except Exception:
                    pass
        return None
