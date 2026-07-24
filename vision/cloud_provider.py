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

logger = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


class CloudVisionProvider:
    """Cloud Vision provider using Google GenAI (Gemini multimodal)."""

    def __init__(self, api_key: str = None, model: str = None):
        self._api_key = api_key or getattr(config, "GEMINI_API_KEY", "")
        self.model = model or getattr(config, "GEMINI_MODEL", "gemini-2.5-flash")
        self._client = None

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            if not self._api_key:
                raise RuntimeError("GEMINI_API_KEY not configured in .env file.")
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def check_availability(self) -> Dict[str, Any]:
        """Check if Gemini Cloud API is configured."""
        if not self._api_key:
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

    def analyze(self, image_bytes: bytes, mime_type: str = "image/jpeg", prompt_extra: str = "") -> PersonAnalysisResult:
        """Analyze image using Gemini Multimodal Vision."""
        if not self._api_key:
            raise RuntimeError("GEMINI_API_KEY is missing. Please add it to your .env file.")

        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type=mime_type
        )

        user_text = "Perform detailed person, physical body, face, and clothing visual recognition on this image."
        if prompt_extra:
            user_text += f"\nUser extra context: {prompt_extra}"

        full_prompt = f"{PERSON_ANALYSIS_SYSTEM_PROMPT}\n\nUser Request: {user_text}"

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=[image_part, full_prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2,
                )
            )

            text_content = response.text or ""
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
