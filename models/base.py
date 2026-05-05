"""
Abstract base class for all LLM backends.
Provides a unified interface so agents don't care about the provider.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
import time
import json
import logging
import re
import ast

import config

logger = logging.getLogger(__name__)


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
            try:
                return self.generate(
                    prompt=prompt,
                    system=system,
                    schema=schema,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=stream,
                )
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                # For rate limits, use longer delay
                if 'rate' in err_str and 'limit' in err_str:
                    wait = max(delay, 15)  # At least 15s for rate limits
                else:
                    wait = delay
                if attempt < max_retries - 1:
                    logger.warning(
                        f"[{self.get_name()}] Attempt {attempt + 1}/{max_retries} "
                        f"failed: {e}. Retrying in {wait:.1f}s..."
                    )
                    time.sleep(wait)
                    delay = min(delay * config.RETRY_BACKOFF_FACTOR, config.RETRY_MAX_DELAY)

        raise RuntimeError(
            f"[{self.get_name()}] All {max_retries} attempts failed. "
            f"Last error: {last_error}"
        )
