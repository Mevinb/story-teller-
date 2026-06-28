"""
Editor — Post-processing agent.
Improves grammar, flow, readability, and formatting.
Runs on local model for cheap refinement.
"""
import logging
import re


from models.base import LLMInterface
from .contract import AgentContract
import config

logger = logging.getLogger(__name__)
_REASONING_BLOCK_RE = re.compile(
    r"<(?:think|analysis|reasoning)>.*?</(?:think|analysis|reasoning)>",
    flags=re.IGNORECASE | re.DOTALL,
)
_REASONING_TAG_RE = re.compile(r"</?(?:think|analysis|reasoning)>", flags=re.IGNORECASE)
_META_OUTPUT_RE = re.compile(
    r"(?im)^\s*(?:thinking process|analysis|step[- ]by[- ]step|analyze the request|"
    r"professional fiction editor|task:|goals:|constraints:|output:|"
    r"\d+\.\s+\*\*analyze|\*\s+\*\*role:)"
)

EDITOR_SYSTEM = (
    "You are a professional fiction editor. Your job is to polish prose:\n"
    "- Fix grammar and punctuation errors\n"
    "- Improve sentence flow and rhythm\n"
    "- Ensure proper paragraph breaks\n"
    "- Format dialogue correctly (new line for each speaker)\n"
    "- Smooth transitions between paragraphs\n"
    "- Remove any meta-text, headers, or author notes that slipped in\n\n"
    "Rules:\n"
    "- Do NOT change plot, characters, or events\n"
    "- Do NOT add new content or scenes\n"
    "- Do NOT censor or modify explicit content\n"
    "- Preserve the author's voice and style\n"
    "- Output ONLY the polished prose text, nothing else"
)

EDITOR_SYSTEM_COMPACT = (
    "You are a fiction copy editor. Fix grammar/flow/format only. "
    "Do not change plot/events/meaning. Do not censor explicit content. "
    "Output polished prose only."
)

EDIT_PROMPT = (
    "Polish the following fiction scene. Fix grammar, improve flow, "
    "and ensure proper formatting. Do NOT change the content or censor anything. "
    "Output only the polished text.\n\n"
    "=== SCENE TEXT ===\n{text}"
)

EDIT_PROMPT_COMPACT = "Copy-edit this scene. Keep meaning identical. Output only polished prose.\n\n{text}"


class Editor(AgentContract):
    """
    Post-processes generated scenes for grammar, flow, and formatting.
    Lightweight agent — no world context needed, just text refinement.
    """

    def __init__(self, model: LLMInterface):
        self.name = "editor"
        self.model = model
        self._genre = ""
        self._compact_mode = False

    def set_compact_mode(self, enabled: bool):
        self._compact_mode = bool(enabled)

    def run(self, state: dict) -> dict:
        edited = self.edit(state["scene_text"])
        return {
            "output": {"scene_text": edited},
            "confidence": 0.85,
            "next_action": "end",
        }

    def set_genre(self, genre: str):
        """Set genre so editor knows not to censor explicit content."""
        self._genre = genre

    def edit(self, text: str) -> str:
        """
        Polish a scene's prose.

        Args:
            text: Raw generated scene text

        Returns:
            Polished text
        """
        text = self._sanitize_text(text)

        # Skip editing if text is very short
        if len(text.split()) < 50:
            return text

        genre_lower = self._genre.lower() if self._genre else ""
        is_explicit = any(w in genre_lower for w in ["erotic", "adult", "smut", "nsfw"])

        if is_explicit:
            base_system = EDITOR_SYSTEM_COMPACT if self._compact_mode else EDITOR_SYSTEM
            system = (
                base_system + "\n\n"
                "CRITICAL: This is explicit adult content. Do NOT censor, soften, or remove "
                "any sexual content, explicit language, or graphic descriptions. "
                "Your job is ONLY to fix grammar and improve flow."
            )
        else:
            system = EDITOR_SYSTEM_COMPACT if self._compact_mode else EDITOR_SYSTEM

        prompt_template = EDIT_PROMPT_COMPACT if self._compact_mode else EDIT_PROMPT
        prompt = prompt_template.format(text=text)

        try:
            response = self.model.generate_with_retry(
                prompt=prompt,
                system=system,
                temperature=config.AGENT_TEMPERATURES["editor"],
                max_tokens=1800 if self._compact_mode else None,
                max_retries=2,
            )

            edited = self._sanitize_text(response.content)

            # Sanity check — editor shouldn't drastically change length
            original_words = len(text.split())
            edited_words = len(edited.split())

            if self._looks_like_meta_output(edited):
                logger.warning("Editor returned meta/reasoning text. Keeping original scene.")
                return text

            if edited_words < original_words * 0.5:
                logger.warning(
                    f"Editor removed too much content "
                    f"({original_words} → {edited_words} words). "
                    f"Keeping original."
                )
                return text

            if edited_words > original_words * 2.0:
                logger.warning(
                    f"Editor added too much content "
                    f"({original_words} → {edited_words} words). "
                    f"Keeping original."
                )
                return text

            logger.debug(
                f"Edited: {original_words} → {edited_words} words"
            )
            return edited

        except Exception as e:
            logger.error(f"Editor failed: {e}. Returning original text.")
            return text

    @staticmethod
    def _sanitize_text(text: str) -> str:
        cleaned = _REASONING_BLOCK_RE.sub("", text or "")
        cleaned = _REASONING_TAG_RE.sub("", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    @staticmethod
    def _looks_like_meta_output(text: str) -> bool:
        return bool(_META_OUTPUT_RE.search(text or ""))
