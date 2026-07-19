"""
Quality Controller — Phase 2 & 3.

Centralises all scene/chapter quality logic extracted from orchestrator.py:
  - Scene completion validation (word count + sentence termination)
  - Truncation detection and auto-fix
  - Multi-dimensional scene scoring (Phase 3 quality gates):
      * Coherence   — does the scene follow logically from context?
      * Pacing      — word-count / arc-position fit
      * Voice       — dialogue density, POV consistency
      * Temporal    — time / location transition markers
  - Auto-rewrite trigger (score < threshold → request revision)

The controller is stateless (pure functions + configurable thresholds)
and designed to be dependency-injected into the orchestrator.
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

import config

logger = logging.getLogger(__name__)


# Sentence Boundary Regex
# Matches: . ! ? … followed by optional closing quotes (straight or curly),
# markdown emphasis markers, spaces — same pattern as orchestrator._validate_scene_completion
_SENTENCE_END_RE = re.compile(
    r'[.!?\u2026]["\u201d\u2019\u2018\u2018*_\s]*\s*$',
    re.UNICODE,
)

# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class ScoreResult:
    """Multi-dimensional quality score for a generated scene."""
    coherence: float = 1.0       # 0-1: logical flow from context
    pacing: float = 1.0          # 0-1: word count vs arc position
    voice: float = 1.0           # 0-1: dialogue density / POV consistency
    temporal: float = 1.0        # 0-1: time/location transition markers
    issues: list[str] = field(default_factory=list)
    word_count: int = 0

    @property
    def overall(self) -> float:
        return (self.coherence + self.pacing + self.voice + self.temporal) / 4.0

    @property
    def passes(self) -> bool:
        return self.overall >= QualityController.QUALITY_THRESHOLD

    def as_dict(self) -> dict:
        return {
            "coherence": round(self.coherence, 3),
            "pacing": round(self.pacing, 3),
            "voice": round(self.voice, 3),
            "temporal": round(self.temporal, 3),
            "overall": round(self.overall, 3),
            "passes": self.passes,
            "word_count": self.word_count,
            "issues": self.issues,
        }


@dataclass
class ValidationResult:
    """Result of scene-completion validation."""
    ok: bool
    reason: str = ""
    words: int = 0
    minimum: int = 0
    truncated: bool = False


# ─── Quality Controller ───────────────────────────────────────────────────────

class QualityController:
    """
    Centralized scene quality checks.

    Parameters
    ----------
    quality_threshold : float
        Minimum ``ScoreResult.overall`` to pass the quality gate.
        Scenes below this score are flagged for revision.
    min_scene_words : int
        Override for the minimum word count. Defaults to ``config.MIN_SCENE_WORDS``.
    """

    QUALITY_THRESHOLD: float = 0.55   # class-level default (used by ScoreResult.passes)

    def __init__(
        self,
        quality_threshold: float = 0.55,
        min_scene_words: Optional[int] = None,
    ) -> None:
        self.quality_threshold = quality_threshold
        QualityController.QUALITY_THRESHOLD = quality_threshold  # keep class attr in sync
        self._min_scene_words = min_scene_words or getattr(config, "MIN_SCENE_WORDS", 500)

    # ─── Completion Validation ─────────────────────────────────────────────────

    def validate(self, scene_text: str, scene: dict, chapter_num: int) -> ValidationResult:
        """
        Check whether a generated scene meets minimum completeness requirements.

        Returns a :class:`ValidationResult` instead of raising, so callers
        can decide how to handle failures without try/except everywhere.
        """
        text = scene_text or ""
        words = len(text.split())
        scene_num = scene.get("scene_number", "?")

        # ── Word count ────────────────────────────────────────────────────────
        try:
            target = int(scene.get("word_target", self._min_scene_words))
        except (TypeError, ValueError):
            target = self._min_scene_words
        minimum = max(120, min(self._min_scene_words, int(target * 0.45)))

        if words < minimum:
            return ValidationResult(
                ok=False,
                reason=(
                    f"Scene {scene_num} ch{chapter_num} is too short "
                    f"({words} words; expected ≥ {minimum})"
                ),
                words=words,
                minimum=minimum,
            )

        # ── Sentence termination ──────────────────────────────────────────────
        if not _SENTENCE_END_RE.search(text):
            return ValidationResult(
                ok=False,
                reason=f"Scene {scene_num} ch{chapter_num} ends mid-sentence (truncated)",
                words=words,
                minimum=minimum,
                truncated=True,
            )

        return ValidationResult(ok=True, words=words, minimum=minimum)

    # ─── Multi-dimensional Scoring ─────────────────────────────────────────────

    def score(
        self,
        scene_text: str,
        scene: dict,
        previous_ending: str = "",
        context: str = "",
    ) -> ScoreResult:
        """
        Score a generated scene across four quality dimensions.

        This is a fast, deterministic (no LLM) first-pass gate.
        All scorers return 0.0-1.0; issues list human-readable warnings.
        """
        text = scene_text or ""
        words = len(text.split())
        issues: list[str] = []

        coherence = self._score_coherence(text, previous_ending, scene, issues)
        pacing = self._score_pacing(text, words, scene, issues)
        voice = self._score_voice(text, scene, issues)
        temporal = self._score_temporal(text, previous_ending, scene, issues)

        result = ScoreResult(
            coherence=coherence,
            pacing=pacing,
            voice=voice,
            temporal=temporal,
            issues=issues,
            word_count=words,
        )
        return result

    # ── Coherence ─────────────────────────────────────────────────────────────

    def _score_coherence(
        self,
        text: str,
        previous_ending: str,
        scene: dict,
        issues: list[str],
    ) -> float:
        score = 1.0

        # Check for meta-output leakage (model explaining itself)
        _META_RE = re.compile(
            r"(?im)^\s*(?:thinking process|analysis|step[- ]by[- ]step|"
            r"analyze the request|professional fiction editor|task:|goals:|"
            r"constraints:|output:|\d+\.\s+\*\*analyze|\*\s+\*\*role:)"
        )
        if _META_RE.search(text):
            score -= 0.4
            issues.append("Meta-output detected (model narrating its own process)")

        # Check for repeated reference sentences from previous ending
        if previous_ending and len(previous_ending) > 30:
            prev_words = set(previous_ending.lower().split())
            text_words = set(text[:300].lower().split())
            overlap = len(prev_words & text_words) / max(len(prev_words), 1)
            if overlap > 0.7:
                score -= 0.3
                issues.append(f"Opening heavily mirrors previous ending ({overlap:.0%} overlap)")

        # Check scene covers key events
        key_events = scene.get("key_events", [])
        if key_events:
            covered = 0
            text_lower = text.lower()
            for event in key_events[:3]:
                event_tokens = set(re.findall(r"[a-z]{4,}", str(event).lower()))
                if event_tokens and len(event_tokens & set(re.findall(r"[a-z]{4,}", text_lower))) >= 2:
                    covered += 1
            coverage = covered / min(len(key_events), 3)
            if coverage < 0.5:
                score -= 0.2
                issues.append(f"Key events coverage low ({covered}/{min(len(key_events), 3)} detected)")

        return max(0.0, min(1.0, score))

    # ── Pacing ────────────────────────────────────────────────────────────────

    def _score_pacing(
        self,
        text: str,
        words: int,
        scene: dict,
        issues: list[str],
    ) -> float:
        score = 1.0
        target = int(scene.get("word_target", getattr(config, "WORDS_PER_SCENE_MIN", 500)))
        scene_type = scene.get("type", "build_tension")

        # Word count vs target
        ratio = words / max(target, 1)
        if ratio < 0.5:
            score -= 0.4
            issues.append(f"Scene very short ({words} words, target {target})")
        elif ratio < 0.7:
            score -= 0.2
            issues.append(f"Scene below target ({words}/{target} words)")
        elif ratio > 2.5:
            score -= 0.15
            issues.append(f"Scene unusually long ({words} words, target {target})")

        # Paragraph variety
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        if len(paragraphs) < 2 and words > 300:
            score -= 0.15
            issues.append("Very few paragraph breaks — may feel monotonous")

        # Dialogue check
        dialogue_lines = len(re.findall(r'[""]', text))
        if scene_type in ("setup", "resolution") and dialogue_lines == 0 and words > 200:
            score -= 0.05   # Minor deduction only — some scenes are pure prose
            issues.append("No dialogue in setup/resolution scene")

        return max(0.0, min(1.0, score))

    # ── Voice ─────────────────────────────────────────────────────────────────

    def _score_voice(
        self,
        text: str,
        scene: dict,
        issues: list[str],
    ) -> float:
        score = 1.0

        # Check for POV consistency (crude: detect first-person bleed in third-person text)
        # Most scenes are third-person limited.
        first_person_count = len(re.findall(r"\bI\b(?! love| am| had| was|'m|'d|'ll|'ve)", text))
        third_person_count = len(re.findall(r"\b(?:he|she|they)\b", text.lower()))
        if first_person_count > 5 and third_person_count > 5:
            score -= 0.25
            issues.append(
                f"POV inconsistency: {first_person_count} first-person + "
                f"{third_person_count} third-person references mixed"
            )

        # Repetition check — repeated phrases
        words = text.lower().split()
        if len(words) > 50:
            trigrams: dict[str, int] = {}
            for i in range(len(words) - 2):
                tri = " ".join(words[i:i+3])
                if all(len(w) > 3 for w in words[i:i+3]):
                    trigrams[tri] = trigrams.get(tri, 0) + 1
            repeated = {t: c for t, c in trigrams.items() if c >= 4}
            if repeated:
                worst = max(repeated, key=lambda t: repeated[t])
                score -= min(0.3, len(repeated) * 0.05)
                issues.append(f"Repetitive phrases (e.g. '{worst}' × {repeated[worst]})")

        return max(0.0, min(1.0, score))

    # ── Temporal / Spatial ────────────────────────────────────────────────────

    def _score_temporal(
        self,
        text: str,
        previous_ending: str,
        scene: dict,
        issues: list[str],
    ) -> float:
        score = 1.0
        prev_location = (scene.get("prev_location") or "").strip()
        current_location = (scene.get("location") or "").strip()

        # Location changed but no transition signal?
        if (
            prev_location
            and current_location
            and prev_location.lower() != current_location.lower()
        ):
            transition_markers = (
                "later", "minutes later", "hours later", "next morning",
                "the next day", "meanwhile", "across town", "in the",
                "at the", "arriving", "walked into", "entered", "stepped into",
                "driving", "on the way", "outside", "upstairs", "downstairs",
                "she arrived", "he arrived", "they arrived",
            )
            text_lower = text.lower()
            has_transition = any(m in text_lower for m in transition_markers)
            # Also check if the current location name itself appears in the opening
            loc_in_text = current_location.lower() in text_lower[:500]
            if not has_transition and not loc_in_text:
                score -= 0.3
                issues.append(
                    f"Location jump ({prev_location!r} → {current_location!r}) "
                    "without transition text"
                )

        # Time jump check — detect large time skips without grounding
        time_skip_markers = re.findall(
            r"\b(?:years?|months?|weeks?|decades?)\s+(?:later|passed|had passed|earlier)\b",
            text.lower(),
        )
        if time_skip_markers and len(text.split()) < 200:
            score -= 0.2
            issues.append(
                f"Large time skip ({time_skip_markers[0]}) in a short scene — "
                "may feel abrupt"
            )

        return max(0.0, min(1.0, score))

    # ─── Truncation Fix ───────────────────────────────────────────────────────

    def fix_truncated_ending(
        self,
        scene_text: str,
        generate_fn,  # callable(prompt, system, max_tokens) -> str
    ) -> str:
        """
        Attempt to complete a truncated scene by calling ``generate_fn``.

        Returns the fixed text, or the original if the fix fails / is not needed.
        The ``generate_fn`` must accept (prompt: str, system: str, max_tokens: int)
        and return a string.
        """
        if not scene_text:
            return scene_text
        if _SENTENCE_END_RE.search(scene_text):
            return scene_text  # Not truncated

        tail = scene_text[-300:].strip()
        fix_prompt = (
            "The following prose scene was cut off mid-sentence. "
            "Write ONLY the remaining 1-3 sentences to complete the final thought naturally. "
            "Do NOT repeat any of the existing text. Output ONLY the missing ending.\n\n"
            f"=== SCENE ENDING (cut off) ===\n{tail}"
        )
        try:
            completion = generate_fn(
                fix_prompt,
                "Complete the scene's final sentence naturally. Output only the missing text.",
                200,
            )
            completion = (completion or "").strip()
            # Strip reasoning tags if model leaks them
            completion = re.sub(
                r"<(?:think|analysis|reasoning)>.*?</(?:think|analysis|reasoning)>",
                "",
                completion,
                flags=re.IGNORECASE | re.DOTALL,
            ).strip()
            if completion and len(completion.split()) < 80:
                fixed = f"{scene_text.rstrip()} {completion.lstrip()}".strip()
                if _SENTENCE_END_RE.search(fixed):
                    return fixed
        except Exception as exc:
            logger.warning("Truncation fix failed: %s", exc)
        return scene_text

    # ─── Summary ──────────────────────────────────────────────────────────────

    def summarise(self, score: ScoreResult) -> str:
        """One-line summary of a score for logging."""
        emoji = "✅" if score.passes else "⚠️"
        return (
            f"{emoji} Quality {score.overall:.2f} "
            f"(coherence={score.coherence:.2f}, pacing={score.pacing:.2f}, "
            f"voice={score.voice:.2f}, temporal={score.temporal:.2f}) "
            f"words={score.word_count}"
        )
