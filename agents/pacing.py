"""
Pacing Agent — Post-generation pacing analyzer.

Analyzes generated prose for structural pacing issues:
- Exposition dumps (too much telling, not enough showing)
- Dialogue-heavy scenes (all talk, no action)
- Action density (rapid fire events with no breathing room)
- Abrupt endings (scene cuts off without proper closure)
- Paragraph length variance (monotonous rhythm)

Returns a pacing report with scores and recommendations that the
orchestrator can use to request targeted rewrites.
"""
import logging
import re
from collections import Counter

from .contract import AgentContract

logger = logging.getLogger(__name__)

# Thresholds for pacing detection
_DIALOGUE_HEAVY_THRESHOLD = 0.55   # > 55% dialogue lines = too dialogue-heavy
_EXPOSITION_DUMP_THRESHOLD = 0.70  # > 70% non-dialogue long paragraphs = exposition dump
_SHORT_SCENE_WORDS = 150           # Scenes shorter than this are likely abrupt
_MONOTONE_PARA_VARIANCE = 15      # If paragraph length variance < 15 words, monotonous


class PacingAgent(AgentContract):
    """
    Analyzes prose for structural pacing balance.

    Unlike the Consistency Engine (which checks factual correctness),
    the Pacing Agent checks for *structural* writing quality:
    - Is the scene mostly dialogue with no action beats?
    - Is there an exposition dump?
    - Does the scene end too abruptly?
    - Is the paragraph rhythm monotonous?

    This is a lightweight, deterministic agent — no LLM calls needed.
    """

    def __init__(self):
        self.name = "pacing"

    def run(self, state: dict) -> dict:
        """Execute pacing analysis within the agent contract interface."""
        scene_text = state.get("scene_text", "")
        scene_plan = state.get("scene_plan", {})
        report = self.analyze(scene_text, scene_plan)

        has_issues = any(
            issue.get("severity") in {"high", "medium"}
            for issue in report.get("issues", [])
        )

        return {
            "output": {"pacing_report": report},
            "confidence": 0.90 if not has_issues else 0.60,
            "next_action": "critic",  # Always proceed to critic after pacing
        }

    def analyze(self, scene_text: str, scene_plan: dict = None) -> dict:
        """
        Run all pacing checks on a scene.

        Args:
            scene_text: The generated prose to analyze.
            scene_plan: Optional scene plan for context.

        Returns:
            Pacing report dict with overall_pace, scores, and issues.
        """
        scene_plan = scene_plan or {}
        issues = []
        scores = {}

        if not scene_text or len(scene_text.strip()) < 50:
            return {
                "overall_pace": "unknown",
                "scores": {},
                "issues": [],
                "word_count": 0,
            }

        word_count = len(scene_text.split())

        # ── Check 1: Dialogue ratio ─────────────────────────────────
        dialogue_ratio = self._dialogue_ratio(scene_text)
        scores["dialogue_ratio"] = round(dialogue_ratio, 3)

        if dialogue_ratio > _DIALOGUE_HEAVY_THRESHOLD:
            issues.append({
                "type": "dialogue_heavy",
                "detail": (
                    f"Scene is {dialogue_ratio:.0%} dialogue. "
                    "Consider adding action beats, sensory details, "
                    "or internal thoughts between dialogue lines."
                ),
                "severity": "medium" if dialogue_ratio > 0.70 else "low",
                "recommendation": "interleave_action",
            })

        # ── Check 2: Exposition density ──────────────────────────────
        exposition_ratio = self._exposition_ratio(scene_text)
        scores["exposition_ratio"] = round(exposition_ratio, 3)

        if exposition_ratio > _EXPOSITION_DUMP_THRESHOLD:
            issues.append({
                "type": "exposition_dump",
                "detail": (
                    f"Scene has {exposition_ratio:.0%} exposition density. "
                    "Too much telling — add dialogue, action, or break "
                    "up long descriptive passages."
                ),
                "severity": "medium",
                "recommendation": "break_exposition",
            })

        # ── Check 3: Action density ──────────────────────────────────
        action_density = self._action_density(scene_text)
        scores["action_density"] = round(action_density, 3)

        if action_density > 0.6:
            issues.append({
                "type": "action_overload",
                "detail": (
                    "Scene has very high action density with little "
                    "breathing room. Consider adding a moment of "
                    "reflection or dialogue between action beats."
                ),
                "severity": "low",
                "recommendation": "add_breathing_room",
            })

        # ── Check 4: Abrupt ending ───────────────────────────────────
        if self._has_abrupt_ending(scene_text, scene_plan):
            issues.append({
                "type": "abrupt_ending",
                "detail": (
                    "Scene appears to end abruptly without proper closure. "
                    "Add a transitional sentence or moment of reflection."
                ),
                "severity": "medium",
                "recommendation": "extend_ending",
            })

        # ── Check 5: Paragraph rhythm monotony ──────────────────────
        variance = self._paragraph_length_variance(scene_text)
        scores["paragraph_variance"] = round(variance, 1)

        if variance < _MONOTONE_PARA_VARIANCE and word_count > 200:
            issues.append({
                "type": "monotonous_rhythm",
                "detail": (
                    f"Paragraph lengths are very uniform (variance: {variance:.0f} words). "
                    "Mix short punchy paragraphs with longer descriptive ones."
                ),
                "severity": "low",
                "recommendation": "vary_paragraph_length",
            })

        # ── Compute overall pace ─────────────────────────────────────
        overall = self._classify_pace(dialogue_ratio, exposition_ratio, action_density)
        scores["overall_pace"] = overall

        return {
            "overall_pace": overall,
            "scores": scores,
            "issues": issues,
            "word_count": word_count,
        }

    @staticmethod
    def _dialogue_ratio(text: str) -> float:
        """Calculate the ratio of dialogue lines to total lines."""
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if not lines:
            return 0.0

        dialogue_lines = sum(
            1 for line in lines
            if line.startswith('"') or line.startswith("'")
            or line.startswith("\u201c")  # "
            or re.match(r'^[A-Z][a-z]+ said', line)
            or '," ' in line or '." ' in line
        )
        return dialogue_lines / len(lines)

    @staticmethod
    def _exposition_ratio(text: str) -> float:
        """
        Estimate exposition density.
        Long paragraphs (>50 words) without dialogue markers indicate exposition.
        """
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            return 0.0

        exposition_words = 0
        total_words = 0

        for para in paragraphs:
            words = len(para.split())
            total_words += words

            has_dialogue = '"' in para or "\u201c" in para or "'" in para
            if words > 50 and not has_dialogue:
                exposition_words += words

        return exposition_words / max(1, total_words)

    @staticmethod
    def _action_density(text: str) -> float:
        """Estimate action density using verb-heavy sentence detection."""
        _ACTION_VERBS = {
            "ran", "jumped", "hit", "struck", "dodged", "slashed", "threw",
            "grabbed", "kicked", "punched", "charged", "leaped", "swung",
            "crashed", "slammed", "pulled", "pushed", "rolled", "sprinted",
            "fired", "blocked", "ducked", "lunged", "tackled", "fell",
        }
        sentences = re.split(r"[.!?]+", text.lower())
        if not sentences:
            return 0.0

        action_sentences = sum(
            1 for s in sentences
            if any(verb in s.split() for verb in _ACTION_VERBS)
        )
        return action_sentences / len(sentences)

    @staticmethod
    def _has_abrupt_ending(text: str, scene_plan: dict) -> bool:
        """Check if the scene ends too abruptly."""
        text = text.rstrip()
        if not text:
            return False

        # Very short scenes are likely abrupt
        if len(text.split()) < _SHORT_SCENE_WORDS:
            return True

        # Check if ending is a complete thought
        last_para = text.split("\n\n")[-1].strip()
        if len(last_para.split()) < 8:
            return True

        # Scene ends mid-sentence
        if not text.endswith((".", "!", "?", '"', "'", "\u201d")):
            return True

        return False

    @staticmethod
    def _paragraph_length_variance(text: str) -> float:
        """Calculate the variance of paragraph word counts."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if len(paragraphs) < 3:
            return 100.0  # Not enough data, assume fine

        lengths = [len(p.split()) for p in paragraphs]
        mean = sum(lengths) / len(lengths)
        variance = sum((l - mean) ** 2 for l in lengths) / len(lengths)
        return variance ** 0.5  # Standard deviation

    @staticmethod
    def _classify_pace(
        dialogue_ratio: float,
        exposition_ratio: float,
        action_density: float,
    ) -> str:
        """Classify the overall pacing of the scene."""
        if action_density > 0.4:
            return "fast"
        if dialogue_ratio > 0.50:
            return "conversational"
        if exposition_ratio > 0.60:
            return "slow"
        if action_density > 0.2 and dialogue_ratio > 0.25:
            return "balanced"
        return "moderate"

    @staticmethod
    def format_pacing_feedback(report: dict) -> str:
        """Format pacing issues into a string for writer prompt injection."""
        issues = report.get("issues", [])
        if not issues:
            return ""

        lines = ["PACING FEEDBACK:"]
        for issue in issues:
            severity = issue.get("severity", "low").upper()
            detail = issue.get("detail", "")
            lines.append(f"  [{severity}] {detail}")

        return "\n".join(lines)
