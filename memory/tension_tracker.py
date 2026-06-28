"""
Narrative Tension Tracker — Per-chapter tension curve management.

Maintains a tension score (0.0–1.0) for each chapter, enabling the
Story Architect and Scene Planner to make pacing decisions based on
where the story is in its emotional arc.

Expected curve shapes:
  - Rising action: 0.2 → 0.4 → 0.6 → 0.8
  - Climax:        0.9 → 1.0
  - Resolution:    0.6 → 0.3 → 0.1

Usage:
    tracker = TensionTracker()
    tracker.record(chapter=1, tension=0.3, label="introduction")
    summary = tracker.get_tension_summary(state)
    # Inject `summary` into architect prompt context
"""
import logging
import re
from copy import deepcopy

logger = logging.getLogger(__name__)

# Default tension values for narrative phases
PHASE_TENSION_DEFAULTS = {
    "introduction": 0.15,
    "escalation": 0.40,
    "midpoint": 0.55,
    "collapse": 0.70,
    "climax": 0.90,
    "aftermath": 0.30,
}

# Keywords that indicate high tension in prose
_HIGH_TENSION_MARKERS = {
    "scream", "blood", "fight", "attack", "betray", "death", "kill",
    "escape", "chase", "explosion", "ambush", "threat", "danger",
    "panic", "terror", "confront", "battle", "clash", "desperate",
    "shatter", "collapse", "fury", "rage", "war",
}

# Keywords that indicate low tension / calm
_LOW_TENSION_MARKERS = {
    "peaceful", "calm", "rest", "sleep", "gentle", "serene", "quiet",
    "comfort", "laugh", "smile", "relax", "safe", "home", "warm",
    "embrace", "tender", "soothe", "morning", "garden", "sunset",
}


class TensionTracker:
    """Tracks and analyzes narrative tension across chapters."""

    @staticmethod
    def estimate_tension_from_text(scene_text: str) -> float:
        """
        Estimate tension level from prose using keyword density analysis.
        Returns a float between 0.0 and 1.0.
        """
        if not scene_text:
            return 0.5

        words = re.findall(r"[a-z]+", scene_text.lower())
        if not words:
            return 0.5

        word_set = set(words)
        high_count = len(word_set & _HIGH_TENSION_MARKERS)
        low_count = len(word_set & _LOW_TENSION_MARKERS)

        # Exclamation marks and short sentences indicate intensity
        exclamations = scene_text.count("!")
        sentences = re.split(r"[.!?]+", scene_text)
        short_sentences = sum(1 for s in sentences if 1 <= len(s.split()) <= 6)

        # Calculate raw tension score
        intensity_bonus = min(0.15, exclamations * 0.02)
        pacing_bonus = min(0.10, short_sentences * 0.01)

        if high_count + low_count == 0:
            raw = 0.5
        else:
            raw = high_count / (high_count + low_count)

        tension = min(1.0, max(0.0, raw + intensity_bonus + pacing_bonus))
        return round(tension, 3)

    @staticmethod
    def record_chapter_tension(
        state: dict,
        chapter_num: int,
        tension: float,
        label: str = "",
    ) -> dict:
        """
        Record a tension measurement for a chapter in state.

        Args:
            state: The full state dict (will be modified in place).
            chapter_num: Chapter number.
            tension: Tension score 0.0–1.0.
            label: Optional narrative phase label.

        Returns:
            The updated state dict.
        """
        tension = max(0.0, min(1.0, float(tension)))
        plot = state.setdefault("plot", {})
        tension_curve = plot.setdefault("tension_curve", [])

        entry = {
            "chapter": chapter_num,
            "tension": round(tension, 3),
        }
        if label:
            entry["label"] = label.strip().lower()

        # Replace existing entry for this chapter, or append
        for i, existing in enumerate(tension_curve):
            if existing.get("chapter") == chapter_num:
                tension_curve[i] = entry
                return state

        tension_curve.append(entry)
        tension_curve.sort(key=lambda e: e.get("chapter", 0))
        return state

    @staticmethod
    def get_tension_curve(state: dict) -> list[dict]:
        """Return the tension curve from state."""
        return state.get("plot", {}).get("tension_curve", [])

    @classmethod
    def get_current_tension(cls, state: dict) -> float:
        """Return the most recent tension value, or 0.5 if no data."""
        curve = cls.get_tension_curve(state)
        if not curve:
            return 0.5
        return float(curve[-1].get("tension", 0.5))

    @classmethod
    def get_tension_trend(cls, state: dict, window: int = 3) -> str:
        """
        Analyze the recent tension trend.
        Returns: 'rising', 'falling', 'flat', or 'peak'.
        """
        curve = cls.get_tension_curve(state)
        if len(curve) < 2:
            return "flat"

        recent = curve[-window:]
        tensions = [e.get("tension", 0.5) for e in recent]

        if len(tensions) < 2:
            return "flat"

        # Check for peak (last value is local maximum)
        if tensions[-1] >= 0.85 and tensions[-1] >= max(tensions[:-1]):
            return "peak"

        avg_diff = sum(
            tensions[i + 1] - tensions[i]
            for i in range(len(tensions) - 1)
        ) / (len(tensions) - 1)

        if avg_diff > 0.05:
            return "rising"
        elif avg_diff < -0.05:
            return "falling"
        return "flat"

    @classmethod
    def get_recommended_tension(
        cls,
        state: dict,
        chapter_num: int,
        total_premise_steps: int,
    ) -> float:
        """
        Recommend a target tension for the next chapter based on
        narrative position and existing curve.
        """
        if total_premise_steps <= 0:
            return 0.5

        # Narrative position as fraction 0.0–1.0
        progress = min(1.0, chapter_num / max(1, total_premise_steps))

        # Standard tension arc: slow rise → peak at 75-85% → resolution
        if progress < 0.25:
            target = 0.15 + progress * 1.0  # 0.15 → 0.40
        elif progress < 0.50:
            target = 0.40 + (progress - 0.25) * 1.2  # 0.40 → 0.70
        elif progress < 0.80:
            target = 0.70 + (progress - 0.50) * 1.0  # 0.70 → 1.00
        else:
            target = 1.0 - (progress - 0.80) * 3.5  # 1.00 → 0.30

        return round(max(0.1, min(1.0, target)), 2)

    @classmethod
    def get_tension_summary(cls, state: dict, max_entries: int = 8) -> str:
        """
        Format a human-readable tension summary for prompt injection.
        """
        curve = cls.get_tension_curve(state)
        if not curve:
            return ""

        trend = cls.get_tension_trend(state)
        current = cls.get_current_tension(state)

        lines = [f"[Tension Curve] Current: {current:.2f} | Trend: {trend}"]

        # Show recent entries
        recent = curve[-max_entries:]
        points = []
        for entry in recent:
            ch = entry.get("chapter", "?")
            t = entry.get("tension", 0.5)
            label = entry.get("label", "")
            bar = "█" * int(t * 10)
            label_str = f" ({label})" if label else ""
            points.append(f"  Ch{ch}: {bar} {t:.2f}{label_str}")

        lines.extend(points)
        return "\n".join(lines)
