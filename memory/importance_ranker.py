"""
Importance Ranker — Dynamic character importance scoring.

Calculates and maintains character importance based on:
- mention_frequency (how often the character appears)
- dialogue_density (how much they speak)
- plot_impact (involvement in major events)
- recency (how recently they appeared)

This drives retrieval prioritization and context budget allocation.
"""
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ImportanceRanker:
    """Calculates dynamic importance scores for characters."""

    # Scoring weights
    WEIGHT_MENTIONS = 0.30
    WEIGHT_DIALOGUE = 0.20
    WEIGHT_PLOT_IMPACT = 0.30
    WEIGHT_RECENCY = 0.20

    # Dialogue detection pattern
    _DIALOGUE_RE = re.compile(
        r'(?:'
        r'"[^"]{4,}"'        # double-quoted speech
        r"|'[^']{4,}'"       # single-quoted speech
        r'|\u201c[^\u201d]{4,}\u201d'  # smart quotes
        r')',
        re.DOTALL,
    )

    @classmethod
    def calculate_importance(
        cls,
        char_data: dict,
        current_chapter: int,
        max_mentions_norm: int = 100,
    ) -> float:
        """
        Compute a [0..1] importance score for a single character.

        Args:
            char_data: Character dict from state (must include mention_count,
                       first_seen, last_seen, status, role).
            current_chapter: The chapter currently being generated.
            max_mentions_norm: Normalization ceiling for mention frequency.

        Returns:
            Float in [0, 1].
        """
        if not isinstance(char_data, dict):
            return 0.0

        # Dead/retired characters decay faster
        status = str(char_data.get("status", "active")).lower()
        if status == "dead":
            return 0.05
        status_multiplier = 1.0 if status == "active" else 0.6

        # Mention frequency (normalized 0–1)
        mentions = max(0, int(char_data.get("mention_count", 0)))
        mention_score = min(1.0, mentions / max(1, max_mentions_norm))

        # Dialogue density — approximated from emotional_history richness
        # (actual dialogue counting happens in update_mention_count)
        dialogue_score = min(1.0, float(char_data.get("dialogue_density", 0.0)))

        # Plot impact — derived from arc_progression length + event involvement
        arc_entries = char_data.get("arc_progression", [])
        events = char_data.get("story_events_involved", 0)
        if isinstance(arc_entries, list):
            arc_count = len(arc_entries)
        else:
            arc_count = 1 if arc_entries else 0
        plot_impact = min(1.0, (arc_count * 0.15 + events * 0.25))

        # Recency — how recently the character was seen
        last_seen = int(char_data.get("last_seen", 0))
        if current_chapter <= 0 or last_seen <= 0:
            recency = 0.5  # Default for new characters
        else:
            gap = max(0, current_chapter - last_seen)
            recency = max(0.0, 1.0 - (gap * 0.15))

        # Role bonus
        role = str(char_data.get("role", "supporting")).lower()
        role_bonus = 0.15 if role == "main" else 0.0

        raw_score = (
            cls.WEIGHT_MENTIONS * mention_score
            + cls.WEIGHT_DIALOGUE * dialogue_score
            + cls.WEIGHT_PLOT_IMPACT * plot_impact
            + cls.WEIGHT_RECENCY * recency
            + role_bonus
        )

        return round(min(1.0, max(0.0, raw_score * status_multiplier)), 3)

    @classmethod
    def rank_characters(
        cls,
        characters: dict,
        current_chapter: int,
    ) -> list[tuple[str, float]]:
        """
        Return all characters sorted by importance (descending).

        Returns:
            List of (name, score) tuples.
        """
        scored = []
        for name, data in (characters or {}).items():
            score = cls.calculate_importance(data, current_chapter)
            scored.append((name, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    @classmethod
    def get_top_characters(
        cls,
        characters: dict,
        current_chapter: int,
        n: int = 5,
    ) -> list[str]:
        """Return the top N character names by importance."""
        ranked = cls.rank_characters(characters, current_chapter)
        return [name for name, _ in ranked[:n]]

    @classmethod
    def update_mention_count(
        cls,
        char_name: str,
        scene_text: str,
        char_data: dict,
        current_chapter: int,
    ) -> dict:
        """
        Count character mentions and dialogue density in a scene.
        Mutates and returns char_data with updated counts.
        """
        if not isinstance(char_data, dict):
            char_data = {}

        text_lower = (scene_text or "").lower()
        name_lower = str(char_name or "").lower()

        # Count name mentions
        pattern = r"\b" + re.escape(name_lower) + r"\b"
        name_matches = len(re.findall(pattern, text_lower))

        # Also count alias mentions
        for alias in char_data.get("aliases", []):
            alias_lower = str(alias).lower()
            if alias_lower != name_lower:
                alias_pattern = r"\b" + re.escape(alias_lower) + r"\b"
                name_matches += len(re.findall(alias_pattern, text_lower))

        prev_count = int(char_data.get("mention_count", 0))
        char_data["mention_count"] = prev_count + name_matches

        # Estimate dialogue density
        # Find dialogue lines that follow the character's name within 100 chars
        dialogue_matches = cls._DIALOGUE_RE.findall(scene_text or "")
        total_dialogue_chars = sum(len(d) for d in dialogue_matches)
        total_text_chars = max(1, len(scene_text or ""))

        # Rough heuristic: if this character is mentioned near dialogue
        char_dialogue = 0
        for match in cls._DIALOGUE_RE.finditer(scene_text or ""):
            start = max(0, match.start() - 100)
            preceding = scene_text[start:match.start()].lower()
            if name_lower in preceding:
                char_dialogue += len(match.group())

        if total_dialogue_chars > 0:
            density = min(1.0, char_dialogue / max(1, total_dialogue_chars))
        else:
            density = 0.0

        # Exponential moving average with previous density
        prev_density = float(char_data.get("dialogue_density", 0.0))
        char_data["dialogue_density"] = round(prev_density * 0.6 + density * 0.4, 3)

        # Update last_seen
        if name_matches > 0:
            char_data["last_seen"] = current_chapter

        return char_data

    @classmethod
    def update_all_characters(
        cls,
        characters: dict,
        scene_text: str,
        current_chapter: int,
    ) -> dict:
        """
        Update mention counts and importance scores for all characters
        after a scene. Mutates and returns the characters dict.
        """
        for name, data in (characters or {}).items():
            cls.update_mention_count(name, scene_text, data, current_chapter)
            data["importance_score"] = cls.calculate_importance(data, current_chapter)
        return characters
