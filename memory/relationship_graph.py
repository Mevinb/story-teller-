"""
Relationship Graph — Dynamic relationship evolution tracking.

Replaces flat "[Character A]": "friend" relationships with rich structures:
{
    "[Character A]": {
        "type": "ally",
        "trust": 0.61,
        "romantic_tension": 0.34,
        "hostility": 0.12,
        "history": [
            {"chapter": 2, "event": "saved her life"},
            {"chapter": 6, "event": "lied about mission"}
        ]
    }
}

This allows the retriever and writer to understand relationship DYNAMICS,
not just static labels.
"""
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)

# Default relationship template
_DEFAULT_RELATIONSHIP = {
    "type": "neutral",
    "trust": 0.5,
    "romantic_tension": 0.0,
    "hostility": 0.0,
    "history": [],
}

# Valid relationship types
RELATIONSHIP_TYPES = {
    "ally", "friend", "rival", "enemy", "lover", "family",
    "mentor", "subordinate", "neutral", "complicated",
}


class RelationshipGraph:
    """Manages evolving character relationships."""

    @staticmethod
    def ensure_rich_relationship(rel_value) -> dict:
        """
        Convert a legacy flat relationship string into a rich structure.
        If already rich, return as-is.
        """
        if isinstance(rel_value, dict) and "type" in rel_value:
            # Already a rich relationship
            result = deepcopy(_DEFAULT_RELATIONSHIP)
            result.update(rel_value)
            # Ensure history is a list
            if not isinstance(result.get("history"), list):
                result["history"] = []
            return result

        if isinstance(rel_value, str):
            # Legacy flat string — migrate
            rel_type = rel_value.strip().lower()
            if rel_type not in RELATIONSHIP_TYPES:
                # Map common legacy values
                mapping = {
                    "love interest": "lover",
                    "best friend": "friend",
                    "partner": "ally",
                    "teacher": "mentor",
                    "student": "subordinate",
                    "classmate": "neutral",
                    "colleague": "neutral",
                }
                rel_type = mapping.get(rel_type, "neutral")

            result = deepcopy(_DEFAULT_RELATIONSHIP)
            result["type"] = rel_type

            # Set initial scores based on type
            type_defaults = {
                "ally": {"trust": 0.7, "hostility": 0.05},
                "friend": {"trust": 0.8, "hostility": 0.0},
                "rival": {"trust": 0.2, "hostility": 0.5},
                "enemy": {"trust": 0.0, "hostility": 0.9},
                "lover": {"trust": 0.75, "romantic_tension": 0.8},
                "family": {"trust": 0.6, "hostility": 0.1},
                "mentor": {"trust": 0.7, "hostility": 0.0},
                "subordinate": {"trust": 0.5, "hostility": 0.1},
                "neutral": {"trust": 0.5, "hostility": 0.0},
                "complicated": {"trust": 0.3, "romantic_tension": 0.3, "hostility": 0.3},
            }
            defaults = type_defaults.get(rel_type, {})
            result.update(defaults)
            return result

        return deepcopy(_DEFAULT_RELATIONSHIP)

    @classmethod
    def update_relationship(
        cls,
        relationships: dict,
        target_name: str,
        update: dict,
        chapter: int,
    ) -> dict:
        """
        Update a character's relationship with target_name.
        Merges scores and appends history events.

        Args:
            relationships: The character's relationships dict.
            target_name: The other character's name.
            update: Dict with optional keys: type, trust, romantic_tension,
                    hostility, event (str describing what happened).
            chapter: Current chapter number.

        Returns:
            The updated relationships dict.
        """
        if not isinstance(relationships, dict):
            relationships = {}

        existing = relationships.get(target_name)
        rich = cls.ensure_rich_relationship(existing)

        # Update type if provided
        if "type" in update:
            new_type = str(update["type"]).strip().lower()
            if new_type in RELATIONSHIP_TYPES:
                rich["type"] = new_type

        # Update numeric scores — blend with existing (EMA-style)
        for key in ("trust", "romantic_tension", "hostility"):
            if key in update:
                try:
                    new_val = float(update[key])
                    new_val = max(0.0, min(1.0, new_val))
                    old_val = float(rich.get(key, 0.5))
                    # Weighted blend: 40% old, 60% new (recent events matter more)
                    rich[key] = round(old_val * 0.4 + new_val * 0.6, 3)
                except (TypeError, ValueError):
                    pass

        # Append history event
        event_text = str(update.get("event", "")).strip()
        if event_text:
            rich.setdefault("history", []).append({
                "chapter": chapter,
                "event": event_text,
            })
            # Keep history manageable — last 20 events
            if len(rich["history"]) > 20:
                rich["history"] = rich["history"][-20:]

        relationships[target_name] = rich
        return relationships

    @classmethod
    def get_relationship_summary(
        cls,
        char_name: str,
        target_name: str,
        relationships: dict,
    ) -> str:
        """
        Format a human-readable relationship summary for prompt injection.
        """
        rel = relationships.get(target_name)
        if not rel:
            return ""

        rich = cls.ensure_rich_relationship(rel)
        parts = [f"{char_name} → {target_name}: {rich['type']}"]

        scores = []
        if rich.get("trust", 0.5) != 0.5:
            scores.append(f"trust={rich['trust']:.1f}")
        if rich.get("romantic_tension", 0) > 0.1:
            scores.append(f"romance={rich['romantic_tension']:.1f}")
        if rich.get("hostility", 0) > 0.1:
            scores.append(f"hostility={rich['hostility']:.1f}")
        if scores:
            parts.append(f"({', '.join(scores)})")

        history = rich.get("history", [])
        if history:
            recent = history[-3:]
            events = "; ".join(f"Ch{h['chapter']}: {h['event']}" for h in recent)
            parts.append(f"Recent: {events}")

        return " | ".join(parts)

    @classmethod
    def get_active_relationships(
        cls,
        relationships: dict,
        min_significance: float = 0.3,
    ) -> dict:
        """
        Filter relationships to those with significant dynamics.
        Significance = max(|trust - 0.5|, romantic_tension, hostility).
        """
        active = {}
        for name, rel in (relationships or {}).items():
            rich = cls.ensure_rich_relationship(rel)
            significance = max(
                abs(float(rich.get("trust", 0.5)) - 0.5),
                float(rich.get("romantic_tension", 0)),
                float(rich.get("hostility", 0)),
            )
            if significance >= min_significance:
                active[name] = rich
        return active

    @classmethod
    def decay_stale_relationships(
        cls,
        relationships: dict,
        current_chapter: int,
        decay_rate: float = 0.05,
    ) -> dict:
        """
        Decay relationship scores toward neutral for characters not
        interacting recently. Applies per-chapter decay.
        """
        for name, rel in (relationships or {}).items():
            rich = cls.ensure_rich_relationship(rel)
            history = rich.get("history", [])

            if history:
                last_chapter = max(h.get("chapter", 0) for h in history)
            else:
                last_chapter = 0

            gap = max(0, current_chapter - last_chapter)
            if gap < 3:
                continue  # Only decay after 3+ chapters of no interaction

            decay = decay_rate * (gap - 2)
            # Decay toward neutral (0.5 for trust, 0 for others)
            for key, neutral in [("trust", 0.5), ("romantic_tension", 0.0), ("hostility", 0.0)]:
                current = float(rich.get(key, neutral))
                if current > neutral:
                    rich[key] = round(max(neutral, current - decay), 3)
                elif current < neutral:
                    rich[key] = round(min(neutral, current + decay), 3)

            relationships[name] = rich
        return relationships

    @classmethod
    def get_all_relationship_summaries(
        cls,
        char_name: str,
        relationships: dict,
        max_entries: int = 8,
    ) -> str:
        """Format all significant relationships for prompt injection."""
        active = cls.get_active_relationships(relationships, min_significance=0.15)
        if not active:
            return ""

        lines = []
        # Sort by significance (descending)
        sorted_rels = sorted(
            active.items(),
            key=lambda x: max(
                abs(float(x[1].get("trust", 0.5)) - 0.5),
                float(x[1].get("romantic_tension", 0)),
                float(x[1].get("hostility", 0)),
            ),
            reverse=True,
        )

        for target, _ in sorted_rels[:max_entries]:
            summary = cls.get_relationship_summary(char_name, target, relationships)
            if summary:
                lines.append(summary)

        return "\n".join(lines)
