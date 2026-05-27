"""
Event Extractor — Major story event extraction and tracking.

Extracts and categorizes significant plot events from scene text:
{
    "type": "betrayal",
    "characters": ["Elise", "Marcus"],
    "impact": "high",
    "chapter": 5,
    "scene": 2,
    "description": "Marcus betrayed Elise by revealing her location",
    "resolved": false
}

This enables smarter unresolved plot tracking and event-based retrieval.
"""
import logging

logger = logging.getLogger(__name__)

# Event type categories
EVENT_TYPES = {
    "betrayal", "alliance", "death", "injury", "discovery",
    "revelation", "confrontation", "escape", "arrival", "departure",
    "romance", "breakup", "reunion", "sacrifice", "transformation",
    "power_shift", "secret_revealed", "promise", "threat",
    "loss", "victory", "capture", "rescue", "training",
}


class EventExtractor:
    """Extracts and manages structured story events."""

    @staticmethod
    def create_event(
        event_type: str,
        characters: list,
        description: str,
        chapter: int,
        scene: int,
        impact: str = "medium",
        resolved: bool = False,
    ) -> dict:
        """Create a standardized event dict."""
        valid_type = event_type.strip().lower().replace(" ", "_")
        if valid_type not in EVENT_TYPES:
            valid_type = "discovery"  # Default fallback

        valid_impact = str(impact).strip().lower()
        if valid_impact not in {"high", "medium", "low"}:
            valid_impact = "medium"

        return {
            "type": valid_type,
            "characters": [str(c).strip() for c in (characters or []) if str(c).strip()],
            "description": str(description).strip(),
            "chapter": int(chapter),
            "scene": int(scene),
            "impact": valid_impact,
            "resolved": bool(resolved),
        }

    @staticmethod
    def normalize_events(raw_events: list) -> list:
        """
        Normalize a list of raw event dicts (e.g., from LLM output)
        into standardized event format.
        """
        normalized = []
        for raw in (raw_events or []):
            if not isinstance(raw, dict):
                continue

            event = {
                "type": str(raw.get("type", "discovery")).strip().lower().replace(" ", "_"),
                "characters": [],
                "description": str(raw.get("description", raw.get("event", ""))).strip(),
                "chapter": int(raw.get("chapter", 0)),
                "scene": int(raw.get("scene", 0)),
                "impact": str(raw.get("impact", "medium")).strip().lower(),
                "resolved": bool(raw.get("resolved", False)),
            }

            # Normalize type
            if event["type"] not in EVENT_TYPES:
                event["type"] = "discovery"

            # Normalize impact
            if event["impact"] not in {"high", "medium", "low"}:
                event["impact"] = "medium"

            # Normalize characters
            chars = raw.get("characters", [])
            if isinstance(chars, str):
                chars = [c.strip() for c in chars.split(",") if c.strip()]
            elif isinstance(chars, list):
                chars = [str(c).strip() for c in chars if str(c).strip()]
            else:
                chars = []
            event["characters"] = chars

            if event["description"]:
                normalized.append(event)

        return normalized

    @staticmethod
    def get_unresolved_events(events: list) -> list:
        """Return only unresolved events."""
        return [
            e for e in (events or [])
            if isinstance(e, dict) and not e.get("resolved", False)
        ]

    @staticmethod
    def get_high_impact_events(events: list) -> list:
        """Return only high-impact events."""
        return [
            e for e in (events or [])
            if isinstance(e, dict) and e.get("impact") == "high"
        ]

    @staticmethod
    def get_character_events(events: list, char_name: str) -> list:
        """Return events involving a specific character."""
        name_lower = str(char_name).lower()
        return [
            e for e in (events or [])
            if isinstance(e, dict) and any(
                str(c).lower() == name_lower for c in e.get("characters", [])
            )
        ]

    @staticmethod
    def mark_resolved(events: list, description_fragment: str) -> list:
        """
        Mark events matching a description fragment as resolved.
        Returns the updated events list.
        """
        fragment_lower = description_fragment.lower()
        for event in (events or []):
            if isinstance(event, dict):
                desc = str(event.get("description", "")).lower()
                if fragment_lower in desc:
                    event["resolved"] = True
        return events

    @classmethod
    def get_events_summary(
        cls,
        events: list,
        max_events: int = 10,
        unresolved_only: bool = False,
    ) -> str:
        """Format events as a summary string for prompt injection."""
        if unresolved_only:
            events = cls.get_unresolved_events(events)

        if not events:
            return ""

        # Sort by impact (high first), then by chapter (recent first)
        impact_order = {"high": 0, "medium": 1, "low": 2}
        sorted_events = sorted(
            events,
            key=lambda e: (
                impact_order.get(e.get("impact", "medium"), 1),
                -int(e.get("chapter", 0)),
            ),
        )

        lines = []
        for event in sorted_events[:max_events]:
            chars = ", ".join(event.get("characters", []))
            status = "OPEN" if not event.get("resolved") else "resolved"
            lines.append(
                f"[{event.get('type', '?').upper()}] "
                f"Ch{event.get('chapter', '?')}: "
                f"{event.get('description', 'Unknown event')} "
                f"({chars}) [{status}]"
            )

        return "\n".join(lines)

    @staticmethod
    def count_character_event_involvement(events: list, char_name: str) -> int:
        """Count how many events a character is involved in."""
        name_lower = str(char_name).lower()
        count = 0
        for event in (events or []):
            if isinstance(event, dict):
                chars = [str(c).lower() for c in event.get("characters", [])]
                if name_lower in chars:
                    count += 1
        return count
