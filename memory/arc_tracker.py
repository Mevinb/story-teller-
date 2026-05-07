"""
Arc Tracker — Character arc progression tracking.

Manages the transformation journey of each character:
{
    "core_wound": "fear of abandonment",
    "false_belief": "trust makes people weak",
    "current_phase": "slowly opening emotionally",
    "arc_events": [
        {"chapter": 3, "event": "betrayed"},
        {"chapter": 9, "event": "trusted ally again"}
    ],
    "arc_direction": "healing"
}

This enables the system to understand character TRANSFORMATIONS,
not just static traits. The writer can reference where a character
is in their journey and write accordingly.
"""
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)

# Default arc template
_DEFAULT_ARC = {
    "core_wound": "",
    "false_belief": "",
    "current_phase": "introduction",
    "arc_events": [],
    "arc_direction": "neutral",
}

# Valid arc phases (narrative progression)
ARC_PHASES = [
    "introduction",       # Character is established
    "comfort_zone",       # Status quo before disruption
    "disruption",         # Inciting incident for this character
    "resistance",         # Character resists change
    "exploration",        # Tentatively exploring new territory
    "commitment",         # Committing to change
    "ordeal",            # Major test/crisis
    "transformation",    # Core belief shifts
    "integration",       # New identity settles in
    "mastery",           # Character embodies transformation
]

# Valid arc directions
ARC_DIRECTIONS = {
    "positive",    # Moving toward growth/healing
    "negative",    # Moving toward corruption/fall
    "flat",        # Character stays the same, changes world around them
    "healing",     # Recovering from trauma
    "corrupting",  # Being corrupted by power/events
    "neutral",     # Direction not yet established
}


class ArcTracker:
    """Tracks and manages character arc progression."""

    @staticmethod
    def ensure_arc_data(char_data: dict) -> dict:
        """
        Ensure a character has arc_progression_data (rich arc structure).
        Migrates from legacy arc_progression list if needed.
        """
        if not isinstance(char_data, dict):
            return deepcopy(_DEFAULT_ARC)

        arc_data = char_data.get("arc_progression_data")
        if isinstance(arc_data, dict) and "arc_events" in arc_data:
            # Already has rich arc data
            result = deepcopy(_DEFAULT_ARC)
            result.update(arc_data)
            if not isinstance(result["arc_events"], list):
                result["arc_events"] = []
            return result

        # Migrate from legacy arc_progression (list of strings)
        result = deepcopy(_DEFAULT_ARC)
        legacy = char_data.get("arc_progression", [])
        if isinstance(legacy, list):
            for i, entry in enumerate(legacy):
                if isinstance(entry, str) and entry.strip():
                    result["arc_events"].append({
                        "chapter": i + 1,
                        "event": entry.strip(),
                    })
        elif isinstance(legacy, str) and legacy.strip():
            result["arc_events"].append({
                "chapter": 1,
                "event": legacy.strip(),
            })

        return result

    @classmethod
    def update_arc(
        cls,
        char_data: dict,
        arc_update: dict,
        chapter: int,
    ) -> dict:
        """
        Merge new arc data into a character's arc progression.

        Args:
            char_data: The full character dict.
            arc_update: Dict with optional keys: core_wound, false_belief,
                        current_phase, arc_direction, event (str).
            chapter: Current chapter number.

        Returns:
            Updated char_data.
        """
        arc = cls.ensure_arc_data(char_data)

        # Update core narrative elements
        for key in ("core_wound", "false_belief"):
            if key in arc_update and str(arc_update[key]).strip():
                arc[key] = str(arc_update[key]).strip()

        # Update phase
        if "current_phase" in arc_update:
            phase = str(arc_update["current_phase"]).strip().lower().replace(" ", "_")
            if phase in ARC_PHASES:
                arc["current_phase"] = phase
            else:
                # Accept freeform phases too
                arc["current_phase"] = str(arc_update["current_phase"]).strip()

        # Update direction
        if "arc_direction" in arc_update:
            direction = str(arc_update["arc_direction"]).strip().lower()
            if direction in ARC_DIRECTIONS:
                arc["arc_direction"] = direction

        # Append arc event
        event_text = str(arc_update.get("event", "")).strip()
        if event_text:
            arc["arc_events"].append({
                "chapter": chapter,
                "event": event_text,
            })
            # Keep manageable — last 30 events
            if len(arc["arc_events"]) > 30:
                arc["arc_events"] = arc["arc_events"][-30:]

        # Auto-detect phase from events if not explicitly set
        if "current_phase" not in arc_update and arc["arc_events"]:
            detected = cls.detect_arc_phase(arc["arc_events"])
            if detected:
                arc["current_phase"] = detected

        char_data["arc_progression_data"] = arc
        return char_data

    @classmethod
    def detect_arc_phase(cls, arc_events: list) -> str:
        """
        Attempt to detect the current arc phase from event history.
        Uses simple heuristics based on event count and keywords.
        """
        if not arc_events:
            return "introduction"

        count = len(arc_events)
        last_event = str(arc_events[-1].get("event", "")).lower()

        # Keyword-based detection
        crisis_markers = {"betray", "death", "loss", "crisis", "fight", "attack", "fail", "break"}
        growth_markers = {"trust", "forgive", "accept", "open", "heal", "love", "grow", "learn"}
        disruption_markers = {"discover", "reveal", "shock", "surprise", "encounter", "meet"}

        if count <= 2:
            if any(m in last_event for m in disruption_markers):
                return "disruption"
            return "comfort_zone"

        if any(m in last_event for m in crisis_markers):
            return "ordeal"

        if any(m in last_event for m in growth_markers):
            if count >= 6:
                return "integration"
            return "exploration"

        # Phase by event density
        if count <= 3:
            return "disruption"
        if count <= 5:
            return "resistance"
        if count <= 8:
            return "exploration"
        if count <= 12:
            return "commitment"
        return "transformation"

    @classmethod
    def get_arc_summary(cls, char_name: str, char_data: dict) -> str:
        """
        Format a human-readable arc summary for prompt injection.
        """
        arc = cls.ensure_arc_data(char_data)
        parts = [f"[{char_name} Arc]"]

        if arc.get("core_wound"):
            parts.append(f"Core wound: {arc['core_wound']}")
        if arc.get("false_belief"):
            parts.append(f"False belief: {arc['false_belief']}")

        parts.append(f"Phase: {arc.get('current_phase', 'introduction')}")
        parts.append(f"Direction: {arc.get('arc_direction', 'neutral')}")

        events = arc.get("arc_events", [])
        if events:
            recent = events[-4:]
            event_strs = [f"Ch{e['chapter']}: {e['event']}" for e in recent]
            parts.append(f"Recent arc events: {'; '.join(event_strs)}")

        return " | ".join(parts)

    @classmethod
    def get_all_arc_summaries(
        cls,
        characters: dict,
        max_chars: int = 5,
    ) -> str:
        """Format arc summaries for the top characters."""
        summaries = []
        for name, data in list(characters or {}).items()[:max_chars]:
            arc = cls.ensure_arc_data(data)
            # Only include characters with meaningful arcs
            if arc.get("arc_events") or arc.get("core_wound"):
                summaries.append(cls.get_arc_summary(name, data))
        return "\n".join(summaries)
