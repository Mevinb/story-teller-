"""
Memory Compressor — Multi-tier memory management.

Implements a tiered memory system:
- SHORT_TERM: Last 2 chapters, full detail (from vector store)
- LEGEND_MEMORY: Permanent truths (major deaths, world rules,
  betrayals, marriages, power systems)

This prevents context collapse on long stories by ensuring
critical facts are never lost while managing memory budget.
"""
import logging

logger = logging.getLogger(__name__)

# Event types that are always legend-worthy
LEGEND_EVENT_TYPES = {
    "death", "betrayal", "marriage", "power_shift",
    "transformation", "secret_revealed",
}

# High-impact threshold for auto-legend promotion
LEGEND_IMPACT_THRESHOLD = "high"


class MemoryCompressor:
    """Manages multi-tier memory for long-form story generation."""

    @staticmethod
    def get_legend_events(state: dict) -> list:
        """
        Extract legend-worthy events from the state.
        These are permanent truths that should never be forgotten.
        """
        plot = state.get("plot", {})
        legend = list(plot.get("legend_memory", []))
        return legend

    @staticmethod
    def should_be_legend(event: dict) -> bool:
        """
        Determine if an event should be promoted to legend memory.
        """
        if not isinstance(event, dict):
            return False

        # High-impact events are always legend-worthy
        if event.get("impact") == LEGEND_IMPACT_THRESHOLD:
            return True

        # Certain event types are always legend-worthy
        event_type = str(event.get("type", "")).lower()
        if event_type in LEGEND_EVENT_TYPES:
            return True

        return False

    @classmethod
    def promote_events_to_legend(
        cls,
        state: dict,
        new_events: list,
    ) -> list:
        """
        Check new events and promote legend-worthy ones to legend memory.
        Returns list of newly promoted events.
        """
        legend = state.setdefault("plot", {}).setdefault("legend_memory", [])
        promoted = []

        existing_descs = {
            str(e.get("description", "")).lower()
            for e in legend
            if isinstance(e, dict)
        }

        for event in (new_events or []):
            if not isinstance(event, dict):
                continue
            if not cls.should_be_legend(event):
                continue

            desc = str(event.get("description", "")).lower()
            if desc and desc not in existing_descs:
                legend_entry = {
                    "type": event.get("type", "unknown"),
                    "description": event.get("description", ""),
                    "characters": event.get("characters", []),
                    "chapter": event.get("chapter", 0),
                    "permanent": True,
                }
                legend.append(legend_entry)
                promoted.append(legend_entry)
                existing_descs.add(desc)

        if promoted:
            logger.info("Promoted %d events to legend memory", len(promoted))

        return promoted

    @classmethod
    def build_legend_context(cls, state: dict, max_entries: int = 15) -> str:
        """
        Format legend memory for prompt injection.
        These facts must ALWAYS be included in context.
        """
        legend = cls.get_legend_events(state)
        if not legend:
            return ""

        lines = ["=== PERMANENT STORY FACTS (LEGEND MEMORY) ==="]

        # Character deaths
        deaths = [e for e in legend if e.get("type") == "death"]
        if deaths:
            lines.append("Deaths:")
            for d in deaths[:5]:
                chars = ", ".join(d.get("characters", []))
                lines.append(f"  - {chars}: {d.get('description', 'died')}")

        # Betrayals and power shifts
        critical = [
            e for e in legend
            if e.get("type") in {"betrayal", "power_shift", "secret_revealed"}
        ]
        if critical:
            lines.append("Critical events:")
            for c in critical[:5]:
                lines.append(f"  - Ch{c.get('chapter', '?')}: {c.get('description', '')}")

        # Other legend events
        others = [
            e for e in legend
            if e.get("type") not in {"death", "betrayal", "power_shift", "secret_revealed"}
        ]
        if others:
            lines.append("Other permanent facts:")
            for o in others[:5]:
                lines.append(f"  - {o.get('description', '')}")

        return "\n".join(lines[:max_entries + 1])

    @classmethod
    def get_character_lifecycle_summary(cls, characters: dict) -> str:
        """
        Summarize character lifecycle states for context.
        Only includes non-active characters.
        """
        non_active = []
        for name, data in (characters or {}).items():
            status = str(data.get("status", "active")).lower()
            if status != "active":
                non_active.append(f"{name}: {status}")

        if not non_active:
            return ""

        return "Character statuses: " + "; ".join(non_active)

    @classmethod
    def build_tiered_context(
        cls,
        state: dict,
        current_chapter: int,
        max_chars: int = 1500,
    ) -> str:
        """
        Assemble tiered memory context for prompt injection.

        Priority order:
        1. Legend memory (always included)
        2. Character lifecycle states
        3. Recent chapter summaries (short-term)
        """
        parts = []
        char_count = 0

        # Tier 1: Legend memory (always)
        legend_ctx = cls.build_legend_context(state)
        if legend_ctx:
            parts.append(legend_ctx)
            char_count += len(legend_ctx)

        # Tier 2: Character lifecycle
        lifecycle = cls.get_character_lifecycle_summary(state.get("characters", {}))
        if lifecycle and char_count + len(lifecycle) < max_chars:
            parts.append(lifecycle)
            char_count += len(lifecycle)

        # Tier 3: Recent chapter summaries (last 2)
        summaries = state.get("plot", {}).get("chapter_summaries", [])
        if summaries:
            recent = summaries[-2:]
            summary_lines = ["Recent chapters:"]
            for s in recent:
                line = f"  Ch{s.get('chapter', '?')}: {s.get('summary', '')[:200]}"
                if char_count + len(line) < max_chars:
                    summary_lines.append(line)
                    char_count += len(line)
            if len(summary_lines) > 1:
                parts.append("\n".join(summary_lines))

        return "\n\n".join(parts)
