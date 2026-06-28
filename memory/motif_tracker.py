"""
Motif Tracker — Symbol and recurring imagery tracking.

Tracks recurring objects, imagery, and thematic elements across chapters:
{
    "red_scarf": {
        "mentions": 7,
        "meaning": "lost love",
        "chapters": [1, 3, 5, 8],
        "contexts": ["she clutched the red scarf", "the scarf lay abandoned"]
    }
}

This enables:
- Recurring imagery injection into writer prompts
- Thematic consistency validation
- Foreshadowing and callback support
"""
import logging
import re
from collections import Counter

logger = logging.getLogger(__name__)


class MotifTracker:
    """Tracks and manages recurring symbols, motifs, and thematic elements."""

    @staticmethod
    def extract_motifs_from_text(
        scene_text: str,
        existing_motifs: dict,
        min_word_length: int = 4,
    ) -> dict:
        """
        Scan prose for recurring objects/phrases that match existing tracked motifs.
        Returns dict of motif_key -> list of context snippets found in this text.

        This does NOT discover new motifs — it only checks if known motifs appear.
        New motif discovery is handled by the LLM evolution engine.
        """
        if not scene_text or not existing_motifs:
            return {}

        text_lower = scene_text.lower()
        found = {}

        for motif_key, motif_data in existing_motifs.items():
            if not isinstance(motif_data, dict):
                continue

            # Build search terms from the motif key and any aliases
            search_terms = [motif_key.lower().replace("_", " ")]
            aliases = motif_data.get("aliases", [])
            if isinstance(aliases, list):
                search_terms.extend(a.lower() for a in aliases if a)

            for term in search_terms:
                if term in text_lower:
                    # Extract surrounding context (±40 chars around match)
                    idx = text_lower.find(term)
                    start = max(0, idx - 40)
                    end = min(len(scene_text), idx + len(term) + 40)
                    context = scene_text[start:end].strip()
                    context = re.sub(r"\s+", " ", context)

                    found.setdefault(motif_key, []).append(context)
                    break  # One match per motif per scene is enough

        return found

    @staticmethod
    def register_motif(
        state: dict,
        motif_key: str,
        meaning: str = "",
        aliases: list = None,
    ) -> dict:
        """
        Register a new motif in state, or update an existing one.

        Args:
            state: The full state dict.
            motif_key: Unique key (e.g., 'red_scarf', 'silver_ring').
            meaning: Thematic meaning (e.g., 'lost love', 'hidden power').
            aliases: Alternative names for the same symbol.

        Returns:
            The updated state dict.
        """
        motif_key = motif_key.strip().lower().replace(" ", "_")
        if not motif_key:
            return state

        world = state.setdefault("world", {})
        motifs = world.setdefault("motifs", {})

        if motif_key in motifs:
            # Update existing
            if meaning:
                motifs[motif_key]["meaning"] = meaning.strip()
            if aliases:
                existing_aliases = motifs[motif_key].get("aliases", [])
                new_aliases = [a for a in aliases if a not in existing_aliases]
                motifs[motif_key]["aliases"] = existing_aliases + new_aliases
        else:
            motifs[motif_key] = {
                "meaning": meaning.strip() if meaning else "",
                "aliases": aliases or [],
                "mentions": 0,
                "chapters": [],
                "contexts": [],
            }

        return state

    @staticmethod
    def record_motif_appearance(
        state: dict,
        motif_key: str,
        chapter_num: int,
        context: str = "",
    ) -> dict:
        """Record that a motif appeared in a specific chapter."""
        motif_key = motif_key.strip().lower().replace(" ", "_")
        world = state.setdefault("world", {})
        motifs = world.setdefault("motifs", {})

        if motif_key not in motifs:
            motifs[motif_key] = {
                "meaning": "",
                "aliases": [],
                "mentions": 0,
                "chapters": [],
                "contexts": [],
            }

        motif = motifs[motif_key]
        motif["mentions"] = motif.get("mentions", 0) + 1

        chapters = motif.setdefault("chapters", [])
        if chapter_num not in chapters:
            chapters.append(chapter_num)

        if context:
            contexts = motif.setdefault("contexts", [])
            contexts.append(context.strip()[:120])
            # Keep last 10 contexts
            if len(contexts) > 10:
                motif["contexts"] = contexts[-10:]

        return state

    @staticmethod
    def get_active_motifs(state: dict, min_mentions: int = 2) -> dict:
        """Return motifs with enough mentions to be considered recurring."""
        motifs = state.get("world", {}).get("motifs", {})
        return {
            key: data
            for key, data in motifs.items()
            if isinstance(data, dict) and data.get("mentions", 0) >= min_mentions
        }

    @classmethod
    def get_motif_summary(cls, state: dict, max_motifs: int = 6) -> str:
        """Format active motifs for prompt injection."""
        active = cls.get_active_motifs(state, min_mentions=1)
        if not active:
            return ""

        # Sort by mention count (most recurring first)
        sorted_motifs = sorted(
            active.items(),
            key=lambda x: x[1].get("mentions", 0),
            reverse=True,
        )

        lines = ["[Recurring Motifs & Symbols]"]
        for key, data in sorted_motifs[:max_motifs]:
            name = key.replace("_", " ").title()
            meaning = data.get("meaning", "")
            mentions = data.get("mentions", 0)
            chapters = data.get("chapters", [])

            parts = [f"  {name} (×{mentions})"]
            if meaning:
                parts.append(f"meaning: {meaning}")
            if chapters:
                parts.append(f"in Ch{', '.join(str(c) for c in chapters[-5:])}")

            lines.append(" | ".join(parts))

        return "\n".join(lines)

    @classmethod
    def update_motifs_after_scene(
        cls,
        state: dict,
        scene_text: str,
        chapter_num: int,
    ) -> int:
        """
        Scan scene text for known motifs and update their tracking data.
        Returns the number of motif appearances recorded.
        """
        motifs = state.get("world", {}).get("motifs", {})
        if not motifs:
            return 0

        found = cls.extract_motifs_from_text(scene_text, motifs)
        count = 0
        for motif_key, contexts in found.items():
            for context in contexts:
                cls.record_motif_appearance(state, motif_key, chapter_num, context)
                count += 1

        return count
