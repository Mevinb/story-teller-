"""
Retriever — Context assembly engine.
Combines structured state (JSON) + semantic search (FAISS)
to build a compact, relevant context window for generation.
"""
import logging
from typing import List, Optional

import config
from .state_manager import StateManager
from .vector_store import VectorStore, SearchResult

logger = logging.getLogger(__name__)


class Retriever:
    """
    Assembles context for scene generation by combining:
    1. Structured character/world state from state.json
    2. Semantically relevant past content from FAISS
    """

    def __init__(self, state_manager: StateManager, vector_store: VectorStore):
        self.state = state_manager
        self.vectors = vector_store

    def retrieve_context(
        self,
        scene_plan: dict,
        chapter_num: int,
        extra_query: str = "",
        top_k: int = None,
        max_chars: int = None,
    ) -> str:
        """
        Build the context window for a scene.

        Args:
            scene_plan: The scene plan dict from the Scene Planner
            chapter_num: Current chapter number
            extra_query: Additional search terms
            top_k: Override default retrieval count
            max_chars: Override default context budget (in chars)

        Returns:
            Formatted context string ready for prompt injection
        """
        top_k = top_k or config.TOP_K_RETRIEVAL
        max_chars = max_chars or config.CONTEXT_TOKEN_BUDGET * 4  # ~4 chars per token

        parts = []
        char_count = 0

        # ─── Part 1: Structured State ────────────────────────────────
        characters = scene_plan.get("characters_present", [])
        state_context = self.state.get_context_window(characters=characters or None)
        parts.append("=== STORY STATE ===")
        parts.append(state_context)
        char_count += len(state_context)

        # ─── Part 2: Scene-Specific Character Details ─────────────────
        for char_name in characters:
            char_data = self.state.get_character(char_name)
            if not char_data:
                continue
            char_parts = []
            if char_data.get("description"):
                char_parts.append(f"Description: {char_data['description']}")
            if char_data.get("traits"):
                char_parts.append(f"Traits: {', '.join(char_data['traits'])}")
            if char_data.get("state"):
                state_str = ", ".join(f"{k}={v}" for k, v in char_data["state"].items() if v)
                if state_str:
                    char_parts.append(f"Current state: {state_str}")
            if char_data.get("relationships"):
                rels = ", ".join(f"{k}: {v}" for k, v in char_data["relationships"].items())
                char_parts.append(f"Relationships: {rels}")
            if char_data.get("arc_progression"):
                recent_arc = char_data["arc_progression"][-3:]
                if isinstance(recent_arc, list):
                    char_parts.append(f"Recent arc: {'; '.join(str(a) for a in recent_arc)}")
            if char_parts:
                detail = f"\n[{char_name}] " + " | ".join(char_parts)
                parts.append(detail)
                char_count += len(detail)

        # ─── Part 3: Semantic Retrieval ───────────────────────────────
        query = self._build_search_query(scene_plan, extra_query)
        if query and self.vectors.index.ntotal > 0:
            results = self.vectors.search(query, top_k=top_k)

            if results:
                parts.append("\n=== RELEVANT PAST CONTENT ===")
                for result in results:
                    chunk_text = result.text
                    meta = result.metadata

                    # Budget check
                    if char_count + len(chunk_text) > max_chars:
                        # Truncate this chunk to fit
                        remaining = max_chars - char_count - 50
                        if remaining > 100:
                            chunk_text = chunk_text[:remaining] + "..."
                        else:
                            break

                    source = f"[Ch{meta.chapter}/Sc{meta.scene}]"
                    parts.append(f"\n{source} {chunk_text}")
                    char_count += len(chunk_text) + len(source) + 2

        # ─── Part 4: Chapter Summaries (if available) ──────────────────
        plot = self.state.get_plot()
        summaries = plot.get("chapter_summaries", [])
        if summaries and char_count < max_chars - 200:
            # Include last 2 chapter summaries
            recent_summaries = summaries[-2:]
            parts.append("\n=== RECENT CHAPTER SUMMARIES ===")
            for s in recent_summaries:
                summary_text = f"Chapter {s['chapter']}: {s['summary']}"
                if char_count + len(summary_text) < max_chars:
                    parts.append(summary_text)
                    char_count += len(summary_text)

        context = "\n".join(parts)
        logger.debug(
            f"Context assembled: {len(context)} chars, "
            f"{len(parts)} sections"
        )
        return context

    def _build_search_query(self, scene_plan: dict, extra: str = "") -> str:
        """Build a semantic search query from the scene plan."""
        query_parts = []

        if scene_plan.get("summary"):
            query_parts.append(scene_plan["summary"])

        if scene_plan.get("characters_present"):
            query_parts.append(
                "Characters: " + ", ".join(scene_plan["characters_present"])
            )

        if scene_plan.get("location"):
            query_parts.append(f"Location: {scene_plan['location']}")

        if scene_plan.get("key_events"):
            query_parts.append(
                "Events: " + ", ".join(scene_plan["key_events"])
            )

        if extra:
            query_parts.append(extra)

        return " ".join(query_parts)

    def retrieve_for_consistency(
        self,
        scene_text: str,
        characters: list = None,
    ) -> str:
        """
        Retrieve context specifically for the Consistency Engine.
        Focuses on character states and relevant past content.
        """
        parts = []

        # Full character states
        parts.append("=== CHARACTER STATES ===")
        all_chars = self.state.get_characters()
        target_chars = {k: v for k, v in all_chars.items() if k in (characters or [])} \
            if characters else all_chars

        for name, info in target_chars.items():
            parts.append(f"\n{name}:")
            parts.append(f"  Traits: {', '.join(info.get('traits', []))}")
            if info.get("relationships"):
                for other, rel in info["relationships"].items():
                    parts.append(f"  → {other}: {rel}")
            if info.get("state"):
                for k, v in info["state"].items():
                    if v:
                        parts.append(f"  {k}: {v}")

        # Relevant past content
        if self.vectors.index.ntotal > 0:
            results = self.vectors.search(scene_text[:500], top_k=3)
            if results:
                parts.append("\n=== RELATED PAST SCENES ===")
                for r in results:
                    parts.append(
                        f"[Ch{r.metadata.chapter}/Sc{r.metadata.scene}] "
                        f"{r.text[:300]}"
                    )

        # World rules
        world = self.state.get_world()
        if world.get("rules"):
            parts.append("\n=== WORLD RULES ===")
            for rule in world["rules"]:
                parts.append(f"- {rule}")

        return "\n".join(parts)
