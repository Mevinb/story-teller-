"""
Retriever — Context assembly engine.
Combines structured state (JSON) + semantic search (FAISS)
to build a compact, relevant context window for generation.

Upgraded with hybrid narrative retrieval:
- Temporal weighting (recency)
- Character overlap weighting
- Unresolved thread weighting
- Emotional relevance weighting
- Evolution context injection (arcs, relationships, transitions, phase)
"""
import logging

import config
from .state_manager import StateManager
from .vector_store import VectorStore
from .relationship_graph import RelationshipGraph
from .arc_tracker import ArcTracker
from .importance_ranker import ImportanceRanker
from .event_extractor import EventExtractor
from .memory_compressor import MemoryCompressor

logger = logging.getLogger(__name__)


class Retriever:
    """
    Assembles context for scene generation by combining:
    1. Structured character/world state from state.json
    2. Semantically relevant past content from FAISS
    3. Evolution context (arcs, relationships, transitions, phase)
    """

    def __init__(self, state_manager: StateManager, vector_store: VectorStore):
        self.state = state_manager
        self.vectors = vector_store

    def retrieve_context(
        self,
        scene_plan: dict,
        chapter_num: int,
        previous_ending: str = "",
        extra_query: str = "",
        top_k: int = None,
        max_chars: int = None,
    ) -> str:
        """
        Build the context window for a scene with hybrid scoring.

        Returns:
            Formatted context string ready for prompt injection
        """
        top_k = top_k or config.TOP_K_RETRIEVAL
        max_chars = max_chars or config.CONTEXT_TOKEN_BUDGET * 4

        parts = []
        char_count = 0

        # ─── Part 1: Structured State ────────────────────────────────
        characters = scene_plan.get("characters_present", [])
        state_context = self.state.get_context_window(characters=characters or None)
        parts.append("=== STORY STATE ===")
        parts.append(state_context)
        char_count += len(state_context)

        # ─── Part 1.5: Narrative Phase ───────────────────────────────
        phase = self.state.get_narrative_phase()
        if phase and phase != "introduction":
            phase_block = f"\n=== NARRATIVE PHASE: {phase.upper()} ==="
            parts.append(phase_block)
            char_count += len(phase_block)

        # ─── Part 2: Scene-Specific Character Details + Evolution ─────
        all_chars = self.state.get_characters()
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

            # Status (if non-active)
            status = char_data.get("status", "active")
            if status != "active":
                char_parts.append(f"Status: {status}")

            # Rich relationships
            rels = char_data.get("relationships", {})
            if rels:
                rel_summary = RelationshipGraph.get_all_relationship_summaries(
                    char_name, rels, max_entries=4,
                )
                if rel_summary:
                    char_parts.append(f"Relationships:\n{rel_summary}")

            # Emotional trajectory
            emotional_history = char_data.get("emotional_history", [])
            if emotional_history:
                recent = emotional_history[-3:]
                emo_strs = [f"Ch{e.get('chapter','?')}: {e.get('emotion','')} ({e.get('cause','')})" for e in recent]
                char_parts.append(f"Emotional arc: {'; '.join(emo_strs)}")

            # Arc progression
            arc_summary = ArcTracker.get_arc_summary(char_name, char_data)
            if arc_summary and char_data.get("arc_progression_data", {}).get("arc_events"):
                char_parts.append(arc_summary)

            # Legacy arc_progression
            elif char_data.get("arc_progression"):
                recent_arc = char_data["arc_progression"][-3:]
                if isinstance(recent_arc, list):
                    char_parts.append(f"Recent arc: {'; '.join(str(a) for a in recent_arc)}")

            if char_parts:
                detail = f"\n[{char_name}] " + " | ".join(char_parts)
                parts.append(detail)
                char_count += len(detail)

        # ─── Part 2.3: Character Importance Ranking ──────────────────
        if all_chars and char_count < max_chars - 200:
            ranked = ImportanceRanker.rank_characters(all_chars, chapter_num)
            top_names = [f"{name}({score:.1f})" for name, score in ranked[:6]]
            if top_names:
                rank_block = f"\n=== CHARACTER IMPORTANCE ===\n{', '.join(top_names)}"
                parts.append(rank_block)
                char_count += len(rank_block)

        # ─── Part 2.5: Transition State ──────────────────────────────
        transition = self.state.get_latest_transition()
        if transition and char_count < max_chars - 300:
            trans_parts = []
            if transition.get("time_elapsed"):
                trans_parts.append(f"Time elapsed: {transition['time_elapsed']}")
            phys = transition.get("physical_states", {})
            if phys:
                phys_strs = [f"{k}: {v}" for k, v in phys.items()]
                trans_parts.append(f"Physical: {'; '.join(phys_strs)}")
            convos = transition.get("open_conversations", [])
            if convos:
                trans_parts.append(f"Open threads: {'; '.join(convos[:3])}")
            env = transition.get("environmental_carryover", [])
            if env:
                trans_parts.append(f"Environment: {'; '.join(env[:3])}")
            emo_carry = transition.get("emotional_carryover", [])
            if emo_carry:
                trans_parts.append(f"Lingering: {'; '.join(emo_carry[:3])}")
            if trans_parts:
                trans_block = "\n=== TRANSITION FROM PREVIOUS SCENE ===\n" + "\n".join(trans_parts)
                parts.append(trans_block)
                char_count += len(trans_block)

        # ─── Part 2.7: Legend Memory ─────────────────────────────────
        legend_ctx = MemoryCompressor.build_legend_context(self.state.state)
        if legend_ctx and char_count < max_chars - 200:
            parts.append(f"\n{legend_ctx}")
            char_count += len(legend_ctx)

        # ─── Part 2.8: Unresolved Events ─────────────────────────────
        story_events = self.state.state.get("plot", {}).get("story_events", [])
        unresolved = EventExtractor.get_unresolved_events(story_events)
        if unresolved and char_count < max_chars - 200:
            events_summary = EventExtractor.get_events_summary(unresolved, max_events=5, unresolved_only=True)
            if events_summary:
                events_block = f"\n=== UNRESOLVED PLOT THREADS ===\n{events_summary}"
                parts.append(events_block)
                char_count += len(events_block)

        # ─── Part 2.9: Continuity Anchor ─────────────────────────────
        if previous_ending:
            anchor = " ".join(previous_ending.split())
            if len(anchor) > 420:
                anchor = anchor[-420:].lstrip()
            anchor_block = (
                "\n=== IMMEDIATE CONTINUITY ANCHOR ===\n"
                "Continue directly from this moment before moving to new beats:\n"
                f"{anchor}"
            )
            if char_count + len(anchor_block) < max_chars:
                parts.append(anchor_block)
                char_count += len(anchor_block)

        # ─── Part 3: Semantic Retrieval with Hybrid Scoring ───────────
        continuity_query = ""
        if previous_ending:
            continuity_query = "Previous ending: " + " ".join(previous_ending.split())[:240]
        merged_query = " ".join(q for q in [continuity_query, extra_query] if q).strip()
        query = self._build_search_query(scene_plan, merged_query)
        if query and self.vectors.index.ntotal > 0:
            results = self.vectors.search(query, top_k=top_k * 2)  # Over-fetch for re-ranking

            if results:
                # Re-rank with hybrid scoring
                reranked = self._hybrid_rerank(
                    results, scene_plan, chapter_num,
                    self.state.get_characters(),
                    self.state.state.get("plot", {}).get("unresolved_threads", []),
                )

                parts.append("\n=== RELEVANT PAST CONTENT (REFERENCE ONLY - DO NOT COPY PROSE) ===")
                for result in reranked[:top_k]:
                    chunk_text = self._compact_excerpt(result.text)
                    meta = result.metadata

                    if char_count + len(chunk_text) > max_chars:
                        remaining = max_chars - char_count - 50
                        if remaining > 100:
                            chunk_text = chunk_text[:remaining] + "..."
                        else:
                            break

                    source = f"[Ch{meta.chapter}/Sc{meta.scene}]"
                    parts.append(f"\n{source} {chunk_text}")
                    char_count += len(chunk_text) + len(source) + 2

        # ─── Part 4: Chapter Summaries ─────────────────────────────────
        plot = self.state.get_plot()
        summaries = plot.get("chapter_summaries", [])
        if summaries and char_count < max_chars - 200:
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

    def _hybrid_rerank(
        self,
        results,
        scene_plan: dict,
        chapter_num: int,
        characters: dict,
        unresolved_threads: list,
        backstory_query: bool = False,
    ) -> list:
        """Re-rank search results using recency-weighted hybrid scoring.

        Primary formula (per spec):
            recency_score = 1.0 - 0.05 * (current_chapter - chunk_chapter)
            final_score   = semantic_score * 0.7 + recency_score * 0.3

        Secondary tie-breaker signals (character overlap, thread relevance,
        emotional keywords) are applied within the remaining headroom so that
        recent chapters almost always outweigh older ones unless the query
        explicitly references backstory.
        """
        scene_chars = set(c.lower() for c in scene_plan.get("characters_present", []))
        thread_text = " ".join(unresolved_threads).lower()

        scored = []
        for result in results:
            semantic = float(result.score)
            meta = result.metadata

            # ── Recency weight (spec-mandated formula) ────────────────────
            chapter_gap = max(0, chapter_num - meta.chapter)
            recency_score = max(0.0, 1.0 - 0.05 * chapter_gap)

            # When the query is explicitly about backstory, recency penalty is
            # halved so that older chapters can surface naturally.
            if backstory_query:
                recency_score = max(0.0, 1.0 - 0.025 * chapter_gap)

            # ── Spec-mandated combination ─────────────────────────────────
            base_score = semantic * 0.7 + recency_score * 0.3

            # ── Secondary tie-breaker signals (small weight) ──────────────
            chunk_chars = set(c.lower() for c in (meta.characters or []))
            if scene_chars and chunk_chars:
                overlap = len(scene_chars & chunk_chars) / max(1, len(scene_chars))
            else:
                overlap = 0.0

            text_lower = result.text.lower()
            unresolved_score = 0.0
            if thread_text:
                thread_words = set(thread_text.split())
                text_words = set(text_lower.split())
                common = len(thread_words & text_words)
                unresolved_score = min(1.0, common / max(1, len(thread_words)) * 3)

            emotional_keywords = {"afraid", "angry", "love", "betray", "trust", "grief", "joy", "despair"}
            emo_count = sum(1 for kw in emotional_keywords if kw in text_lower)
            emotional_score = min(1.0, emo_count * 0.25)

            # Tie-breaker: up to +0.05 bonus — never overrides recency dominance
            tiebreaker = (overlap * 0.5 + unresolved_score * 0.35 + emotional_score * 0.15) * 0.05

            result._hybrid_score = base_score + tiebreaker
            scored.append(result)

        scored.sort(key=lambda r: r._hybrid_score, reverse=True)
        return scored

    def _build_search_query(self, scene_plan: dict, extra: str = "") -> str:
        """Build a semantic search query from the scene plan."""
        query_parts = []
        if scene_plan.get("summary"):
            query_parts.append(scene_plan["summary"])
        if scene_plan.get("characters_present"):
            query_parts.append("Characters: " + ", ".join(scene_plan["characters_present"]))
        if scene_plan.get("location"):
            query_parts.append(f"Location: {scene_plan['location']}")
        if scene_plan.get("key_events"):
            query_parts.append("Events: " + ", ".join(scene_plan["key_events"]))
        if extra:
            query_parts.append(extra)
        return " ".join(query_parts)

    @staticmethod
    def _compact_excerpt(text: str, max_chars: int = 420) -> str:
        """Keep retrieved memory useful as facts without inviting verbatim copying."""
        cleaned = " ".join((text or "").split())
        if len(cleaned) <= max_chars:
            return cleaned
        return cleaned[:max_chars].rsplit(" ", 1)[0].rstrip() + "..."

    def retrieve_for_consistency(
        self,
        scene_text: str,
        characters: list = None,
    ) -> str:
        """Retrieve context specifically for the Consistency Engine."""
        parts = []

        parts.append("=== CHARACTER STATES ===")
        all_chars = self.state.get_characters()
        target_chars = {k: v for k, v in all_chars.items() if k in (characters or [])} \
            if characters else all_chars

        for name, info in target_chars.items():
            parts.append(f"\n{name}:")
            parts.append(f"  Traits: {', '.join(info.get('traits', []))}")
            if info.get("relationships"):
                for other, rel in info["relationships"].items():
                    if isinstance(rel, dict):
                        parts.append(f"  → {other}: {rel.get('type', 'neutral')} (trust={rel.get('trust', 0.5):.1f})")
                    else:
                        parts.append(f"  → {other}: {rel}")
            if info.get("state"):
                for k, v in info["state"].items():
                    if v:
                        parts.append(f"  {k}: {v}")
            status = info.get("status", "active")
            if status != "active":
                parts.append(f"  Status: {status}")

        if self.vectors.index.ntotal > 0:
            results = self.vectors.search(scene_text[:500], top_k=3)
            if results:
                parts.append("\n=== RELATED PAST SCENES ===")
                for r in results:
                    parts.append(
                        f"[Ch{r.metadata.chapter}/Sc{r.metadata.scene}] "
                        f"{r.text[:300]}"
                    )

        world = self.state.get_world()
        if world.get("rules"):
            parts.append("\n=== WORLD RULES ===")
            for rule in world["rules"]:
                parts.append(f"- {rule}")

        return "\n".join(parts)
