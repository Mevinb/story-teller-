"""
Narrative Evolution Engine — Central post-scene evolution orchestrator.

This is the heart of the dynamic storytelling system. After each scene,
it orchestrates:
1. Entity extraction (new characters, aliases)
2. Emotional timeline updates
3. Relationship evolution
4. Major event extraction
5. Character importance scoring
6. Arc progression tracking
7. Scene transition state generation
8. Narrative phase detection

Uses a single structured LLM call to extract all evolution data at once,
then dispatches updates to the specialized modules.
"""
import json
import re
import logging
from typing import Optional

from models.base import LLMInterface
from .relationship_graph import RelationshipGraph
from .arc_tracker import ArcTracker
from .event_extractor import EventExtractor
from .importance_ranker import ImportanceRanker
from .memory_compressor import MemoryCompressor

logger = logging.getLogger(__name__)

# ─── LLM Prompt for Evolution Extraction ─────────────────────────────

EVOLUTION_SYSTEM = (
    "You are a Narrative Evolution Analyzer. After reading a scene, you extract "
    "structured data about character development, relationships, emotions, events, "
    "and story progression. You ONLY output valid JSON. No commentary."
)

EVOLUTION_PROMPT = """Analyze this scene and extract narrative evolution data.

=== SCENE TEXT ===
{scene_text}

=== SCENE PLAN ===
{scene_plan}

=== KNOWN CHARACTERS ===
{known_characters}

=== CHAPTER/SCENE ===
Chapter {chapter}, Scene {scene}

=== INSTRUCTIONS ===
Extract ALL of the following from the scene. Be thorough but concise.

Return this exact JSON structure:
{{
    "entities": [
        {{
            "name": "character name (canonical form)",
            "aliases": ["any aliases or nicknames used"],
            "role": "main|supporting|minor",
            "status": "active|missing|dead|retired|imprisoned|corrupted",
            "is_new": true
        }}
    ],
    "emotional_updates": [
        {{
            "character": "name",
            "emotion": "current emotion after scene",
            "cause": "what caused this emotion",
            "intensity": 0.8
        }}
    ],
    "relationship_updates": [
        {{
            "character_a": "name",
            "character_b": "name",
            "type": "ally|friend|rival|enemy|lover|family|mentor|neutral|complicated",
            "trust_change": 0.1,
            "romantic_tension_change": 0.0,
            "hostility_change": -0.1,
            "event": "what happened between them"
        }}
    ],
    "events": [
        {{
            "type": "betrayal|alliance|death|injury|discovery|revelation|confrontation|escape|arrival|departure|romance|breakup|reunion|sacrifice|transformation|power_shift|secret_revealed|promise|threat|loss|victory|capture|rescue|training",
            "characters": ["involved characters"],
            "description": "what happened",
            "impact": "high|medium|low"
        }}
    ],
    "arc_updates": [
        {{
            "character": "name",
            "event": "arc-relevant event description",
            "arc_direction": "positive|negative|healing|corrupting|flat|neutral",
            "current_phase": "what phase this puts them in"
        }}
    ],
    "transition_state": {{
        "time_elapsed": "how much time passed during scene",
        "physical_states": {{
            "character_name": "physical state after scene"
        }},
        "open_conversations": ["unfinished dialogue threads"],
        "environmental_carryover": ["weather, location state, etc."],
        "emotional_carryover": ["lingering tensions or emotions"]
    }},
    "narrative_phase": "introduction|escalation|midpoint|collapse|climax|aftermath|null"
}}

RULES:
- Only include entities that actually appear or are mentioned in the scene.
- For emotional_updates, describe the CURRENT emotion after the scene events, not before.
- For relationship_updates, only include relationships that CHANGED in this scene.
- For events, only include SIGNIFICANT plot events, not routine actions.
- For arc_updates, focus on character TRANSFORMATIONS, not static traits.
- Set narrative_phase to null if the scene doesn't clearly shift the story phase.
- For trust/romantic/hostility changes, use small increments (-0.3 to +0.3).
"""

EVOLUTION_PROMPT_COMPACT = """Analyze scene for narrative evolution. Chapter {chapter}, Scene {scene}.

SCENE: {scene_text}
CHARACTERS: {known_characters}

Return JSON with: entities[], emotional_updates[], relationship_updates[], events[], arc_updates[], transition_state{{}}, narrative_phase.
Keep concise. Only include data that CHANGED in this scene."""


# Narrative phase keywords for auto-detection
_PHASE_KEYWORDS = {
    "introduction": {"introduce", "meet", "discover", "begin", "arrive", "first"},
    "escalation": {"tension", "conflict", "challenge", "pursue", "confront", "threaten"},
    "midpoint": {"revelation", "twist", "turning point", "realize", "truth"},
    "collapse": {"fail", "lose", "betray", "break", "fall", "despair"},
    "climax": {"final", "decisive", "battle", "showdown", "ultimate"},
    "aftermath": {"aftermath", "resolve", "peace", "rebuild", "heal", "ending"},
}


class EvolutionEngine:
    """
    Central orchestrator for post-scene narrative evolution.

    Call evolve_after_scene() after each scene to update:
    - Character entities and aliases
    - Emotional histories
    - Relationship dynamics
    - Story events
    - Importance scores
    - Character arcs
    - Transition states
    - Narrative phase
    """

    def __init__(self, compact_mode: bool = False):
        self._compact_mode = compact_mode

    def set_compact_mode(self, enabled: bool):
        self._compact_mode = bool(enabled)

    def evolve_after_scene(
        self,
        scene_text: str,
        scene_plan: dict,
        chapter_num: int,
        scene_num: int,
        state_manager,
        llm: LLMInterface,
        progress_callback=None,
    ) -> dict:
        """
        Run the full evolution pipeline after a scene.

        Args:
            scene_text: The generated scene prose.
            scene_plan: The scene plan dict.
            chapter_num: Current chapter number.
            scene_num: Current scene number.
            state_manager: StateManager instance.
            llm: LLM interface for extraction.
            progress_callback: Optional callback for progress updates.

        Returns:
            Summary dict of what was updated.
        """
        cb = progress_callback or (lambda msg: None)
        summary = {
            "entities_found": 0,
            "emotions_updated": 0,
            "relationships_updated": 0,
            "events_extracted": 0,
            "arcs_updated": 0,
            "legend_promoted": 0,
            "narrative_phase": None,
        }

        if not scene_text or len(scene_text.strip()) < 50:
            logger.warning("Scene text too short for evolution analysis")
            return summary

        # Step 1: Extract evolution data via LLM
        cb("Analyzing narrative evolution...")
        evolution_data = self._extract_evolution_data(
            scene_text=scene_text,
            scene_plan=scene_plan,
            chapter_num=chapter_num,
            scene_num=scene_num,
            state_manager=state_manager,
            llm=llm,
        )

        if not evolution_data:
            logger.warning("Evolution extraction returned no data")
            # Still update importance scores from text analysis
            self._update_importance_scores(state_manager, scene_text, chapter_num)
            return summary

        # Step 2: Process entities (new characters, aliases)
        cb("Processing character entities...")
        entities = evolution_data.get("entities", [])
        summary["entities_found"] = self._process_entities(
            entities, state_manager, chapter_num,
        )

        # Step 3: Update emotional histories
        cb("Updating emotional timelines...")
        emotional_updates = evolution_data.get("emotional_updates", [])
        summary["emotions_updated"] = self._process_emotional_updates(
            emotional_updates, state_manager, chapter_num,
        )

        # Step 4: Update relationships
        cb("Evolving relationship dynamics...")
        relationship_updates = evolution_data.get("relationship_updates", [])
        summary["relationships_updated"] = self._process_relationship_updates(
            relationship_updates, state_manager, chapter_num,
        )

        # Step 5: Extract and store events
        cb("Extracting story events...")
        events = evolution_data.get("events", [])
        summary["events_extracted"] = self._process_events(
            events, state_manager, chapter_num, scene_num,
        )

        # Step 6: Update character arcs
        cb("Tracking character arcs...")
        arc_updates = evolution_data.get("arc_updates", [])
        summary["arcs_updated"] = self._process_arc_updates(
            arc_updates, state_manager, chapter_num,
        )

        # Step 7: Store transition state
        cb("Saving transition state...")
        transition = evolution_data.get("transition_state", {})
        if isinstance(transition, dict) and transition:
            self._process_transition_state(
                transition, state_manager, chapter_num, scene_num,
            )

        # Step 8: Update narrative phase
        phase = evolution_data.get("narrative_phase")
        if phase and phase != "null" and isinstance(phase, str):
            state_manager.set_narrative_phase(phase.strip().lower())
            summary["narrative_phase"] = phase.strip().lower()

        # Step 9: Update importance scores (always, from text analysis)
        cb("Updating importance scores...")
        self._update_importance_scores(state_manager, scene_text, chapter_num)

        # Step 10: Promote legend events
        promoted = MemoryCompressor.promote_events_to_legend(
            state_manager.state,
            self._get_story_events(state_manager),
        )
        summary["legend_promoted"] = len(promoted)
        if promoted:
            state_manager.save()

        logger.info(
            "Evolution complete: %d entities, %d emotions, %d relationships, "
            "%d events, %d arcs, %d legends",
            summary["entities_found"],
            summary["emotions_updated"],
            summary["relationships_updated"],
            summary["events_extracted"],
            summary["arcs_updated"],
            summary["legend_promoted"],
        )

        return summary

    def _extract_evolution_data(
        self,
        scene_text: str,
        scene_plan: dict,
        chapter_num: int,
        scene_num: int,
        state_manager,
        llm: LLMInterface,
    ) -> Optional[dict]:
        """Use LLM to extract structured evolution data from scene text."""
        # Build known characters list
        characters = state_manager.get_characters()
        char_list = []
        for name, data in characters.items():
            status = data.get("status", "active")
            role = data.get("role", "supporting")
            char_list.append(f"{name} ({role}, {status})")
        known_chars = ", ".join(char_list) if char_list else "None known yet"

        # Format scene plan
        plan_str = ""
        if isinstance(scene_plan, dict):
            plan_str = f"Summary: {scene_plan.get('summary', '')}"
            if scene_plan.get("characters_present"):
                plan_str += f"\nCharacters: {', '.join(scene_plan['characters_present'])}"

        # Build prompt
        text_limit = 2000 if self._compact_mode else 3500
        if self._compact_mode:
            prompt = EVOLUTION_PROMPT_COMPACT.format(
                scene_text=scene_text[:text_limit],
                known_characters=known_chars,
                chapter=chapter_num,
                scene=scene_num,
            )
        else:
            prompt = EVOLUTION_PROMPT.format(
                scene_text=scene_text[:text_limit],
                scene_plan=plan_str[:500],
                known_characters=known_chars,
                chapter=chapter_num,
                scene=scene_num,
            )

        schema = {
            "type": "object",
            "properties": {
                "entities": {"type": "array"},
                "emotional_updates": {"type": "array"},
                "relationship_updates": {"type": "array"},
                "events": {"type": "array"},
                "arc_updates": {"type": "array"},
                "transition_state": {"type": "object"},
                "narrative_phase": {"type": "string"},
            },
        }

        try:
            response = llm.generate_with_retry(
                prompt=prompt,
                system=EVOLUTION_SYSTEM,
                schema=schema,
                temperature=0.1,  # Low temp for analysis
                max_tokens=1200 if self._compact_mode else 2000,
                max_retries=2,
            )
            result = response.as_json()
            if isinstance(result, dict):
                return result
            logger.warning("Evolution LLM returned non-dict: %s", type(result))
            return None
        except Exception as e:
            logger.error("Evolution extraction failed: %s", e)
            return None

    def _process_entities(
        self,
        entities: list,
        state_manager,
        chapter_num: int,
    ) -> int:
        """Process extracted entities — register new characters and aliases."""
        if not isinstance(entities, list):
            return 0

        count = 0
        for entity in entities:
            if not isinstance(entity, dict):
                continue

            name = str(entity.get("name", "")).strip()
            if not name or len(name) < 2:
                continue

            is_new = bool(entity.get("is_new", False))
            aliases = entity.get("aliases", [])
            if isinstance(aliases, str):
                aliases = [aliases]
            aliases = [str(a).strip() for a in aliases if str(a).strip()]

            role = str(entity.get("role", "supporting")).strip().lower()
            if role not in {"main", "supporting", "minor"}:
                role = "supporting"

            status = str(entity.get("status", "active")).strip().lower()
            valid_statuses = {"active", "missing", "dead", "retired", "imprisoned", "corrupted"}
            if status not in valid_statuses:
                status = "active"

            # Check if character already exists
            existing = state_manager.get_character(name)
            if existing:
                # Update aliases and status
                updates = {}
                if aliases:
                    existing_aliases = existing.get("aliases", [])
                    new_aliases = [a for a in aliases if a not in existing_aliases]
                    if new_aliases:
                        updates["aliases"] = existing_aliases + new_aliases
                if status != "active" and status != existing.get("status", "active"):
                    updates["status"] = status
                updates["last_seen"] = chapter_num
                if updates:
                    state_manager.update_character(name, updates)
                    count += 1
            elif is_new:
                # Register new character
                new_char = {
                    "description": "",
                    "role": role,
                    "status": status,
                    "aliases": aliases,
                    "first_seen": chapter_num,
                    "last_seen": chapter_num,
                    "mention_count": 0,
                    "importance_score": 0.3 if role == "main" else 0.1,
                    "generated": True,
                }
                state_manager.apply_state_update({"characters": {name: new_char}})
                count += 1
                logger.info("Evolution: registered new character '%s' (%s)", name, role)

        return count

    def _process_emotional_updates(
        self,
        updates: list,
        state_manager,
        chapter_num: int,
    ) -> int:
        """Process emotional state updates for characters."""
        if not isinstance(updates, list):
            return 0

        count = 0
        for update in updates:
            if not isinstance(update, dict):
                continue

            char_name = str(update.get("character", "")).strip()
            emotion = str(update.get("emotion", "")).strip()
            cause = str(update.get("cause", "")).strip()

            if not char_name or not emotion:
                continue

            # Update current emotional state
            state_update = {"emotion": emotion}
            state_manager.update_character(char_name, {"state": state_update})

            # Append to emotional history
            intensity = 0.5
            try:
                intensity = float(update.get("intensity", 0.5))
                intensity = max(0.0, min(1.0, intensity))
            except (TypeError, ValueError):
                pass

            history_entry = {
                "chapter": chapter_num,
                "emotion": emotion,
                "cause": cause,
                "intensity": intensity,
            }
            state_manager.update_emotional_history(char_name, history_entry)
            count += 1

        return count

    def _process_relationship_updates(
        self,
        updates: list,
        state_manager,
        chapter_num: int,
    ) -> int:
        """Process relationship evolution updates."""
        if not isinstance(updates, list):
            return 0

        count = 0
        for update in updates:
            if not isinstance(update, dict):
                continue

            char_a = str(update.get("character_a", "")).strip()
            char_b = str(update.get("character_b", "")).strip()
            if not char_a or not char_b:
                continue

            rel_update = {}
            if "type" in update:
                rel_update["type"] = update["type"]
            if "event" in update:
                rel_update["event"] = update["event"]

            # Process score changes
            for key, field in [
                ("trust_change", "trust"),
                ("romantic_tension_change", "romantic_tension"),
                ("hostility_change", "hostility"),
            ]:
                if key in update:
                    try:
                        change = float(update[key])
                        # Get current value from character's relationship
                        char_data = state_manager.get_character(char_a)
                        if char_data:
                            rels = char_data.get("relationships", {})
                            existing = RelationshipGraph.ensure_rich_relationship(
                                rels.get(char_b)
                            )
                            current = float(existing.get(field, 0.5))
                            rel_update[field] = max(0.0, min(1.0, current + change))
                    except (TypeError, ValueError):
                        pass

            if rel_update:
                # Update A's view of B
                char_data = state_manager.get_character(char_a)
                if char_data:
                    rels = char_data.get("relationships", {})
                    rels = RelationshipGraph.update_relationship(
                        rels, char_b, rel_update, chapter_num,
                    )
                    state_manager.update_character(char_a, {"relationships": rels})

                # Update B's view of A (mirror relationship)
                char_data_b = state_manager.get_character(char_b)
                if char_data_b:
                    rels_b = char_data_b.get("relationships", {})
                    rels_b = RelationshipGraph.update_relationship(
                        rels_b, char_a, rel_update, chapter_num,
                    )
                    state_manager.update_character(char_b, {"relationships": rels_b})

                count += 1

        return count

    def _process_events(
        self,
        events: list,
        state_manager,
        chapter_num: int,
        scene_num: int,
    ) -> int:
        """Process and store extracted story events."""
        if not isinstance(events, list):
            return 0

        # Add chapter/scene to each event
        for event in events:
            if isinstance(event, dict):
                event["chapter"] = chapter_num
                event["scene"] = scene_num

        normalized = EventExtractor.normalize_events(events)
        if not normalized:
            return 0

        for event in normalized:
            state_manager.add_story_event(event)

            # Also update the character's event involvement count
            for char_name in event.get("characters", []):
                char_data = state_manager.get_character(char_name)
                if char_data:
                    involvement = int(char_data.get("story_events_involved", 0))
                    state_manager.update_character(
                        char_name,
                        {"story_events_involved": involvement + 1},
                    )

        return len(normalized)

    def _process_arc_updates(
        self,
        updates: list,
        state_manager,
        chapter_num: int,
    ) -> int:
        """Process character arc progression updates."""
        if not isinstance(updates, list):
            return 0

        count = 0
        for update in updates:
            if not isinstance(update, dict):
                continue

            char_name = str(update.get("character", "")).strip()
            if not char_name:
                continue

            char_data = state_manager.get_character(char_name)
            if not char_data:
                continue

            arc_update = {}
            if "event" in update:
                arc_update["event"] = update["event"]
            if "arc_direction" in update:
                arc_update["arc_direction"] = update["arc_direction"]
            if "current_phase" in update:
                arc_update["current_phase"] = update["current_phase"]
            if "core_wound" in update:
                arc_update["core_wound"] = update["core_wound"]
            if "false_belief" in update:
                arc_update["false_belief"] = update["false_belief"]

            if arc_update:
                updated_data = ArcTracker.update_arc(char_data, arc_update, chapter_num)
                # Persist arc_progression_data
                state_manager.update_character(
                    char_name,
                    {"arc_progression_data": updated_data.get("arc_progression_data", {})},
                )
                count += 1

        return count

    def _process_transition_state(
        self,
        transition: dict,
        state_manager,
        chapter_num: int,
        scene_num: int,
    ) -> None:
        """Store the scene transition state."""
        transition_entry = {
            "from_chapter": chapter_num,
            "from_scene": scene_num,
            "to_scene": scene_num + 1,
            "time_elapsed": str(transition.get("time_elapsed", "")).strip(),
            "physical_states": transition.get("physical_states", {}),
            "open_conversations": transition.get("open_conversations", []),
            "environmental_carryover": transition.get("environmental_carryover", []),
            "emotional_carryover": transition.get("emotional_carryover", []),
        }
        state_manager.add_transition_state(transition_entry)

    def _update_importance_scores(
        self,
        state_manager,
        scene_text: str,
        chapter_num: int,
    ) -> None:
        """Update importance scores for all characters based on scene text."""
        characters = state_manager.get_characters()
        updated = ImportanceRanker.update_all_characters(
            characters, scene_text, chapter_num,
        )
        # Persist updated scores
        for name, data in updated.items():
            state_manager.update_character(name, {
                "mention_count": data.get("mention_count", 0),
                "dialogue_density": data.get("dialogue_density", 0.0),
                "importance_score": data.get("importance_score", 0.5),
                "last_seen": data.get("last_seen", 0),
            })

    @staticmethod
    def _get_story_events(state_manager) -> list:
        """Get all story events from state."""
        plot = state_manager.state.get("plot", {})
        return plot.get("story_events", [])


def evolve_after_scene(
    scene_text: str,
    scene_plan: dict,
    chapter_num: int,
    scene_num: int,
    state_manager,
    llm: LLMInterface,
    compact_mode: bool = False,
    progress_callback=None,
) -> dict:
    """
    Convenience function — create an EvolutionEngine and run evolution.
    This is the main entry point called from the orchestrator.
    """
    engine = EvolutionEngine(compact_mode=compact_mode)
    return engine.evolve_after_scene(
        scene_text=scene_text,
        scene_plan=scene_plan,
        chapter_num=chapter_num,
        scene_num=scene_num,
        state_manager=state_manager,
        llm=llm,
        progress_callback=progress_callback,
    )
