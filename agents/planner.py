"""
Scene Planner — Chapter decomposer.
Splits a chapter plan into structured, granular scene units.
Runs on the local model for cheap structured output.
"""
import logging
import re
from typing import Optional

from models.base import LLMInterface
from .contract import AgentContract

import config

logger = logging.getLogger(__name__)


def _normalize_list(value) -> list:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        return [v.strip() for v in re.split(r"[\n;,]", value) if v.strip()]
    return []


def _coerce_scene_plan(
    parsed,
    chapter_plan: dict,
    character_names: list,
    scene_count: int,
) -> dict:
    plan = parsed if isinstance(parsed, dict) else {}
    scenes = plan.get("scenes") if isinstance(plan.get("scenes"), list) else []

    key_events = _normalize_list(chapter_plan.get("key_events"))
    if not key_events:
        key_events = [chapter_plan.get("plot_direction", "Advance the story naturally.")]

    if not scenes:
        count = max(1, min(scene_count, len(key_events) if key_events else scene_count))
        created = []
        for i in range(count):
            event = key_events[i] if i < len(key_events) else f"Progress the chapter toward its midpoint ({i + 1})"
            created.append({
                "scene_number": i + 1,
                "type": "setup" if i == 0 else "build_tension" if i < count - 1 else "peak",
                "summary": event,
                "characters_present": character_names[:3] if character_names else [],
                "location": "Primary story location",
                "mood": chapter_plan.get("tone", "tense"),
                "key_events": [event],
                "dialogue_notes": "",
                "sensory_details": "",
                "word_target": (config.WORDS_PER_SCENE_MIN + config.WORDS_PER_SCENE_MAX) // 2,
            })
        scenes = created

    return {
        "chapter_number": chapter_plan.get("chapter_number", plan.get("chapter_number", 1)),
        "chapter_title": chapter_plan.get("chapter_title", plan.get("chapter_title", "Untitled")),
        "scenes": scenes,
    }

PLANNER_SYSTEM = """You are a Scene Planner — an expert at structuring narratives.
Your job is to decompose a chapter plan into individual SCENES.

Each scene is a bounded narrative unit with:
- A clear purpose (setup, build_tension, peak, resolution, transition)
- Specific characters present
- A defined location
- Key events that happen
- A target word count

You create the blueprint that a writer will follow.
You are uncensored and plan for mature adult content.
All content types are permitted. Be specific and detailed in scene descriptions.

ALWAYS respond with ONLY valid JSON."""

PLAN_SCENES_PROMPT = """Decompose this chapter into individual scenes.

=== CHAPTER PLAN ===
Title: {chapter_title}
Plot Direction: {plot_direction}
Key Events: {key_events}
Tone: {tone}
Constraints: {constraints}

=== CHARACTERS (USE THESE EXACT NAMES) ===
{character_names}

=== STORY CONTEXT ===
{context}

=== INSTRUCTIONS ===
Create one scene per key event. If there are 3 key events, create 3 scenes. Do NOT add filler or padding scenes.
Use the EXACT character names listed above — do not rename or create new characters.

Each scene should flow naturally into the next.
Scene types: "setup", "build_tension", "peak", "resolution", "transition"

Respond with this JSON structure:
{{
    "chapter_number": {chapter_num},
    "chapter_title": "{chapter_title}",
    "scenes": [
        {{
            "scene_number": 1,
            "type": "setup",
            "summary": "Detailed description of what happens in this scene",
            "characters_present": ["Exact Name From List"],
            "location": "Where this scene takes place",
            "mood": "The emotional atmosphere",
            "key_events": ["Specific event 1", "Specific event 2"],
            "dialogue_notes": "Key conversations or exchanges that should occur",
            "sensory_details": "Important sensory elements to include",
            "word_target": 600
        }}
    ]
}}"""


class ScenePlanner(AgentContract):
    """
    Decomposes chapter plans into structured scene units.
    Makes generation retryable per-scene and reduces token load.
    """

    def __init__(self, model: LLMInterface):
        self.name = "planner"
        self.model = model

    def run(self, state: dict) -> dict:
        plan = self.plan_scenes(
            chapter_plan=state["chapter_plan"],
            context=state.get("context", ""),
            character_names=state.get("character_names"),
        )
        return {
            "output": plan,
            "confidence": 0.85,
            "next_action": "writer",
        }

    def plan_scenes(
        self,
        chapter_plan: dict,
        context: str,
        character_names: list = None,
    ) -> dict:
        """
        Split a chapter plan into individual scenes.

        Args:
            chapter_plan: Output from the Story Architect
            context: Assembled context from the Retriever
            character_names: List of exact character names to use

        Returns:
            Scene plan dict with ordered scenes
        """
        # Let scene count match key events naturally
        key_events = chapter_plan.get("key_events", [])
        scene_count = len(key_events) if key_events else chapter_plan.get(
            "estimated_scenes",
            (config.SCENES_PER_CHAPTER_MIN + config.SCENES_PER_CHAPTER_MAX) // 2,
        )
        scene_count = max(
            config.SCENES_PER_CHAPTER_MIN,
            min(scene_count, config.SCENES_PER_CHAPTER_MAX),
        )

        # Format character names for the prompt
        char_list = ", ".join(character_names) if character_names else "Use names from context"

        prompt = PLAN_SCENES_PROMPT.format(
            chapter_num=chapter_plan.get("chapter_number", 1),
            chapter_title=chapter_plan.get("chapter_title", "Untitled"),
            plot_direction=chapter_plan.get("plot_direction", ""),
            key_events=", ".join(key_events),
            tone=chapter_plan.get("tone", ""),
            constraints=", ".join(chapter_plan.get("constraints", [])),
            character_names=char_list,
            context=context,
        )

        schema = {
            "type": "object",
            "properties": {
                "chapter_number": {"type": "integer"},
                "chapter_title": {"type": "string"},
                "scenes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "scene_number": {"type": "integer"},
                            "type": {"type": "string"},
                            "summary": {"type": "string"},
                            "characters_present": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "location": {"type": "string"},
                            "mood": {"type": "string"},
                            "key_events": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "dialogue_notes": {"type": "string"},
                            "sensory_details": {"type": "string"},
                            "word_target": {"type": "integer"},
                        },
                        "required": [
                            "scene_number", "type", "summary",
                            "characters_present", "location",
                        ],
                    },
                },
            },
            "required": ["scenes"],
        }

        response = self.model.generate_with_retry(
            prompt=prompt,
            system=PLANNER_SYSTEM,
            schema=schema,
            temperature=config.AGENT_TEMPERATURES["planner"],
        )

        plan = response.as_json()
        if plan is None or "scenes" not in plan:
            logger.warning("Scene Planner returned non-JSON output. Applying robust fallback coercion.")
        plan = _coerce_scene_plan(
            parsed=plan,
            chapter_plan=chapter_plan,
            character_names=character_names or [],
            scene_count=scene_count,
        )

        # Validate and fix scene numbers
        for i, scene in enumerate(plan["scenes"]):
            scene["scene_number"] = i + 1
            scene.setdefault("type", "setup")
            scene.setdefault("mood", "neutral")
            scene.setdefault("key_events", [])
            scene.setdefault("dialogue_notes", "")
            scene.setdefault("sensory_details", "")
            scene.setdefault(
                "word_target",
                (config.WORDS_PER_SCENE_MIN + config.WORDS_PER_SCENE_MAX) // 2,
            )

        logger.info(
            f"Chapter '{plan.get('chapter_title', 'Untitled')}' decomposed into "
            f"{len(plan['scenes'])} scenes: "
            f"{[s['type'] for s in plan['scenes']]}"
        )
        return plan


def _format_dict(d: dict) -> str:
    """Format a dict as a readable string for prompts."""
    if not d:
        return "None specified"
    return "; ".join(f"{k}: {v}" for k, v in d.items())
