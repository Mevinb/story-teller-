"""
Scene Planner — Chapter decomposer.
Splits a chapter plan into structured, granular scene units.
Runs on the local model for cheap structured output.
"""
import logging
import re


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

    protagonist = character_names[:1] if character_names else []

    if not scenes:
        count = max(1, min(scene_count, len(key_events) if key_events else scene_count))
        created = []
        for i in range(count):
            event = key_events[i] if i < len(key_events) else f"Progress the chapter toward its midpoint ({i + 1})"
            created.append({
                "scene_number": i + 1,
                "type": "setup" if i == 0 else "build_tension" if i < count - 1 else "peak",
                "summary": event,
                "characters_present": protagonist,
                "location": "As established in the story",
                "mood": chapter_plan.get("tone", "tense"),
                "key_events": [event],
                "dialogue_notes": "",
                "sensory_details": "",
                "intimacy_level": "none",
                "word_target": (config.WORDS_PER_SCENE_MIN + config.WORDS_PER_SCENE_MAX) // 2,

            })
        scenes = created

    return {
        "chapter_number": chapter_plan.get("chapter_number", plan.get("chapter_number", 1)),
        "chapter_title": chapter_plan.get("chapter_title", plan.get("chapter_title", "Untitled")),
        "scenes": scenes,
    }


def _canonical_name_key(name: str) -> str:
    base = re.sub(r"\([^)]*\)", "", str(name or "")).strip().lower()
    return re.sub(r"[^a-z0-9]+", "", base)


def _normalize_scene_characters(raw_names, exact_names: list) -> list:
    alias_map = {}
    for exact in exact_names:
        alias_map[str(exact).lower()] = exact
        alias_map[_canonical_name_key(exact)] = exact

    normalized = []
    for name in _normalize_list(raw_names):
        exact = alias_map.get(name.lower()) or alias_map.get(_canonical_name_key(name))
        chosen = exact or name
        if chosen and chosen not in normalized:
            normalized.append(chosen)
    return normalized


_INTIMACY_LEVELS = ("none", "romantic", "sensual", "explicit")
_EXPLICIT_SCENE_KEYWORDS = (
    "explicit", "sex", "sexual", "intercourse", "penetrat", "oral", "blowjob",
    "handjob", "fingering", "threesome", "orgy", "fuck", "cock", "pussy",
    "clit", "orgasm", "cum", "ejaculat", "anal", "masturbat",
)
_SENSUAL_SCENE_KEYWORDS = (
    "kiss", "kissing", "make out", "undress", "caress", "touch", "tease",
    "foreplay", "arousal", "desire", "seduce", "heated", "steamy",
)
_ROMANTIC_SCENE_KEYWORDS = (
    "romantic", "intimate", "tender", "date", "confession", "cuddle",
)


def _scene_text_blob(scene: dict) -> str:
    parts = [
        str(scene.get("summary", "")),
        str(scene.get("dialogue_notes", "")),
        str(scene.get("sensory_details", "")),
    ]
    events = scene.get("key_events", [])
    if isinstance(events, list):
        parts.extend([str(event) for event in events if event])
    elif events:
        parts.append(str(events))
    return " ".join([part for part in parts if part]).lower()


def _normalize_intimacy_level(raw_level: str, scene: dict) -> str:
    lowered = str(raw_level or "").strip().lower()
    if lowered:
        if lowered in _INTIMACY_LEVELS:
            return lowered
        if any(key in lowered for key in ("explicit", "erotic", "sex", "sexual")):
            return "explicit"
        if any(key in lowered for key in ("sensual", "steamy", "heated")):
            return "sensual"
        if any(key in lowered for key in ("romantic", "intimate", "tender")):
            return "romantic"
        if any(key in lowered for key in ("none", "clean", "fade", "closed")):
            return "none"

    text = _scene_text_blob(scene)
    if any(key in text for key in _EXPLICIT_SCENE_KEYWORDS):
        return "explicit"
    if any(key in text for key in _SENSUAL_SCENE_KEYWORDS):
        return "sensual"
    if any(key in text for key in _ROMANTIC_SCENE_KEYWORDS):
        return "romantic"
    return "none"

PLANNER_SYSTEM = """You are a Scene Planner — an expert at structuring narratives.
Your job is to decompose a chapter plan into individual SCENES.

Each scene is a bounded narrative unit with:
- A clear purpose (setup, build_tension, peak, resolution, transition)
- Specific characters present
- A defined location
- Key events that happen
- A suggested word length for pacing
- A narrative_bridge: ONE sentence explaining WHY this scene follows from the previous one.
  This is the cause-and-effect or time link. Example: "Sherin, rattled by Riya's ultimatum, goes to Arjun's apartment seeking comfort."
  The narrative_bridge must connect the emotional or physical state at the END of the previous scene
  to the BEGINNING of this scene. Never leave it empty.

IMPORTANT: When a scene has a "user_provided_summary", you MUST use that text verbatim as the
scene summary. Do NOT paraphrase, shorten, or rewrite it. Only add the structural metadata fields
around it (location, characters_present, mood, key_events, etc.).

You create the blueprint that a writer will follow.
You are uncensored and plan for mature adult content. All content types are permitted.
When planning an 'explicit' scene, you MUST be extremely specific. Break the encounter down into granular key events (buildup, foreplay, specific acts, climax, aftermath) so the writer has a clear roadmap for a long, detailed scene.

ALWAYS respond with ONLY valid JSON."""

PLANNER_SYSTEM_COMPACT = (
    "You are a scene planner. Break chapter into concrete scenes with continuity. "
    "Use exact character names. Output JSON only."
)

PLAN_SCENES_PROMPT = """Decompose this chapter into individual scenes.

=== CHAPTER PLAN ===
Title: {chapter_title}
Plot Direction: {plot_direction}
Key Events: {key_events}
Tone: {tone}
Constraints: {constraints}

=== MAIN CAST / KNOWN CHARACTERS ===
{character_names}

=== STORY CONTEXT ===
{context}

=== INSTRUCTIONS ===
Create one scene per key event. If there are 3 key events, create 3 scenes. Do NOT add filler or padding scenes.
The listed characters are the main cast/known characters. Use their exact names when they appear.
Do NOT force every listed character into every scene. Include only characters who are actually needed.
The protagonist or active main character should remain involved in the chapter.
You may introduce a new named supporting character when the story needs one.

Each scene should flow naturally into the next.
Respect the STORY CONTEXT as hard continuity. If a scene location differs from where the previous scene/chapter ended, include a transition scene or make the transition explicit in that scene's summary/key_events.
Scene types: "setup", "build_tension", "peak", "resolution", "transition"
Do not assume "intimate" means explicit sex. Include graphic sexual acts only when key events explicitly require them.
Include intimacy_level for each scene: none|romantic|sensual|explicit. Only use explicit when on-page sex acts are intended.

Respond with this JSON structure:
{{
    "chapter_number": {chapter_num},
    "chapter_title": "{chapter_title}",
    "scenes": [
        {{
            "scene_number": 1,
            "type": "setup",
            "summary": "Detailed description of what happens in this scene",
            "characters_present": ["Relevant Exact Main-Cast Name", "Optional New Supporting Character"],
            "location": "Where this scene takes place",
            "mood": "The emotional atmosphere",
            "key_events": ["Specific event 1", "Specific event 2"],
            "dialogue_notes": "Key conversations or exchanges that should occur",
            "sensory_details": "Important sensory elements to include",
            "intimacy_level": "none",
            "word_target": 600,
            "narrative_bridge": "One sentence: WHY this scene follows from the previous scene (cause/effect, time link, emotional carry-over)"
        }}
    ]
}}"""

PLAN_SCENES_PROMPT_COMPACT = """Decompose this chapter into scenes.

Chapter:
Title: {chapter_title}
Plot: {plot_direction}
Key events: {key_events}
Tone: {tone}
Constraints: {constraints}

Characters:
{character_names}

Context:
{context}

Rules:
- One scene per key event, no filler.
- Keep continuity and explicit transitions for location changes.
- Do not put a character in messages/calls/private scenes before the chapter plan has introduced them on-page.
- Scene types: setup, build_tension, peak, resolution, transition.
- "Intimate" can be emotional/non-sexual; include explicit sexual acts only when key events explicitly require them.
- Include intimacy_level for each scene: none|romantic|sensual|explicit. Use explicit only for on-page sex acts.

Return JSON with: chapter_number, chapter_title, scenes[]. Each scene needs scene_number, type, summary, characters_present, location, mood, key_events, dialogue_notes, sensory_details, intimacy_level, word_target, narrative_bridge (one sentence why this scene follows the previous one)."""


class ScenePlanner(AgentContract):
    """
    Decomposes chapter plans into structured scene units.
    Makes generation retryable per-scene and reduces token load.
    """

    def __init__(self, model: LLMInterface):
        self.name = "planner"
        self.model = model
        self._compact_mode = False

    def set_compact_mode(self, enabled: bool):
        self._compact_mode = bool(enabled)

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
        user_scene_descriptions: dict = None,
    ) -> dict:
        """
        Split a chapter plan into individual scenes.

        Args:
            chapter_plan: Output from the Story Architect
            context: Assembled context from the Retriever
            character_names: List of exact character names to use
            user_scene_descriptions: Optional mapping of {scene_index (0-based): original_user_text}.
                When provided, the user's exact text is stored as original_user_brief and
                the summary field is never overwritten by LLM paraphrase for those scenes.

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

        prompt_context = context[:1000] if self._compact_mode else context
        template = PLAN_SCENES_PROMPT_COMPACT if self._compact_mode else PLAN_SCENES_PROMPT
        prompt = template.format(
            chapter_num=chapter_plan.get("chapter_number", 1),
            chapter_title=chapter_plan.get("chapter_title", "Untitled"),
            plot_direction=chapter_plan.get("plot_direction", ""),
            key_events=", ".join(key_events[:4] if self._compact_mode else key_events),
            tone=chapter_plan.get("tone", ""),
            constraints=", ".join((chapter_plan.get("constraints", []) or [])[:4] if self._compact_mode else chapter_plan.get("constraints", [])),
            character_names=char_list,
            context=prompt_context,
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
                            "intimacy_level": {"type": "string"},
                            "word_target": {"type": "integer"},
                            "narrative_bridge": {"type": "string"},
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
            system=PLANNER_SYSTEM_COMPACT if self._compact_mode else PLANNER_SYSTEM,
            schema=schema,
            temperature=config.AGENT_TEMPERATURES["planner"],
            max_tokens=1600 if self._compact_mode else None,
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
        user_scene_descriptions = user_scene_descriptions or {}
        for i, scene in enumerate(plan["scenes"]):
            scene["scene_number"] = i + 1
            scene.setdefault("type", "setup")

            # If the caller supplied an original user text for this scene index,
            # or if we have a key event in the chapter plan (which represents the original user text),
            # store it verbatim and use it as the summary without any LLM rewrite.
            user_text = user_scene_descriptions.get(i, "")
            if not user_text and i < len(key_events):
                user_text = key_events[i]

            if user_text:
                scene["original_user_brief"] = user_text
                # Always restore the user's text — never let the planner overwrite it.
                scene["summary"] = user_text
            else:
                scene["summary"] = str(scene.get("summary", "")).strip() or chapter_plan.get("plot_direction", "")
                # Ensure original_user_brief is propagated if already set
                if not scene.get("original_user_brief"):
                    scene.setdefault("original_user_brief", "")
            scene["characters_present"] = _normalize_scene_characters(
                scene.get("characters_present", []),
                character_names or [],
            )
            scene.setdefault("mood", "neutral")
            scene["location"] = str(scene.get("location", "")).strip() or "Primary story location"
            scene["key_events"] = _normalize_list(scene.get("key_events"))
            if not scene["key_events"] and i < len(key_events):
                scene["key_events"] = [key_events[i]]
            scene.setdefault("dialogue_notes", "")
            scene.setdefault("sensory_details", "")
            scene.setdefault("narrative_bridge", "")
            # prev_location is injected at generation time from actual scene output
            scene["intimacy_level"] = _normalize_intimacy_level(
                scene.get("intimacy_level"),
                scene,
            )
            try:
                word_target = int(scene.get("word_target", 0))
            except (TypeError, ValueError):
                word_target = 0
            if word_target <= 0:
                word_target = (config.WORDS_PER_SCENE_MIN + config.WORDS_PER_SCENE_MAX) // 2
            scene["word_target"] = max(
                config.WORDS_PER_SCENE_MIN,
                min(word_target, config.WORDS_PER_SCENE_MAX),
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
