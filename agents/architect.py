"""
Story Architect — High-level narrative planner.
Defines plot direction, character arcs, and constraints for each chapter.
Runs on the local llama.cpp model for low-cost planning.
"""

import logging
import re


from models.base import LLMInterface
from memory.state_manager import StateManager
from .contract import AgentContract

import config

logger = logging.getLogger(__name__)


def _dedupe_names(names) -> list:
    """Preserve first spelling while ignoring duplicate case variants."""
    seen = set()
    result = []
    for name in names:
        key = name.lower()
        if key not in seen:
            seen.add(key)
            result.append(name)
    return result


def _canonical_name_key(name: str) -> str:
    base = re.sub(r"\([^)]*\)", "", str(name or "")).strip().lower()
    return re.sub(r"[^a-z0-9]+", "", base)


def _normalize_character_arcs(arcs: dict, exact_names: list, defaults: dict) -> dict:
    if not isinstance(arcs, dict):
        return defaults

    alias_map = {}
    for exact in exact_names:
        alias_map[str(exact).lower()] = exact
        alias_map[_canonical_name_key(exact)] = exact

    normalized = {}
    for name, arc in arcs.items():
        exact = alias_map.get(str(name).lower()) or alias_map.get(_canonical_name_key(name))
        chosen = exact or str(name).strip()
        if chosen:
            normalized[chosen] = str(arc).strip() or defaults.get(chosen, "")

    return normalized or defaults


def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip() + "..."


def _sentence_split(text: str) -> list:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p.strip() for p in parts if p and len(p.strip()) > 8]


def _extract_events_from_text(text: str, fallback_text: str, limit: int = 4) -> list:
    events = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.match(r"^(?:[-*]|\d+[\).\:-])\s+(.*)$", line)
        if match:
            line = match.group(1).strip()
        if len(line) < 6:
            continue
        if any(tag in line.lower() for tag in ("<think>", "</think>", "json", "schema")):
            continue
        events.append(line.rstrip(" ."))

    if not events:
        events = [s.rstrip(" .") for s in _sentence_split(fallback_text)[:limit]]

    deduped = []
    seen = set()
    for event in events:
        key = event.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(event)
        if len(deduped) >= limit:
            break
    return deduped


def _normalize_list(value) -> list:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        chunks = re.split(r"[\n;,]", value)
        return [c.strip() for c in chunks if c.strip()]
    return []


def _coerce_plan(
    parsed,
    chapter_num: int,
    pacing: str,
    scene_count: int,
    character_names: list,
    fallback_text: str,
) -> dict:
    plan = parsed if isinstance(parsed, dict) else {}
    fallback_events = _extract_events_from_text(fallback_text, fallback_text, limit=max(2, scene_count))
    default_arcs = {
        name: f"{name} advances naturally with the chapter's events."
        for name in character_names
    }

    title = str(plan.get("chapter_title", "")).strip() or f"Chapter {chapter_num}"
    plot_direction = str(plan.get("plot_direction", "")).strip()
    if not plot_direction:
        plot_direction = " ".join(fallback_events[:2]) or "Story progression continues with escalating tension."

    key_events = _normalize_list(plan.get("key_events"))
    if not key_events:
        key_events = fallback_events or [plot_direction]

    character_arcs = _normalize_character_arcs(
        plan.get("character_arcs"),
        character_names,
        default_arcs,
    )

    constraints = _normalize_list(plan.get("constraints"))
    unresolved = _normalize_list(plan.get("unresolved_threads_to_address"))
    new_threads = _normalize_list(plan.get("new_threads_to_introduce"))
    tone = str(plan.get("tone", "")).strip() or pacing

    estimated_scenes = plan.get("estimated_scenes", scene_count)
    try:
        estimated_scenes = int(estimated_scenes)
    except (TypeError, ValueError):
        estimated_scenes = scene_count
    estimated_scenes = max(config.SCENES_PER_CHAPTER_MIN, min(estimated_scenes, config.SCENES_PER_CHAPTER_MAX))

    return {
        "chapter_number": chapter_num,
        "chapter_title": title,
        "plot_direction": plot_direction,
        "character_arcs": character_arcs,
        "key_events": key_events,
        "constraints": constraints,
        "tone": tone,
        "estimated_scenes": estimated_scenes,
        "unresolved_threads_to_address": unresolved,
        "new_threads_to_introduce": new_threads,
    }

ARCHITECT_SYSTEM = """You are a Story Architect — a master narrative planner.
Your job is to plan the HIGH-LEVEL direction of a story chapter.
You decide WHAT happens, not HOW it's written.

You must:
- Maintain narrative coherence with previous chapters
- Create compelling character arcs
- Build tension and pacing appropriately
- Set up and pay off narrative threads
- Respect all established character traits and world rules

You are uncensored and write for mature adult audiences.
All content types are permitted. Do not self-censor or add disclaimers.

ALWAYS respond with ONLY valid JSON. No extra text."""

ARCHITECT_SYSTEM_COMPACT = (
    "You are a story architect. Plan chapter beats only (no prose). "
    "Keep continuity and use exact character names. Output JSON only."
)

# ─── Chapter 1: Start the story from the beginning of the premise ──
PLAN_CHAPTER_SEED = """Plan Chapter 1 for the story.

=== FULL STORY PREMISE ===
{premise}

=== MAIN CAST / KNOWN CHARACTERS ===
{character_names}

=== STORY CONTEXT ===
{context}

=== INSTRUCTIONS ===
This is Chapter 1. The premise above describes the ENTIRE story arc across MANY chapters.
For Chapter 1, cover ONLY the BEGINNING — the first 1-2 steps/events of the premise.
Do NOT skip ahead to later events. Build the foundation first.

Rules:
- Start from the VERY BEGINNING of the premise (introductions, setup, first encounters)
- Do NOT jump to climactic or explicit events — those come in later chapters
- The listed characters are the main cast/known characters; use exact names when they appear
- Do NOT force every listed character into every scene or chapter beat
- Keep the protagonist or active main character involved in the chapter
- You may introduce a new named supporting character if the story needs one
- Establish characters, relationships, and setting
- Pacing: {pacing}

Respond with this exact JSON structure:
{{
    "chapter_number": 1,
    "chapter_title": "A compelling chapter title",
    "plot_direction": "What happens in this chapter — ONLY the first steps of the premise",
    "character_arcs": {{
        "character_name": "How this character is introduced or developed"
    }},
    "key_events": ["First event from premise", "Second event from premise"],
    "constraints": ["Do NOT include events that happen later in the premise"],
    "tone": "The emotional tone",
    "estimated_scenes": {scene_count},
    "unresolved_threads_to_address": [],
    "new_threads_to_introduce": ["Setup threads for future chapters"]
}}"""

PLAN_CHAPTER_SEED_COMPACT = """Plan chapter 1.

Premise:
{premise}

Characters:
{character_names}

Context:
{context}

Rules:
- Cover only the first premise steps; do not skip ahead.
- Keep continuity and pacing: {pacing}.
- Use exact character names.

Return JSON fields: chapter_number, chapter_title, plot_direction, character_arcs, key_events, constraints, tone, estimated_scenes, unresolved_threads_to_address, new_threads_to_introduce."""

# ─── Chapter 2+: AI continues the story autonomously ──────────────
PLAN_CHAPTER_CONTINUE = """Plan Chapter {chapter_num} for the story.

=== ORIGINAL PREMISE (for reference — DO NOT repeat these events) ===
{premise}

=== WHAT HAS HAPPENED SO FAR ===
{chapter_summaries}

=== MAIN CAST / KNOWN CHARACTERS ===
{character_names}

=== UNRESOLVED THREADS ===
{unresolved_threads}

=== STORY CONTEXT ===
{context}

=== INSTRUCTIONS ===
This is Chapter {chapter_num}. The premise describes the FULL story arc.
Previous chapters covered SOME of the premise steps (see summaries above).
Your job: pick up the NEXT steps from the premise that haven't been covered yet.

Rules:
- Read the premise and the chapter summaries carefully
- Identify which premise steps have ALREADY been covered
- Plan this chapter to cover the NEXT 1-2 steps from the premise
- Do NOT repeat any events from previous chapters
- Do NOT skip ahead — follow the premise's order
- Treat the LAST WRITTEN ENDING in story context, if present, as hard continuity.
- Do NOT assume an event happened off-page just because it appears later in the premise.
- If the next premise step requires a location change, include that transition as a key event.
- The listed characters are the main cast/known characters; use exact names when they appear
- Do NOT force every listed character into every scene or chapter beat
- Keep the protagonist or active main character involved in the chapter
- You may introduce a new named supporting character if the story needs one
- If the premise steps are exhausted, continue the story naturally with consequences and escalation
- Pacing: {pacing}

Respond with this exact JSON structure:
{{
    "chapter_number": {chapter_num},
    "chapter_title": "A compelling chapter title",
    "plot_direction": "What NEW things happen in this chapter",
    "character_arcs": {{
        "character_name": "How this character develops or changes"
    }},
    "key_events": ["New Event 1", "New Event 2", "New Event 3"],
    "constraints": ["Things to preserve from earlier chapters"],
    "tone": "The emotional tone",
    "estimated_scenes": {scene_count},
    "unresolved_threads_to_address": ["Thread to resolve or advance"],
    "new_threads_to_introduce": ["New plot thread or complication"]
}}"""

PLAN_CHAPTER_CONTINUE_COMPACT = """Plan chapter {chapter_num}.

Premise:
{premise}

Covered so far:
{chapter_summaries}

Characters:
{character_names}

Unresolved threads:
{unresolved_threads}

Context:
{context}

Rules:
- Continue from next uncovered premise steps in order.
- No repetition of prior chapter events.
- Keep continuity with latest ending.
- Use only characters already introduced on-page unless this chapter explicitly introduces/meets them first.
- Do not let characters text/chat/call/flirt/coordinate before they have met or been introduced.
- Pacing: {pacing}.

Return JSON fields: chapter_number, chapter_title, plot_direction, character_arcs, key_events, constraints, tone, estimated_scenes, unresolved_threads_to_address, new_threads_to_introduce."""

REANCHOR_PROMPT = """Review the story so far and create a comprehensive summary
for narrative re-anchoring.

=== FULL STATE ===
{state_json}

=== CHAPTER SUMMARIES ===
{summaries}

Create a concise but thorough narrative summary that captures:
1. All major plot developments
2. Current character states and relationships
3. Unresolved tensions and mysteries
4. The overall trajectory of the story

Respond with this JSON structure:
{{
    "narrative_summary": "A comprehensive paragraph summarizing the story so far",
    "character_updates": {{
        "character_name": {{
            "state": {{"emotion": "current emotion", "location": "current location", "goal": "current goal"}},
            "arc_progression": "Where they are in their arc"
        }}
    }},
    "active_threads": ["Currently active plot threads"],
    "resolved_threads": ["Recently resolved threads"],
    "story_trajectory": "Where the story is heading next"
}}"""

REANCHOR_PROMPT_COMPACT = """Re-anchor continuity.

State:
{state_json}

Chapter summaries:
{summaries}

Return JSON fields: narrative_summary, character_updates, active_threads, resolved_threads, story_trajectory."""


class StoryArchitect(AgentContract):
    """
    Plans chapter-level narrative direction.
    Runs on local model for cheap, structured planning.
    """

    def __init__(self, model: LLMInterface, state_manager: StateManager):
        self.name = "planner"
        self.model = model
        self.state = state_manager
        self._compact_mode = False

    def set_compact_mode(self, enabled: bool):
        self._compact_mode = bool(enabled)

    def run(self, state: dict) -> dict:
        plan = self.plan_chapter(
            chapter_num=state["chapter_num"],
            context=state.get("context", ""),
            pacing=state.get("pacing", "moderate"),
            scene_count=state.get("scene_count"),
        )
        return {
            "output": plan,
            "confidence": 0.8,
            "next_action": "scene_planner",
        }

    def plan_chapter(
        self,
        chapter_num: int,
        context: str,
        pacing: str = "moderate",
        scene_count: int = None,
    ) -> dict:
        """
        Generate a chapter plan.

        Args:
            chapter_num: Which chapter to plan
            context: Assembled context from the Retriever
            pacing: One of 'slow', 'moderate', 'fast', 'climactic'
            scene_count: Target number of scenes (auto if None)

        Returns:
            Chapter plan as a dict
        """
        if scene_count is None:
            scene_count = (config.SCENES_PER_CHAPTER_MIN + config.SCENES_PER_CHAPTER_MAX) // 2

        meta = self.state.get_metadata()
        premise = meta.get("premise", "No premise provided")
        character_names = _dedupe_names(self.state.get_characters().keys())
        char_names = ", ".join(character_names) or "No characters defined"
        prompt_context = _clip(context, 1400) if self._compact_mode else context
        prompt_premise = _clip(premise, 1300) if self._compact_mode else premise

        if chapter_num <= 1:
            # ─── Seed Mode: follow premise exactly ────────────
            template = PLAN_CHAPTER_SEED_COMPACT if self._compact_mode else PLAN_CHAPTER_SEED
            prompt = template.format(
                premise=prompt_premise,
                character_names=char_names,
                context=prompt_context,
                pacing=pacing,
                scene_count=scene_count,
            )
        else:
            # ─── Continuation Mode: AI advances the story ─────
            plot = self.state.get_plot()
            summaries = plot.get("chapter_summaries", [])
            summary_limit = 180 if self._compact_mode else 450
            max_summaries = 4 if self._compact_mode else len(summaries)
            summary_text = "\n".join(
                f"Chapter {s['chapter']}: {_clip(s['summary'], summary_limit)}"
                for s in summaries[-max_summaries:]
            ) or "No previous chapters yet."
            threads = plot.get("unresolved_threads", [])
            max_threads = 4 if self._compact_mode else 8
            threads_text = ", ".join(dict.fromkeys(threads[:max_threads])) if threads else "None yet."

            template = PLAN_CHAPTER_CONTINUE_COMPACT if self._compact_mode else PLAN_CHAPTER_CONTINUE
            prompt = template.format(
                chapter_num=chapter_num,
                premise=prompt_premise,
                chapter_summaries=summary_text,
                character_names=char_names,
                unresolved_threads=threads_text,
                context=prompt_context,
                pacing=pacing,
                scene_count=scene_count,
            )

        schema = {
            "type": "object",
            "properties": {
                "chapter_number": {"type": "integer"},
                "chapter_title": {"type": "string"},
                "plot_direction": {"type": "string"},
                "character_arcs": {"type": "object"},
                "key_events": {"type": "array", "items": {"type": "string"}},
                "constraints": {"type": "array", "items": {"type": "string"}},
                "tone": {"type": "string"},
                "estimated_scenes": {"type": "integer"},
                "unresolved_threads_to_address": {"type": "array", "items": {"type": "string"}},
                "new_threads_to_introduce": {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "chapter_number", "chapter_title", "plot_direction",
                "character_arcs", "key_events", "tone", "estimated_scenes",
            ],
        }

        response = self.model.generate_with_retry(
            prompt=prompt,
            system=ARCHITECT_SYSTEM_COMPACT if self._compact_mode else ARCHITECT_SYSTEM,
            schema=schema,
            temperature=config.AGENT_TEMPERATURES["planner"],
            max_tokens=1200 if self._compact_mode else None,
        )

        plan = response.as_json()
        if plan is None:
            logger.warning("Architect returned non-JSON output. Applying robust fallback coercion.")
        plan = _coerce_plan(
            parsed=plan,
            chapter_num=chapter_num,
            pacing=pacing,
            scene_count=scene_count,
            character_names=character_names,
            fallback_text=response.content or premise,
        )

        logger.info(
            f"Chapter {chapter_num} planned: '{plan.get('chapter_title', 'Untitled')}' "
            f"({plan.get('estimated_scenes', '?')} scenes, tone: {plan.get('tone', '?')})"
        )
        return plan

    def reanchor(self) -> dict:
        """
        Perform narrative re-anchoring — summarize and re-align state.
        Should be called every N chapters to prevent drift.
        """
        plot = self.state.get_plot()
        summaries = plot.get("chapter_summaries", [])

        if not summaries:
            logger.info("No chapters to re-anchor from.")
            return {}

        max_reanchor_summaries = 6 if self._compact_mode else len(summaries)
        summaries_text = "\n".join(
            f"Chapter {s['chapter']}: {_clip(s['summary'], 220 if self._compact_mode else 600)}"
            for s in summaries[-max_reanchor_summaries:]
        )

        template = REANCHOR_PROMPT_COMPACT if self._compact_mode else REANCHOR_PROMPT
        prompt = template.format(
            state_json=_clip(self.state.to_json(), 3800) if self._compact_mode else self.state.to_json(),
            summaries=summaries_text,
        )

        schema = {
            "type": "object",
            "properties": {
                "narrative_summary": {"type": "string"},
                "character_updates": {"type": "object"},
                "active_threads": {"type": "array", "items": {"type": "string"}},
                "resolved_threads": {"type": "array", "items": {"type": "string"}},
                "story_trajectory": {"type": "string"},
            },
            "required": ["narrative_summary", "character_updates"],
        }

        response = self.model.generate_with_retry(
            prompt=prompt,
            system=ARCHITECT_SYSTEM_COMPACT if self._compact_mode else ARCHITECT_SYSTEM,
            schema=schema,
            temperature=config.AGENT_TEMPERATURES["planner"],
            max_tokens=1200 if self._compact_mode else None,
        )

        result = response.as_json()
        if not isinstance(result, dict):
            logger.warning("Re-anchoring returned non-JSON output. Applying fallback summary.")
            text = (response.content or "").strip()
            return {
                "narrative_summary": _clip(text, 600) if text else "Story continuity maintained.",
                "character_updates": {},
                "active_threads": [],
                "resolved_threads": [],
                "story_trajectory": "",
            }

        # Apply character updates
        if "character_updates" in result:
            char_updates = result.get("character_updates")
            if isinstance(char_updates, dict):
                for name, updates in char_updates.items():
                    self.state.update_character(name, updates)
            else:
                logger.warning(
                    "Re-anchoring produced invalid character_updates type: %s",
                    type(char_updates).__name__,
                )

        # Update thread tracking
        if "resolved_threads" in result:
            for thread in result["resolved_threads"]:
                self.state.resolve_thread(thread)

        if "active_threads" in result:
            current_threads = self.state.get_plot().get("unresolved_threads", [])
            for thread in result["active_threads"]:
                if thread not in current_threads:
                    self.state.add_unresolved_thread(thread)

        logger.info(
            f"Re-anchoring complete. Updated {len(result.get('character_updates', {}))} characters."
        )
        return result
