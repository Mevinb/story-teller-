"""
Premise Architect — Intelligent Narrative Timeline & Story Blueprint Engine.

Specialized in:
1. Deeply grasping half-made stories, manuscripts, and narrative drafts.
2. Continuing and completing unfinished story drafts to a satisfying climax and resolution.
3. Re-architecting complete master story arcs from scratch using extracted premise points.
4. Automatically extracting characters, world setting, and thematic lore for the Story Bible.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from models.base import (
    LLMInterface,
    LLMResponse,
    ModelUnavailableError,
    RateLimitExhaustedError,
    QuotaExhaustedError,
    ContentBlockedError,
)
from pipeline.errors import PipelineCancelledError
from logger import get_logger

logger = get_logger("premise")


def clean_reasoning_and_fences(text: str) -> str:
    """Strip <think> blocks, markdown code fences, and extraneous metadata."""
    if not text:
        return ""
    # Strip <think>...</think> and similar reasoning tags
    text = re.sub(r"<(?:think|thought|reasoning|analysis)>.*?</(?:think|thought|reasoning|analysis)>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Strip code block fences
    text = re.sub(r"^```(?:json|markdown|text)?\s*$", "", text, flags=re.MULTILINE | re.IGNORECASE)
    text = text.replace("```", "")
    return text.strip()


def normalize_beat_text(beat: str) -> str:
    """Normalize a story beat by stripping numbering, labels, and formatting artifacts."""
    if not beat:
        return ""
    s = beat.strip()
    # Strip leading markdown bold or bullet formatting
    s = re.sub(r"^[\*\-\•\>]+\s*", "", s)
    # Strip prefixes like "Step 1:", "Beat 01 -", "Scene 1.", "**Beat 1:**"
    s = re.sub(r"^\*{0,2}(?:step|beat|scene|part|act)\s*\d+[\s\:\.\-]*\*{0,2}\s*", "", s, flags=re.IGNORECASE)
    # Strip numeric prefixes like "1.", "1)", "1:"
    s = re.sub(r"^\d+[\).\:-]\s*", "", s)
    # Strip any remaining leading bold asterisks
    s = re.sub(r"^\*{1,2}(.*?)\*{1,2}\s*:\s*", r"\1: ", s)
    return s.strip()


def repair_truncated_json(text: str) -> Optional[dict]:
    """
    Repair truncated or slightly broken JSON by balancing strings, arrays, and objects.
    Enables salvaging full story outlines even if output hit model token limits.
    """
    if not text:
        return None
    text = text.strip()
    start = text.find("{")
    if start == -1:
        return None
    text = text[start:]

    # 1. Try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2. Rebalance quotes, brackets, and braces
    stack = []
    in_string = False
    escape = False
    cleaned_chars = []

    for ch in text:
        if escape:
            escape = False
            cleaned_chars.append(ch)
            continue
        if ch == "\\":
            escape = True
            cleaned_chars.append(ch)
            continue
        if ch == '"':
            in_string = not in_string
            cleaned_chars.append(ch)
            continue
        if not in_string:
            if ch in ("{", "["):
                stack.append(ch)
            elif ch == "}":
                if stack and stack[-1] == "{":
                    stack.pop()
            elif ch == "]":
                if stack and stack[-1] == "[":
                    stack.pop()
        cleaned_chars.append(ch)

    candidate = "".join(cleaned_chars)
    if in_string:
        candidate += '"'

    # Strip trailing commas
    trimmed = candidate.rstrip()
    if trimmed.endswith(","):
        candidate = trimmed[:-1]

    closing = ""
    for opener in reversed(stack):
        closing += "}" if opener == "{" else "]"

    try:
        return json.loads(candidate + closing)
    except Exception:
        pass

    # 3. Fallback: Trim to last comma or quote and close
    idx = max(candidate.rfind(","), candidate.rfind('"'))
    if idx > 0:
        sub = candidate[:idx].rstrip().rstrip(",")
        sub_stack = []
        s_instr = False
        s_esc = False
        for ch in sub:
            if s_esc:
                s_esc = False
                continue
            if ch == "\\":
                s_esc = True
                continue
            if ch == '"':
                s_instr = not s_instr
                continue
            if not s_instr:
                if ch in ("{", "["):
                    sub_stack.append(ch)
                elif ch == "}" and sub_stack and sub_stack[-1] == "{":
                    sub_stack.pop()
                elif ch == "]" and sub_stack and sub_stack[-1] == "[":
                    sub_stack.pop()
        if s_instr:
            sub += '"'
        for o in reversed(sub_stack):
            sub += "}" if o == "{" else "]"
        try:
            return json.loads(sub)
        except Exception:
            pass

    return None


def extract_fields_via_regex(raw_text: str) -> dict:
    """
    Robust regex-based extractor that recovers narrative blueprint fields
    even if the response is raw text, unstructured markdown, or invalid JSON.
    """
    title_match = re.search(r'"title"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', raw_text) or \
                  re.search(r'(?:^|\n)\s*(?:title|story title)\s*[:=]\s*([^\n\r"]+)', raw_text, re.IGNORECASE)

    genre_match = re.search(r'"genre"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', raw_text) or \
                  re.search(r'(?:^|\n)\s*(?:genre)\s*[:=]\s*([^\n\r"]+)', raw_text, re.IGNORECASE)

    setting_match = re.search(r'"setting"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', raw_text) or \
                    re.search(r'(?:^|\n)\s*(?:setting|world)\s*[:=]\s*([^\n\r"]+)', raw_text, re.IGNORECASE)

    summary_match = re.search(r'"summary"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', raw_text)

    # Extract steps
    steps = []
    # Try finding beat strings like "Beat 1: ..." or "1. ..."
    raw_step_matches = re.findall(r'^\s*(?:\d+[\.\)]|Beat\s*\d+[:\.]?)\s*([^\n\r"]+)', raw_text, re.MULTILINE)
    for s in raw_step_matches:
        norm = normalize_beat_text(s)
        if norm and len(norm) > 15:
            steps.append(norm)

    if not steps:
        # Try JSON array items
        json_array_matches = re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', raw_text)
        for s in json_array_matches:
            if any(s.strip().lower().startswith(prefix) for prefix in ("beat ", "scene ", "chapter ")) and len(s) > 15:
                steps.append(normalize_beat_text(s))

    # Extract characters
    characters = {}
    char_blocks = re.findall(r'"([^"]{2,30})"\s*:\s*\{\s*"role"[^}]*\}', raw_text)
    for cb in char_blocks:
        characters[cb] = {"description": "Story cast member", "traits": []}

    return {
        "title": title_match.group(1).strip() if title_match else "",
        "genre": genre_match.group(1).strip() if genre_match else "",
        "setting": setting_match.group(1).strip() if setting_match else "",
        "summary": summary_match.group(1).strip() if summary_match else "",
        "steps": steps,
        "characters": characters,
    }


def is_prose_draft(text: str) -> bool:
    """
    Heuristic to determine if input text looks like a half-made story draft
    (continuous prose narrative, dialogue, scenes) rather than short bullet notes.
    """
    if not text:
        return False
    words = text.split()
    if len(words) >= 120:
        # Check for dialogue quotes
        quote_count = text.count('"') + text.count('“') + text.count('”')
        # Check for paragraph structure
        paragraphs = [p for p in text.split("\n\n") if len(p.split()) > 15]
        if quote_count >= 2 or len(paragraphs) >= 2 or len(words) > 300:
            return True
    return False


class PremiseArchitect:
    """
    Intelligent story architect agent for designing, completing, and refining
    narrative timelines and extracting rich Story Bible lore from story drafts.
    """

    DEFAULT_BEATS = 18

    GENERATION_SCHEMA = {
        "title": "Story title",
        "genre": "Genre name",
        "summary": "1-2 sentence overall summary",
        "setting": "World setting",
        "themes": ["theme 1", "theme 2"],
        "story_analysis": {
            "detected_type": "half_made_story_draft or concept_outline",
            "cutoff_point": "Description of cutoff or N/A",
            "central_conflict": "Core dramatic stakes",
        },
        "characters": {
            "Character Name": {
                "role": "Role",
                "description": "Description",
                "traits": ["trait 1", "trait 2"]
            }
        },
        "steps": ["Beat 1: description...", "Beat 2: description..."]
    }

    @classmethod
    def generate(
        cls,
        idea_text: str,
        mode: str = "auto",
        target_beats: Optional[int] = None,
        characters: Optional[dict] = None,
        setting: str = "",
        themes: Optional[list] = None,
        llm: Optional[LLMInterface] = None,
    ) -> dict:
        """
        Generate a complete narrative timeline from an idea or half-made story draft.
        """
        if not llm:
            raise ValueError("An active LLMInterface must be provided to PremiseArchitect.generate")

        text = (idea_text or "").strip()
        if not text:
            raise ValueError("Story idea or draft text cannot be empty.")

        detected_draft = is_prose_draft(text)
        resolved_mode = mode
        if resolved_mode == "auto":
            resolved_mode = "continue" if detected_draft else "rearchitect"

        text_len = len(text)
        num_beats = None
        if target_beats and str(target_beats).lower() not in ("auto", "0", "none", "null"):
            try:
                num_beats = int(target_beats)
            except (ValueError, TypeError):
                num_beats = None

        if not num_beats:
            if text_len > 120000:
                num_beats = 90
            elif text_len > 80000:
                num_beats = 75
            elif text_len > 40000:
                num_beats = 60
            elif text_len > 20000:
                num_beats = 45
            elif text_len > 10000:
                num_beats = 36
            else:
                num_beats = cls.DEFAULT_BEATS

        # Detect explicit chapter/scene markers in the input text to ensure beat count matches or exceeds them
        detected_scene_markers = len(re.findall(r'(?im)^\s*(?:chapter|scene|part|act)\s+\d+', text))
        if detected_scene_markers > 0:
            min_from_markers = max(detected_scene_markers * 2, detected_scene_markers + 4)
            if min_from_markers > num_beats:
                logger.info(
                    "[PremiseArchitect] Detected %d chapter/scene markers in draft; elevating target beats from %d to %d",
                    detected_scene_markers, num_beats, min_from_markers
                )
                num_beats = min_from_markers

        # Enforce minimum scene-density beat floors for large drafts so scenes are never dropped or summarized away
        if detected_draft or resolved_mode == "continue":
            min_floor = 0
            if text_len > 120000:
                min_floor = 75
            elif text_len > 80000:
                min_floor = 60
            elif text_len > 45000:
                min_floor = 45
            elif text_len > 20000:
                min_floor = 30
            elif text_len > 10000:
                min_floor = 24
            if min_floor > num_beats:
                logger.info(
                    "[PremiseArchitect] Elevating target beats from %d to %d to preserve all scenes in %d-char draft",
                    num_beats, min_floor, text_len
                )
                num_beats = min_floor

        num_beats = max(6, min(150, num_beats))

        # Format existing context
        chars_txt = cls._format_characters_context(characters)
        themes_txt = ", ".join(themes) if isinstance(themes, list) else str(themes or "")

        prompt = cls._build_generation_prompt(
            mode=resolved_mode,
            idea_text=text,
            target_beats=num_beats,
            chars_txt=chars_txt,
            setting=setting,
            themes_txt=themes_txt,
            is_draft=detected_draft,
        )

        system_instruction = (
            "You are a master story architect, developmental editor, and Hollywood script doctor. "
            "You excel at analyzing unfinished manuscripts, grasping their characters and narrative DNA, "
            "and architecting complete, compelling, chronologically sound narrative timelines."
        )

        logger.info(
            "[PremiseArchitect] Dispatching generation request (mode=%s, target_beats=%d, input_chars=%d, is_draft=%s)",
            resolved_mode, num_beats, len(text), detected_draft,
        )

        try:
            response = llm.generate(
                prompt=prompt,
                system=system_instruction,
                schema=cls.GENERATION_SCHEMA,
                max_tokens=8192,
            )
            return cls._parse_generation_response(response, text, resolved_mode, num_beats)
        except (ModelUnavailableError, RateLimitExhaustedError, QuotaExhaustedError, ContentBlockedError, PipelineCancelledError):
            # Fail fast: do NOT retry without schema if the model is unavailable or rate limited!
            raise
        except Exception as e:
            logger.warning("[PremiseArchitect] First generation attempt failed (%s). Retrying without strict schema...", e)
            response = llm.generate(
                prompt=prompt,
                system=system_instruction,
                max_tokens=8192,
            )
            return cls._parse_generation_response(response, text, resolved_mode, num_beats)

    @classmethod
    def refine(
        cls,
        steps: list[str],
        characters: Optional[dict] = None,
        setting: str = "",
        themes: Optional[list] = None,
        llm: Optional[LLMInterface] = None,
    ) -> list[str]:
        """Refine existing beats for dramatic tension, pacing, continuity, and character agency."""
        if not llm:
            raise ValueError("An active LLMInterface must be provided to PremiseArchitect.refine")
        if not steps:
            return []

        chars_txt = cls._format_characters_context(characters)
        themes_txt = ", ".join(themes) if isinstance(themes, list) else str(themes or "")
        steps_txt = "\n".join(f"{i+1}. {step}" for i, step in enumerate(steps))

        prompt = (
            "You are a master developmental editor. Your task is to polish and refine the sequential story beats of a narrative outline.\n\n"
            "=== INPUT DETAILS ===\n"
            f"CHARACTERS:\n{chars_txt or 'Extract from beats'}\n"
            f"SETTING: {setting or 'Derived from beats'}\n"
            f"THEMES: {themes_txt or 'Derived from beats'}\n\n"
            f"CURRENT STORY BEATS:\n{steps_txt}\n\n"
            "=== REQUIREMENTS ===\n"
            "1. Fix narrative drift, sudden jumps in logic, pacing slumps, or premature resolutions.\n"
            "2. Ensure strong cause-and-effect progression ('Because Beat X happened, therefore Beat Y occurs').\n"
            "3. Ensure character motivations remain consistent and active.\n"
            "4. Return the refined list of beats formatted in a clean JSON object:\n"
            "{\n"
            '  "steps": [\n'
            '    "Beat 1 description...",\n'
            '    "Beat 2 description..."\n'
            "  ]\n"
            "}\n"
            "Do NOT include act headings or conversational commentary. Output valid JSON only."
        )

        response = llm.generate(prompt=prompt, system="You are a developmental editor refining narrative pacing.")
        cleaned = clean_reasoning_and_fences(response.content or "")
        
        parsed = response.as_json()
        if isinstance(parsed, dict) and isinstance(parsed.get("steps"), list):
            refined = [normalize_beat_text(s) for s in parsed["steps"] if str(s).strip()]
            if refined:
                return refined

        # Fallback to line parsing
        from pipeline.orchestrator import PipelineOrchestrator
        raw_steps = PipelineOrchestrator._premise_steps(cleaned)
        return [normalize_beat_text(s) for s in raw_steps if normalize_beat_text(s)] or steps

    @classmethod
    def expand(
        cls,
        steps: list[str],
        characters: Optional[dict] = None,
        setting: str = "",
        themes: Optional[list] = None,
        target_beats: Optional[int] = None,
        llm: Optional[LLMInterface] = None,
    ) -> list[str]:
        """Expand narrative beats by inserting intermediate scene beats, character interactions, and escalations."""
        if not llm:
            raise ValueError("An active LLMInterface must be provided to PremiseArchitect.expand")
        if not steps:
            return []

        chars_txt = cls._format_characters_context(characters)
        themes_txt = ", ".join(themes) if isinstance(themes, list) else str(themes or "")
        steps_txt = "\n".join(f"{i+1}. {step}" for i, step in enumerate(steps))
        desired_count = target_beats or (len(steps) + max(4, len(steps) // 2))

        prompt = (
            "You are a master creative novelist. Your task is to expand an existing story timeline by inserting "
            "compelling intermediate scene beats, relationship developments, and suspense escalations.\n\n"
            "=== INPUT DETAILS ===\n"
            f"CHARACTERS:\n{chars_txt or 'From beats'}\n"
            f"SETTING: {setting or 'From beats'}\n"
            f"THEMES: {themes_txt or 'From beats'}\n\n"
            f"CURRENT STORY BEATS ({len(steps)} steps):\n{steps_txt}\n\n"
            f"TARGET COUNT: Approximately {desired_count} total sequential beats.\n\n"
            "=== REQUIREMENTS ===\n"
            "1. Insert transitional beats, investigation scenes, character conflict, and rising stakes between the major anchors.\n"
            "2. Preserve the core trajectory and climax of the original beats.\n"
            "3. Format as a clean JSON object:\n"
            "{\n"
            '  "steps": [\n'
            '    "Beat 1 description...",\n'
            '    "Beat 2 description..."\n'
            "  ]\n"
            "}\n"
            "Output valid JSON only with no commentary."
        )

        response = llm.generate(prompt=prompt, system="You are an expert novelist expanding narrative outlines.")
        cleaned = clean_reasoning_and_fences(response.content or "")

        parsed = response.as_json()
        if isinstance(parsed, dict) and isinstance(parsed.get("steps"), list):
            expanded = [normalize_beat_text(s) for s in parsed["steps"] if str(s).strip()]
            if expanded:
                return expanded

        from pipeline.orchestrator import PipelineOrchestrator
        raw_steps = PipelineOrchestrator._premise_steps(cleaned)
        return [normalize_beat_text(s) for s in raw_steps if normalize_beat_text(s)] or steps

    # ─── Internal Prompt Builders & Parsers ───────────────────────────

    @classmethod
    def _format_characters_context(cls, characters: Optional[Any]) -> str:
        if not characters:
            return ""
        chars_txt = ""
        if isinstance(characters, dict):
            for cname, cinfo in characters.items():
                if isinstance(cinfo, dict):
                    desc = cinfo.get("description", "")
                    traits = ", ".join(cinfo.get("traits", []))
                    chars_txt += f"- {cname}: {desc} (Traits: {traits})\n"
                elif isinstance(cinfo, str):
                    chars_txt += f"- {cname}: {cinfo}\n"
        elif isinstance(characters, list):
            for c in characters:
                if isinstance(c, dict):
                    name = c.get("name", "Unknown")
                    desc = c.get("description", "")
                    traits = ", ".join(c.get("traits", []))
                    chars_txt += f"- {name}: {desc} (Traits: {traits})\n"
                elif isinstance(c, str):
                    chars_txt += f"- {c}\n"
        return chars_txt.strip()

    @classmethod
    def _build_generation_prompt(
        cls,
        mode: str,
        idea_text: str,
        target_beats: int,
        chars_txt: str,
        setting: str,
        themes_txt: str,
        is_draft: bool,
    ) -> str:
        if mode == "continue":
            mode_instructions = (
                "MODE: CONTINUE & COMPLETE HALF-MADE STORY (Finish Unfinished Story)\n"
                "The author has provided an UNFINISHED STORY DRAFT or partial narrative prose that ends prematurely.\n"
                "Your objective:\n"
                "1. COMPREHEND WHAT HAPPENED SO FAR: Deeply read the draft. Identify all established characters, "
                "world rules, relationships, secrets, and events that already occurred up to the cutoff point.\n"
                "2. GRANULAR SCENE-BY-SCENE DRAFT MAPPING (Phase 1):\n"
                "   - Map EVERY single scene, encounter, conversation, confrontation, and event from the provided draft into sequential story beats.\n"
                "   - ABSOLUTE ZERO-SCENE-DROPPING MANDATE: Do NOT summarize away, gloss over, skip, or merge scenes! "
                "Every major conversation, discovery, decision, and location change in the author's draft MUST have its own dedicated beat "
                "so that ZERO written scenes are omitted or lost.\n"
                "3. PINPOINT THE EXACT CUTOFF: Identify the exact scene or cliffhanger where the author's writing paused.\n"
                "4. ARCHITECT THE CONTINUATION & CONCLUSION (Phase 2): From the cutoff point forward, invent and plot "
                "the subsequent beats. Escalate the core conflict, introduce necessary complications/midpoint twists, "
                "bring the storyline through a gripping climax, and provide a satisfying emotional resolution.\n"
                f"5. TOTAL BEAT COUNT: Output approximately {target_beats} granular beats covering all draft scenes and continuation without dropping scenes."
            )
        else:
            mode_instructions = (
                "MODE: RE-ARCHITECT FULL STORY FROM BEGINNING (Comprehensive High-Density Master Arc)\n"
                "The author has provided story text, draft scenes, or conceptual ideas.\n"
                "Your objective:\n"
                "1. EXTRACT ALL DRAMATIC ASSETS: Extract all characters, relationships, locations, lore, subplots, and conflicts from the input.\n"
                "2. HIGH SCENE DENSITY MASTER TIMELINE:\n"
                "   - Architect a rich, comprehensive story timeline spanning from the opening hook to the final resolution.\n"
                "   - ABSOLUTE ZERO-SCENE-DROPPING MANDATE: Do NOT skim or jump over intermediate steps. Include all pivotal dialogue scenes, character vulnerabilities, "
                "secondary character interactions, investigations, and escalating obstacles.\n"
                f"3. TOTAL BEAT COUNT: Output approximately {target_beats} chronological scene beats with high scene coverage."
            )

        json_schema_example = (
            "{\n"
            '  "title": "A compelling, evocative title for this story",\n'
            '  "genre": "Genre (e.g., Dark Fantasy, Sci-Fi / Cyberpunk, Psychological Thriller, Horror, Romance, Mystery)",\n'
            '  "summary": "1-2 sentence compelling summary of the entire completed story arc",\n'
            '  "setting": "World setting, time period, location, and atmosphere",\n'
            '  "themes": ["theme 1", "theme 2", "theme 3"],\n'
            '  "story_analysis": {\n'
            '    "detected_type": "half_made_story_draft" or "concept_outline",\n'
            '    "cutoff_point": "Description of where the user draft stopped and how the continuation begins (or N/A)",\n'
            '    "central_conflict": "The core dramatic engine and stakes",\n'
            '    "unresolved_tensions": ["key open question 1", "key open question 2"]\n'
            "  },\n"
            '  "characters": {\n'
            '    "Character Name": {\n'
            '      "role": "Protagonist / Antagonist / Ally / Mentor",\n'
            '      "description": "Background, motive, and story arc",\n'
            '      "traits": ["trait1", "trait2", "trait3"]\n'
            "    }\n"
            "  },\n"
            '  "steps": [\n'
            f'    "Beat 1: Clear, concrete scene description (who does what, where, and consequence)...",\n'
            f'    "...sequential beats up to ~{target_beats} total..."\n'
            "  ]\n"
            "}"
        )

        prompt = (
            "You are an elite master story architect. Analyze the provided story text and construct a complete, "
            "chronologically ordered, beat-by-beat narrative timeline.\n\n"
            "=== INPUT DETAILS ===\n"
            f"KNOWN CHARACTERS (if any):\n{chars_txt or 'None provided — extract from story text'}\n\n"
            f"SETTING CONTEXT (if any):\n{setting or 'None provided — extract from story text'}\n\n"
            f"THEMES (if any):\n{themes_txt or 'None provided — extract from story text'}\n\n"
            f"INPUT STORY TEXT / DRAFT:\n\"\"\"\n{idea_text}\n\"\"\"\n\n"
            f"=== ARCHITECTURE DIRECTIVE ===\n"
            f"{mode_instructions}\n\n"
            "=== BEAT QUALITY & GRANULARITY REQUIREMENTS ===\n"
            "1. Each beat must describe a concrete, observable scene event with character agency, obstacle, and consequence.\n"
            "2. HIGH SCENE FIDELITY: Never compress multiple scenes into a single vague beat. If the story has multiple encounters, dialogue exchanges, or investigations, give EACH distinct scene its own beat so that no scenes are missed.\n"
            "3. Avoid vague generalities like 'Tension rises' or 'They talk about plans'. Instead write: 'Kael confronts Vane at the sunken docks, demanding the deciphered ledger, but Vane reveals the seal belongs to Kael’s father.'\n"
            f"4. BEAT TARGET: Aim to output approximately {target_beats} complete sequential beats to ensure full scene coverage.\n"
            "5. The beats must flow in strict chronological sequence.\n"
            "6. Do NOT output act headers (like 'Act 1:'), chapter labels, or markdown formatting inside the steps array.\n"
            "7. Automatically extract ALL significant characters present in the text into the characters object.\n\n"
            "=== OUTPUT FORMAT ===\n"
            "You MUST respond ONLY with a valid JSON object matching this exact schema:\n"
            f"{json_schema_example}\n"
            "Do NOT include any text outside the JSON object. Do not wrap in ```json fences."
        )
        return prompt

    @classmethod
    def _parse_generation_response(
        cls,
        response: LLMResponse,
        raw_input: str,
        resolved_mode: str,
        num_beats: int,
    ) -> dict:
        content = response.content or ""
        cleaned = clean_reasoning_and_fences(content)

        parsed_json = response.as_json()
        if not isinstance(parsed_json, dict):
            # Try repairing truncated JSON or extracting JSON object
            parsed_json = repair_truncated_json(cleaned)
            if parsed_json:
                logger.info("[PremiseArchitect] Successfully repaired and parsed JSON output from model.")
            else:
                logger.warning("[PremiseArchitect] Standard JSON parsing failed; running regex field extraction on raw response.")

        steps: list[str] = []
        characters: dict = {}
        setting: str = ""
        themes: list = []
        summary: str = ""
        title: str = ""
        genre: str = ""
        story_analysis: dict = {}

        if isinstance(parsed_json, dict):
            title = str(parsed_json.get("title") or "").strip()
            genre = str(parsed_json.get("genre") or "").strip()
            raw_steps = parsed_json.get("steps") or []
            if isinstance(raw_steps, list):
                for s in raw_steps:
                    norm = normalize_beat_text(str(s))
                    if norm and len(norm) > 10:
                        steps.append(norm)

            summary = str(parsed_json.get("summary") or "").strip()
            setting = str(parsed_json.get("setting") or "").strip()
            
            raw_themes = parsed_json.get("themes") or []
            if isinstance(raw_themes, list):
                themes = [str(t).strip() for t in raw_themes if str(t).strip()]
            elif isinstance(raw_themes, str):
                themes = [t.strip() for t in raw_themes.split(",") if t.strip()]

            raw_chars = parsed_json.get("characters") or {}
            if isinstance(raw_chars, dict):
                for cname, cinfo in raw_chars.items():
                    cname_clean = str(cname).strip()
                    if not cname_clean:
                        continue
                    if isinstance(cinfo, dict):
                        desc = str(cinfo.get("description") or cinfo.get("role") or "").strip()
                        raw_traits = cinfo.get("traits") or []
                        traits = [str(t).strip() for t in raw_traits if str(t).strip()] if isinstance(raw_traits, list) else []
                        characters[cname_clean] = {"description": desc, "traits": traits}
                    elif isinstance(cinfo, str):
                        characters[cname_clean] = {"description": cinfo.strip(), "traits": []}

            raw_analysis = parsed_json.get("story_analysis") or {}
            if isinstance(raw_analysis, dict):
                story_analysis = raw_analysis

        # Fallback 1: Regex field extraction if primary fields are missing
        if not steps or not title:
            regex_data = extract_fields_via_regex(cleaned)
            if not title and regex_data.get("title"):
                title = regex_data["title"]
            if not genre and regex_data.get("genre"):
                genre = regex_data["genre"]
            if not setting and regex_data.get("setting"):
                setting = regex_data["setting"]
            if not summary and regex_data.get("summary"):
                summary = regex_data["summary"]
            if not steps and regex_data.get("steps"):
                steps = regex_data["steps"]
            if not characters and regex_data.get("characters"):
                characters = regex_data["characters"]

        # Fallback 2: Line-based premise steps
        if not steps:
            from pipeline.orchestrator import PipelineOrchestrator
            raw_parsed = PipelineOrchestrator._premise_steps(cleaned)
            for s in raw_parsed:
                norm = normalize_beat_text(s)
                if norm and len(norm) > 10:
                    steps.append(norm)

        # Fallback 3: Title derivation from input seeds if still blank
        if not title:
            first_line = raw_input.strip().split("\n")[0].strip()
            first_line = re.sub(r"^(?:user|assistant|prompt|system|human|story|draft)\s*:\s*", "", first_line, flags=re.IGNORECASE)
            first_line = re.sub(r"^[#\*\-\s]+", "", first_line).strip()
            if 3 < len(first_line) <= 60 and not first_line.endswith(":"):
                title = first_line
            elif characters:
                main_char = list(characters.keys())[0]
                title = f"The Chronicles of {main_char}"
            else:
                title = "Untitled Master Story"

        if not genre:
            genre = "Fantasy / Speculative Fiction"

        # Validate that generation actually produced content
        if not steps and not characters and not setting:
            logger.error("[PremiseArchitect] Model produced no beats, characters, or setting from draft input.")
            raise RuntimeError(
                "Failed to auto-generate story details from the provided draft. "
                "The AI model produced no narrative beats or characters. Please check your model or try again."
            )

        # Ensure analysis status is clear
        if not story_analysis:
            story_analysis = {
                "detected_type": "half_made_story_draft" if is_prose_draft(raw_input) else "concept_outline",
                "mode_applied": resolved_mode,
                "cutoff_point": "Draft scenes mapped to initial beats; continuation architected." if resolved_mode == "continue" else "Full narrative re-architected from premise seeds.",
            }
        else:
            story_analysis["mode_applied"] = resolved_mode

        logger.info(
            "[PremiseArchitect] Generation complete | title=%r | genre=%r | steps=%d | characters=%d | setting=%r",
            title, genre, len(steps), len(characters), setting[:40] if setting else "",
        )

        return {
            "status": "ok",
            "title": title,
            "genre": genre,
            "steps": steps,
            "characters": characters,
            "setting": setting,
            "themes": themes,
            "summary": summary,
            "story_analysis": story_analysis,
        }
