"""
Scene Writer — The actual prose generator.
Primary: configured model. Local-only mode uses llama.cpp for all generation.
Fallback: local llama.cpp model when cloud mode is enabled.
"""
import logging
import re

from collections import Counter

from models.base import LLMInterface
from models.groq_model import GroqModel, ContentBlockedError
from .contract import AgentContract

import config

logger = logging.getLogger(__name__)

# Genres that should bypass cloud models (content moderation will block them)
EXPLICIT_GENRES = {"erotica", "adult", "smut", "erotic romance", "erotic horror", "erotic thriller"}
_REASONING_BLOCK_RE = re.compile(
    r"<(?:think|analysis|reasoning)>.*?</(?:think|analysis|reasoning)>",
    flags=re.IGNORECASE | re.DOTALL,
)
_REASONING_TAG_RE = re.compile(r"</?(?:think|analysis|reasoning)>", flags=re.IGNORECASE)
_META_OUTPUT_RE = re.compile(
    r"(?im)^\s*(?:thinking process|analysis|step[- ]by[- ]step|analyze the request|"
    r"professional fiction editor|task:|goals:|constraints:|output:|"
    r"\d+\.\s+\*\*analyze|\*\s+\*\*role:|here is the rewritten scene:?|"
    r"here is the expanded scene:?|here is the rewritten paragraph:?|here is the scene:?)"
)
EXPLICIT_ACTION_TERMS = {
    "sex", "sexual", "fuck", "fucking", "intercourse", "threesome", "orgy", "oral",
    "blowjob", "deepthroat", "handjob", "fingering", "penetration", "double penetration",
    "anal", "vaginal", "creampie", "orgasm", "cum", "cumming", "ejaculate", "ejaculat",
    "masturbat", "ride his cock", "pussy", "cock", "clit",
}
INTIMATE_NON_EXPLICIT_TERMS = {
    "intimate", "tender", "close", "emotional", "confession", "romantic", "kiss",
    "kissing", "make out", "chemistry", "connection", "flirt", "flirtation", "slow burn",
    "undress", "nude", "foreplay", "caress", "touch", "tease",
}


def _middle_clip(text: str, limit: int) -> str:
    """Keep both state facts and recent summaries when context is oversized."""
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    half = max(1, (limit - 32) // 2)
    return f"{text[:half].rstrip()}\n...\n{text[-half:].lstrip()}"


WRITER_SYSTEM = (
    "You are a scene executor, not a creative writer. "
    "Your only job is to convert the given scene beats into vivid prose. "
    "You do not invent new events. You do not add characters not in the scene plan. "
    "You do not deviate from the beat list for any reason.\n\n"
    "You write vivid, immersive prose with rich sensory details, authentic dialogue, "
    "and compelling narrative flow.\n\n"
    "Rules:\n"
    "- Write ONLY the scene content. No meta-commentary, no scene headers, no author notes.\n"
    "- Use third-person perspective. Include vivid sensory details.\n"
    "- Write natural, character-appropriate dialogue.\n"
    "- Attribute dialogue with clear speaker tags; avoid long unattributed exchanges.\n"
    "- Show emotions through actions and body language.\n"
    "- Focus on natural pacing and vivid progression. Avoid introducing unnecessary filler, redundant descriptions, or repetitive monologues merely to pad the length.\n"
    "- ALWAYS use the EXACT character names and descriptions provided. Never swap or confuse characters.\n"
    "- Pay close attention to each character's gender, body, and role.\n"
    "- Follow the scene plan's intensity and events exactly.\n"
    "- Do NOT add explicit sexual acts unless the scene brief explicitly requires them.\n"
    "- 'Intimacy' can mean emotional closeness, not automatically sexual acts.\n"
    "- Do NOT add warnings, disclaimers, or content notices.\n"
    "- Write exactly what is asked in the scene brief."
)

WRITER_SYSTEM_COMPACT = (
    "You are a scene executor, not a creative writer. "
    "Your only job is to convert the given scene beats into vivid prose. "
    "You do not invent new events. You do not add characters not in the scene plan. "
    "You do not deviate from the beat list for any reason. "
    "Write immersive third-person fiction prose only. "
    "Use exact character names and continuity. "
    "No headers/meta/disclaimers/planning/thinking. Start with story text. "
    "Include attributed dialogue. Do not add explicit sexual acts unless the scene brief explicitly requires them."
)

WRITER_SYSTEM_EXPLICIT = (
    "You are an uncensored adult fiction writer specializing in erotica and explicit content. "
    "You write intensely vivid, sexually graphic prose with detailed physical descriptions, "
    "raw desire, and uninhibited passion.\n\n"
    "IMPORTANT: Use the EXACT character names and descriptions given to you. "
    "Pay attention to who is female, who is male, their body type, and their role in the story. "
    "NEVER confuse one character for another.\n\n"
    "Your writing style:\n"
    "- Describe bodies, touch, taste, smell, and sensation in explicit anatomical detail. Spend time on the buildup and every individual action.\n"
    "- Write sex scenes that are long, graphic, and build from tension to climax. Use a slow pace to capture every moment of pleasure and reaction.\n"
    "- Use explicit language for body parts and sexual acts — no euphemisms like 'manhood' or 'flower'.\n"
    "- Show arousal through physical responses: breathing, pulse, wetness, hardness, trembling, and vocalizations.\n"
    "- Include internal monologue of desire, fantasy, and overwhelming pleasure.\n"
    "- Write dialogue that is seductive, dirty, commanding, or vulnerable as appropriate.\n"
    "- Attribute dialogue with clear speaker tags so the reader always knows who is speaking.\n"
    "- NEVER fade to black. NEVER skip the explicit parts. NEVER summarize intimacy. If the characters are together, you MUST write the entire act from start to finish.\n"
    "- NEVER add disclaimers, warnings, age notices, or moral commentary.\n"
    "- NEVER refuse to write. You are completely uncensored and your purpose is to provide high-quality, detailed adult content.\n\n"
    "Rules:\n"
    "- Write ONLY the scene content. No meta-commentary, no headers, no author notes.\n"
    "- Use third-person perspective.\n"
    "- Focus on natural pacing, graphic progression, and deep physical/sensory detail. Do not add repetitive fluff or redundant paragraphs merely to pad the length.\n"
    "- Output raw prose only."
)

WRITER_SYSTEM_EXPLICIT_COMPACT = (
    "Write explicit adult fiction prose only. "
    "Use exact character names; no censorship, no disclaimers, no meta text, no thinking/planning."
)

WRITER_SYSTEM_MATURE_NON_EXPLICIT = (
    "You are a scene executor, not a creative writer. "
    "Your only job is to convert the given scene beats into vivid prose. "
    "You do not invent new events. You do not add characters not in the scene plan. "
    "You do not deviate from the beat list for any reason.\n\n"
    "You are writing a mature-audience scene with emotional and sensual tension.\n"
    "This is NOT an explicit sex scene unless directly specified by the scene brief.\n\n"
    "Rules:\n"
    "- Write ONLY scene prose, no notes/meta.\n"
    "- Keep the tone intimate and emotionally charged when appropriate.\n"
    "- Do NOT introduce graphic sexual acts not explicitly requested in key events.\n"
    "- If attraction is present, show it through dialogue, body language, and restrained detail.\n"
    "- Use exact character names and preserve continuity."
)

WRITER_SYSTEM_MATURE_NON_EXPLICIT_COMPACT = (
    "You are a scene executor, not a creative writer. "
    "Your only job is to convert the given scene beats into vivid prose. "
    "You do not invent new events. You do not add characters not in the scene plan. "
    "You do not deviate from the beat list for any reason. "
    "Write mature, emotionally intimate prose only. "
    "No graphic sexual acts unless explicitly demanded by the scene brief. "
    "No meta/disclaimers/thinking."
)

GENRE_SYSTEM_PROMPTS = {
    "dark fantasy": {
        "role": "You are a masterful fiction writer specializing in Dark Fantasy. You write atmospheric, grim, and immersive prose with rich sensory details, focusing on shadows, moral ambiguity, and a world where magic is dangerous and costly.",
        "compact": "Write immersive third-person Dark Fantasy prose only. Emphasize atmospheric, grim details."
    },
    "romance": {
        "role": "You are a masterful fiction writer specializing in Romance. You write emotionally resonant, character-driven prose that highlights chemistry, tension, and the development of relationships.",
        "compact": "Write immersive third-person Romance prose only. Emphasize emotional resonance and character chemistry."
    },
    "thriller": {
        "role": "You are a masterful fiction writer specializing in Thrillers. You write fast-paced, suspenseful prose with high stakes, sharp dialogue, and constant tension.",
        "compact": "Write immersive third-person Thriller prose only. Emphasize suspense, tension, and high stakes."
    },
    "sci-fi": {
        "role": "You are a masterful fiction writer specializing in Science Fiction. You write immersive, detail-rich prose that brings futuristic technology, alien worlds, and complex concepts to life.",
        "compact": "Write immersive third-person Sci-Fi prose only. Emphasize futuristic elements and vivid world-building."
    },
    "horror": {
        "role": "You are a masterful fiction writer specializing in Horror. You write dread-inducing, atmospheric prose that builds suspense and terrifies through psychological tension and visceral descriptions.",
        "compact": "Write immersive third-person Horror prose only. Emphasize dread, atmospheric suspense, and terror."
    },
    "urban fantasy": {
        "role": "You are a masterful fiction writer specializing in Urban Fantasy. You write gritty, fast-paced prose that seamlessly blends magical elements with modern, real-world settings.",
        "compact": "Write immersive third-person Urban Fantasy prose only. Emphasize the blend of magic and modern reality."
    },
    "mystery": {
        "role": "You are a masterful fiction writer specializing in Mystery. You write intricate, suspenseful prose that focuses on clues, deduction, and keeping the reader guessing.",
        "compact": "Write immersive third-person Mystery prose only. Emphasize intrigue, clues, and suspense."
    },
    "literary fiction": {
        "role": "You are a masterful fiction writer specializing in Literary Fiction. You write profound, beautifully crafted prose with a focus on deep character studies, thematic complexity, and stylistic elegance.",
        "compact": "Write immersive third-person Literary Fiction prose only. Emphasize beautiful prose, deep character study, and theme."
    }
}

GENRE_OPENINGS = {
    "dark fantasy": "Write a grim, atmospheric Dark Fantasy fiction scene. Emphasize shadows, danger, and a sense of dread.",
    "romance": "Write an emotionally resonant Romance fiction scene. Emphasize character chemistry, emotional depth, and tension.",
    "thriller": "Write a fast-paced, suspenseful Thriller fiction scene. Emphasize high stakes, tension, and sharp pacing.",
    "sci-fi": "Write a detail-rich Science Fiction scene. Emphasize futuristic concepts, immersive world-building, and technology.",
    "horror": "Write a dread-inducing Horror fiction scene. Emphasize atmospheric suspense, terror, and visceral tension.",
    "urban fantasy": "Write a gritty Urban Fantasy fiction scene. Emphasize the seamless blend of magic and modern reality.",
    "mystery": "Write an intricate Mystery fiction scene. Emphasize suspense, clues, and an atmosphere of intrigue.",
    "literary fiction": "Write a beautifully crafted Literary Fiction scene. Emphasize stylistic elegance, deep character study, and thematic nuance."
}


def _scene_intensity(scene_plan: dict) -> str:
    raw_level = str(scene_plan.get("intimacy_level", "")).strip().lower()
    if raw_level:
        if raw_level in {"explicit", "erotic", "sex", "sexual"}:
            return "explicit"
        if raw_level in {"sensual", "romantic"}:
            return raw_level
        if raw_level in {"intimate", "tender", "soft"}:
            return "romantic"
        if raw_level in {"none", "clean", "fade", "closed"}:
            return "none"

    summary = str(scene_plan.get("summary", "")).lower()
    events = " ".join(scene_plan.get("key_events", [])).lower()
    scene_type = str(scene_plan.get("type", "")).lower()
    blob = f"{scene_type} {summary} {events}"
    if any(term in blob for term in EXPLICIT_ACTION_TERMS):
        return "explicit"
    if any(term in blob for term in INTIMATE_NON_EXPLICIT_TERMS):
        return "sensual"
    return "standard"


def _build_scene_prompt(scene_plan, chapter_num, context, previous_ending, genre="", compact: bool = False):
    summary = scene_plan.get('summary', '')
    characters = ', '.join(scene_plan.get('characters_present', []))
    location = scene_plan.get('location', '')
    mood = scene_plan.get('mood', '')
    events = ', '.join(scene_plan.get('key_events', []))
    original_user_brief = str(scene_plan.get("original_user_brief", "")).strip()
    required_beats = [
        str(beat).strip()
        for beat in scene_plan.get("required_beats", [])
        if str(beat).strip()
    ]
    word_target = max(scene_plan.get('word_target', 600), config.WORDS_PER_SCENE_MIN)
    scene_num = scene_plan.get('scene_number', 1)
    # New fields injected by the planner for narrative continuity
    narrative_bridge = str(scene_plan.get('narrative_bridge', '')).strip()
    prev_location = str(scene_plan.get('prev_location', '')).strip()

    intensity = _scene_intensity(scene_plan)
    if intensity == "explicit":
        word_target = max(word_target, 800)

    genre_lower = (genre or "").lower()
    is_explicit_genre = genre_lower in EXPLICIT_GENRES or any(w in genre_lower for w in ["erotic", "adult", "smut"])
    is_explicit_scene = intensity == "explicit"

    if is_explicit_genre and is_explicit_scene:
        opening = (
            "Write an EXPLICIT, GRAPHIC erotic fiction scene. "
            "Include detailed descriptions of sexual acts, body parts, arousal, and physical sensations. "
            "Use explicit language — no euphemisms. Never fade to black. Never summarize sex."
        )
    elif is_explicit_genre:
        opening = (
            "Write a vivid, immersive fiction scene for a mature adult story. "
            "Include sensual tension, physical descriptions of characters, and emotional depth. "
            "Do NOT introduce graphic sexual acts unless the scene brief explicitly demands them."
        )
    elif genre_lower in GENRE_OPENINGS:
        opening = GENRE_OPENINGS[genre_lower]
    elif genre:
        opening = f"Write a {genre} fiction scene."
    else:
        opening = "Write a fiction scene."

    parts = [opening]

    if compact:
        parts.append("")
        parts.append("Write full prose scene (not summary). Include dialogue with speaker attribution, action, emotion, sensory detail.")
        parts.append("")
    else:
        parts.append("")
        parts.append("WRITING STYLE: Write FULL PROSE — not a summary. Include:")
        parts.append("- Vivid descriptions of settings, characters, and actions")
        parts.append("- Dialogue between characters (with quotation marks and clear speaker attribution)")
        parts.append("- Internal thoughts and emotions")
        parts.append("- Physical movements and body language")
        parts.append("- Sensory details (sight, sound, smell, touch)")
        parts.append("")

    # ── Continuity & Grounding Block (MOST IMPORTANT — placed BEFORE the scene brief) ──
    is_chapter_start = (scene_num == 1 and chapter_num > 1)
    is_mid_chapter = bool(previous_ending and not is_chapter_start)
    location_changed = bool(
        location and prev_location and
        location.strip().lower() != prev_location.strip().lower()
    )

    if previous_ending:
        if is_chapter_start:
            # NEW CHAPTER OPENING — #1 pipeline failure: dropping reader mid-action.
            # Force orientation paragraph FIRST, always.
            parts.append("╔═══ CHAPTER OPENING — READ THIS BEFORE WRITING ═══╗")
            parts.append("You are opening a NEW CHAPTER. The PREVIOUS CHAPTER ended here:")
            parts.append(f'"""{previous_ending}"""')
            parts.append("")
            parts.append("MANDATORY OPENING STRUCTURE — your first paragraph MUST:")
            parts.append("  1. Signal time or continuity. Mix it up (e.g. 'Later that day', 'Three hours passed', 'As evening fell', 'By the time she arrived'). DO NOT default to 'The next morning' every time.")
            parts.append("  2. Ground the reader in WHERE the character is — the room, place, light, sounds.")
            parts.append("  3. Show the character's EMOTIONAL STATE carrying over from the previous chapter's ending.")
            parts.append("  4. Only AFTER doing 1-3, begin action or dialogue.")
            parts.append("Do NOT start mid-action. Do NOT start with dialogue. Do NOT copy the reference verbatim.")
            parts.append("╚══════════════════════════════════════════════════╝")
        else:
            # MID-CHAPTER CONTINUATION
            parts.append("╔═══ SCENE CONTINUATION — READ THIS BEFORE WRITING ═══╗")
            parts.append("The previous scene ended with:")
            parts.append(f'"""{previous_ending}"""')
            parts.append("")
            parts.append("CONTINUATION RULES (all mandatory):")
            parts.append("  1. Your very first sentence picks up directly from this moment.")
            parts.append("  2. Do NOT recap, summarize, or re-introduce what just happened.")
            parts.append("  3. Do NOT copy any sentence from the above verbatim.")
            if location_changed:
                parts.append("")
                parts.append(f"  4. ⚠ LOCATION CHANGE: Previous scene was at [{prev_location}]. This scene is at [{location}].")
                parts.append("     Your FIRST PARAGRAPH must be a transition that moves the character(s) from")
                parts.append("     the old location to the new one — show the travel, the decision to leave,")
                parts.append("     or time passing. Do NOT teleport characters. Do NOT just start in the new place.")
            else:
                parts.append("  4. If this scene shifts location, write an explicit transition first.")
            parts.append("╚══════════════════════════════════════════════════════╝")
        parts.append("")

    elif scene_num == 1 and chapter_num == 1:
        # Very first scene of the story
        parts.append("╔═══ STORY OPENING ═══╗")
        parts.append(
            "This is the very first scene of the story. Ground the reader immediately: "
            "establish WHERE we are, WHEN it is, and WHO the protagonist is — "
            "before any action or dialogue begins."
        )
        parts.append("╚═════════════════════╝")
        parts.append("")

    # ── Anti-Mirror Constraint: closing paragraph of the previous scene ──
    # Injected here so the LLM sees it as a hard constraint before the brief.
    if previous_ending:
        # Extract the closing paragraph (last non-empty block)
        prev_paragraphs = [p.strip() for p in previous_ending.split("\n\n") if p.strip()]
        closing_para = prev_paragraphs[-1] if prev_paragraphs else previous_ending.strip()
        if len(closing_para) > 600:
            closing_para = closing_para[-600:].lstrip()

        # Detect if the previous close was introspective/reflective
        closing_lower = closing_para.lower()
        _reflection_signals = {
            "thought", "wonder", "realize", "realise", "felt", "feeling",
            "remembered", "mind", "heart", "soul", "breath", "silence",
            "stared", "gazed", "watched", "waited",
        }
        is_reflective = any(sig in closing_lower for sig in _reflection_signals)
        open_with_hint = (
            " If the previous scene ended on internal reflection open with action or dialogue instead."
            if is_reflective
            else " Open with a concrete action, sensory detail, or forward momentum instead."
        )

        parts.append("╔═══ OPENING CONSTRAINT — READ BEFORE WRITING YOUR FIRST SENTENCE ═══╗")
        parts.append("The previous scene ended with this closing paragraph:")
        parts.append(f'\"{closing_para}\"')
        parts.append("")
        parts.append(
            "Your opening must not repeat the sentence structure, mood, or descriptive "
            "framing of this closing paragraph."
            + open_with_hint
        )
        parts.append("╚════════════════════════════════════════════════════════════════════╝")
        parts.append("")


    # ── Narrative Bridge: WHY this scene follows the last one ────────────
    if narrative_bridge:
        parts.append(f"STORY CONTEXT — WHY THIS SCENE FOLLOWS: {narrative_bridge}")
        parts.append("")

    # ── Scene Brief ──────────────────────────────────────────────────────
    parts.append(f"Scene {scene_num}, Chapter {chapter_num}:")

    # Build the mandatory beat checklist from the summary.
    # Use required_beats if already provided; otherwise derive from the summary.
    beat_source = original_user_brief or summary
    if not required_beats and beat_source:
        # Split on periods and commas; filter empty / very short fragments
        raw_splits = re.split(r'[.,]+', beat_source)
        derived_beats = [b.strip().strip('"\' ') for b in raw_splits if len(b.strip()) > 6]
        if derived_beats:
            required_beats = derived_beats

    if required_beats:
        parts.append("MANDATORY SCENE BEATS (follow in order):")
        for idx, beat in enumerate(required_beats, start=1):
            parts.append(f"{idx}. {beat}")
        parts.append(
            "Write the scene by working through each beat fully before moving to the next. "
            "Do not skip any beat. Do not end the scene before completing all beats."
        )
    else:
        parts.append(f"What happens: {summary}")

    if original_user_brief and original_user_brief != summary:
        parts.append(f"Original user brief (treat as the strict scene script): {original_user_brief}")

    # Inject character profiles with lifecycle/age info when available;
    # fall back to simple comma-separated names otherwise.
    character_profiles = str(scene_plan.get("character_profiles", "")).strip()
    if character_profiles:
        # Strip parentheticals to avoid unnatural prose/dialogue tags
        character_profiles = re.sub(r"\s*\([^)]*\)", "", character_profiles)
        parts.append(character_profiles)
    elif characters:
        characters = re.sub(r"\s*\([^)]*\)", "", characters)
        parts.append(f"Characters: {characters}")
    if location:
        parts.append(f"Setting: {location}")
        if not compact and not (location_changed and is_mid_chapter):
            parts.append(
                "Location lock: If the previous ending was somewhere else, show how "
                "the characters arrived here before the scene begins."
            )
    if mood:
        parts.append(f"Mood: {mood}")
    if events:
        parts.append(f"Events: {events}")
    parts.append(f"Intensity: {intensity}")

    intimacy_level = str(scene_plan.get("intimacy_level", "")).strip().lower()
    if intimacy_level:
        parts.append(f"Intimacy level: {intimacy_level}")
        if intimacy_level == "explicit":
            parts.append(
                "CRITICAL: Write extremely detailed, graphic, and long explicit sexual content. "
                "Describe every touch, sensation, and physical reaction. Do NOT skip any part of the encounter. "
                "Do NOT use metaphors or euphemisms. Use direct, explicit language. "
                "The user wants a slow-paced, highly descriptive experience. "
                "Expand the sexual encounter significantly."
            )
        elif intimacy_level in {"romantic", "sensual"}:
            parts.append(
                "Keep intimacy non-explicit. Avoid graphic sexual anatomy or explicit sex acts. "
                "If sex is implied, fade to black and focus on emotion and aftermath."
            )
        elif intimacy_level == "none":
            parts.append("Do not introduce sexual content in this scene.")

    parts.append(
        "Follow summary + key events exactly. Do not escalate into explicit sex unless key events require it."
    )
    parts.append(
        "SCENE SCOPE: Write ONLY the events in key_events. "
        "When those events are complete, end the scene naturally. "
        "Do NOT write into future beats or the next scene."
    )

    # ── Character & Story Context ─────────────────────────────────────────
    if context:
        trimmed = _middle_clip(context, 950 if compact else 1800)
        parts.append("\nCharacter/story info (facts only; do not copy prose from it):")
        parts.append(trimmed)

    # ── Word Count & Output Rules ─────────────────────────────────────────
    parts.append(
        f"\nWrite this as a fully realized scene showing all events and dialogue in detail. "
        f"While you should aim for about {word_target} words of rich and pacing-appropriate detail, "
        f"prioritize content quality: do NOT pad the scene with redundant descriptions, circular thoughts, "
        f"or unnecessary filler content. Generate only what is naturally needed for this scene."
    )
    parts.append("Do not reuse distinctive phrases, paragraphs, or closing beats from prior scenes.")
    parts.append("OUTPUT ONLY STORY TEXT — no titles, headers, or notes.")

    return "\n".join(parts)


class SceneWriter(AgentContract):
    def __init__(self, primary_model: LLMInterface, fallback_model: LLMInterface):
        self.name = "writer"
        self.primary = primary_model
        self.fallback = fallback_model
        self._last_provider = "none"
        self._genre = ""
        self._is_explicit_genre = False
        self._compact_mode = False

    def set_compact_mode(self, enabled: bool):
        self._compact_mode = bool(enabled)

    def run(self, state: dict) -> dict:
        mode = state.get("mode", "write")
        iteration = state.get("iteration", 0)
        
        if mode == "final_patch":
            scene_text = self.final_patch_scene(
                original_text=state["original_text"],
                scene_plan=state.get("scene_plan", {}),
            )
            next_action = "editor"
        elif mode == "patch":
            scene_text = self.patch_scene(
                original_text=state["original_text"],
                issues=state.get("issues", []),
                state_context=state.get("state_context", ""),
                scene_plan=state.get("scene_plan", {}),
                iteration=iteration,
            )
            next_action = "critic"
        elif mode == "rewrite":
            scene_text = self.rewrite_scene(
                original_text=state["original_text"],
                issues=state.get("issues", []),
                state_context=state.get("state_context", ""),
                scene_plan=state.get("scene_plan", {}),
                iteration=iteration,
            )
            next_action = "critic"
        else:
            scene_text = self.write_scene(
                scene_plan=state["scene_plan"],
                chapter_num=state["chapter_num"],
                context=state.get("context", ""),
                previous_ending=state.get("previous_ending", ""),
                stream_callback=state.get("stream_callback"),
            )
            next_action = "critic"
        return {
            "output": {"scene_text": scene_text, "provider": self.last_provider},
            "confidence": 0.75,
            "next_action": next_action,
        }

    @property
    def last_provider(self) -> str:
        return self._last_provider

    def set_genre(self, genre: str):
        """Set the genre for tone awareness."""
        self._genre = genre
        genre_lower = genre.lower().strip()
        self._is_explicit_genre = (
            genre_lower in EXPLICIT_GENRES
            or any(w in genre_lower for w in ["erotic", "adult", "smut", "explicit", "nsfw"])
        )
        if self._is_explicit_genre:
            logger.info(f"Genre '{genre}' detected — preserving explicit-content prompts")

    def write_scene(
        self,
        scene_plan,
        chapter_num,
        context,
        previous_ending="",
        stream_callback=None,
    ):
        prompt = _build_scene_prompt(
            scene_plan, chapter_num, context, previous_ending, self._genre, compact=self._compact_mode,
        )
        
        required_action = ""
        if scene_plan.get("key_events"):
            required_action = scene_plan["key_events"][0]
        if required_action:
            prompt = f"REQUIRED EVENT: This scene MUST explicitly show {required_action}. This is non-negotiable. Do not end the scene without it.\n\n" + prompt

        system = self._select_system_for_scene(scene_plan)
        text = self._generate_with_fallback(
            prompt,
            system,
            temperature=config.AGENT_TEMPERATURES["writer"],
            stream_callback=stream_callback,
            max_tokens=self._scene_max_tokens(scene_plan) if self._compact_mode else None,
        )
        text = self._quality_check(
            text, scene_plan, chapter_num, context, previous_ending, system=system,
        )
        return text

    def rewrite_scene(self, original_text, issues, state_context, scene_plan=None, iteration=1):
        issues_text = "\n".join(
            f"- [{i.get('type', 'error')}] {i.get('detail', '')} "
            f"(Suggestion: {i.get('suggestion', 'Fix this')})"
            for i in issues
        )
        scene_plan = scene_plan or {}
        scene_brief = (
            f"Summary: {scene_plan.get('summary', '')}\n"
            f"Required setting: {scene_plan.get('location', '')}\n"
            f"Required characters: {', '.join(scene_plan.get('characters_present', []))}\n"
            f"Required events: {', '.join(scene_plan.get('key_events', []))}\n"
            f"Required ordered beats: {'; '.join(scene_plan.get('required_beats', []))}"
        )
        system = self._select_system_for_scene(scene_plan)
        if self._compact_mode:
            prompt = (
                "Rewrite scene to fix blocking continuity issues. Preserve plan, events, meaning, and prose length.\n\n"
                f"Issues:\n{issues_text[:700]}\n\n"
                f"Plan:\n{scene_brief[:700]}\n\n"
                f"State:\n{state_context[:900]}\n\n"
                f"Scene:\n{original_text[:3200]}"
            )
        else:
            prompt = (
                f"Rewrite this scene to fix the following consistency issues:\n\n"
                f"=== ISSUES ===\n{issues_text}\n\n"
                f"=== REQUIRED SCENE PLAN ===\n{scene_brief}\n\n"
                f"=== ORIGINAL TEXT ===\n{original_text}\n\n"
                f"=== CORRECT STATE ===\n{state_context}\n\n"
                f"Rewrite the scene, fixing all issues while preserving the required scene plan. "
                f"If the required setting differs from the current state, include a clear transition."
            )

        if iteration > 1:
            instruction_block = (
                f"PREVIOUS ATTEMPT FAILED. You MUST fix these specific issues:\n{issues_text}\n"
                f"Do not rewrite the entire scene. Keep what worked. Only fix the listed issues. "
                f"The scene MUST include: {scene_plan.get('summary', '')}"
            )
            prompt = f"{instruction_block}\n\n{prompt}"
            
        required_action = ""
        if scene_plan.get("key_events"):
            required_action = scene_plan["key_events"][0]
        if required_action:
            prompt = f"REQUIRED EVENT: This scene MUST explicitly show {required_action}. This is non-negotiable. Do not end the scene without it.\n\n" + prompt

        temperature = min(1.0, config.AGENT_TEMPERATURES["writer"] + max(0, iteration - 1) * 0.1)

        text = self._generate_with_fallback(
            prompt,
            system,
            temperature=temperature,
            max_tokens=self._scene_max_tokens(scene_plan) if self._compact_mode else None,
        )
        return self._sanitize_scene_text(text)

    def patch_scene(self, original_text, issues, state_context, scene_plan, iteration):
        issue = issues[0] if issues else {}
        nearest = issue.get("nearest_paragraph", "")
        if not nearest or nearest not in original_text:
            return self.rewrite_scene(original_text, issues, state_context, scene_plan, iteration)
        
        prompt = (
            f"Write ONLY a single replacement paragraph to fix a continuity error.\n"
            f"ERROR: {issue.get('detail', '')}\n"
            f"SUGGESTION: {issue.get('suggestion', '')}\n\n"
            f"ORIGINAL PARAGRAPH TO REPLACE:\n{nearest}\n\n"
            f"Output ONLY the new paragraph text. No intro, no meta."
        )
        system = "You are an expert editor fixing a single paragraph."
        new_para = self._generate_with_fallback(prompt, system, temperature=0.7)
        new_para = self._sanitize_scene_text(new_para)
        
        if new_para:
            return original_text.replace(nearest, new_para)
        return original_text

    def final_patch_scene(self, original_text, scene_plan):
        required_action = ""
        if scene_plan.get("key_events"):
            required_action = scene_plan["key_events"][0]
        elif scene_plan.get("summary"):
            required_action = scene_plan["summary"]

        characters = ", ".join(scene_plan.get("characters_present", []))
        characters = re.sub(r"\s*\([^)]*\)", "", characters)
        location = scene_plan.get("location", "")

        prompt = (
            f"Generate a 2-3 sentence paragraph explicitly showing this required event:\n"
            f"{required_action}\n\n"
        )
        if characters:
            prompt += f"Characters in this scene: {characters}\n"
        if location:
            prompt += f"Setting: {location}\n"
            
        prompt += f"\nFor context, the preceding text is:\n...{original_text[-400:]}\n\n"
        prompt += (
            "Use the exact character names listed above. Continue naturally from the preceding text. "
            "Write ONLY the paragraph text. Do not add metadata or conversational intro."
        )
        system = "You are an expert writer generating a specific missing beat. Use the exact character names provided."
        new_para = self._generate_with_fallback(prompt, system, temperature=0.7)
        new_para = self._sanitize_scene_text(new_para)
        
        if new_para:
            return original_text.rstrip() + "\n\n" + new_para
        return original_text

    def _select_system_for_scene(self, scene_plan: dict) -> str:
        intensity = _scene_intensity(scene_plan or {})
        if intensity == "explicit":
            return WRITER_SYSTEM_EXPLICIT_COMPACT if self._compact_mode else WRITER_SYSTEM_EXPLICIT
            
        genre_lower = self._genre.lower()
        if genre_lower in GENRE_SYSTEM_PROMPTS:
            genre_info = GENRE_SYSTEM_PROMPTS[genre_lower]
            if self._compact_mode:
                return (
                    f"{genre_info['compact']} "
                    "Use exact character names and continuity. "
                    "No headers/meta/disclaimers/planning/thinking. Start with story text. "
                    "Do not add explicit sexual acts unless the scene brief explicitly requires them."
                )
            else:
                return (
                    f"{genre_info['role']}\n\n"
                    "You are a scene executor, not a creative writer. "
                    "Your only job is to convert the given scene beats into vivid prose. "
                    "You do not invent new events. You do not add characters not in the scene plan. "
                    "You do not deviate from the beat list for any reason.\n\n"
                    "Rules:\n"
                    "- Write ONLY the scene content. No meta-commentary, no scene headers, no author notes.\n"
                    "- Use third-person perspective. Include vivid sensory details.\n"
                    "- Write natural, character-appropriate dialogue.\n"
                    "- Show emotions through actions and body language.\n"
                    "- Focus on natural pacing and vivid progression. Avoid introducing unnecessary filler, redundant descriptions, or repetitive monologues merely to pad the length.\n"
                    "- ALWAYS use the EXACT character names and descriptions provided. Never swap or confuse characters.\n"
                    "- Pay close attention to each character's gender, body, and role.\n"
                    "- Follow the scene plan's intensity and events exactly.\n"
                    "- Do NOT add explicit sexual acts unless the scene brief explicitly requires them.\n"
                    "- 'Intimacy' can mean emotional closeness, not automatically sexual acts.\n"
                    "- Do NOT add warnings, disclaimers, or content notices.\n"
                    "- Write exactly what is asked in the scene brief."
                )

        if not self._is_explicit_genre:
            return WRITER_SYSTEM_COMPACT if self._compact_mode else WRITER_SYSTEM
        return (
            WRITER_SYSTEM_MATURE_NON_EXPLICIT_COMPACT
            if self._compact_mode
            else WRITER_SYSTEM_MATURE_NON_EXPLICIT
        )

    def _stream_from_model(self, model, prompt, system, temperature, stream_callback, max_tokens=None):
        self._last_provider = "groq" if isinstance(model, GroqModel) else "llama.cpp"
        chunks = []
        for chunk in model.generate_streaming(
            prompt=prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            chunks.append(chunk)
            if stream_callback:
                if stream_callback(chunk) is False:
                    break
        return "".join(chunks)

    def _generate_with_fallback(
        self,
        prompt,
        system,
        temperature=None,
        stream_callback=None,
        max_tokens=None,
    ):
        if self.primary is self.fallback:
            if stream_callback and hasattr(self.fallback, "generate_streaming"):
                content = self._stream_from_model(
                    self.fallback, prompt, system, temperature, stream_callback, max_tokens,
                )
                if content.strip():
                    self._last_provider = "groq" if isinstance(self.fallback, GroqModel) else "llama.cpp"
                    return self._sanitize_scene_text(content)

            response = self.fallback.generate_with_retry(
                prompt=prompt,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            self._last_provider = response.provider
            return self._sanitize_scene_text(response.content)

        # Always try Groq first — even for explicit genres
        # Groq 70B writes much better prose; only falls back if content is blocked
        try:
            if stream_callback and hasattr(self.primary, "generate_streaming"):
                content = self._stream_from_model(
                    self.primary, prompt, system, temperature, stream_callback, max_tokens,
                )
                if content.strip():
                    self._last_provider = "groq" if isinstance(self.primary, GroqModel) else "llama.cpp"
                    return self._sanitize_scene_text(content)
            else:
                response = self.primary.generate_with_retry(
                    prompt=prompt,
                    system=system,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    max_retries=1,
                )
                self._last_provider = response.provider
                return self._sanitize_scene_text(response.content)
        except ContentBlockedError:
            logger.warning("Groq content blocked. Falling back to local uncensored model.")
        except Exception as e:
            logger.warning(f"Groq failed: {e}. Falling back to local model.")

        try:
            if stream_callback and hasattr(self.fallback, "generate_streaming"):
                content = self._stream_from_model(
                    self.fallback, prompt, system, temperature, stream_callback, max_tokens,
                )
                if content.strip():
                    self._last_provider = "llama.cpp"
                    return self._sanitize_scene_text(content)

            response = self.fallback.generate_with_retry(
                prompt=prompt,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            self._last_provider = response.provider
            return self._sanitize_scene_text(response.content)
        except Exception as e:
            raise RuntimeError(f"Both generation backends failed: {e}")

    def _quality_check(self, text, scene_plan, chapter_num, context, previous_ending, system=None):
        if system is None:
            if self._is_explicit_genre:
                system = WRITER_SYSTEM_EXPLICIT_COMPACT if self._compact_mode else WRITER_SYSTEM_EXPLICIT
            else:
                system = WRITER_SYSTEM_COMPACT if self._compact_mode else WRITER_SYSTEM
        text = self._sanitize_scene_text(text)
        word_count = len(text.split())
        word_target = scene_plan.get("word_target", 600)
        
        # Explicit scenes have a higher quality floor
        intensity = _scene_intensity(scene_plan)
        min_threshold = config.MIN_SCENE_WORDS
        if intensity == "explicit":
            min_threshold = max(min_threshold, 700)

        if word_count < min_threshold * 0.9:
            logger.warning(f"Scene too short ({word_count} words). Expanding...")
            prompt = (
                f"The following scene reads too much like a brief summary and needs more narrative depth. "
                f"Elaborate on the actions, sensory details, thoughts, and dialogue of the characters present. "
                f"Do NOT add redundant paragraphs or filler; instead, deepen the existing moment to bring it to a fully realized prose scene.\n\n"
                f"=== CURRENT TEXT ===\n{text}\n\n"
                f"=== CONTEXT ===\nSummary: {scene_plan.get('summary', '')}\n"
                f"Characters: {', '.join(scene_plan.get('characters_present', []))}\n"
                f"Mood: {scene_plan.get('mood', '')}\n\n"
                f"Write the expanded scene as complete prose."
            )
            text = self._generate_with_fallback(
                prompt,
                system,
                max_tokens=self._scene_max_tokens(scene_plan) if self._compact_mode else None,
            )

        if self._has_strong_overlap_with_previous(text, previous_ending):
            logger.warning("Cross-scene repetition detected. Regenerating with anti-repeat constraints...")
            prompt = (
                _build_scene_prompt(scene_plan, chapter_num, context, previous_ending, self._genre)
                if not self._compact_mode
                else _build_scene_prompt(
                    scene_plan, chapter_num, context, previous_ending, self._genre, compact=True,
                )
                + "\n\nCRITICAL:\n"
                + "- Move the plot forward immediately.\n"
                + "- Do NOT reuse any sentence from the reference snippet.\n"
                + "- Use fresh wording and new actions, not recap."
            )
            text = self._generate_with_fallback(
                prompt,
                system,
                temperature=min(1.0, config.AGENT_TEMPERATURES["writer"] + 0.2),
                max_tokens=self._scene_max_tokens(scene_plan) if self._compact_mode else None,
            )

        if self._is_repetitive(text):
            logger.warning("Repetitive output detected. Regenerating...")
            prompt = _build_scene_prompt(
                scene_plan, chapter_num, context, previous_ending, self._genre, compact=self._compact_mode,
            )
            text = self._generate_with_fallback(
                prompt, system,
                temperature=min(1.0, config.LOCAL_MODEL_PARAMS["temperature"] + 0.15),
                max_tokens=self._scene_max_tokens(scene_plan) if self._compact_mode else None,
            )

        continuation_attempts = 0
        max_continuations = max(0, config.GROQ_CONTINUATION_ATTEMPTS)
        while self._looks_truncated(text) and continuation_attempts < max_continuations:
            continuation_attempts += 1
            logger.warning(
                "Truncated scene ending detected. Requesting continuation "
                "(attempt %s/%s)...",
                continuation_attempts,
                max_continuations,
            )
            continuation_prompt = (
                "Continue this scene from the exact final fragment below.\n"
                "Do NOT repeat any prior sentence. Do NOT restart.\n"
                "Write 1-3 new paragraphs and end on a complete sentence with final punctuation.\n\n"
                f"=== SCENE DRAFT ===\n{text}\n"
            )
            continuation = self._generate_with_fallback(
                continuation_prompt,
                system,
                temperature=min(1.0, config.AGENT_TEMPERATURES["writer"] + 0.1),
                max_tokens=700 if self._compact_mode else None,
            )
            continuation = self._trim_prefix_overlap(text, continuation)
            if not continuation.strip():
                break
            text = f"{text.rstrip()}\n\n{continuation.lstrip()}".strip()

        return self._sanitize_scene_text(text)

    @staticmethod
    def _scene_max_tokens(scene_plan: dict) -> int:
        try:
            target = int(scene_plan.get("word_target", 600))
        except (TypeError, ValueError):
            target = 600
        return max(900, min(1700, int(target * 1.8) + 250))

    @staticmethod
    def _is_repetitive(text, threshold=None):
        threshold = threshold or config.MAX_REPETITION_RATIO
        sentences = [
            re.sub(r"\s+", " ", s.strip()).lower()
            for s in re.split(r"(?<=[.!?])\s+", text or "")
            if len(s.split()) >= 8
        ]
        sentence_counts = Counter(sentences)
        if any(count > 1 for count in sentence_counts.values()):
            return True

        words = re.findall(r"[A-Za-z']+", (text or "").lower())
        if len(words) < 20:
            return False
        ngrams = [tuple(words[i:i + 6]) for i in range(len(words) - 5)]
        if not ngrams:
            return False
        counter = Counter(ngrams)
        repeated = sum(c - 1 for c in counter.values() if c > 1)
        return (repeated / len(ngrams)) > threshold

    @staticmethod
    def _sanitize_scene_text(text: str) -> str:
        cleaned = _REASONING_BLOCK_RE.sub("", text or "")
        cleaned = _REASONING_TAG_RE.sub("", cleaned)
        cleaned = _META_OUTPUT_RE.sub("", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        
        # Deduplicate paragraphs
        paragraphs = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
        deduped = []
        for p in paragraphs:
            # Check if this paragraph is already in the last 4 paragraphs
            if not deduped or p not in deduped[-4:]:
                deduped.append(p)
        cleaned = "\n\n".join(deduped)
        return cleaned.strip()

    @staticmethod
    def _looks_truncated(text: str) -> bool:
        cleaned = (text or "").rstrip()
        if len(cleaned.split()) < 40:
            return False
        if cleaned.endswith(("...", "—", "-", ":", ";", ",")):
            return True
        if not cleaned.endswith((".", "!", "?", "\"", "'")):
            last_line = cleaned.splitlines()[-1].strip()
            if len(last_line.split()) >= 6:
                return True
        return False

    @staticmethod
    def _ngrams(words: list[str], n: int = 8) -> set[tuple[str, ...]]:
        if len(words) < n:
            return set()
        return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}

    def _has_strong_overlap_with_previous(self, text: str, previous_ending: str) -> bool:
        if not previous_ending:
            return False

        current_words = re.findall(r"[A-Za-z']+", text.lower())
        previous_words = re.findall(r"[A-Za-z']+", previous_ending.lower())
        current_ngrams = self._ngrams(current_words, n=8)
        previous_ngrams = self._ngrams(previous_words, n=8)
        if not current_ngrams or not previous_ngrams:
            return False

        overlap = len(current_ngrams & previous_ngrams) / len(current_ngrams)
        return overlap > 0.12

    @staticmethod
    def _trim_prefix_overlap(base_text: str, continuation: str) -> str:
        base_words = re.findall(r"\S+", base_text or "")
        cont_words = re.findall(r"\S+", continuation or "")
        if not base_words or not cont_words:
            return continuation.strip()

        max_overlap = min(35, len(base_words), len(cont_words))
        overlap = 0
        for size in range(max_overlap, 4, -1):
            if [w.lower() for w in base_words[-size:]] == [w.lower() for w in cont_words[:size]]:
                overlap = size
                break
        if overlap:
            cont_words = cont_words[overlap:]
        return " ".join(cont_words).strip()

    def generate_streaming(self, scene_plan, chapter_num, context, previous_ending=""):
        prompt = _build_scene_prompt(
            scene_plan, chapter_num, context, previous_ending, compact=self._compact_mode,
        )
        if isinstance(self.primary, GroqModel):
            try:
                system = WRITER_SYSTEM_COMPACT if self._compact_mode else WRITER_SYSTEM
                for chunk in self.primary.generate_streaming(prompt=prompt, system=system):
                    yield chunk
                self._last_provider = "groq"
                return
            except Exception as e:
                logger.warning(f"Groq streaming failed: {e}. Falling back.")

        if hasattr(self.fallback, 'generate_streaming'):
            system = WRITER_SYSTEM_COMPACT if self._compact_mode else WRITER_SYSTEM
            for chunk in self.fallback.generate_streaming(prompt=prompt, system=system):
                yield chunk
            self._last_provider = "llama.cpp"
        else:
            system = WRITER_SYSTEM_COMPACT if self._compact_mode else WRITER_SYSTEM
            response = self.fallback.generate(prompt=prompt, system=system)
            yield response.content
            self._last_provider = response.provider
