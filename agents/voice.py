"""
Character Voice Agent — Voice profile management and validation.

Maintains per-character voice profiles:
{
    "speech_patterns": ["short sentences", "sarcastic", "avoids contractions"],
    "vocabulary": ["mate", "bloody"],
    "quirks": ["always clears throat before speaking"],
    "formality": "informal",
    "sentence_length": "short"
}

These profiles are injected into the Scene Writer's prompt to ensure
each character has a distinct, consistent voice across the entire story.
"""
import logging
import re
from copy import deepcopy

from .contract import AgentContract

logger = logging.getLogger(__name__)

# Default voice profile template
_DEFAULT_VOICE = {
    "speech_patterns": [],
    "vocabulary": [],
    "quirks": [],
    "formality": "neutral",     # formal / neutral / informal / crude
    "sentence_length": "medium",  # short / medium / long / mixed
    "emotional_range": "moderate",  # restrained / moderate / expressive
}

# Formality detection keywords
_FORMAL_MARKERS = {
    "indeed", "certainly", "perhaps", "nevertheless", "furthermore",
    "moreover", "consequently", "therefore", "whom", "shall",
    "one must", "i believe", "if you would", "pardon",
}
_INFORMAL_MARKERS = {
    "yeah", "gonna", "wanna", "kinda", "dunno", "nah", "hey",
    "cool", "awesome", "dude", "bro", "man", "like", "stuff",
    "whatever", "y'know", "ain't", "gotta", "lemme",
}


class VoiceAgent(AgentContract):
    """
    Manages character voice profiles and generates voice guidance
    for injection into the Scene Writer's system prompt.

    This is NOT an LLM agent — it works purely from structured data
    in state.json and deterministic text analysis.
    """

    def __init__(self):
        self.name = "voice"

    def run(self, state: dict) -> dict:
        """Agent contract interface — build voice guidance for a scene."""
        scene_plan = state.get("scene_plan", {})
        characters = state.get("characters", {})

        guidance = self.build_voice_guidance(
            scene_plan.get("characters_present", []),
            characters,
        )

        return {
            "output": {"voice_guidance": guidance},
            "confidence": 0.95,
            "next_action": "writer",
        }

    @staticmethod
    def ensure_voice_profile(char_data: dict) -> dict:
        """
        Ensure a character has a voice profile.
        Initializes from personality traits if available.
        """
        if not isinstance(char_data, dict):
            return deepcopy(_DEFAULT_VOICE)

        voice = char_data.get("voice_profile")
        if isinstance(voice, dict) and "speech_patterns" in voice:
            result = deepcopy(_DEFAULT_VOICE)
            result.update(voice)
            return result

        # Initialize from personality traits
        profile = deepcopy(_DEFAULT_VOICE)
        personality = char_data.get("personality", [])
        if isinstance(personality, list):
            trait_text = " ".join(str(t).lower() for t in personality)

            # Infer speech patterns from personality
            if "shy" in trait_text or "quiet" in trait_text:
                profile["speech_patterns"].append("speaks softly, often hesitates")
                profile["sentence_length"] = "short"
                profile["emotional_range"] = "restrained"
            if "bold" in trait_text or "confident" in trait_text:
                profile["speech_patterns"].append("speaks with authority and directness")
                profile["emotional_range"] = "expressive"
            if "sarcastic" in trait_text or "witty" in trait_text:
                profile["speech_patterns"].append("uses dry humor and sarcasm")
            if "formal" in trait_text or "noble" in trait_text:
                profile["formality"] = "formal"
            if "crude" in trait_text or "rough" in trait_text:
                profile["formality"] = "crude"
            if "intelligent" in trait_text or "scholarly" in trait_text:
                profile["speech_patterns"].append("uses precise vocabulary")
                profile["sentence_length"] = "long"

        return profile

    @classmethod
    def update_voice_profile(
        cls,
        char_data: dict,
        updates: dict,
    ) -> dict:
        """
        Merge updates into a character's voice profile.

        Args:
            char_data: The full character dict.
            updates: Dict with optional voice profile fields.

        Returns:
            Updated char_data with merged voice profile.
        """
        profile = cls.ensure_voice_profile(char_data)

        for key in ("formality", "sentence_length", "emotional_range"):
            if key in updates:
                profile[key] = str(updates[key]).strip().lower()

        for list_key in ("speech_patterns", "vocabulary", "quirks"):
            if list_key in updates:
                new_items = updates[list_key]
                if isinstance(new_items, str):
                    new_items = [new_items]
                if isinstance(new_items, list):
                    existing = profile.get(list_key, [])
                    for item in new_items:
                        item = str(item).strip()
                        if item and item not in existing:
                            existing.append(item)
                    profile[list_key] = existing[-10:]  # Cap at 10

        char_data["voice_profile"] = profile
        return char_data

    @classmethod
    def analyze_dialogue_voice(cls, scene_text: str, char_name: str) -> dict:
        """
        Analyze a character's dialogue in prose to detect their speech patterns.
        Returns detected attributes that can be used to initialize or update a profile.
        """
        # Extract dialogue lines attributed to this character
        name_lower = char_name.lower()
        lines = scene_text.split("\n")
        char_dialogue = []

        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue

            # Check if this dialogue line is near the character's name
            line_lower = line_stripped.lower()
            has_name = name_lower in line_lower

            # Check for dialogue markers
            has_dialogue = '"' in line_stripped or "\u201c" in line_stripped

            if has_name and has_dialogue:
                # Extract the quoted text
                quotes = re.findall(r'"([^"]+)"', line_stripped)
                quotes.extend(re.findall(r'\u201c([^\u201d]+)\u201d', line_stripped))
                char_dialogue.extend(quotes)

        if not char_dialogue:
            return {}

        # Analyze detected dialogue
        all_words = []
        sentence_lengths = []
        for line in char_dialogue:
            words = line.split()
            all_words.extend(words)
            sentences = re.split(r"[.!?]+", line)
            for s in sentences:
                s_words = s.strip().split()
                if s_words:
                    sentence_lengths.append(len(s_words))

        word_set = set(w.lower() for w in all_words)

        # Detect formality
        formal_count = len(word_set & _FORMAL_MARKERS)
        informal_count = len(word_set & _INFORMAL_MARKERS)
        if formal_count > informal_count + 1:
            formality = "formal"
        elif informal_count > formal_count + 1:
            formality = "informal"
        else:
            formality = "neutral"

        # Detect sentence length tendency
        avg_length = sum(sentence_lengths) / max(1, len(sentence_lengths))
        if avg_length < 6:
            sent_len = "short"
        elif avg_length > 15:
            sent_len = "long"
        else:
            sent_len = "medium"

        # Detect contraction usage
        uses_contractions = any(
            "'" in w and w not in {"i'm", "i've", "i'll", "i'd"}
            for w in word_set
        )

        result = {
            "formality": formality,
            "sentence_length": sent_len,
        }

        if not uses_contractions and len(char_dialogue) >= 3:
            result.setdefault("speech_patterns", []).append("avoids contractions")

        return result

    @classmethod
    def build_voice_guidance(
        cls,
        characters_present: list,
        characters_data: dict,
        max_chars: int = 4,
    ) -> str:
        """
        Build a VOICE PROFILE block for injection into the Scene Writer prompt.

        Args:
            characters_present: List of character names in this scene.
            characters_data: Full characters dict from state.
            max_chars: Maximum number of voice profiles to include.

        Returns:
            Formatted string for prompt injection, or empty string if no profiles.
        """
        if not characters_present or not characters_data:
            return ""

        profiles = []
        for name in characters_present[:max_chars]:
            char = characters_data.get(name, {})
            if not isinstance(char, dict):
                continue

            voice = cls.ensure_voice_profile(char)

            # Only include if the profile has meaningful content
            has_content = (
                voice.get("speech_patterns")
                or voice.get("vocabulary")
                or voice.get("quirks")
                or voice.get("formality", "neutral") != "neutral"
            )
            if not has_content:
                continue

            parts = [f"  [{name}]"]
            if voice.get("formality", "neutral") != "neutral":
                parts.append(f"    Formality: {voice['formality']}")
            if voice.get("sentence_length", "medium") != "medium":
                parts.append(f"    Sentences: {voice['sentence_length']}")
            if voice.get("emotional_range", "moderate") != "moderate":
                parts.append(f"    Emotional range: {voice['emotional_range']}")
            if voice.get("speech_patterns"):
                patterns = "; ".join(voice["speech_patterns"][:4])
                parts.append(f"    Patterns: {patterns}")
            if voice.get("vocabulary"):
                vocab = ", ".join(voice["vocabulary"][:6])
                parts.append(f"    Vocabulary: {vocab}")
            if voice.get("quirks"):
                quirks = "; ".join(voice["quirks"][:3])
                parts.append(f"    Quirks: {quirks}")

            profiles.append("\n".join(parts))

        if not profiles:
            return ""

        header = (
            "=== CHARACTER VOICE PROFILES ===\n"
            "Each character below has a distinct voice. When writing their dialogue,\n"
            "match their speech patterns, vocabulary, and formality level.\n"
        )
        return header + "\n".join(profiles) + "\n=== END VOICE PROFILES ==="
