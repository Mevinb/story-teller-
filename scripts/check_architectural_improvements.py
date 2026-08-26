import os
import sys
import logging
from memory.state_manager import StateManager
from memory.tension_tracker import TensionTracker
from memory.motif_tracker import MotifTracker
from agents.pacing import PacingAgent
from agents.voice import VoiceAgent
from agents.consistency import ConsistencyEngine

logging.basicConfig(level=logging.INFO)

print("1. Testing TensionTracker...")
state = {"plot": {}}
# Estimate tension from high-tension text
high_tension_text = "Scream! Blood everywhere! He had to escape the dangerous fight before death found him! Panic!"
tension_high = TensionTracker.estimate_tension_from_text(high_tension_text)
print(f"High tension score: {tension_high}")
assert tension_high > 0.6

# Estimate tension from low-tension text
low_tension_text = "It was a calm, peaceful morning in the sunny garden. She smiled and relaxed with a warm embrace."
tension_low = TensionTracker.estimate_tension_from_text(low_tension_text)
print(f"Low tension score: {tension_low}")
assert tension_low < 0.4

# Record and trend
state = TensionTracker.record_chapter_tension(state, chapter_num=1, tension=0.2, label="intro")
state = TensionTracker.record_chapter_tension(state, chapter_num=2, tension=0.45, label="escalation")
state = TensionTracker.record_chapter_tension(state, chapter_num=3, tension=0.8, label="climax")
trend = TensionTracker.get_tension_trend(state)
print(f"Detected trend: {trend}")
assert trend in ("rising", "peak")

summary = TensionTracker.get_tension_summary(state)
print("Tension Summary:")
print(summary)
assert "Ch3" in summary
print("TensionTracker OK.")

print("\n2. Testing MotifTracker...")
state = {
    "world": {
        "motifs": {
            "red_scarf": {
                "meaning": "lost love",
                "aliases": ["crimson scarf", "silk scarf"],
                "mentions": 0,
                "chapters": [],
                "contexts": [],
            }
        }
    }
}
scene_text = "She wore the crimson scarf around her neck, remembering him."
found = MotifTracker.extract_motifs_from_text(scene_text, state["world"]["motifs"])
assert "red_scarf" in found
assert len(found["red_scarf"]) == 1

# Update motifs in state
motifs_updated = MotifTracker.update_motifs_after_scene(state, scene_text, chapter_num=1)
assert motifs_updated == 1
assert state["world"]["motifs"]["red_scarf"]["mentions"] == 1
assert 1 in state["world"]["motifs"]["red_scarf"]["chapters"]

summary = MotifTracker.get_motif_summary(state)
print("Motif Summary:")
print(summary)
assert "Red Scarf" in summary
print("MotifTracker OK.")

print("\n3. Testing PacingAgent...")
pacing_agent = PacingAgent()

# Dialogue heavy test
dialogue_heavy = '"Hello," she said.\n"Hi," he replied.\n"How are you?" she asked.\n"Good," he said.'
report = pacing_agent.analyze(dialogue_heavy)
print(f"Dialogue heavy pace: {report['overall_pace']}")
assert any(i["type"] == "dialogue_heavy" for i in report["issues"])

# Exposition dump test
expo_dump = "The kingdom had been ruled by three kings who all loved gold more than they loved the people. The people lived in absolute squalor, building high walls and farming dry dirt while the dragons watched from the mountains. Each year, the tax collector would arrive with twenty guards to take everything that had been saved, forcing families to hide in the deep dark caves under the forest."
report = pacing_agent.analyze(expo_dump)
print(f"Exposition dump pace: {report['overall_pace']}")
assert any(i["type"] == "exposition_dump" for i in report["issues"])
print("PacingAgent OK.")

print("\n4. Testing VoiceAgent...")
voice_agent = VoiceAgent()
char_data = {
    "personality": ["shy", "formal"],
    "voice_profile": {
        "speech_patterns": ["stutters when nervous"],
        "vocabulary": ["indeed", "pardon"],
        "quirks": [],
        "formality": "formal",
        "sentence_length": "short",
    }
}
guidance = voice_agent.build_voice_guidance(["Alice"], {"Alice": char_data})
print("Voice Guidance:")
print(guidance)
assert "Formality: formal" in guidance
assert "indeed" in guidance
print("VoiceAgent OK.")

from models.base import LLMResponse

class DummyModel:
    def generate_with_retry(self, *args, **kwargs):
        # Return a valid LLMResponse indicating no issues
        return LLMResponse(
            content='{"is_consistent": true, "issues": [], "state_updates": {}}',
            model="dummy",
            provider="dummy",
        )

critic = ConsistencyEngine(DummyModel())
# Single character present, referred to only by pronoun in prose - should not fail!
report = critic.validate(
    scene_text="She walked to the window and sighed, feeling lonely.",
    state_context="",
    scene_plan={"characters_present": ["Alice"], "summary": "Alice is lonely."},
)
print(f"Pronoun-only single character validation is_consistent: {report['is_consistent']}")
assert report["is_consistent"] is True

# Two characters present, one missing completely - should fail
report = critic.validate(
    scene_text="She walked to the window and sighed.",
    state_context="",
    scene_plan={"characters_present": ["Alice", "Bob"], "summary": "Alice and Bob talk."},
)
print(f"Two characters, one missing validation is_consistent: {report['is_consistent']}")
# Since Bob is missing by name and there is no pronoun matching Bob's presence, it should flag low-severity warnings or block.
# Let's inspect issues.
for issue in report["issues"]:
    print(f" - {issue['detail']} (blocking: {issue['blocking']}, severity: {issue['severity']})")
assert any(i["type"] == "character_contradiction" for i in report["issues"])
print("Pronoun-Aware Check OK.")

print("\n6. Testing Semantic Beat Validation...")
# Should match semantic equivalents: "John confesses his love." vs "He admitted he loved her."
# We'll use the actual sentence-transformer model in ConsistencyEngine
critic_real = ConsistencyEngine(DummyModel())
report = critic_real.validate(
    scene_text="John finally looked up at Alice, his voice trembling. He admitted he loved her, that he had for years.",
    state_context="",
    scene_plan={
        "characters_present": ["John", "Alice"],
        "summary": "John and Alice meet.",
        "required_beats": ["John confesses his love."],
    }
)
print(f"Semantic beat validation is_consistent: {report['is_consistent']}")
for issue in report["issues"]:
    print(f" - {issue['detail']}")
assert report["is_consistent"] is True
print("Semantic Beat Validation OK.")

print("\nAll architectural improvement tests passed successfully!")
