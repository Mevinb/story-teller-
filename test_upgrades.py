import os
import sys
import logging
from memory.state_manager import StateManager
from memory.vector_store import VectorStore
from memory.retriever import Retriever
from agents.consistency import ConsistencyEngine
from memory.event_extractor import EventExtractor

logging.basicConfig(level=logging.INFO)

print("1. Testing StateManager Character Profiles...")
sm = StateManager("./projects/test_project")
sm.initialize(
    title="Test", 
    genre="Sci-Fi", 
    premise="A test premise", 
    characters={"Alice": {"personality": ["brave", "smart"], "goals": ["survive"]}}
)
char = sm.get_character("Alice")
assert "brave" in char["personality"]
assert "survive" in char["goals"]
ctx = sm.get_context_window()
assert "Personality: brave" in ctx
assert "Goals: survive" in ctx
print("StateManager OK.")

print("\n2. Testing VectorStore CrossEncoder & Retriever...")
vs = VectorStore("./projects/test_project")
vs.add_text("Alice walked to the store.", chapter=1, scene=1)
vs.add_text("Alice fought the dragon.", chapter=1, scene=2)
vs.add_text("Bob went to sleep.", chapter=1, scene=3)

retriever = Retriever(sm, vs)
# Try semantic retrieval
res = vs.search("Alice combat", top_k=2)
for r in res:
    print(f" - {r.text} (score: {r.score})")

# Test retriever context
scene_plan = {"characters_present": ["Alice"], "summary": "Alice prepares for battle."}
context = retriever.retrieve_context(scene_plan=scene_plan, chapter_num=2, extra_query="Alice battle")
print(f"Retrieved context length: {len(context)}")
assert "Alice fought the dragon" in context or "Alice walked" in context
print("Retriever & CrossEncoder OK.")

print("\n3. Testing ConsistencyEngine Fast Validator...")
class DummyModel:
    def generate_with_retry(self, *args, **kwargs):
        raise ValueError("LLM was called, but should have been skipped!")
        
critic = ConsistencyEngine(DummyModel())
# Missing character should fail fast
report = critic.validate(
    scene_text="Bob walked alone in the forest.",
    state_context="",
    scene_plan={"characters_present": ["Alice", "Bob"], "summary": "Bob and Alice walk."},
)
assert report["is_consistent"] is False
assert any(i["type"] == "character_contradiction" for i in report["issues"])
print("ConsistencyEngine Fast Validator OK.")

print("\nAll tests passed successfully.")
