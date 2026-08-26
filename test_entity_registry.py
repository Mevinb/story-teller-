"""
Tests for the entity registry in StateManager.
Run: python3 -m pytest test_entity_registry.py -v --tb=short
"""
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from memory.state_manager import StateManager, _empty_state, _canonical_character_key


def make_project():
    tmp = tempfile.mkdtemp(prefix="st_entities_")
    return tmp


class TestEntityRegistry(unittest.TestCase):

    def setUp(self):
        self.dir = make_project()
        self.sm = StateManager(self.dir)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_empty_state_has_entities_field(self):
        state = _empty_state()
        self.assertIn("entities", state)
        self.assertEqual(state["entities"], {})

    def test_register_new_entity(self):
        self.sm.initialize(
            title="T", genre="drama", premise="p",
            characters={"Alice": {"role": "main"}},
        )
        record = self.sm.register_entity("Bob", entity_type="character", chapter=1)
        self.assertEqual(record["canonical_name"], "Bob")
        self.assertEqual(record["type"], "character")
        self.assertEqual(record["status"], "active")
        entities = self.sm.get_entities()
        self.assertIn(_canonical_character_key("Bob"), entities)

    def test_register_is_idempotent(self):
        self.sm.initialize(title="T", genre="drama", premise="p")
        first = self.sm.register_entity("Bob", aliases=["Bobby"])
        second = self.sm.register_entity("Bob", aliases=["Bobster"])
        self.assertEqual(first["canonical_name"], second["canonical_name"])
        aliases = self.sm.get_entity_status_typed("Bob").get("aliases", [])
        self.assertIn("Bobby", aliases)
        self.assertIn("Bobster", aliases)

    def test_resolve_entity_alias(self):
        self.sm.initialize(title="T", genre="drama", premise="p")
        self.sm.register_entity("Elizabeth", aliases=["Liz", "Lizzy"])
        self.assertEqual(self.sm.resolve_entity_name("Liz"), "Elizabeth")
        self.assertEqual(self.sm.resolve_entity_name("Elizabeth"), "Elizabeth")
        self.assertIsNone(self.sm.resolve_entity_name("Nobody"))

    def test_get_entity_status(self):
        self.sm.initialize(
            title="T", genre="drama", premise="p",
            characters={"Alice": {"role": "main"}},
        )
        self.assertEqual(self.sm.get_entity_status("Alice"), "active")
        self.assertIsNone(self.sm.get_entity_status("UnknownPerson"))

    def test_update_entity_status_syncs_character(self):
        self.sm.initialize(
            title="T", genre="drama", premise="p",
            characters={"Alice": {"role": "main"}},
        )
        self.sm.update_entity_status("Alice", "dead")
        self.assertEqual(self.sm.get_entity_status("Alice"), "dead")
        char = self.sm.get_character("Alice")
        self.assertEqual(char["status"], "dead")

    def test_backfill_migrates_legacy_state(self):
        legacy = {
            "metadata": {"title": "T", "genre": "drama", "premise": "p",
                         "current_chapter": 1, "state_version": 3},
            "characters": {
                "Alice": {"role": "main", "status": "active", "aliases": ["Al"]},
                "Bob": {"role": "supporting", "status": "dead"},
            },
            "world": {
                "locations": {"The Castle": {"description": "big"}},
                "rules": [], "timeline": [],
            },
            "plot": {"major_events": [], "unresolved_threads": [],
                     "foreshadowing": [], "chapter_summaries": [],
                     "story_events": [], "legend_memory": []},
            "transitions": [],
        }
        with open(os.path.join(self.dir, "state.json"), "w") as f:
            json.dump(legacy, f)

        sm = StateManager(self.dir)
        loaded = sm.load()
        entities = loaded.get("entities", {})
        self.assertIn(_canonical_character_key("Alice"), entities)
        self.assertIn(_canonical_character_key("Bob"), entities)
        bob = entities[_canonical_character_key("Bob")]
        self.assertEqual(bob["status"], "dead")
        castle = entities[_canonical_character_key("The Castle")]
        self.assertEqual(castle["type"], "location")

    def test_backfill_preserves_aliases(self):
        legacy = {
            "metadata": {"title": "T", "genre": "drama", "premise": "p",
                         "current_chapter": 1, "state_version": 3},
            "characters": {
                "Elizabeth": {"role": "main", "status": "active", "aliases": ["Liz", "Lizzy"]},
            },
            "world": {"locations": {}, "rules": [], "timeline": []},
            "plot": {"major_events": [], "unresolved_threads": [],
                     "foreshadowing": [], "chapter_summaries": [],
                     "story_events": [], "legend_memory": []},
            "transitions": [],
        }
        with open(os.path.join(self.dir, "state.json"), "w") as f:
            json.dump(legacy, f)

        sm = StateManager(self.dir)
        entities = sm.load().get("entities", {})
        elizabeth = entities[_canonical_character_key("Elizabeth")]
        self.assertIn("Liz", elizabeth["aliases"])
        self.assertEqual(sm.resolve_entity_name("Lizzy"), "Elizabeth")


class TestEntityRegistryNaming(unittest.TestCase):

    def test_canonical_key(self):
        self.assertEqual(_canonical_character_key("Alice Smith"), "alicesmith")
        self.assertEqual(_canonical_character_key("Raj (Bus Driver)"), "raj")
        self.assertEqual(_canonical_character_key("Liz"), "elizabeth")


if __name__ == "__main__":
    unittest.main()
