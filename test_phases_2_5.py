"""
Comprehensive test suite for Phases 2-5 improvements.
Run: python3 -m pytest test_phases_2_5.py -v --tb=short
"""
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Ensure project root on path
sys.path.insert(0, os.path.dirname(__file__))


# ─── Phase 2: Agent Registry ──────────────────────────────────────────────────

class TestAgentRegistry(unittest.TestCase):

    def test_register_and_get(self):
        from pipeline.agent_registry import AgentRegistry
        reg = AgentRegistry()
        mock_agent = MagicMock()
        mock_agent.run = lambda inputs: {"output": "ok"}
        reg.register("writer", mock_agent)
        reg._built = True  # mark built so get() works
        self.assertIs(reg.get("writer"), mock_agent)

    def test_get_before_build_raises(self):
        from pipeline.agent_registry import AgentRegistry
        reg = AgentRegistry()
        with self.assertRaises(RuntimeError):
            reg.get("architect")

    def test_get_unknown_raises(self):
        from pipeline.agent_registry import AgentRegistry
        reg = AgentRegistry()
        reg._built = True
        with self.assertRaises(KeyError):
            reg.get("nonexistent")

    def test_compact_mode_applied_to_capable_agents(self):
        from pipeline.agent_registry import AgentRegistry
        reg = AgentRegistry()
        mock_writer = MagicMock()
        mock_writer.set_compact_mode = MagicMock()
        reg._agents["writer"] = mock_writer
        reg._built = True
        reg.set_compact_mode(True)
        mock_writer.set_compact_mode.assert_called_once_with(True)

    def test_compact_mode_false(self):
        from pipeline.agent_registry import AgentRegistry
        reg = AgentRegistry()
        mock_agent = MagicMock()
        mock_agent.set_compact_mode = MagicMock()
        reg._agents["architect"] = mock_agent
        reg._built = True
        reg.set_compact_mode(False)
        mock_agent.set_compact_mode.assert_called_once_with(False)

    def test_describe_returns_class_names(self):
        from pipeline.agent_registry import AgentRegistry
        reg = AgentRegistry()
        mock = MagicMock(spec_set=["run"])
        reg.register("writer", mock)
        reg._built = True
        desc = reg.describe()
        self.assertIn("writer", desc)
        self.assertIsInstance(desc["writer"], str)

    def test_all_agents_returns_copy(self):
        from pipeline.agent_registry import AgentRegistry
        reg = AgentRegistry()
        m = MagicMock()
        reg.register("pacing", m)
        reg._built = True
        agents = reg.all_agents()
        agents["pacing"] = None  # mutate copy
        self.assertIs(reg.get("pacing"), m)  # original unchanged

    def test_register_replaces_existing(self):
        from pipeline.agent_registry import AgentRegistry
        reg = AgentRegistry()
        reg._built = True
        m1 = MagicMock()
        m2 = MagicMock()
        reg.register("writer", m1)
        reg.register("writer", m2)
        self.assertIs(reg.get("writer"), m2)

    def test_agent_protocol_satisfied(self):
        from pipeline.agent_registry import Agent
        class MyAgent:
            def run(self, inputs: dict) -> dict:
                return {}
        self.assertIsInstance(MyAgent(), Agent)

    def test_repr_shows_status(self):
        from pipeline.agent_registry import AgentRegistry
        reg = AgentRegistry()
        self.assertIn("empty", repr(reg))
        reg._built = True
        self.assertIn("built", repr(reg))


# ─── Phase 2/3: Quality Controller ───────────────────────────────────────────

class TestQualityControllerValidation(unittest.TestCase):

    def setUp(self):
        from pipeline.quality_controller import QualityController
        self.qc = QualityController()

    def _scene(self, **kw):
        return {"scene_number": 1, "word_target": 600, "type": "build_tension", **kw}

    def test_valid_scene_passes(self):
        text = ("The sun set slowly over the mountains. " * 50).strip() + " She smiled."
        r = self.qc.validate(text, self._scene(), chapter_num=1)
        self.assertTrue(r.ok)
        self.assertEqual(r.truncated, False)

    def test_short_scene_fails(self):
        text = "Short."
        r = self.qc.validate(text, self._scene(), chapter_num=1)
        self.assertFalse(r.ok)
        self.assertFalse(r.truncated)

    def test_truncated_scene_flagged(self):
        text = ("Word " * 300) + "she walked into the"
        r = self.qc.validate(text, self._scene(), chapter_num=2)
        self.assertFalse(r.ok)
        self.assertTrue(r.truncated)

    def test_ellipsis_not_truncated(self):
        # Need enough words (> minimum ~270) to pass word-count check
        text = ("Word " * 300).strip() + "\u2026"
        r = self.qc.validate(text, self._scene(), chapter_num=1)
        self.assertTrue(r.ok, f"Should pass but got: {r.reason}")

    def test_quoted_ending_not_truncated(self):
        # 50 * 4 words = 200 < 270, need 300+
        text = ("He said something important once. " * 80) + '"Goodbye."'
        r = self.qc.validate(text, self._scene(), chapter_num=1)
        self.assertTrue(r.ok, f"Should pass but got: {r.reason}")

    def test_markdown_bold_ending_not_truncated(self):
        """Scenes ending with **emphasis** should not be rejected."""
        text = ("She moved forward through the corridor. " * 50) + "The door was **locked**."
        r = self.qc.validate(text, self._scene(), chapter_num=1)
        self.assertTrue(r.ok, f"Should pass but got: {r.reason}")

    def test_word_count_returned(self):
        text = "The cat sat on the mat. " * 40
        r = self.qc.validate(text, self._scene(), chapter_num=1)
        self.assertGreater(r.words, 0)

    def test_minimum_reflects_target(self):
        # target=200 → minimum = max(120, min(500, 90)) = 120
        scene = self._scene(word_target=200)
        text = ("Word " * 130).strip() + "."
        r = self.qc.validate(text, scene, chapter_num=1)
        self.assertTrue(r.ok)


class TestQualityControllerScoring(unittest.TestCase):

    def setUp(self):
        from pipeline.quality_controller import QualityController
        self.qc = QualityController()

    def _good_text(self, n=80):
        return ("She walked through the corridor. He followed closely behind. "
                "The tension was palpable as they entered the room. ") * n

    def test_score_returns_all_dimensions(self):
        text = self._good_text()
        result = self.qc.score(text, {"scene_number": 1, "word_target": 500, "type": "build_tension"})
        for dim in ("coherence", "pacing", "voice", "temporal", "overall", "passes", "word_count", "issues"):
            self.assertIn(dim, result.as_dict())

    def test_overall_in_range(self):
        text = self._good_text()
        result = self.qc.score(text, {"word_target": 500, "type": "build_tension"})
        self.assertGreaterEqual(result.overall, 0.0)
        self.assertLessEqual(result.overall, 1.0)

    def test_meta_output_lowers_coherence(self):
        text = "Thinking process: I will analyze the request. Step by step analysis. " + self._good_text(5)
        result = self.qc.score(text, {"word_target": 100, "type": "build_tension"})
        self.assertLess(result.coherence, 0.9)
        self.assertTrue(any("Meta" in i for i in result.issues))

    def test_short_scene_lowers_pacing(self):
        text = "She sat down. He left." * 3  # very short
        result = self.qc.score(text, {"word_target": 600, "type": "build_tension"})
        self.assertLess(result.pacing, 0.8)

    def test_location_jump_without_transition_lowers_temporal(self):
        # Text does NOT contain the target location name or any transition marker
        text = ("She sat quietly thinking about what had happened. "
                "The memories flooded back one by one. ") * 10
        scene = {
            "word_target": 300,
            "type": "build_tension",
            "prev_location": "The Kitchen",
            "location": "Distant Mountain Peak",  # unique name, not in text
        }
        result = self.qc.score(text, scene)
        self.assertLess(result.temporal, 1.0)

    def test_pov_inconsistency_lowers_voice(self):
        # Mix lots of "I" with lots of "she/he"
        text = ("I walked in. She followed me. He saw me. I turned around. "
                "She called out to me. He nodded at me. ") * 20
        result = self.qc.score(text, {"word_target": 300, "type": "build_tension"})
        self.assertLess(result.voice, 1.0)

    def test_summarise_returns_string(self):
        text = self._good_text()
        result = self.qc.score(text, {"word_target": 400, "type": "build_tension"})
        summary = self.qc.summarise(result)
        self.assertIsInstance(summary, str)
        self.assertIn("Quality", summary)

    def test_passes_threshold_default(self):
        from pipeline.quality_controller import QualityController, ScoreResult
        sc = ScoreResult(coherence=0.8, pacing=0.8, voice=0.8, temporal=0.8)
        self.assertTrue(sc.passes)

    def test_fails_threshold_low_scores(self):
        from pipeline.quality_controller import ScoreResult
        sc = ScoreResult(coherence=0.1, pacing=0.1, voice=0.1, temporal=0.1)
        self.assertFalse(sc.passes)

    def test_score_result_as_dict_complete(self):
        from pipeline.quality_controller import ScoreResult
        sc = ScoreResult(word_count=500, issues=["test issue"])
        d = sc.as_dict()
        self.assertEqual(d["word_count"], 500)
        self.assertIn("test issue", d["issues"])


# ─── Phase 5: World Bible Generator ──────────────────────────────────────────

class TestWorldBible(unittest.TestCase):

    def setUp(self):
        self.project_dir = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.project_dir, "chapters"), exist_ok=True)
        self._write_state()

    def tearDown(self):
        shutil.rmtree(self.project_dir, ignore_errors=True)

    def _write_state(self, title="Test Story", genre="fantasy"):
        state = {
            "metadata": {"title": title, "genre": genre},
            "characters": {
                "Alice": {
                    "description": "Hero",
                    "relationships": {"Bob": "best friend"},  # dict format
                },
                "Bob": {"description": "Mentor"},
            },
        }
        with open(os.path.join(self.project_dir, "state.json"), "w") as f:
            json.dump(state, f)

    def _write_chapter(self, num: int, text: str):
        path = os.path.join(self.project_dir, "chapters", f"chapter_{num:03d}.md")
        with open(path, "w") as f:
            f.write(text)

    def test_generates_empty_bible_no_chapters(self):
        from pipeline.world_bible import WorldBibleGenerator
        gen = WorldBibleGenerator(self.project_dir)
        bible = gen.generate()
        self.assertEqual(bible.title, "Test Story")
        self.assertEqual(bible.genre, "fantasy")
        self.assertIsInstance(bible.locations, list)

    def test_saves_markdown_and_json(self):
        from pipeline.world_bible import WorldBibleGenerator
        gen = WorldBibleGenerator(self.project_dir)
        gen.generate()
        self.assertTrue(os.path.exists(os.path.join(self.project_dir, "world_bible.md")))
        self.assertTrue(os.path.exists(os.path.join(self.project_dir, "world_bible.json")))

    def test_regex_extraction_finds_locations(self):
        self._write_chapter(1, "Alice arrived at the Grand Hotel. She entered the hospital lobby.")
        from pipeline.world_bible import WorldBibleGenerator
        gen = WorldBibleGenerator(self.project_dir)
        bible = gen.generate()
        names = [e.name.lower() for e in bible.locations]
        self.assertTrue(any("hotel" in n or "hospital" in n for n in names))

    def test_character_relationships_extracted(self):
        # Relationships stored as dict {other_char: description}
        self._write_chapter(1, "Alice and Bob met.")
        from pipeline.world_bible import WorldBibleGenerator
        gen = WorldBibleGenerator(self.project_dir)
        bible = gen.generate()
        # Alice has relationships: {"Bob": "best friend"}
        self.assertIn("Alice", bible.relationships)
        self.assertTrue(len(bible.relationships["Alice"]) > 0)

    def test_deduplication(self):
        from pipeline.world_bible import WorldBibleGenerator, WorldEntry
        gen = WorldBibleGenerator(self.project_dir)
        entries = [
            WorldEntry("location", "The Forest", "A dark forest.", [1]),
            WorldEntry("location", "the forest", "A mystical forest.", [2]),
        ]
        deduped = gen._deduplicate(entries)
        self.assertEqual(len(deduped), 1)
        self.assertIn(2, deduped[0].source_chapters)

    def test_to_markdown_has_title(self):
        from pipeline.world_bible import WorldBible
        bible = WorldBible(title="My Epic", genre="sci-fi")
        md = bible.to_markdown()
        self.assertIn("My Epic", md)
        self.assertIn("sci-fi", md)

    def test_to_dict_serializable(self):
        from pipeline.world_bible import WorldBible
        bible = WorldBible(title="Test", genre="drama")
        data = bible.to_dict()
        # Should be JSON-serializable
        json.dumps(data)

    def test_max_chapters_limit(self):
        self._write_chapter(1, "At the castle, the king spoke.")
        self._write_chapter(2, "Inside the dungeon, the prisoner waited.")
        from pipeline.world_bible import WorldBibleGenerator
        gen = WorldBibleGenerator(self.project_dir)
        bible = gen.generate(max_chapters=1)
        # Chapter 2 should not be included
        all_names = [e.name.lower() for e in bible.all_entries()]
        self.assertFalse(any("dungeon" in n for n in all_names))

    def test_progress_callback_called(self):
        from pipeline.world_bible import WorldBibleGenerator
        calls = []
        gen = WorldBibleGenerator(self.project_dir, progress_callback=calls.append)
        gen.generate()
        self.assertGreater(len(calls), 0)


# ─── Phase 5: Story Exporter ─────────────────────────────────────────────────

class TestStoryExporter(unittest.TestCase):

    def setUp(self):
        self.project_dir = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.project_dir, "chapters"), exist_ok=True)
        self._write_state()
        self._write_chapters()

    def tearDown(self):
        shutil.rmtree(self.project_dir, ignore_errors=True)

    def _write_state(self):
        state = {"metadata": {"title": "Epic Quest", "genre": "Fantasy"}}
        with open(os.path.join(self.project_dir, "state.json"), "w") as f:
            json.dump(state, f)

    def _write_chapters(self):
        for i in range(1, 4):
            path = os.path.join(self.project_dir, "chapters", f"chapter_{i:03d}.md")
            with open(path, "w") as f:
                f.write(f"# Chapter {i}: The Journey Continues\n\nShe walked forward boldly. " * 10)

    def test_markdown_export_creates_file(self):
        from pipeline.exporter import StoryExporter
        exp = StoryExporter(self.project_dir)
        path = exp.export_markdown()
        self.assertTrue(os.path.exists(path))
        self.assertTrue(path.endswith(".md"))

    def test_markdown_contains_all_chapters(self):
        from pipeline.exporter import StoryExporter
        exp = StoryExporter(self.project_dir)
        path = exp.export_markdown()
        with open(path) as f:
            content = f.read()
        self.assertIn("Chapter 1", content)
        self.assertIn("Chapter 2", content)
        self.assertIn("Chapter 3", content)

    def test_markdown_has_title(self):
        from pipeline.exporter import StoryExporter
        exp = StoryExporter(self.project_dir)
        path = exp.export_markdown()
        with open(path) as f:
            content = f.read()
        self.assertIn("Epic Quest", content)

    def test_txt_export_creates_file(self):
        from pipeline.exporter import StoryExporter
        exp = StoryExporter(self.project_dir)
        path = exp.export_txt()
        self.assertTrue(os.path.exists(path))
        self.assertTrue(path.endswith(".txt"))

    def test_txt_strips_markdown_headings(self):
        from pipeline.exporter import StoryExporter
        path = StoryExporter(self.project_dir).export_txt()
        with open(path) as f:
            content = f.read()
        # The headings from files that start with '# Chapter N:' should be stripped.
        # We check that our chapter text blocks start with the plain title,
        # not with the '# ' prefix on a standalone line.
        lines = content.splitlines()
        heading_lines_with_hash = [l for l in lines if l.startswith("# ")]
        self.assertEqual(heading_lines_with_hash, [], f"Found un-stripped headings: {heading_lines_with_hash[:3]}")

    def test_strip_markdown_removes_formatting(self):
        from pipeline.exporter import StoryExporter
        text = "# Heading\n**bold** and _italic_ text.\n* * *\n[link](http://x.com)"
        result = StoryExporter._strip_markdown(text)
        self.assertNotIn("#", result)
        self.assertNotIn("**", result)
        self.assertNotIn("_italic_", result)
        self.assertNotIn("[link]", result)
        self.assertIn("bold", result)
        self.assertIn("italic", result)

    def test_safe_filename_removes_special_chars(self):
        from pipeline.exporter import StoryExporter
        exp = StoryExporter(self.project_dir)
        name = exp._safe_filename("My Epic: A Quest!")
        self.assertNotIn(":", name)
        self.assertNotIn("!", name)

    def test_custom_output_dir(self):
        from pipeline.exporter import StoryExporter
        out_dir = tempfile.mkdtemp()
        try:
            exp = StoryExporter(self.project_dir, output_dir=out_dir)
            path = exp.export_markdown()
            self.assertTrue(path.startswith(out_dir))
        finally:
            shutil.rmtree(out_dir)

    def test_export_all_returns_dict(self):
        from pipeline.exporter import StoryExporter
        exp = StoryExporter(self.project_dir)
        results = exp.export_all(include_epub=False)
        self.assertIn("markdown", results)
        self.assertIn("txt", results)

    def test_md_to_epub_html_extracts_title(self):
        from pipeline.exporter import StoryExporter
        text = "# Chapter 1: The Beginning\n\nShe walked in.\n\nHe followed."
        title, html = StoryExporter._md_to_epub_html(text)
        self.assertEqual(title, "Chapter 1: The Beginning")
        self.assertIn("<p>", html)

    def test_epub_available_returns_bool(self):
        from pipeline.exporter import epub_available
        self.assertIsInstance(epub_available(), bool)


# ─── Phase 4: CLI ─────────────────────────────────────────────────────────────

class TestCLI(unittest.TestCase):

    def setUp(self):
        self.project_dir = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.project_dir, "chapters"), exist_ok=True)
        self._write_state()
        import config
        self._orig_projects_dir = config.PROJECTS_DIR
        self._project_name = os.path.basename(self.project_dir)
        config.PROJECTS_DIR = os.path.dirname(self.project_dir)

    def tearDown(self):
        import config
        config.PROJECTS_DIR = self._orig_projects_dir
        shutil.rmtree(self.project_dir, ignore_errors=True)

    def _write_state(self, missing_fields=False):
        state = {
            "metadata": {
                "title": "Test CLI Story",
                "genre": "drama",
                "current_chapter": 0,
                "total_scenes_written": 0,
                "premise": "- Hero rises\n- Final battle",
                "premise_step_completed": -1,
            },
            "characters": {"Alice": {"description": "Hero"}},
            "plot": {"chapter_summaries": [], "major_events": []},
        }
        if missing_fields:
            del state["plot"]
        with open(os.path.join(self.project_dir, "state.json"), "w") as f:
            json.dump(state, f)

    def _run(self, argv):
        from pipeline.__main__ import main
        return main(argv)

    def test_validate_healthy_project(self):
        rc = self._run(["validate", self._project_name])
        self.assertEqual(rc, 0)

    def test_validate_missing_key_returns_error(self):
        self._write_state(missing_fields=True)
        rc = self._run(["validate", self._project_name])
        self.assertEqual(rc, 1)

    def test_migrate_adds_missing_fields(self):
        # Remove premise_step_completed to simulate old schema
        state_path = os.path.join(self.project_dir, "state.json")
        with open(state_path) as f:
            state = json.load(f)
        del state["metadata"]["premise_step_completed"]
        with open(state_path, "w") as f:
            json.dump(state, f)
        rc = self._run(["migrate", self._project_name])
        self.assertEqual(rc, 0)
        with open(state_path) as f:
            migrated = json.load(f)
        self.assertIn("premise_step_completed", migrated["metadata"])

    def test_migrate_idempotent(self):
        rc1 = self._run(["migrate", self._project_name])
        rc2 = self._run(["migrate", self._project_name])
        self.assertEqual(rc1, 0)
        self.assertEqual(rc2, 0)

    def test_stats_command(self):
        rc = self._run(["stats", self._project_name])
        self.assertEqual(rc, 0)

    def test_world_bible_command(self):
        rc = self._run(["world-bible", self._project_name])
        self.assertEqual(rc, 0)
        self.assertTrue(os.path.exists(os.path.join(self.project_dir, "world_bible.md")))
        self.assertTrue(os.path.exists(os.path.join(self.project_dir, "world_bible.json")))

    def test_export_md_command(self):
        rc = self._run(["export", self._project_name, "--format", "md"])
        self.assertEqual(rc, 0)

    def test_export_txt_command(self):
        rc = self._run(["export", self._project_name, "--format", "txt"])
        self.assertEqual(rc, 0)

    def test_no_command_returns_zero(self):
        rc = self._run([])
        self.assertEqual(rc, 0)

    def test_unknown_project_exits(self):
        with self.assertRaises(SystemExit):
            self._run(["validate", "nonexistent_project_xyz"])


# ─── Phase 4: Structured Logging ─────────────────────────────────────────────

class TestStructuredLogging(unittest.TestCase):
    """Verify the orchestrator's _log produces structured entries."""

    def _make_orch(self):
        from pipeline.orchestrator import PipelineOrchestrator
        orch = object.__new__(PipelineOrchestrator)
        orch._events = []
        orch._progress_cb = lambda event, data=None, **kw: orch._events.append((event, data))
        return orch

    def test_log_emits_log_event(self):
        orch = self._make_orch()
        orch._log("test message", level="info", details={"key": "value"})
        self.assertTrue(any(ev == "log" for ev, _ in orch._events))

    def test_log_entry_has_required_fields(self):
        orch = self._make_orch()
        orch._log("hello world", level="warn")
        entries = [d for ev, d in orch._events if ev == "log"]
        self.assertTrue(len(entries) > 0)
        entry = entries[0]
        self.assertIn("timestamp", entry)
        self.assertIn("level", entry)
        self.assertIn("message", entry)
        self.assertEqual(entry["level"], "warn")
        self.assertEqual(entry["message"], "hello world")

    def test_log_details_included_when_provided(self):
        orch = self._make_orch()
        orch._log("test", details={"chapter": 3})
        entries = [d for ev, d in orch._events if ev == "log"]
        self.assertIn("details", entries[0])
        self.assertEqual(entries[0]["details"]["chapter"], 3)

    def test_emit_fires_callback(self):
        orch = self._make_orch()
        orch._emit("chapter_start", {"chapter": 1})
        self.assertIn(("chapter_start", {"chapter": 1}), orch._events)


# ─── Integration: Full Pipeline round-trip (no LLM) ──────────────────────────

class TestOrchestratorWIPRoundTrip(unittest.TestCase):
    """End-to-end WIP save/load/clear cycle without any LLM calls."""

    def setUp(self):
        self.project_dir = tempfile.mkdtemp()
        self.chapters_dir = os.path.join(self.project_dir, "chapters")
        os.makedirs(self.chapters_dir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.project_dir, ignore_errors=True)

    def _make_orch(self):
        from pipeline.orchestrator import PipelineOrchestrator
        orch = object.__new__(PipelineOrchestrator)
        orch.chapters_dir = self.chapters_dir
        orch.logs_dir = os.path.join(self.project_dir, "logs")
        os.makedirs(orch.logs_dir, exist_ok=True)
        orch._progress_cb = lambda **kw: None
        return orch

    def test_save_load_clear_roundtrip(self):
        orch = self._make_orch()
        scenes = [{"scene": 1, "text": "Alice walked in."}, {"scene": 2, "text": "Bob ran away."}]
        orch._save_wip(7, scenes, {"chapter_title": "Test"}, [])
        loaded = orch._load_wip(7)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["chapter_num"], 7)
        self.assertEqual(len(loaded["completed_scenes"]), 2)
        orch._clear_wip(7)
        self.assertIsNone(orch._load_wip(7))

    def test_multiple_chapters_wip(self):
        orch = self._make_orch()
        for ch in (1, 2, 3):
            orch._save_wip(ch, [{"scene": 1, "text": f"Ch{ch}"}], {}, [])
        for ch in (1, 2, 3):
            data = orch._load_wip(ch)
            self.assertEqual(data["chapter_num"], ch)
        orch._clear_wip(2)
        self.assertIsNone(orch._load_wip(2))
        self.assertIsNotNone(orch._load_wip(1))
        self.assertIsNotNone(orch._load_wip(3))

    def test_wip_path_format(self):
        orch = self._make_orch()
        path = orch._wip_path(5)
        self.assertIn("wip_chapter_005", path)

    def test_no_tmp_file_after_save(self):
        orch = self._make_orch()
        orch._save_wip(1, [], {}, [])
        tmp = orch._wip_path(1) + ".tmp"
        self.assertFalse(os.path.exists(tmp))


if __name__ == "__main__":
    unittest.main(verbosity=2)
