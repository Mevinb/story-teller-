"""Offline tests for the Phase C UX feature set.

Covers (all mocked — no network, no models):
- Manual scene regeneration prep: brief recovery, last-scene-only rule,
  typed-scene guard, session bookkeeping after removal
- Branch options generation: JSON parsing, prompt grounding, error paths
- Continuity report: threads/seeds/characters/motifs, aging warnings
- Flask routes: continuity, branch-options, export download safety,
  manual scene regeneration validation
"""

import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ──────────────────────────────────────────────────────────────────────
# Manual scene regeneration prep
# ──────────────────────────────────────────────────────────────────────

def _make_session():
    return {
        "chapter_num": 3,
        "chapter_title": "The Long Night",
        "character_names": ["Alice", "Bob"],
        "pacing": "moderate",
        "scene_counter": 2,
        "completed_scenes": [
            "First scene text with tension.",
            "Second scene text with a reveal.",
        ],
        "completed_scene_plans": [
            {"type": "rising", "summary": "Alice finds the letter"},
            {"type": "peak", "summary": "Bob confronts the intruder"},
        ],
        "previous_ending": "the reveal",
    }


def _make_orch(session):
    from pipeline.orchestrator import PipelineOrchestrator

    orch = object.__new__(PipelineOrchestrator)
    orch._log = lambda *a, **kw: None
    orch._cancelled = False
    orch._scene_cancelled = False
    # delete_manual_scene dependencies
    orch._extract_ending = lambda text, max_chars=400: text[-50:]
    orch._previous_chapter_ending = lambda ch: "previous chapter ending"
    orch._build_manual_chapter_plan_stub = (
        lambda num, title, plans, chars, pacing: {"scenes": []}
    )
    orch._save_wip = lambda *a, **kw: None
    orch._sanitize_generated_text = lambda text: text
    base = tempfile.mkdtemp()
    orch.project_dir = base
    orch.chapters_dir = os.path.join(base, "chapters")
    os.makedirs(orch.chapters_dir, exist_ok=True)
    orch._session = session
    return orch


class TestPrepareManualSceneRegeneration(unittest.TestCase):

    def test_recovers_brief_from_plan_and_removes_scene(self):
        session = _make_session()
        orch = _make_orch(session)

        brief = orch.prepare_manual_scene_regeneration(session, 1)

        self.assertEqual(brief, "Bob confronts the intruder")
        self.assertEqual(len(session["completed_scenes"]), 1)
        self.assertEqual(session["scene_counter"], 1)
        # previous_ending recalculated from remaining scene
        self.assertIn("First scene text", session["previous_ending"])

    def test_explicit_brief_overrides_plan_summary(self):
        session = _make_session()
        orch = _make_orch(session)

        brief = orch.prepare_manual_scene_regeneration(
            session, 1, scene_brief="A quiet ritual before the storm"
        )

        self.assertEqual(brief, "A quiet ritual before the storm")
        self.assertEqual(len(session["completed_scenes"]), 1)

    def test_middle_scene_rejected(self):
        session = _make_session()
        session["completed_scenes"].append("Third scene.")
        session["completed_scene_plans"].append({"type": "falling", "summary": "x"})
        orch = _make_orch(session)

        with self.assertRaises(ValueError):
            orch.prepare_manual_scene_regeneration(session, 1)

    def test_index_out_of_range(self):
        session = _make_session()
        orch = _make_orch(session)

        with self.assertRaises(IndexError):
            orch.prepare_manual_scene_regeneration(session, 5)

    def test_typed_scene_requires_explicit_brief(self):
        session = _make_session()
        session["completed_scene_plans"][1] = {
            "type": "custom", "summary": "[User-typed scene] blah",
        }
        orch = _make_orch(session)

        with self.assertRaises(ValueError):
            orch.prepare_manual_scene_regeneration(session, 1)

        brief = orch.prepare_manual_scene_regeneration(
            session, 1, scene_brief="Rewrite as a storm chase"
        )
        self.assertEqual(brief, "Rewrite as a storm chase")

    def test_missing_plan_summary_requires_brief(self):
        session = _make_session()
        session["completed_scene_plans"][1] = {"type": "rising"}
        orch = _make_orch(session)

        with self.assertRaises(ValueError):
            orch.prepare_manual_scene_regeneration(session, 1)


# ──────────────────────────────────────────────────────────────────────
# Branch options
# ──────────────────────────────────────────────────────────────────────

class TestGenerateBranchOptions(unittest.TestCase):

    def _make_orch(self, response_json):
        from pipeline.orchestrator import PipelineOrchestrator

        orch = object.__new__(PipelineOrchestrator)
        orch._log = lambda *a, **kw: None
        orch._cancelled = False
        orch._scene_cancelled = False

        orch.state_manager = MagicMock()
        orch.state_manager.load.return_value = {
            "metadata": {
                "premise": "A city under eternal rain.",
                "current_chapter": 4,
            },
            "plot": {
                "chapter_summaries": [
                    {"chapter": 1, "summary": "Rain begins."},
                    {"chapter": 2, "summary": "The flood."},
                ],
                "unresolved_threads": ["Who controls the rain?"],
                "foreshadowing": [
                    {"chapter": 2, "text": "The umbrella salesman", "status": "planted"},
                    {"chapter": 1, "text": "The resolved locket", "status": "paid"},
                ],
            },
        }

        model = MagicMock()
        response = MagicMock()
        response.as_json.return_value = response_json
        model.generate_with_retry.return_value = response
        orch._active_planning_model = lambda: model

        base = tempfile.mkdtemp()
        orch.chapters_dir = os.path.join(base, "chapters")
        os.makedirs(orch.chapters_dir, exist_ok=True)
        with open(os.path.join(orch.chapters_dir, "chapter_002.md"), "w") as f:
            f.write("# Chapter 2\n\nThe flood swallowed the square.")

        captured = {}

        def capture_retry(**kwargs):
            captured.update(kwargs)
            return response

        model.generate_with_retry.side_effect = capture_retry
        orch._captured = captured
        return orch

    def test_parses_options_and_grounded_prompt(self):
        payload = {
            "options": [
                {
                    "title": "The Umbrella Man",
                    "premise": "The salesman reveals his role.",
                    "new_threads": "Is he friend or foe?",
                    "risk": "May deflate the mystery early.",
                },
                {
                    "title": "Flood Exodus",
                    "premise": "The city evacuates.",
                },
                {"title": "", "premise": ""},  # junk row filtered by title default
            ]
        }
        orch = self._make_orch(payload)

        result = orch.generate_branch_options(count=3)

        self.assertEqual(result["status"], "ok")
        titles = [o["title"] for o in result["options"]]
        self.assertIn("The Umbrella Man", titles)
        self.assertIn("Flood Exodus", titles)
        self.assertEqual(len(result["options"]), 3)  # junk row kept as Untitled
        self.assertEqual(result["options"][2]["title"], "Untitled branch")

        prompt = orch._captured["prompt"]
        self.assertIn("Who controls the rain?", prompt)
        self.assertIn("The umbrella salesman", prompt)
        self.assertNotIn("The resolved locket", prompt)  # paid seeds are excluded
        self.assertIn("The flood swallowed the square.", prompt)
        self.assertIn("eternal rain", prompt)

    def test_empty_response_is_error(self):
        orch = self._make_orch({"options": []})
        result = orch.generate_branch_options(count=3)
        self.assertEqual(result["status"], "error")

    def test_none_response_is_error(self):
        orch = self._make_orch(None)
        result = orch.generate_branch_options(count=3)
        self.assertEqual(result["status"], "error")

    def test_direction_hint_included_in_prompt(self):
        orch = self._make_orch({"options": [{"title": "t", "premise": "p"}]})
        orch.generate_branch_options(count=2, direction_hint="Lean into horror")
        self.assertIn("Lean into horror", orch._captured["prompt"])

    def test_count_clamped(self):
        orch = self._make_orch({"options": [{"title": "t", "premise": "p"}]})
        orch.generate_branch_options(count=99)
        self.assertIn("5 DISTINCT", orch._captured["prompt"])


# ──────────────────────────────────────────────────────────────────────
# Continuity report
# ──────────────────────────────────────────────────────────────────────

class TestContinuityReport(unittest.TestCase):

    def _make_orch(self, state):
        from pipeline.orchestrator import PipelineOrchestrator

        orch = object.__new__(PipelineOrchestrator)
        orch._log = lambda *a, **kw: None
        orch.state_manager = MagicMock()
        orch.state_manager.load.return_value = state
        return orch

    def _state(self):
        return {
            "metadata": {
                "current_chapter": 5,
                "narrative_phase": "rising_action",
            },
            "plot": {
                "chapter_summaries": [{"chapter": i} for i in range(1, 5)],
                "unresolved_threads": ["The missing heir", "The debt"],
                "foreshadowing": [
                    {"chapter": 2, "text": "A locked drawer", "status": "planted"},
                    {"chapter": 4, "text": "A fresh seed", "status": "planted"},
                    {"chapter": 1, "text": "Paid off", "status": "paid"},
                ],
            },
            "characters": {
                "Alice": {
                    "role": "protagonist",
                    "status": "active",
                    "state": {"emotion": "determined", "goal": "find the heir", "location": "library"},
                },
                "Bob": {
                    "role": "supporting",
                    "status": "missing",
                    "state": {},
                },
            },
        }

    def test_report_structure(self):
        orch = self._make_orch(self._state())
        report = orch.continuity_report()

        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["current_chapter"], 5)
        self.assertEqual(report["narrative_phase"], "rising_action")
        self.assertEqual(report["chapters_written"], 4)
        self.assertEqual(report["unresolved_threads"], ["The missing heir", "The debt"])
        seed_texts = [s["text"] for s in report["planted_seeds"]]
        self.assertIn("A locked drawer", seed_texts)
        self.assertNotIn("Paid off", seed_texts)
        chars = {c["name"]: c for c in report["characters"]}
        self.assertEqual(chars["Alice"]["emotion"], "determined")
        self.assertEqual(chars["Bob"]["status"], "missing")

    def test_aged_seed_warning(self):
        state = self._state()
        orch = self._make_orch(state)
        report = orch.continuity_report()

        aged = [w for w in report["warnings"] if "locked drawer" in w]
        self.assertTrue(aged, f"expected aging warning, got: {report['warnings']}")
        self.assertIn("3 chapters", aged[0])

    def test_fresh_seed_no_warning(self):
        state = self._state()
        state["plot"]["foreshadowing"] = [
            {"chapter": 4, "text": "A fresh seed", "status": "planted"},
        ]
        orch = self._make_orch(state)
        report = orch.continuity_report()

        aged = [w for w in report["warnings"] if "fresh seed" in w]
        self.assertFalse(aged)

    def test_non_active_character_warning(self):
        orch = self._make_orch(self._state())
        report = orch.continuity_report()
        self.assertTrue(any("Bob" in w and "missing" in w for w in report["warnings"]))

    def test_motifs_surfaced(self):
        state = self._state()
        state["world"] = {
            "motifs": {
                "rain": {"mentions": 4},
                "mirror": {"mentions": 1},  # below min_mentions=2
            }
        }
        orch = self._make_orch(state)
        report = orch.continuity_report()

        names = [m["name"] for m in report["active_motifs"]]
        self.assertIn("Rain", names)
        self.assertNotIn("Mirror", names)


# ──────────────────────────────────────────────────────────────────────
# Flask routes
# ──────────────────────────────────────────────────────────────────────

class _FakePipeline:
    """Stands in for PipelineOrchestrator in route tests."""

    def __init__(self, name, **kwargs):
        self.name = name
        self.loaded = False

    def load_project(self):
        self.loaded = True

    def continuity_report(self):
        return {"status": "ok", "current_chapter": 2, "warnings": []}

    def generate_branch_options(self, count=3, direction_hint=""):
        if direction_hint == "boom":
            return {"status": "error", "error": "model exploded"}
        return {"status": "ok", "options": [{"title": "T", "premise": "P"}]}

    def export_story(self, fmt="md", output_dir=None):
        return {
            "status": "ok",
            "format": fmt,
            "files": {fmt: os.path.join("/tmp", f"story.{fmt}")},
        }


class TestFlaskRoutes(unittest.TestCase):

    def setUp(self):
        import app as app_module

        self.app_module = app_module
        self.client = app_module.app.test_client()

    def tearDown(self):
        self.app_module._manual_sessions.pop("proj", None)
        self.app_module._active_pipelines.pop("proj", None)

    def test_continuity_route(self):
        with patch.object(self.app_module, "PipelineOrchestrator", _FakePipeline):
            res = self.client.get("/api/project/proj/continuity")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["current_chapter"], 2)

    def test_branch_options_route_ok(self):
        with patch.object(self.app_module, "PipelineOrchestrator", _FakePipeline):
            res = self.client.post(
                "/api/project/proj/branch-options",
                json={"count": 3, "direction_hint": "go west"},
            )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["options"][0]["title"], "T")

    def test_branch_options_route_error_maps_to_502(self):
        with patch.object(self.app_module, "PipelineOrchestrator", _FakePipeline):
            res = self.client.post(
                "/api/project/proj/branch-options",
                json={"direction_hint": "boom"},
            )
        self.assertEqual(res.status_code, 502)
        self.assertEqual(res.get_json()["error"], "model exploded")

    def test_export_story_route_returns_files_map(self):
        with patch.object(self.app_module, "PipelineOrchestrator", _FakePipeline):
            res = self.client.post(
                "/api/project/proj/export-story",
                json={"format": "epub"},
            )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["format"], "epub")
        self.assertIn("story.epub", data["files"]["epub"])

    def test_export_download_rejects_traversal(self):
        res = self.client.get(
            "/api/project/proj/export/download?file=..%2F..%2Fconfig.py"
        )
        self.assertEqual(res.status_code, 400)

    def test_export_download_rejects_bad_extension(self):
        res = self.client.get(
            "/api/project/proj/export/download?file=story.exe"
        )
        self.assertEqual(res.status_code, 400)

    def test_export_download_missing_file_404(self):
        res = self.client.get(
            "/api/project/proj/export/download?file=nope.md"
        )
        self.assertEqual(res.status_code, 404)

    def test_export_download_serves_file(self):
        import config

        project_dir = os.path.join(config.PROJECTS_DIR, "proj")
        os.makedirs(project_dir, exist_ok=True)
        target = os.path.join(project_dir, "story.md")
        try:
            with open(target, "w", encoding="utf-8") as f:
                f.write("# Story\n\nOnce upon a time.")
            res = self.client.get(
                "/api/project/proj/export/download?file=story.md"
            )
            self.assertEqual(res.status_code, 200)
            self.assertIn("Once upon a time", res.get_data(as_text=True))
            self.assertEqual(res.mimetype, "text/markdown")
        finally:
            if os.path.exists(target):
                os.remove(target)
            try:
                os.rmdir(project_dir)
            except OSError:
                pass

    def test_regenerate_requires_active_session(self):
        res = self.client.post(
            "/api/project/proj/generate/manual/scene/0/regenerate",
            json={},
        )
        self.assertEqual(res.status_code, 404)

    def test_regenerate_validation_error(self):
        fake = _FakePipeline("proj")
        fake.prepare_manual_scene_regeneration = (
            lambda session, index, scene_brief="": (_ for _ in ()).throw(
                ValueError("Only the most recent scene can be regenerated.")
            )
        )
        self.app_module._manual_sessions["proj"] = {
            "pipeline": fake,
            "session": {"scene_counter": 0, "completed_scenes": []},
            "eq": None,
        }
        try:
            res = self.client.post(
                "/api/project/proj/generate/manual/scene/0/regenerate",
                json={},
            )
        finally:
            self.app_module._manual_sessions.pop("proj", None)
        self.assertEqual(res.status_code, 400)
        self.assertIn("most recent", res.get_json()["error"])


if __name__ == "__main__":
    unittest.main()
