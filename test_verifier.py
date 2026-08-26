"""
Tests for the post-generation Verifier and its wiring into the scene graph.
Run: python3 -m pytest test_verifier.py -v --tb=short
"""
import os
import shutil
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(__file__))

import config
from agents.verifier import Verifier
from memory.state_manager import StateManager


class VerifierHarness:
    """A StateManager project dir pre-populated with known entities."""

    def __init__(self):
        self.dir = tempfile.mkdtemp(prefix="st_verifier_")
        self.sm = StateManager(self.dir)
        self.sm.initialize(
            title="The Test Story",
            genre="drama",
            premise="A simple premise.",
            characters={
                "Alice": {"role": "main", "status": "active"},
                "Bob": {"role": "main", "status": "dead"},
            },
        )
        self.verifier = Verifier()

    def cleanup(self):
        shutil.rmtree(self.dir, ignore_errors=True)


class TestVerifierDeterministic(unittest.TestCase):

    def setUp(self):
        self.h = VerifierHarness()

    def tearDown(self):
        self.h.cleanup()

    def test_clean_scene_passes(self):
        report = self.h.verifier.verify(
            scene_text="Alice walked through the door and sat down calmly.",
            state_manager=self.h.sm,
            scene_plan={"scene_number": 1, "characters_present": ["Alice"]},
            chapter_num=1,
        )
        self.assertTrue(report["is_consistent"])
        self.assertEqual(report["blocking_issues"], [])

    def test_dead_character_acting_is_blocking(self):
        report = self.h.verifier.verify(
            scene_text="Bob stood up and greeted the crowd with a smile.",
            state_manager=self.h.sm,
            scene_plan={"scene_number": 1, "characters_present": ["Alice", "Bob"]},
            chapter_num=1,
        )
        self.assertFalse(report["is_consistent"])
        blocking = report["blocking_issues"]
        self.assertEqual(len(blocking), 1)
        self.assertEqual(blocking[0]["type"], "dead_character_acting")
        self.assertTrue(blocking[0]["blocking"])

    def test_missing_character_return_is_non_blocking(self):
        self.h.sm.update_entity_status("Bob", "missing")
        report = self.h.verifier.verify(
            scene_text="Bob walked back into the room, unharmed.",
            state_manager=self.h.sm,
            scene_plan={"scene_number": 1, "characters_present": ["Bob"]},
            chapter_num=1,
        )
        # 'missing' is a note, not a blocker — must NOT force a rewrite.
        self.assertTrue(report["is_consistent"])
        non_blocking = report["non_blocking_issues"]
        self.assertTrue(any(i["type"] == "dead_character_acting" for i in non_blocking))

    def test_future_premise_leak_is_blocking(self):
        def leak_fn(scene_text, chapter_num, state):
            return ["'collapsing footbridge' belongs to future premise step 3"]

        report = self.h.verifier.verify(
            scene_text="Alice crossed the collapsing footbridge in a panic.",
            state_manager=self.h.sm,
            scene_plan={"scene_number": 1, "characters_present": ["Alice"]},
            chapter_num=1,
            premise_violation_fn=leak_fn,
        )
        self.assertFalse(report["is_consistent"])
        self.assertTrue(any(i["type"] == "future_step_leak" for i in report["blocking_issues"]))

    def test_implicit_entity_is_non_blocking_note(self):
        report = self.h.verifier.verify(
            scene_text=(
                "Alice met Bimal at the market, and Bimal offered a warm welcome. "
                "Then Alice told Bimal about her long journey."
            ),
            state_manager=self.h.sm,
            scene_plan={"scene_number": 1, "characters_present": ["Alice"]},
            chapter_num=1,
        )
        self.assertTrue(report["is_consistent"])
        notes = report["non_blocking_issues"]
        self.assertTrue(any(i["type"] == "implicit_entity" and "Bimal" in i["detail"] for i in notes))

    def test_llm_pass_adds_issues(self):
        model = MagicMock()
        response = MagicMock()
        response.as_json.return_value = {
            "verified": False,
            "issues": [
                {"type": "identity_swap", "detail": "Alice called Bob by Carol's name.",
                 "severity": "high", "suggestion": "Fix the name."},
            ],
        }
        model.generate_with_retry.return_value = response
        verifier = Verifier(model)

        report = verifier.verify(
            scene_text="Alice spoke to Bob.",
            state_manager=self.h.sm,
            scene_plan={"scene_number": 1, "characters_present": ["Alice", "Bob"]},
            chapter_num=1,
            run_llm=True,
        )
        self.assertFalse(report["is_consistent"])
        self.assertTrue(any(i["type"] == "identity_swap" for i in report["blocking_issues"]))

    def test_llm_failure_is_non_fatal(self):
        model = MagicMock()
        model.generate_with_retry.side_effect = RuntimeError("boom")
        verifier = Verifier(model)
        report = verifier.verify(
            scene_text="Alice stood still.",
            state_manager=self.h.sm,
            scene_plan={"scene_number": 1, "characters_present": ["Alice"]},
            chapter_num=1,
            run_llm=True,
        )
        self.assertTrue(report["is_consistent"])

    def test_issue_signature_stable(self):
        a = {"type": "dead_character_acting", "detail": "Bob is acting but dead."}
        b = {"type": "dead_character_acting", "detail": "Bob is acting but dead!"}
        self.assertEqual(Verifier.issue_signature(a), Verifier.issue_signature(b))


class TestVerifierSceneGraphEscalation(unittest.TestCase):
    """Drive PipelineOrchestrator._run_scene_graph with mocked agents."""

    SCENE_TEXT = "Alice stood at the window, watching the rain fall silently."

    def _make_orchestrator(self, verifier_report):
        from pipeline.orchestrator import PipelineOrchestrator

        orch = object.__new__(PipelineOrchestrator)
        orch._cancelled = False
        orch._scene_cancelled = False
        orch._backend = "local"
        orch._progress_cb = lambda *a, **kw: None
        orch._emit = lambda *a, **kw: None
        orch._log = lambda *a, **kw: None
        orch._record_trace = lambda **kw: None

        def fake_writer(inp):
            return {"output": {"scene_text": self.SCENE_TEXT, "provider": "mock"}}

        orch.writer = MagicMock()
        orch.writer.run = fake_writer
        orch.writer.last_provider = "mock"
        orch.editor = MagicMock()
        orch.editor.run = lambda inp: {"output": {"scene_text": self.SCENE_TEXT}}
        orch.consistency = MagicMock()
        orch.consistency.run = lambda inp: {
            "output": {"report": {"is_consistent": True, "issues": [], "state_updates": {}}}
        }
        orch.consistency.get_blocking_issues = lambda report: []
        orch.consistency.get_non_blocking_issues = lambda report: []
        orch.consistency.issue_signature = Verifier.issue_signature
        orch.retriever = MagicMock()
        orch.retriever.retrieve_for_consistency = lambda **kw: "mock context"
        orch.state_manager = MagicMock()
        orch.state_manager.state = {}
        orch.quality_ctrl = MagicMock()
        orch.quality_ctrl.validate = lambda *a, **kw: SimpleNamespace(
            ok=True, truncated=False, words=700, minimum=500
        )
        orch.quality_ctrl.score = lambda *a, **kw: SimpleNamespace(
            overall=0.9, passes=True, issues=[],
            as_dict=lambda: {"overall": 0.9, "passes": True, "issues": []},
        )
        # Deterministic pacing agent used by the scene-graph quality gate
        orch.pacing = MagicMock()
        orch.pacing.analyze = lambda *a, **kw: {"issues": []}
        orch._future_premise_violations = staticmethod(lambda *a, **kw: [])
        orch._verifier_windows = lambda chapter_num, state: ([], [])
        orch.verifier = MagicMock()
        orch._run_verifier = lambda **kw: verifier_report
        return orch

    def test_blocked_verifier_escalates_to_review(self):
        blocking_issue = {
            "type": "dead_character_acting",
            "detail": "Bob is dead but acting.",
            "severity": "high",
            "blocking": True,
            "source": "verifier",
            "suggestion": "Fix it.",
        }
        report = {
            "is_consistent": False,
            "blocking_issues": [blocking_issue],
            "non_blocking_issues": [],
            "issues": [blocking_issue],
            "signatures": [Verifier.issue_signature(blocking_issue)],
            "checks": ["entity_status"],
            "needs_review": False,
        }
        orig_review_after = config.VERIFIER_REVIEW_AFTER
        orig_hard_stop = config.VERIFIER_HARD_STOP
        try:
            config.VERIFIER_REVIEW_AFTER = 3
            config.VERIFIER_HARD_STOP = False
            orch = self._make_orchestrator(report)
            result = orch._run_scene_graph(
                chapter_num=1,
                scene={"scene_number": 1, "characters_present": ["Alice"]},
                scene_context="ctx",
                previous_ending="The rain continued.",
                trace_entries=[],
                step_counter={"steps": 0},
            )
        finally:
            config.VERIFIER_REVIEW_AFTER = orig_review_after
            config.VERIFIER_HARD_STOP = orig_hard_stop

        self.assertEqual(result["status"], "complete")
        self.assertTrue(result["needs_review"])
        self.assertIsNotNone(result["verifier_report"])

    def test_clean_verifier_passes_immediately(self):
        report = {
            "is_consistent": True,
            "blocking_issues": [],
            "non_blocking_issues": [],
            "issues": [],
            "signatures": [],
            "checks": ["entity_status"],
            "needs_review": False,
        }
        orch = self._make_orchestrator(report)
        result = orch._run_scene_graph(
            chapter_num=1,
            scene={"scene_number": 1, "characters_present": ["Alice"]},
            scene_context="ctx",
            previous_ending="The rain continued.",
            trace_entries=[],
            step_counter={"steps": 0},
        )
        self.assertEqual(result["status"], "complete")
        self.assertFalse(result["needs_review"])


if __name__ == "__main__":
    unittest.main()
