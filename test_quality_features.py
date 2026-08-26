"""Offline tests for the quality/voice/foreshadowing feature set.

Covers (all mocked — no network, no models):
- Selective best-of-N count selection (key vs normal scenes)
- Deterministic quality gate in the scene graph (one guided rewrite)
- Typed vector memory: chunk classification + memory_type filtered search
- StateManager: style-exemplar ring buffer, foreshadowing seeds, pruning
- Retriever: VOICE SAMPLE / STYLE ANCHOR / PLANTED FORESHADOWING blocks
- Architect continue-template keeps roadmap + thread-introduction slots
"""

import tempfile
import threading
import unittest
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Best-of-N scene selection
# ──────────────────────────────────────────────────────────────────────

class TestBestOfNCount(unittest.TestCase):

    def _orch(self):
        from pipeline.orchestrator import PipelineOrchestrator

        orch = object.__new__(PipelineOrchestrator)
        return orch

    def test_first_scene_is_key(self):
        import config

        orch = self._orch()
        self.assertEqual(
            orch._best_of_n_count({"type": "rising"}, 1, 4), config.BEST_OF_N_KEY_SCENES
        )

    def test_peak_scene_is_key(self):
        import config

        orch = self._orch()
        self.assertEqual(
            orch._best_of_n_count({"type": "peak"}, 2, 4), config.BEST_OF_N_KEY_SCENES
        )

    def test_final_scene_is_key(self):
        import config

        orch = self._orch()
        self.assertEqual(
            orch._best_of_n_count({"type": "rising"}, 4, 4), config.BEST_OF_N_KEY_SCENES
        )

    def test_middle_normal_scene_single_sample(self):
        orch = self._orch()
        self.assertEqual(orch._best_of_n_count({"type": "rising"}, 2, 4), 1)

    def test_disabled_when_key_config_is_one(self):
        import config

        original = config.BEST_OF_N_KEY_SCENES
        config.BEST_OF_N_KEY_SCENES = 1
        try:
            orch = self._orch()
            self.assertEqual(orch._best_of_n_count({"type": "peak"}, 1, 1), 1)
        finally:
            config.BEST_OF_N_KEY_SCENES = original


# ──────────────────────────────────────────────────────────────────────
# Scene-graph quality gate (one guided rewrite, then accept)
# ──────────────────────────────────────────────────────────────────────

class TestSceneGraphQualityGate(unittest.TestCase):
    """Drive _run_scene_graph with mocked agents and a failing QC score."""

    SCENE_TEXT = (
        "Alice stood at the window and watched the rain fall over the empty "
        "street below, thinking about everything that had brought her here. "
    ) * 12  # >120 words so best-of-N candidates are viable

    def _make_orchestrator(self, verifier_report, qc_scores):
        """qc_scores: list of ScoreResult-like namespaces consumed in order."""
        from pipeline.orchestrator import PipelineOrchestrator
        from agents.verifier import Verifier

        orch = object.__new__(PipelineOrchestrator)
        orch._cancelled = False
        orch._scene_cancelled = False
        orch._backend = "local"
        orch._progress_cb = lambda *a, **kw: None
        orch._emit = lambda *a, **kw: None
        orch._log = lambda *a, **kw: None
        orch._record_trace = lambda **kw: None

        calls = {"writer": 0, "score": 0}

        def fake_writer(inp):
            calls["writer"] += 1
            mode = inp.get("mode")
            if mode == "write":
                # Assert no issues leaked into the initial write
                self.assertEqual(inp.get("issues"), [])
            elif mode == "rewrite":
                issues = inp.get("issues") or []
                self.assertTrue(
                    any(i.get("type") == "quality" for i in issues),
                    "quality issues must reach the writer on rewrite",
                )
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

        def fake_score(*a, **kw):
            idx = min(calls["score"], len(qc_scores) - 1)
            calls["score"] += 1
            return qc_scores[idx]

        orch.quality_ctrl.score = fake_score
        orch.pacing = MagicMock()
        orch.pacing.analyze = lambda *a, **kw: {"issues": []}
        orch._future_premise_violations = staticmethod(lambda *a, **kw: [])
        orch._verifier_windows = lambda chapter_num, state: ([], [])
        orch.verifier = MagicMock()
        orch._run_verifier = lambda **kw: verifier_report
        return orch, calls

    def test_low_score_triggers_exactly_one_quality_rewrite(self):
        import config

        originals = (config.BEST_OF_N_KEY_SCENES, config.QUALITY_GATE_ENABLED,
                     config.QUALITY_GATE_THRESHOLD)
        config.BEST_OF_N_KEY_SCENES = 1  # isolate the gate from best-of-N
        config.QUALITY_GATE_ENABLED = True
        config.QUALITY_GATE_THRESHOLD = 0.55
        try:
            low = SimpleNamespace(
                overall=0.2, passes=False, issues=["Weak ending", "Thin dialogue"],
                as_dict=lambda: {"overall": 0.2, "passes": False},
            )
            high = SimpleNamespace(
                overall=0.9, passes=True, issues=[],
                as_dict=lambda: {"overall": 0.9, "passes": True},
            )
            report = {
                "is_consistent": True,
                "blocking_issues": [],
                "non_blocking_issues": [],
                "issues": [],
                "signatures": [],
                "checks": [],
                "needs_review": False,
            }
            orch, calls = self._make_orchestrator(report, [low, high])
            result = orch._run_scene_graph(
                chapter_num=1,
                scene={"scene_number": 2, "characters_present": ["Alice"]},
                scene_context="ctx",
                previous_ending="The rain continued.",
                trace_entries=[],
                step_counter={"steps": 0},
            )
            self.assertEqual(result["status"], "complete")
            self.assertFalse(result["needs_review"])
            # Gate scored the draft, flagged it, then scored the rewrite.
            self.assertEqual(calls["score"], 2)
            # Writer ran once for the draft and once for the guided rewrite.
            self.assertEqual(calls["writer"], 2)
            self.assertAlmostEqual(result["quality_scores"]["overall"], 0.9)
        finally:
            (config.BEST_OF_N_KEY_SCENES, config.QUALITY_GATE_ENABLED,
             config.QUALITY_GATE_THRESHOLD) = originals

    def test_gate_disabled_accepts_low_score(self):
        import config

        original = config.QUALITY_GATE_ENABLED
        config.QUALITY_GATE_ENABLED = False
        try:
            low = SimpleNamespace(
                overall=0.1, passes=False, issues=["Bad"],
                as_dict=lambda: {"overall": 0.1},
            )
            report = {
                "is_consistent": True,
                "blocking_issues": [],
                "non_blocking_issues": [],
                "issues": [],
                "signatures": [],
                "checks": [],
                "needs_review": False,
            }
            orch, calls = self._make_orchestrator(report, [low])
            result = orch._run_scene_graph(
                chapter_num=1,
                scene={"scene_number": 2, "characters_present": ["Alice"]},
                scene_context="ctx",
                previous_ending="The rain continued.",
                trace_entries=[],
                step_counter={"steps": 0},
            )
            self.assertEqual(result["status"], "complete")
            self.assertEqual(calls["writer"], 1)
            self.assertIsNone(result["quality_scores"])
        finally:
            config.QUALITY_GATE_ENABLED = original


# ──────────────────────────────────────────────────────────────────────
# Typed vector memory
# ──────────────────────────────────────────────────────────────────────

class TestTypedVectorMemory(unittest.TestCase):

    def _make_store(self, texts, metas):
        from memory.vector_store import VectorStore

        store = VectorStore.__new__(VectorStore)
        store._lock = threading.RLock()
        store._texts = list(texts)
        store._metadata = list(metas)
        store._embedding_cache = OrderedDict()
        store._embedding_cache_hits = 0
        store._embedding_cache_misses = 0
        store._encode = lambda texts: np.zeros((len(texts), 4), dtype="float32")

        fake_index = MagicMock()
        fake_index.ntotal = len(texts)
        n = len(texts)
        fake_index.search = lambda q, k: (
            np.zeros((1, n)), np.array([list(range(n))]),
        )
        store._index = fake_index
        store._save = lambda: None
        return store

    def test_classify_chunk_dialogue(self):
        from memory.vector_store import VectorStore

        self.assertEqual(
            VectorStore._classify_chunk('"Hello," she said. "Are you coming?"'),
            "dialogue",
        )

    def test_classify_chunk_action(self):
        from memory.vector_store import VectorStore

        text = "He ran to the door and slammed it. She grabbed the bag and fled."
        self.assertEqual(VectorStore._classify_chunk(text), "action")

    def test_classify_chunk_description(self):
        from memory.vector_store import VectorStore

        self.assertEqual(
            VectorStore._classify_chunk("The valley lay quiet under pale morning light."),
            "description",
        )

    def test_search_filters_by_memory_type(self):
        from memory.vector_store import ChunkMetadata, SearchResult  # noqa: F401
        from memory.vector_store import VectorStore

        texts = [
            '"I will not go," Alice said.',
            "The valley lay quiet under pale morning light.",
            "He ran and slammed the door.",
        ]
        metas = []
        for i, t in enumerate(texts):
            meta = ChunkMetadata(chapter=1, scene=i + 1, characters=["Alice"],
                                 location="home", memory_type=VectorStore._classify_chunk(t),
                                 chunk_index=i)
            metas.append(meta.to_dict())
        store = self._make_store(texts, metas)

        dialogue = store.search("Alice said", top_k=5, memory_type="dialogue")
        self.assertEqual(len(dialogue), 1)
        self.assertEqual(dialogue[0].metadata.memory_type, "dialogue")

        action = store.search("running", top_k=5, memory_type="action")
        self.assertEqual(len(action), 1)
        self.assertIn("slammed", action[0].text)

    def test_add_text_auto_classifies_each_chunk(self):
        from memory.vector_store import VectorStore

        store = self._make_store([], [])
        dialogue_text = (
            '"Stay back," she whispered. "I mean it."\n\n'
            '"You never listen," he said coldly. "You never did."\n\n'
            '"Then leave," Alice said.'
        )
        action_text = (
            "He ran to the door and slammed it behind him. "
            "Then he grabbed his coat and fled down the stairs."
        )
        descriptive_text = "The valley lay quiet under pale morning light."

        store.add_text(text=dialogue_text, chapter=1, scene=1,
                       characters=["Alice"], location="home", memory_type="auto")
        store.add_text(text=action_text, chapter=1, scene=2,
                       characters=["Alice"], location="home", memory_type="auto")
        store.add_text(text=descriptive_text, chapter=1, scene=3,
                       characters=["Alice"], location="home", memory_type="auto")

        types = [m["memory_type"] for m in store._metadata]
        self.assertIn("dialogue", types)
        self.assertIn("action", types)
        self.assertIn("description", types)


# ──────────────────────────────────────────────────────────────────────
# StateManager: style exemplars + foreshadowing
# ──────────────────────────────────────────────────────────────────────

class TestStyleExemplarAndForeshadowing(unittest.TestCase):

    def setUp(self):
        from memory.state_manager import StateManager

        self.tmp = tempfile.TemporaryDirectory()
        self.sm = StateManager(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_style_exemplar_ring_buffer(self):
        self.sm.record_style_exemplar(1, 0.8, "word " * 400)
        self.sm.record_style_exemplar(2, 0.7, "second chapter passage text")
        self.sm.record_style_exemplar(3, 0.9, "third chapter passage text")

        exemplars = self.sm.get_plot().get("style_exemplars", [])
        self.assertEqual(len(exemplars), 2)  # ring buffer capped
        self.assertEqual([e["chapter"] for e in exemplars], [2, 3])

    def test_style_exemplar_word_cap(self):
        long_text = "word " * 500
        self.sm.record_style_exemplar(1, 0.8, long_text)
        exemplars = self.sm.get_plot().get("style_exemplars", [])
        self.assertEqual(len(exemplars[0]["text"].split()), 320)

    def test_empty_exemplar_ignored(self):
        self.sm.record_style_exemplar(1, 0.8, "   ")
        self.assertEqual(self.sm.get_plot().get("style_exemplars", []), [])

    def test_foreshadowing_records_and_dedupes(self):
        self.sm.add_foreshadowing(["The cracked pocket watch", "A stranger's letter"], 1)
        self.sm.add_foreshadowing(["The cracked pocket watch", "A recurring dream"], 2)

        seeds = self.sm.get_plot().get("foreshadowing", [])
        texts = [s["text"] for s in seeds]
        self.assertEqual(texts, ["The cracked pocket watch", "A stranger's letter",
                                 "A recurring dream"])
        self.assertTrue(all(s["status"] == "planted" for s in seeds))
        self.assertEqual([s["chapter"] for s in seeds], [1, 1, 2])

    def test_foreshadowing_capped(self):
        for i in range(15):
            self.sm.add_foreshadowing([f"seed {i}"], 1)
        seeds = self.sm.get_plot().get("foreshadowing", [])
        self.assertEqual(len(seeds), 12)
        self.assertEqual(seeds[-1]["text"], "seed 14")

    def test_prune_filters_exemplars_and_seeds_by_chapter(self):
        self.sm.record_style_exemplar(1, 0.8, "chapter one style sample")
        self.sm.record_style_exemplar(2, 0.9, "chapter two style sample")
        self.sm.add_foreshadowing(["seed one"], 1)
        self.sm.add_foreshadowing(["seed two"], 2)

        self.sm.prune_after_chapter(1, 3)

        plot = self.sm.get_plot()
        self.assertEqual([e["chapter"] for e in plot["style_exemplars"]], [1])
        self.assertEqual([s["chapter"] for s in plot["foreshadowing"]], [1])


# ──────────────────────────────────────────────────────────────────────
# Retriever context blocks
# ──────────────────────────────────────────────────────────────────────

class TestRetrieverContextBlocks(unittest.TestCase):

    def setUp(self):
        from memory.retriever import Retriever
        from memory.state_manager import StateManager

        self.tmp = tempfile.TemporaryDirectory()
        self.sm = StateManager(self.tmp.name)
        # Seed style anchor + foreshadowing from earlier chapters
        self.sm.record_style_exemplar(2, 0.9, "The wind moved through the tall grass.")
        self.sm.add_foreshadowing(["The cracked pocket watch"], 1)

        self.vectors = MagicMock()
        self.vectors.index.ntotal = 5
        self.vectors.cross_encoder = None
        self.voice_hit = SimpleNamespace(
            text='"You never listen, Alice," he said coldly.',
            metadata=SimpleNamespace(chapter=1, scene=2, memory_type="dialogue",
                                     characters=["Alice"], location="home",
                                     chunk_index=0),
        )

        def fake_search(query, top_k=None, chapter_filter=None, memory_type=None):
            if memory_type == "dialogue":
                return [self.voice_hit]
            return []

        self.vectors.search = MagicMock(side_effect=fake_search)
        self.retriever = Retriever(self.sm, self.vectors)

    def tearDown(self):
        self.tmp.cleanup()

    def _scene_plan(self):
        return {
            "scene_number": 1,
            "type": "rising",
            "summary": "Alice returns home to find the letter.",
            "characters_present": ["Alice"],
            "location": "Alice's home",
        }

    def test_blocks_present(self):
        ctx = self.retriever.retrieve_context(
            scene_plan=self._scene_plan(),
            chapter_num=3,
            previous_ending="She closed the door behind her.",
        )
        self.assertIn("PLANTED FORESHADOWING", ctx)
        self.assertIn("The cracked pocket watch", ctx)
        self.assertIn("STYLE ANCHOR", ctx)
        self.assertIn("The wind moved through the tall grass.", ctx)
        self.assertIn("VOICE SAMPLE: Alice", ctx)

    def test_current_chapter_exemplar_excluded(self):
        self.sm.record_style_exemplar(3, 0.95, "CURRENT CHAPTER PROSE MARKER")
        ctx = self.retriever.retrieve_context(
            scene_plan=self._scene_plan(),
            chapter_num=3,
            previous_ending="She closed the door behind her.",
        )
        self.assertIn("STYLE ANCHOR", ctx)
        self.assertNotIn("CURRENT CHAPTER PROSE MARKER", ctx)

    def test_recurring_motifs_surfaced(self):
        # sm.state returns a deepcopy; mutate the backing dict instead
        state = self.sm._state
        state.setdefault("world", {}).setdefault("motifs", {})["storm"] = {
            "mentions": 3, "meaning": "inner turmoil", "chapters": [1, 2],
        }
        ctx = self.retriever.retrieve_context(
            scene_plan=self._scene_plan(),
            chapter_num=3,
            previous_ending="She closed the door behind her.",
        )
        self.assertIn("RECURRING MOTIFS", ctx)
        self.assertIn("Storm", ctx)


# ──────────────────────────────────────────────────────────────────────
# Architect continue template keeps roadmap + thread slots
# ──────────────────────────────────────────────────────────────────────

class TestArchitectContinueTemplate(unittest.TestCase):

    def test_template_has_upcoming_steps_and_thread_slot(self):
        from agents.architect import PLAN_CHAPTER_CONTINUE

        self.assertIn("{upcoming_steps}", PLAN_CHAPTER_CONTINUE)
        # The model must be ALLOWED to introduce threads (was hardcoded [])
        self.assertNotIn('"new_threads_to_introduce": []', PLAN_CHAPTER_CONTINUE)
        self.assertIn("seed", PLAN_CHAPTER_CONTINUE.lower())

    def test_compact_template_has_upcoming_steps(self):
        from agents.architect import PLAN_CHAPTER_CONTINUE_COMPACT

        self.assertIn("{upcoming_steps}", PLAN_CHAPTER_CONTINUE_COMPACT)


if __name__ == "__main__":
    unittest.main()
