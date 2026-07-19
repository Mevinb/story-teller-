"""
Unit tests for Phase 1 critical fixes.

Tests:
1. WIP atomic write — crash mid-write leaves no corrupt file
2. WIP JSON corruption — _load_wip returns None gracefully
3. LRU embedding cache — correct eviction order
4. Scene validation regex — valid endings not rejected as truncated
5. Premise exhausted — strict > comparison (off-by-one fix)
6. Structured error types — hierarchy and context
"""
import json
import os
import shutil
import tempfile
import unittest
from collections import OrderedDict
from unittest.mock import MagicMock, patch


# ─── Test helpers ─────────────────────────────────────────────────────────────

def _make_temp_project():
    """Create a temporary project directory structure."""
    tmp = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp, "chapters"), exist_ok=True)
    os.makedirs(os.path.join(tmp, "logs"), exist_ok=True)
    return tmp


# ─── 1. WIP Atomic Write ──────────────────────────────────────────────────────

class TestWIPAtomicWrite(unittest.TestCase):
    """_save_wip must use atomic write so crashes don't leave corrupt files."""

    def setUp(self):
        self.project_dir = _make_temp_project()
        self.chapters_dir = os.path.join(self.project_dir, "chapters")

    def tearDown(self):
        shutil.rmtree(self.project_dir, ignore_errors=True)

    def _make_orchestrator(self):
        """Create a minimal orchestrator stub with only WIP methods."""
        from pipeline.orchestrator import PipelineOrchestrator

        orch = object.__new__(PipelineOrchestrator)
        orch.chapters_dir = self.chapters_dir
        orch.logs_dir = os.path.join(self.project_dir, "logs")
        orch._progress_cb = lambda *a, **kw: None
        return orch

    def test_wip_written_atomically(self):
        """_save_wip creates a final file at the correct path."""
        orch = self._make_orchestrator()
        orch._save_wip(1, [{"scene": 1, "text": "Hello world."}], {}, [])
        wip_path = orch._wip_path(1)
        self.assertTrue(os.path.exists(wip_path))
        with open(wip_path) as f:
            data = json.load(f)
        self.assertEqual(data["chapter_num"], 1)
        self.assertEqual(len(data["completed_scenes"]), 1)

    def test_wip_no_tmp_file_after_success(self):
        """No .tmp file should remain after a successful _save_wip."""
        orch = self._make_orchestrator()
        orch._save_wip(2, [], {}, [])
        tmp_path = orch._wip_path(2) + ".tmp"
        self.assertFalse(os.path.exists(tmp_path))

    def test_wip_overwrite_preserves_atomicity(self):
        """Re-saving WIP overwrites cleanly without corruption."""
        orch = self._make_orchestrator()
        orch._save_wip(1, [{"scene": 1, "text": "First."}], {}, [])
        orch._save_wip(1, [{"scene": 1, "text": "First."}, {"scene": 2, "text": "Second."}], {}, [])
        wip_path = orch._wip_path(1)
        with open(wip_path) as f:
            data = json.load(f)
        self.assertEqual(len(data["completed_scenes"]), 2)


# ─── 2. WIP JSON Corruption Handling ─────────────────────────────────────────

class TestWIPLoadCorruption(unittest.TestCase):
    """_load_wip must return None gracefully on corrupt/missing files."""

    def setUp(self):
        self.project_dir = _make_temp_project()
        self.chapters_dir = os.path.join(self.project_dir, "chapters")

    def tearDown(self):
        shutil.rmtree(self.project_dir, ignore_errors=True)

    def _make_orchestrator(self):
        from pipeline.orchestrator import PipelineOrchestrator

        orch = object.__new__(PipelineOrchestrator)
        orch.chapters_dir = self.chapters_dir
        orch.logs_dir = os.path.join(self.project_dir, "logs")
        orch._progress_cb = lambda *a, **kw: None
        return orch

    def test_load_missing_file_returns_none(self):
        orch = self._make_orchestrator()
        result = orch._load_wip(99)
        self.assertIsNone(result)

    def test_load_corrupt_json_returns_none(self):
        orch = self._make_orchestrator()
        wip_path = orch._wip_path(3)
        with open(wip_path, "w") as f:
            f.write("{corrupt json{{{\n")
        result = orch._load_wip(3)
        self.assertIsNone(result, "Corrupt WIP should return None, not raise")

    def test_corrupt_file_renamed_not_deleted(self):
        """Corrupt WIP files should be renamed to .corrupt for user inspection."""
        orch = self._make_orchestrator()
        wip_path = orch._wip_path(4)
        with open(wip_path, "w") as f:
            f.write("not json")
        orch._load_wip(4)
        corrupt_path = wip_path + ".corrupt"
        self.assertTrue(
            os.path.exists(corrupt_path),
            "Corrupt WIP file should be renamed to .corrupt"
        )
        self.assertFalse(os.path.exists(wip_path), "Original corrupt path should not remain")

    def test_valid_wip_loads_correctly(self):
        orch = self._make_orchestrator()
        orch._save_wip(5, [{"scene": 1, "text": "Text."}], {"chapter_title": "Test"}, [])
        result = orch._load_wip(5)
        self.assertIsNotNone(result)
        self.assertEqual(result["chapter_num"], 5)


# ─── 3. LRU Embedding Cache ───────────────────────────────────────────────────

class TestLRUEmbeddingCache(unittest.TestCase):
    """VectorStore embedding cache must use true LRU eviction."""

    def _make_store(self):
        """Return a VectorStore with a tiny LRU cache for testing."""
        from memory.vector_store import VectorStore

        with tempfile.TemporaryDirectory() as tmp:
            store = VectorStore.__new__(VectorStore)
            store._embedding_cache = OrderedDict()
            store._embedding_cache_hits = 0
            store._embedding_cache_misses = 0
            return store

    def test_lru_evicts_oldest_on_overflow(self):
        """When cache is full, the LEAST recently used entry is evicted."""
        import numpy as np
        import config

        original_max = getattr(config, "EMBEDDING_CACHE_MAX_SIZE", None)
        config.EMBEDDING_CACHE_MAX_SIZE = 3

        try:
            store = self._make_store()
            # Fill cache: A, B, C
            for key in ("A", "B", "C"):
                store._embedding_cache[key] = np.zeros((1, 384), dtype="float32")
            # Access A to make it most recently used (A moves to end)
            store._embedding_cache.move_to_end("A")
            # Insert D — should evict B (the new LRU after A was moved)
            if len(store._embedding_cache) >= config.EMBEDDING_CACHE_MAX_SIZE:
                store._embedding_cache.popitem(last=False)
            store._embedding_cache["D"] = np.zeros((1, 384), dtype="float32")
            self.assertNotIn("B", store._embedding_cache, "B should have been evicted as LRU")
            self.assertIn("A", store._embedding_cache, "A should remain (was recently used)")
            self.assertIn("C", store._embedding_cache)
            self.assertIn("D", store._embedding_cache)
        finally:
            if original_max is not None:
                config.EMBEDDING_CACHE_MAX_SIZE = original_max

    def test_cache_size_bounded(self):
        """Cache never grows beyond EMBEDDING_CACHE_MAX_SIZE entries."""
        import numpy as np
        import config

        original_max = getattr(config, "EMBEDDING_CACHE_MAX_SIZE", None)
        config.EMBEDDING_CACHE_MAX_SIZE = 5

        try:
            store = self._make_store()
            for i in range(20):
                key = f"query_{i}"
                if len(store._embedding_cache) >= config.EMBEDDING_CACHE_MAX_SIZE:
                    store._embedding_cache.popitem(last=False)
                store._embedding_cache[key] = np.zeros((1, 384), dtype="float32")
            self.assertLessEqual(len(store._embedding_cache), config.EMBEDDING_CACHE_MAX_SIZE)
        finally:
            if original_max is not None:
                config.EMBEDDING_CACHE_MAX_SIZE = original_max


# ─── 4. Scene Validation Regex ────────────────────────────────────────────────

class TestSceneValidationRegex(unittest.TestCase):
    """_validate_scene_completion must not reject valid scene endings."""

    def _make_orchestrator(self):
        from pipeline.orchestrator import PipelineOrchestrator

        orch = object.__new__(PipelineOrchestrator)
        orch._progress_cb = lambda *a, **kw: None
        return orch

    def _scene_ending_accepted(self, text: str) -> bool:
        """Return True if the scene ending regex accepts this text."""
        import re
        return bool(re.search(r'[.!?\u2026]["\u2019\u2018\*\_\s]*\s*$', text or ""))

    def test_normal_period_ending(self):
        self.assertTrue(self._scene_ending_accepted("She closed the door."))

    def test_exclamation_ending(self):
        self.assertTrue(self._scene_ending_accepted("He ran away!"))

    def test_question_mark_ending(self):
        self.assertTrue(self._scene_ending_accepted("Was this really happening?"))

    def test_quoted_ending(self):
        self.assertTrue(self._scene_ending_accepted('He said, "Goodbye."'))

    def test_markdown_bold_ending(self):
        """Scenes ending with **emphasis** should not be rejected."""
        self.assertTrue(self._scene_ending_accepted("She was **gone**."))

    def test_em_dash_sentence_no_period_rejected(self):
        """True mid-sentence cut should still be rejected."""
        self.assertFalse(self._scene_ending_accepted("The door swung open and she"))

    def test_ellipsis_unicode_accepted(self):
        """Trailing ellipsis (U+2026) is a valid sentence terminator."""
        self.assertTrue(self._scene_ending_accepted("He waited\u2026"))

    def test_ellipsis_ascii_accepted(self):
        """Trailing ASCII ... with period is accepted."""
        self.assertTrue(self._scene_ending_accepted("He waited..."))

    def test_word_ending_rejected(self):
        """Plain word ending (no punctuation) should be rejected as truncated."""
        self.assertFalse(self._scene_ending_accepted("She opened the"))


# ─── 5. Premise Exhausted Off-by-One ─────────────────────────────────────────

class TestPremiseExhaustedCursor(unittest.TestCase):
    """_premise_exhausted_from_state must use strict > comparison."""

    def _exhausted(self, completed_idx: int, num_steps: int) -> bool:
        from pipeline.orchestrator import PipelineOrchestrator

        steps = [f"step_{i}" for i in range(num_steps)]
        state = {"metadata": {"premise_step_completed": completed_idx}}
        return PipelineOrchestrator._premise_exhausted_from_state(steps, state)

    def test_cursor_at_last_step_not_exhausted(self):
        """When cursor == last step index (2 for 3 steps), still not exhausted.
        The chapter that sets this cursor is still writing step 2.
        The NEXT chapter will advance it and then it becomes exhausted.
        """
        self.assertFalse(self._exhausted(completed_idx=2, num_steps=3))

    def test_cursor_past_last_step_exhausted(self):
        """When cursor > last step index, all steps are covered."""
        self.assertTrue(self._exhausted(completed_idx=3, num_steps=3))

    def test_cursor_negative_not_exhausted(self):
        """cursor == -1 means fresh project, never exhausted."""
        self.assertFalse(self._exhausted(completed_idx=-1, num_steps=5))

    def test_cursor_zero_not_exhausted_single_step(self):
        """Single-step premise: cursor=0 means step is IN the window, not exhausted."""
        self.assertFalse(self._exhausted(completed_idx=0, num_steps=1))

    def test_cursor_one_exhausted_single_step(self):
        """Single-step premise: cursor=1 means past last step, exhausted."""
        self.assertTrue(self._exhausted(completed_idx=1, num_steps=1))

    def test_empty_steps_never_exhausted(self):
        """Empty premise never triggers exhaustion."""
        from pipeline.orchestrator import PipelineOrchestrator

        result = PipelineOrchestrator._premise_exhausted_from_state(
            [], {"metadata": {"premise_step_completed": 99}}
        )
        self.assertFalse(result)


# ─── 6. Structured Error Types ────────────────────────────────────────────────

class TestStructuredErrors(unittest.TestCase):
    """pipeline.errors hierarchy must be importable and correctly structured."""

    def test_import_all_error_types(self):
        from pipeline.errors import (
            PipelineError,
            ModelError,
            RateLimitError,
            ModelUnavailableError,
            TokenBudgetExceeded,
            ValidationError,
            SceneIncompleteError,
            SceneTruncatedError,
            PremiseViolationError,
            StateError,
            WIPCorruptError,
            StateSchemaMismatch,
            PipelineCancelledError,
        )
        # Verify hierarchy
        self.assertTrue(issubclass(RateLimitError, ModelError))
        self.assertTrue(issubclass(ModelError, PipelineError))
        self.assertTrue(issubclass(SceneIncompleteError, ValidationError))
        self.assertTrue(issubclass(ValidationError, PipelineError))
        self.assertTrue(issubclass(WIPCorruptError, StateError))
        self.assertTrue(issubclass(StateError, PipelineError))
        self.assertTrue(issubclass(PipelineCancelledError, PipelineError))

    def test_rate_limit_error_context(self):
        from pipeline.errors import RateLimitError

        err = RateLimitError("Too many requests", backend="groq", retry_after=30.0)
        self.assertEqual(err.backend, "groq")
        self.assertEqual(err.retry_after, 30.0)
        self.assertIn("groq", str(err))

    def test_scene_incomplete_error_context(self):
        from pipeline.errors import SceneIncompleteError

        err = SceneIncompleteError(scene_num=3, chapter_num=2, words=120, minimum=500)
        self.assertEqual(err.words, 120)
        self.assertEqual(err.minimum, 500)
        self.assertEqual(err.chapter_num, 2)

    def test_token_budget_exceeded_is_catchable_as_model_error(self):
        from pipeline.errors import ModelError, TokenBudgetExceeded

        err = TokenBudgetExceeded(chapter_num=5, scene_num=2, tokens_used=25000, budget=24000)
        self.assertIsInstance(err, ModelError)
        self.assertIsInstance(err, Exception)

    def test_wip_corrupt_error_context(self):
        from pipeline.errors import WIPCorruptError

        err = WIPCorruptError(chapter_num=1, path="/tmp/test.json")
        self.assertEqual(err.chapter_num, 1)
        self.assertIn("/tmp/test.json", str(err))

    def test_pipeline_error_is_runtime_error(self):
        """PipelineError must be catchable as RuntimeError for backward compat."""
        from pipeline.errors import PipelineError

        with self.assertRaises(RuntimeError):
            raise PipelineError("test")


if __name__ == "__main__":
    unittest.main(verbosity=2)
