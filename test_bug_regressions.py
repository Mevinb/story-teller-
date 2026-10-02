"""Regression coverage for defects found in the 2026-09-20 application audit."""
import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from flask import Flask

from memory.state_manager import StateManager
from proxy_rotator import parse_proxy_list
from vision.router import vision_bp


class StateIntegrityRegressions(unittest.TestCase):
    def setUp(self):
        self.project = tempfile.mkdtemp(prefix="story_state_regression_")
        self.first = StateManager(self.project)
        self.first.initialize(
            title="Original",
            genre="fiction",
            premise="Start",
            characters={
                "Alice": {"traits": ["brave"]},
                "Bob": {"traits": ["kind"]},
            },
        )

    def tearDown(self):
        shutil.rmtree(self.project, ignore_errors=True)

    def test_stale_manager_does_not_overwrite_newer_state(self):
        stale = StateManager(self.project)
        stale.load()
        self.first.apply_state_update({"metadata": {"title": "Updated"}})
        stale.add_event("A later event", chapter=1)
        state = StateManager(self.project).load()
        self.assertEqual(state["metadata"]["title"], "Updated")
        self.assertEqual(state["plot"]["major_events"][-1]["event"], "A later event")

    def test_explicit_character_replacement_removes_and_clears(self):
        self.first.apply_state_update(
            {"characters": {"Alice": {"traits": []}}},
            replace_characters=True,
        )
        state = self.first.load()
        self.assertEqual(state["characters"]["Alice"]["traits"], [])
        self.assertNotIn("Bob", state["characters"])
        self.assertNotIn("bob", state["entities"])

    def test_reset_clears_lifecycle_status_in_character_and_registry(self):
        self.first.update_entity_status("Alice", "dead")
        self.first.reset_generated()
        self.assertEqual(self.first.get_character("Alice")["status"], "active")
        self.assertEqual(self.first.get_entity_status("Alice"), "active")

    def test_rewind_reconstructs_status_from_retained_events(self):
        self.first.add_story_event({
            "type": "death", "description": "Alice fell", "characters": ["Alice"],
            "chapter": 1, "scene": 1,
        })
        self.first.update_entity_status("Alice", "dead")
        self.first.prune_after_chapter(1, 1)
        self.assertEqual(self.first.get_character("Alice")["status"], "dead")
        self.assertEqual(self.first.get_entity_status("Alice"), "dead")


class RequestValidationRegressions(unittest.TestCase):
    def setUp(self):
        app = Flask(__name__)
        app.register_blueprint(vision_bp)
        app.testing = True
        self.client = app.test_client()

    def test_invalid_vision_base64_is_json_400(self):
        for route in ("/api/vision/analyze", "/api/vision/chat/start"):
            response = self.client.post(route, json={"image_base64": "a"})
            self.assertEqual(response.status_code, 400)
            self.assertFalse(response.get_json()["success"])

    def test_proxy_parser_rejects_unimplemented_features(self):
        for value in ("https://proxy.example:443", "http://user:pass@proxy.example:80"):
            with self.assertRaises(ValueError):
                parse_proxy_list(value)


class StreamingAndEmbeddingRegressions(unittest.TestCase):
    def test_event_channel_broadcasts_and_replays(self):
        from app import _EventChannel
        channel = _EventChannel(maxsize=10)
        first = channel.subscribe()
        second = channel.subscribe()
        channel.put_nowait({"type": "status", "payload": "ready"})
        self.assertEqual(first.get(timeout=0.1), second.get(timeout=0.1))
        reconnect = channel.subscribe()
        self.assertEqual(reconnect.get(timeout=0.1)["payload"], "ready")

    def test_gemini_stream_preserves_spaces_and_split_reasoning_tags(self):
        from models.gemini_model import GeminiModel

        chunks = ["Hello ", "<thi", "nk>hidden", "</think>", "world."]
        client = SimpleNamespace(models=SimpleNamespace(
            generate_content_stream=lambda **kwargs: [
                SimpleNamespace(text=value) for value in chunks
            ]
        ))
        model = GeminiModel(model="gemini-test", api_key="test")
        model._client = client
        with patch.object(model, "_throttle", return_value=None):
            output = "".join(model.generate_streaming("prompt"))
        self.assertEqual(output, "Hello world.")

    def test_vector_index_uses_configured_model_dimension(self):
        import memory.vector_store as vector_module

        class FakeEmbedder:
            def __init__(self, *args, **kwargs):
                pass

            def get_embedding_dimension(self):
                return 7

            def encode(self, texts, **kwargs):
                return np.ones((len(texts), 7), dtype="float32")

        project = tempfile.mkdtemp(prefix="story_vector_regression_")
        try:
            with patch.object(vector_module, "SentenceTransformer", FakeEmbedder):
                store = vector_module.VectorStore(project)
                self.assertEqual(store.get_stats()["dimension"], 7)
                store.add_text("A small memory.", chapter=1, scene=1)
                self.assertEqual(store.index.d, 7)
        finally:
            shutil.rmtree(project, ignore_errors=True)


class MaintenanceApiRegressions(unittest.TestCase):
    def test_create_conflict_and_model_free_maintenance(self):
        import app as web_app
        import config

        root = tempfile.mkdtemp(prefix="story_api_regression_")
        old_projects_dir = config.PROJECTS_DIR
        config.PROJECTS_DIR = root
        try:
            web_app._active_pipelines.clear()
            web_app._manual_sessions.clear()
            web_app._active_combines.clear()
            web_app._maintenance_projects.clear()
            client = web_app.create_app().test_client()
            payload = {"title": "Safe Story", "premise": "A beginning"}
            self.assertEqual(client.post("/api/project/create", json=payload).status_code, 200)
            self.assertEqual(client.post("/api/project/create", json=payload).status_code, 409)
            self.assertEqual(
                client.put(
                    "/api/project/safe_story/state",
                    json={"metadata": {"setting": "Moon"}},
                ).status_code,
                200,
            )
            self.assertEqual(client.post("/api/project/safe_story/reset").status_code, 200)
        finally:
            config.PROJECTS_DIR = old_projects_dir
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
