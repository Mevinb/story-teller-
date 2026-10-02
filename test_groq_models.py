"""Regressions for namespaced Groq model discovery and picker catalogs."""
import unittest
from html.parser import HTMLParser
from unittest.mock import patch

import httpx
from openai import OpenAI

import config
from models.groq_model import GroqModel


class GroqAvailabilityTests(unittest.TestCase):
    def check_catalog(self, entries=None, status=200):
        requests = []

        def respond(request):
            requests.append(request)
            if status != 200:
                return httpx.Response(status, json={"error": {"message": "Denied"}})
            return httpx.Response(200, json={"object": "list", "data": entries or []})

        model = GroqModel(model="qwen/qwen3.8-27b", api_key="test-key")
        with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
            with OpenAI(api_key="test-key", base_url=config.GROQ_BASE_URL,
                        http_client=transport, max_retries=0) as client:
                model._client = client
                model._client_key = "test-key"
                available = model.is_available()
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].method, "GET")
        self.assertEqual(requests[0].url.path, "/openai/v1/models")
        return available

    def test_namespaced_qwen_is_available_without_encoded_retrieve_or_inference(self):
        self.assertTrue(self.check_catalog([
            {"id": "qwen/qwen3.8-27b", "object": "model", "active": True},
        ]))

    def test_missing_model_is_unavailable(self):
        self.assertFalse(self.check_catalog([
            {"id": "openai/gpt-oss-20b", "object": "model"},
        ]))

    def test_inactive_model_is_unavailable(self):
        self.assertFalse(self.check_catalog([
            {"id": "qwen/qwen3.8-27b", "object": "model", "active": False},
        ]))

    def test_catalog_without_optional_active_field(self):
        self.assertTrue(self.check_catalog([
            {"id": "qwen/qwen3.8-27b", "object": "model"},
        ]))

    def test_authentication_error_is_unavailable(self):
        self.assertFalse(self.check_catalog(status=401))


class GroqPickerTests(unittest.TestCase):
    def test_initial_picker_and_settings_share_api_catalog(self):
        from app import create_app

        class Options(HTMLParser):
            def __init__(self):
                super().__init__()
                self.in_groq_select = False
                self.model_options = []
                self.buttons = []

            def handle_starttag(self, tag, attrs):
                attrs = dict(attrs)
                if tag == "select":
                    self.in_groq_select = attrs.get("id") == "setting_GROQ_MODEL"
                if tag == "option" and self.in_groq_select:
                    self.model_options.append(attrs.get("value"))
                if tag == "button":
                    self.buttons.append(attrs.get("onclick", ""))

            def handle_endtag(self, tag):
                if tag == "select":
                    self.in_groq_select = False

        app = create_app()
        with patch("app._list_local_models", return_value=[]):
            client = app.test_client()
            page = client.get("/classic")
            catalog = client.get("/api/models").get_json()
        self.assertEqual(page.status_code, 200)
        parser = Options()
        parser.feed(page.get_data(as_text=True))
        expected = [model["id"] for model in catalog["groq_models"]]
        self.assertEqual(parser.model_options, expected)
        for model_id in expected:
            self.assertIn(f'switchModel("groq:{model_id}")', parser.buttons)
        self.assertIn("qwen/qwen3.8-27b", expected)
        self.assertIn("groq:qwen/qwen3.8-27b", catalog["models"])


if __name__ == "__main__":
    unittest.main()
