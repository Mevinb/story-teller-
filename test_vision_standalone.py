"""
Fast standalone unit & integration tests for Person Recognition and Vision Analysis module.
Tests vision_bp directly on a lightweight Flask test app.
"""

import unittest
import json
from flask import Flask

from vision.schema import PersonAnalysisResult, Demographics, FacialFeatures, BodyStructure, ClothingStyle
from vision.analyzer import VisionAnalyzer
from vision.router import vision_bp


class TestVisionStandaloneFast(unittest.TestCase):
    """Fast test suite for vision module without loading main app models."""

    def setUp(self):
        self.app = Flask(__name__)
        self.app.register_blueprint(vision_bp)
        self.client = self.app.test_client()
        self.client.testing = True

    def test_schema_serialization(self):
        """Verify PersonAnalysisResult serializes to and from dict correctly."""
        res = PersonAnalysisResult(
            person_detected=True,
            summary="A detective standing under neon light",
            demographics=Demographics(apparent_age_range="30s", apparent_gender="Male", aesthetic_vibe="Noir"),
            facial_features=FacialFeatures(hair="Black, short", eyes="Sharp gray", face_shape="Square"),
            body_structure=BodyStructure(build="Athletic", height_estimate="Tall", posture="Upright"),
            clothing_style=ClothingStyle(outfit="Trench coat over shirt", style_category="Noir Detective", colors=["Black", "Beige"]),
            narrative_description="He stood motionless in the shadow of the damp alleyway...",
            consistency_tags=["trench coat", "short black hair", "sharp gray eyes", "tall build"],
            backend_used="test-mock"
        )

        d = res.to_dict()
        self.assertTrue(d["person_detected"])
        self.assertEqual(d["demographics"]["aesthetic_vibe"], "Noir")
        self.assertEqual(len(d["consistency_tags"]), 4)

        restored = PersonAnalysisResult.from_dict(d)
        self.assertEqual(restored.summary, res.summary)
        self.assertEqual(restored.facial_features.hair, "Black, short")
        self.assertEqual(restored.body_structure.build, "Athletic")

    def test_vision_status_endpoint(self):
        """Test GET /api/vision/status returns valid JSON structure."""
        resp = self.client.get('/api/vision/status')
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertTrue(data.get("success"))
        self.assertIn("active_mode", data.get("data", {}))
        self.assertIn("local", data.get("data", {}))
        self.assertIn("cloud", data.get("data", {}))

    def test_vision_analyze_endpoint_missing_file(self):
        """Test POST /api/vision/analyze with missing image returns 400."""
        resp = self.client.post('/api/vision/analyze')
        self.assertEqual(resp.status_code, 400)
        data = json.loads(resp.data)
        self.assertFalse(data.get("success"))

    def test_vision_analyzer_availability(self):
        """Test VisionAnalyzer queries backends cleanly without crashing."""
        analyzer = VisionAnalyzer()
        status = analyzer.get_available_backends()
        self.assertIn(status["active_mode"], ["local", "cloud", "auto", "gemini", "ollama"])

    def test_local_provider_auto_pull_signature(self):
        """Verify pull_model method exists and callable on LocalVisionProvider."""
        analyzer = VisionAnalyzer()
        self.assertTrue(hasattr(analyzer.local_provider, "pull_model"))

    def test_plain_text_fallback_parser(self):
        """Verify freeform Moondream output is parsed into structured fields via observations."""
        analyzer = VisionAnalyzer()
        raw_text = "The image features a woman with long dark hair wearing a red top. She is standing in front of some plants."
        observations = {
            "face_hair": "woman with long dark hair",
            "outfit": "wearing a red top",
            "upper_body": "wearing a red top, natural shoulders",
            "midsection": "standing upright",
            "lower_body": "standing in front of plants",
            "overall_vibe": "woman standing in front of some plants"
        }
        combined = "\n".join(f"[{k.upper()}]: {v}" for k, v in observations.items())
        result = analyzer.local_provider._parse_observations_to_result(observations, combined)
        self.assertTrue(result.person_detected)
        self.assertEqual(result.demographics.apparent_gender, "Female")
        self.assertIn("long dark hair", result.facial_features.hair.lower())




if __name__ == "__main__":
    unittest.main()
