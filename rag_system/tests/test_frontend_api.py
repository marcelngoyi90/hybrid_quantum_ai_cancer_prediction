from __future__ import annotations

import unittest
import io
from types import SimpleNamespace
from unittest.mock import patch
from unittest.mock import Mock
import torch
from PIL import Image

from fastapi.testclient import TestClient

from frontend.server import app


class FrontendApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    @patch("pathlib.Path.is_file", return_value=False)
    def test_model_status_reports_missing_private_artifacts(self, _is_file):
        response = self.client.get("/api/model/status")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertFalse(body["ready"])
        missing_paths = {item["path"] for item in body["missing"]}
        self.assertIn(
            "artifacts/features/seer_v2_preprocessor.json",
            missing_paths,
        )
        self.assertIn(
            "artifacts/checkpoints/joint_v2/joint_seed123_best.pt",
            missing_paths,
        )

    @patch("pathlib.Path.is_file", return_value=False)
    def test_prediction_fails_safely_when_model_files_are_missing(
        self,
        _is_file,
    ):
        response = self.client.post(
            "/api/predict",
            data={"clinical": "{}"},
            files={"image": ("patch.png", b"not-an-image", "image/png")},
        )
        self.assertEqual(response.status_code, 503)
        self.assertIn("Prediction is not ready", response.json()["detail"])

    @patch("frontend.server.retrieve")
    def test_background_uses_the_local_knowledge_database(self, retrieve):
        retrieve.return_value = [
            {
                "title": f"Evidence {number}",
                "source_url": f"https://example.org/{number}",
            }
            for number in range(1, 6)
        ]
        response = self.client.get("/api/background")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("reported separately", body["text"])
        self.assertEqual(body["sources"][0]["id"], "S1")
        question = retrieve.call_args.args[0]
        self.assertIn("unpaired", question)

    @patch("frontend.server.answer_question")
    def test_result_explanation_uses_only_safe_model_outputs(self, answer):
        answer.return_value = {
            "answer": "The outputs must be interpreted separately.",
            "sources": [],
        }
        response = self.client.post(
            "/api/explain-results",
            json={
                "image": {
                    "predicted_class": "no_tumour_pattern_detected",
                    "probability": 0.2,
                },
                "clinical": {
                    "predicted_class": "N+",
                    "probability": 0.7,
                },
                "level": "student",
            },
        )
        self.assertEqual(response.status_code, 200)
        question = answer.call_args.args[0]
        self.assertIn("unpaired datasets", question)
        self.assertIn("scores must not be combined", question)
        self.assertNotIn("age at diagnosis", question.lower())
        self.assertNotIn("tumour size", question.lower())

    @patch("frontend.server.model_context")
    @patch("frontend.server.prediction_readiness", return_value={"ready": True})
    def test_image_only_needs_no_clinical_input(self, readiness, context):
        model = Mock()
        model.forward_image.return_value = torch.tensor([0.0])
        context.return_value = SimpleNamespace(model=model, device="cpu")
        image = io.BytesIO()
        Image.new("RGB", (96, 96)).save(image, format="PNG")
        response = self.client.post("/api/predict-image", files={"image": ("patch.png", image.getvalue(), "image/png")})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["image_pathway"]["tumour_pattern_probability"], 0.5)
        self.assertNotIn("clinical_pathway", response.json())
        model.forward_tabular.assert_not_called()

    @patch("frontend.server.answer_question")
    def test_image_only_explanation_does_not_invent_clinical_results(self, answer):
        answer.return_value = {"answer": "Image explanation", "sources": []}
        response = self.client.post("/api/explain-results", json={"image": {"predicted_class": "no_tumour_pattern_detected", "probability": 0.2}})
        self.assertEqual(response.status_code, 200)
        self.assertIn("No clinical pathway was evaluated", answer.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
