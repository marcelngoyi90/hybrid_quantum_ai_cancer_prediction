from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from frontend.server import app


class FrontendApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_model_status_reports_missing_private_artifacts(self):
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

    def test_prediction_fails_safely_when_model_files_are_missing(self):
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


if __name__ == "__main__":
    unittest.main()
