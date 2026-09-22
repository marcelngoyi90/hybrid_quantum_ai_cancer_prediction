import unittest

from src.llm_explanation import format_results


def example_result(image_score, clinical_score):
    return {
        "image_pathway": {
            "tumour_pattern_probability": image_score,
            "threshold": 0.5,
            "predicted_class": (
                "tumour_pattern_detected"
                if image_score >= 0.5
                else "no_tumour_pattern_detected"
            ),
        },
        "clinical_pathway": {
            "n_positive_probability": clinical_score,
            "threshold": 0.313,
            "predicted_class": "N+" if clinical_score >= 0.313 else "N0",
        },
    }


class TestResultFormatting(unittest.TestCase):
    def test_four_classification_combinations(self):
        for image_score, clinical_score in (
            (0.1, 0.2),
            (0.9, 0.2),
            (0.1, 0.8),
            (0.9, 0.8),
        ):
            with self.subTest(image=image_score, clinical=clinical_score):
                result = example_result(image_score, clinical_score)
                text = format_results(result, simulated_scores=True)

                self.assertIn(
                    f"Classification: "
                    f"{result['image_pathway']['predicted_class']}.",
                    text,
                )
                self.assertIn(
                    f"Classification: "
                    f"{result['clinical_pathway']['predicted_class']}.",
                    text,
                )
                self.assertIn("no model inference was performed", text)
                self.assertIn("score for regional-node positivity (N+)", text)

    def test_scores_at_threshold_are_positive(self):
        text = format_results(
            example_result(0.5, 0.313),
            simulated_scores=True,
        )
        self.assertIn("Classification: tumour_pattern_detected.", text)
        self.assertIn("Classification: N+.", text)

    def test_invalid_score_is_rejected(self):
        with self.assertRaises(ValueError):
            format_results(
                example_result(1.2, 0.2),
                simulated_scores=True,
            )

    def test_mismatched_classification_is_rejected(self):
        result = example_result(0.9, 0.2)
        result["clinical_pathway"]["predicted_class"] = "N+"

        with self.assertRaises(ValueError):
            format_results(result, simulated_scores=True)


if __name__ == "__main__":
    unittest.main()