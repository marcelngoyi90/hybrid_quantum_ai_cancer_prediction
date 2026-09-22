from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evidence_retrieval import retrieve_evidence
from src.llm_explanation import generate_explanation


KNOWLEDGE_BASE = (
    PROJECT_ROOT / "knowledge_base" / "breast_node_evidence.json"
)

SCENARIOS = [
    {
        "name": "Both outputs negative",
        "image_probability": 0.10,
        "clinical_probability": 0.20,
        "query": (
            "lymph-node sampling limitations "
            "negative regional lymph-node prediction"
        ),
    },
    {
        "name": "Image positive, clinical negative",
        "image_probability": 0.90,
        "clinical_probability": 0.20,
        "query": (
            "discordant lymph-node predictions "
            "pathological confirmation clinical model limitations"
        ),
    },
]


def main():
    for index, scenario in enumerate(SCENARIOS):
        if index > 0:
            time.sleep(65)
        image_probability = scenario["image_probability"]
        clinical_probability = scenario["clinical_probability"]

        result = {
            "image_pathway": {
                "tumour_pattern_probability": image_probability,
                "threshold": 0.5,
                "predicted_class": (
                    "tumour_pattern_detected"
                    if image_probability >= 0.5
                    else "no_tumour_pattern_detected"
                ),
            },
            "clinical_pathway": {
                "n_positive_probability": clinical_probability,
                "threshold": 0.313,
                "predicted_class": (
                    "N+" if clinical_probability >= 0.313 else "N0"
                ),
            },
            "evidence_retrieval": {
                "documents": retrieve_evidence(
                    query=scenario["query"],
                    knowledge_base_path=KNOWLEDGE_BASE,
                    top_k=3,
                ),
            },
        }

        explanation = generate_explanation(
            result,
            synthetic_demo=True,
            simulated_scores=True
        )

        print(f"\nSIMULATED TEST: {scenario['name']}")
        print("Scores are fabricated, not model predictions.")
        print(explanation["text"])
        print()


if __name__ == "__main__":
    main()