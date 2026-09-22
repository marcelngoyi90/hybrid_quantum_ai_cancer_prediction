from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_explanation import generate_explanation

from scripts.predict_camelyon_patch import load_image
from scripts.predict_seer_patient import (
    collect_patient_information,
    preprocess_record,
)

from src.evidence_retrieval import (
    retrieve_evidence,
)

from src.joint_feature_extraction import (
    load_selected_feature_context,
)


SEED = 123
IMAGE_THRESHOLD = 0.5
CLINICAL_THRESHOLD = 0.313

JOINT_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "joint_model_v2.json"
)
FEATURE_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "seer_features_v2.json"
)
PREPROCESSOR_STATE_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "features"
    / "seer_v2_preprocessor.json"
)
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "checkpoints"
    / "joint_v2"
    / f"joint_seed{SEED}_best.pt"
)
HISTORY_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "joint_v2"
    / f"joint_seed{SEED}_history.json"
)
KNOWLEDGE_BASE_PATH = (
    PROJECT_ROOT
    / "knowledge_base"
    / "breast_node_evidence.json"
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Run the image and clinical pathways "
            "for inputs assumed to belong to one patient."
        )
    )
    parser.add_argument(
        "--image",
        required=True,
        type=Path,
        help=(
            "Path to a 96x96 lymph-node "
            "histopathology patch."
        ),
    )
    parser.add_argument(
        "--explain-synthetic",
        action="store_true",
        help=(
            "Send prediction scores and public evidence to Groq "
            "for an unpaired demo with fictional clinical inputs."
        ),
    )
    return parser.parse_args()


def assess_relationship(
    image_class,
    clinical_class,
):
    if (
        image_class == "tumour_pattern_detected"
        and clinical_class == "N+"
    ):
        return {
            "category": "concordant_positive",
            "summary": (
                "Both model outputs are positive for their "
                "respective prediction tasks."
            ),
            "interpretation": (
                "The image pathway detected a tumour "
                "pattern in this lymph-node patch, and "
                "the clinical pathway estimated N+."
            ),
            "retrieval_topics": [
                "breast cancer lymph-node metastasis",
                "histopathological evidence of nodal metastasis",
                "clinical predictors of positive axillary nodes",
            ],
        }

    if (
        image_class == "no_tumour_pattern_detected"
        and clinical_class == "N0"
    ):
        return {
            "category": "concordant_negative",
            "summary": (
                "Neither pathway produced a positive "
                "signal for the supplied inputs."
            ),
            "interpretation": (
                "The image pathway did not detect a "
                "tumour pattern in this patch, and the "
                "clinical pathway estimated N0. A single "
                "negative patch cannot exclude tumour in "
                "another part of the node or another node."
            ),
            "retrieval_topics": [
                "limitations of lymph-node patch sampling",
                "breast cancer N0 staging",
                "false-negative lymph-node histopathology sampling",
            ],
        }

    if (
        image_class == "tumour_pattern_detected"
        and clinical_class == "N0"
    ):
        return {
            "category": "discordant_image_positive",
            "summary": (
                "The image pathway is positive while "
                "the clinical pathway estimates N0."
            ),
            "interpretation": (
                "The image model predicts a tumour pattern, "
                "while the clinical model predicts N0. "
                "These are different prediction tasks; neither "
                "output is a confirmed pathological finding."
            ),
            "retrieval_topics": [
                "pathological confirmation of nodal metastasis",
                "clinical model false-negative lymph-node prediction",
                "discordant breast cancer staging evidence",
            ],
        }

    if (
        image_class == "no_tumour_pattern_detected"
        and clinical_class == "N+"
    ):
        return {
            "category": "discordant_clinical_positive",
            "summary": (
                "The clinical pathway estimates N+ while "
                "this image patch is negative."
            ),
            "interpretation": (
                "The clinical variables suggest regional "
                "node positivity, but no tumour pattern was "
                "detected in this individual patch. A single "
                "patch cannot represent every region of a "
                "lymph node or every regional node."
            ),
            "retrieval_topics": [
                "lymph-node sampling limitations",
                "clinical prediction of axillary metastasis",
                "heterogeneity of lymph-node metastasis",
            ],
        }

    raise ValueError(
        "Unexpected combination of pathway outputs"
    )


def main():
    arguments = parse_arguments()

    context = load_selected_feature_context(
        joint_config_path=JOINT_CONFIG_PATH,
        feature_config_path=FEATURE_CONFIG_PATH,
        preprocessor_state_path=PREPROCESSOR_STATE_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        history_path=HISTORY_PATH,
        seed=SEED,
        requested_device="cpu",
    )

    (
        image_tensor,
        image_mode,
        image_size,
    ) = load_image(
        arguments.image.expanduser().resolve()
    )

    patient = collect_patient_information(
        context.preprocessor
    )

    (
        numeric_tensor,
        categorical_tensor,
    ) = preprocess_record(
        patient,
        context.preprocessor,
        context.device,
    )

    image_tensor = image_tensor.to(context.device)

    with torch.inference_mode():
        image_logit = context.model.forward_image(
            image_tensor
        ).reshape(-1)[0]

        clinical_logit = context.model.forward_tabular(
            numeric_tensor,
            categorical_tensor,
        ).reshape(-1)[0]

        image_probability = float(
            torch.sigmoid(image_logit).item()
        )
        clinical_probability = float(
            torch.sigmoid(clinical_logit).item()
        )

    image_class = (
        "tumour_pattern_detected"
        if image_probability >= IMAGE_THRESHOLD
        else "no_tumour_pattern_detected"
    )

    clinical_class = (
        "N+"
        if clinical_probability >= CLINICAL_THRESHOLD
        else "N0"
    )

    relationship = assess_relationship(
        image_class,
        clinical_class,
    )

    retrieval_query = " ".join(
    relationship["retrieval_topics"]
)

    retrieved_evidence = retrieve_evidence(
    query=retrieval_query,
    knowledge_base_path=KNOWLEDGE_BASE_PATH,
    top_k=3,
)

    result = {
        "system": (
            "Evidence-Grounded Dual-Pathway "
            "Cancer Decision-Support Prototype"
        ),
        "input_assumption": (
            "Unpaired demonstration with fictional clinical inputs."
            if arguments.explain_synthetic
            else "The relationship between the image and clinical "
                 "inputs has not been verified."
        ),
        "image_pathway": {
            "input_file": arguments.image.name,
            "input_size": [
                image_size[0],
                image_size[1],
            ],
            "input_mode": image_mode,
            "tumour_pattern_probability": round(
                image_probability,
                6,
            ),
            "threshold": IMAGE_THRESHOLD,
            "predicted_class": image_class,
            "unit_of_analysis": (
                "Central region of one 96x96 "
                "lymph-node patch"
            ),
        },
        "clinical_pathway": {
            "patient_input": patient,
            "n_positive_probability": round(
                clinical_probability,
                6,
            ),
            "threshold": CLINICAL_THRESHOLD,
            "predicted_class": clinical_class,
            "unit_of_analysis": (
                "Patient-level regional lymph-node status"
            ),
        },
        "relationship_assessment": relationship,"evidence_retrieval": {
    "query": retrieval_query,
    "documents": retrieved_evidence,
},
        "important_limitations": [
            (
                "This is not a mathematically fused "
                "patient-level probability."
            ),
            (
                "The two original training datasets "
                "were not paired by patient."
            ),
            (
                "The clinical pathway applies to adults "
                "with an existing breast-cancer diagnosis; "
                "it is not a cancer-screening model."
            ),
            (
                "A single negative image patch cannot "
                "exclude tumour elsewhere."
            ),
            (
                "The outputs require clinical and "
                "pathological interpretation."
            ),
        ],
        "warning": (
            "Research use only. This system does not "
            "provide a medical diagnosis."
        ),
    }

    if arguments.explain_synthetic:
        result["llm_explanation"] = generate_explanation(
            result,
            synthetic_demo=True,
        )
    print(
        json.dumps(
            result,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()