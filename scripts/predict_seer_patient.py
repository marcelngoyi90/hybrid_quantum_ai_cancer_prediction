from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.joint_feature_extraction import (
    load_selected_feature_context,
)


SEED = 123

# Selected using internal validation for seed 123.
# This is not a clinical diagnostic threshold.
THRESHOLD = 0.313

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


def ask_integer(prompt, minimum, maximum):
    while True:
        value = input(prompt).strip()

        try:
            number = int(value)
        except ValueError:
            print("Enter a whole number.")
            continue

        if not minimum <= number <= maximum:
            print(
                f"Enter a value from {minimum} "
                f"to {maximum}."
            )
            continue

        return number


def choose_category(
    preprocessor,
    feature,
    title,
):
    vocabulary = (
        preprocessor.category_vocabularies[
            feature
        ]
    )
    options = list(vocabulary.keys())

    print(f"\n{title}")

    for number, option in enumerate(
        options,
        start=1,
    ):
        print(f"{number}. {option}")

    selection = ask_integer(
        "Selection: ",
        minimum=1,
        maximum=len(options),
    )

    return options[selection - 1]


def ask_histology(preprocessor):
    feature = "Histologic Type ICD-O-3"
    vocabulary = (
        preprocessor.category_vocabularies[
            feature
        ]
    )

    print("\nHistological type ICD-O-3")
    print("Common examples:")
    print("8500 - ductal carcinoma")
    print("8520 - lobular carcinoma")
    print("8522 - mixed ductal and lobular carcinoma")
    print("8480 - mucinous adenocarcinoma")

    while True:
        code = input("Enter ICD-O-3 code: ").strip()

        if code in vocabulary:
            return code

        print(
            "That code was not present in the "
            "training vocabulary."
        )


def derive_breast_subtype(
    er_status,
    pr_status,
    her2_status,
):
    known_receptor_values = {
        "Positive",
        "Negative",
    }

    if (
        er_status not in known_receptor_values
        or pr_status not in known_receptor_values
        or her2_status not in known_receptor_values
    ):
        return None

    hormone_receptor_positive = (
        er_status == "Positive"
        or pr_status == "Positive"
    )

    hr_label = (
        "HR+"
        if hormone_receptor_positive
        else "HR-"
    )
    her2_label = (
        "HER2+"
        if her2_status == "Positive"
        else "HER2-"
    )

    return f"{hr_label}/{her2_label}"


def collect_patient_information(preprocessor):
    age = ask_integer(
        "\nAge at diagnosis: ",
        minimum=18,
        maximum=90,
    )

    tumor_size = ask_integer(
        "Tumour size in millimetres: ",
        minimum=0,
        maximum=400,
    )

    race = choose_category(
        preprocessor,
        "Race recode (W, B, AI, API)",
        "Race category",
    )

    origin = choose_category(
        preprocessor,
        (
            "Origin recode NHIA "
            "(Hispanic, Non-Hisp)"
        ),
        "Hispanic origin",
    )

    primary_site = choose_category(
        preprocessor,
        "Primary Site - labeled",
        "Primary breast site",
    )

    histology = ask_histology(
        preprocessor
    )

    laterality = choose_category(
        preprocessor,
        "Laterality",
        "Laterality",
    )

    grade = choose_category(
        preprocessor,
        "grade_unified",
        "Tumour grade",
    )

    er_status = choose_category(
        preprocessor,
        (
            "ER Status Recode "
            "Breast Cancer (1990+)"
        ),
        "ER status",
    )

    pr_status = choose_category(
        preprocessor,
        (
            "PR Status Recode "
            "Breast Cancer (1990+)"
        ),
        "PR status",
    )

    her2_status = choose_category(
        preprocessor,
        "Derived HER2 Recode (2010+)",
        "HER2 status",
    )

    breast_subtype = derive_breast_subtype(
        er_status,
        pr_status,
        her2_status,
    )

    if breast_subtype is None:
        breast_subtype = choose_category(
            preprocessor,
            "Breast Subtype (2010+)",
            "Breast-cancer subtype",
        )
    else:
        print(
            "\nDerived breast-cancer subtype: "
            f"{breast_subtype}"
        )

    t_stage = choose_category(
        preprocessor,
        "t_stage_harmonized",
        "Harmonised T stage",
    )

    m_stage = choose_category(
        preprocessor,
        "m_stage_harmonized",
        "Harmonised M stage",
    )

    return {
        "age_numeric": age,
        "tumor_size_mm": tumor_size,
        "Race recode (W, B, AI, API)": race,
        (
            "Origin recode NHIA "
            "(Hispanic, Non-Hisp)"
        ): origin,
        "Primary Site - labeled": primary_site,
        "Histologic Type ICD-O-3": histology,
        "Laterality": laterality,
        "grade_unified": grade,
        (
            "ER Status Recode "
            "Breast Cancer (1990+)"
        ): er_status,
        (
            "PR Status Recode "
            "Breast Cancer (1990+)"
        ): pr_status,
        "Derived HER2 Recode (2010+)": (
            her2_status
        ),
        "Breast Subtype (2010+)": (
            breast_subtype
        ),
        "tumor_size_status": "measured",
        "t_stage_harmonized": t_stage,
        "m_stage_harmonized": m_stage,
    }


def preprocess_record(
    record,
    preprocessor,
    device,
):
    numeric_values = []

    for feature in preprocessor.numeric_features:
        value = float(record[feature])
        statistics = (
            preprocessor.numeric_statistics[
                feature
            ]
        )

        scaled_value = (
            value - float(statistics["mean"])
        ) / float(statistics["scale"])

        numeric_values.append(
            scaled_value
        )

    categorical_values = []

    for feature in (
        preprocessor.categorical_features
    ):
        value = str(record[feature]).strip()
        vocabulary = (
            preprocessor.category_vocabularies[
                feature
            ]
        )

        if value not in vocabulary:
            raise ValueError(
                f"Unknown value for {feature}: "
                f"{value}"
            )

        categorical_values.append(
            int(vocabulary[value])
        )

    numeric_tensor = torch.tensor(
        [numeric_values],
        dtype=torch.float32,
        device=device,
    )

    categorical_tensor = torch.tensor(
        [categorical_values],
        dtype=torch.int64,
        device=device,
    )

    return (
        numeric_tensor,
        categorical_tensor,
    )


def main():
    context = load_selected_feature_context(
        joint_config_path=JOINT_CONFIG_PATH,
        feature_config_path=FEATURE_CONFIG_PATH,
        preprocessor_state_path=(
            PREPROCESSOR_STATE_PATH
        ),
        checkpoint_path=CHECKPOINT_PATH,
        history_path=HISTORY_PATH,
        seed=SEED,
        requested_device="cpu",
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

    with torch.inference_mode():
        logit = context.model.forward_tabular(
            numeric_tensor,
            categorical_tensor,
        ).reshape(-1)[0]

        probability = float(
            torch.sigmoid(logit).item()
        )

    predicted_class = (
        "N+"
        if probability >= THRESHOLD
        else "N0"
    )

    interpretation = (
        "One or more positive regional lymph "
        "nodes predicted."
        if predicted_class == "N+"
        else
        "No positive regional lymph nodes "
        "predicted."
    )

    result = {
        "patient_input": patient,
        "model_seed": SEED,
        "n_positive_probability": round(
            probability,
            6,
        ),
        "threshold": THRESHOLD,
        "predicted_class": predicted_class,
        "interpretation": interpretation,
        "warning": (
            "Research prediction only. This is "
            "not a medical diagnosis."
        ),
    }

    print("\nPrediction result")
    print(
        json.dumps(
            result,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()