from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.camelyon17_data import build_transform
from src.joint_feature_extraction import (
    load_selected_feature_context,
)


SEED = 123
THRESHOLD = 0.5

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


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen Camelyon17 image pathway "
            "on one 96x96 lymph-node histopathology patch."
        )
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="Path to a 96x96 histopathology patch.",
    )
    return parser.parse_args()


def get_image_path(argument_path):
    if argument_path is not None:
        return argument_path.expanduser().resolve()

    entered_path = input(
        "Path to the 96x96 histopathology patch: "
    ).strip()

    return Path(entered_path).expanduser().resolve()


def load_image(image_path):
    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    with Image.open(image_path) as source:
        original_mode = source.mode
        original_size = source.size
        image = source.convert("RGB")

    if original_size != (96, 96):
        raise ValueError(
            "Expected a 96x96 Camelyon-style patch, "
            f"but received {original_size[0]}x"
            f"{original_size[1]}. Do not use a whole-slide "
            "image, mammogram, photograph, or arbitrary image."
        )

    transform = build_transform(training=False)
    image_tensor = transform(image).unsqueeze(0)

    return image_tensor, original_mode, original_size


def main():
    arguments = parse_arguments()
    image_path = get_image_path(arguments.image)

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
        original_mode,
        original_size,
    ) = load_image(image_path)

    image_tensor = image_tensor.to(context.device)

    with torch.inference_mode():
        logit = context.model.forward_image(
            image_tensor
        ).reshape(-1)[0]

        probability = float(
            torch.sigmoid(logit).item()
        )

    predicted_class = (
        "tumour_pattern_detected"
        if probability >= THRESHOLD
        else "no_tumour_pattern_detected"
    )

    interpretation = (
        "The model detected a tumour pattern in the "
        "central region of this lymph-node patch."
        if predicted_class == "tumour_pattern_detected"
        else
        "The model did not detect a tumour pattern in "
        "the central region of this lymph-node patch."
    )

    result = {
        "image_input": {
            "filename": image_path.name,
            "width": original_size[0],
            "height": original_size[1],
            "original_mode": original_mode,
            "expected_source": (
                "Camelyon-style lymph-node "
                "histopathology patch"
            ),
        },
        "model_seed": SEED,
        "tumour_pattern_probability": round(
            probability,
            6,
        ),
        "threshold": THRESHOLD,
        "predicted_class": predicted_class,
        "interpretation": interpretation,
        "scope": (
            "Patch-level result for the central region; "
            "not a whole-patient cancer or lymph-node-stage "
            "diagnosis."
        ),
        "warning": (
            "Research prediction only. This is not a "
            "medical diagnosis."
        ),
    }

    print(
        json.dumps(
            result,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()