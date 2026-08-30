from pathlib import Path

import numpy as np


DATA_ROOT = (
    Path(__file__).resolve().parents[2]
    / "joint_v2_authorized_handoff"
    / "artifacts"
    / "features"
    / "joint_v2"
)


def load_feature_archive(seed, filename):
    file_path = DATA_ROOT / f"seed{seed}" / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"Data file was not found: {file_path}"
        )

    with np.load(file_path, allow_pickle=False) as archive:
        if set(archive.files) != {"features", "labels"}:
            raise ValueError(
                f"Unexpected arrays in {file_path}: {archive.files}"
            )

        features = archive["features"]
        labels = archive["labels"]

    if features.ndim != 2 or features.shape[1] != 8:
        raise ValueError(
            f"Expected features with shape N x 8, got {features.shape}"
        )

    if labels.ndim != 1:
        raise ValueError(
            f"Expected one-dimensional labels, got {labels.shape}"
        )

    if features.shape[0] != labels.shape[0]:
        raise ValueError(
            "The number of feature rows and labels does not match"
        )

    if features.dtype != np.float32:
        raise ValueError(
            f"Expected float32 features, got {features.dtype}"
        )

    if labels.dtype != np.uint8:
        raise ValueError(
            f"Expected uint8 labels, got {labels.dtype}"
        )

    if not np.isfinite(features).all():
        raise ValueError("The features contain NaN or infinite values")

    if not np.isin(labels, [0, 1]).all():
        raise ValueError("The labels must contain only 0 and 1")

    return features, labels