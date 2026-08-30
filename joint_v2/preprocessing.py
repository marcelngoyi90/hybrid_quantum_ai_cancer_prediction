import numpy as np


def transform_features_to_angles(features):
    features = np.asarray(
        features,
        dtype=np.float32,
    )

    if features.ndim != 2:
        raise ValueError(
            "Features must be a two-dimensional matrix"
        )

    if features.shape[1] != 8:
        raise ValueError(
            "Every row must contain exactly eight features"
        )

    if not np.isfinite(features).all():
        raise ValueError(
            "Features contain NaN or infinite values"
        )

    transformed = (
        2.0 * np.arctan(features)
    ).astype(np.float32)

    return transformed