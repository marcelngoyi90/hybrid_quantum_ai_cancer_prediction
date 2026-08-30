import numpy as np

from data_loader import load_feature_archive
from preprocessing import transform_features_to_angles


features, labels = load_feature_archive(
    seed=42,
    filename="image_train_8d.npz",
)

transformed_features = transform_features_to_angles(
    features
)

print("Original shape:", features.shape)
print("Transformed shape:", transformed_features.shape)
print("Original type:", features.dtype)
print("Transformed type:", transformed_features.dtype)
print(
    "Minimum transformed value:",
    transformed_features.min(),
)
print(
    "Maximum transformed value:",
    transformed_features.max(),
)
print(
    "All transformed values are finite:",
    np.isfinite(transformed_features).all(),
)
print(
    "Labels were not changed:",
    labels.shape == (features.shape[0],),
)