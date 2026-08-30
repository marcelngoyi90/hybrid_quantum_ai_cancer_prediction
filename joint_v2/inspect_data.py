import numpy as np

from data_loader import load_feature_archive


features, labels = load_feature_archive(
    seed=42,
    filename="image_train_8d.npz",
)

classes, counts = np.unique(
    labels,
    return_counts=True,
)

print("Data loaded successfully")
print("Feature shape:", features.shape)
print("Feature type:", features.dtype)
print("Label shape:", labels.shape)
print("Label type:", labels.dtype)
print("Class values:", classes)
print("Class counts:", counts)
print("All features are finite:", np.isfinite(features).all())