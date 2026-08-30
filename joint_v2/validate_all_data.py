import numpy as np

from data_loader import load_feature_archive


SEEDS = [42, 123, 2026]

FILES = [
    "image_train_8d.npz",
    "image_id_val_8d.npz",
    "image_ood_val_8d.npz",
    "tabular_train_8d.npz",
    "tabular_id_val_8d.npz",
    "tabular_temporal_val_8d.npz",
]


for seed in SEEDS:
    print()
    print(f"Checking seed {seed}")

    for filename in FILES:
        features, labels = load_feature_archive(
            seed=seed,
            filename=filename,
        )

        class_counts = np.bincount(
            labels,
            minlength=2,
        )

        print(
            f"{filename}: "
            f"features={features.shape}, "
            f"labels={labels.shape}, "
            f"classes={class_counts.tolist()}"
        )


print()
print("All 18 feature archives passed validation.")