import time

from sklearn.model_selection import train_test_split

from data_loader import load_feature_archive
from models import HybridQuantumModel
from preprocessing import transform_features_to_angles
from training import (
    set_random_seed,
    train_binary_model,
)


seed = 42

print("Loading benchmark data...")

train_features, train_labels = load_feature_archive(
    seed=seed,
    filename="image_train_8d.npz",
)

validation_features, validation_labels = (
    load_feature_archive(
        seed=seed,
        filename="image_ood_val_8d.npz",
    )
)

train_features, _, train_labels, _ = train_test_split(
    train_features,
    train_labels,
    train_size=20000,
    random_state=seed,
    stratify=train_labels,
)

validation_features, _, validation_labels, _ = (
    train_test_split(
        validation_features,
        validation_labels,
        train_size=5000,
        random_state=seed,
        stratify=validation_labels,
    )
)

train_features = transform_features_to_angles(
    train_features
)

validation_features = transform_features_to_angles(
    validation_features
)

set_random_seed(seed)
model = HybridQuantumModel()

start_time = time.perf_counter()

train_binary_model(
    model=model,
    train_features=train_features,
    train_labels=train_labels,
    validation_features=validation_features,
    validation_labels=validation_labels,
    seed=seed,
    learning_rate=0.01,
    batch_size=1024,
    maximum_epochs=1,
    patience=1,
)

elapsed_seconds = time.perf_counter() - start_time

print()
print(
    "Benchmark rows:",
    len(train_labels),
)
print(
    "Elapsed seconds:",
    round(elapsed_seconds, 2),
)
print(
    "Estimated seconds for one full Camelyon epoch:",
    round(
        elapsed_seconds
        * 302436
        / len(train_labels),
        2,
    ),
)
print(
    "Estimated seconds for one full SEER epoch:",
    round(
        elapsed_seconds
        * 461292
        / len(train_labels),
        2,
    ),
)
print("Quantum benchmark completed.")