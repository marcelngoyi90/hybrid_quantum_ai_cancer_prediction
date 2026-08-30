from sklearn.model_selection import train_test_split

from data_loader import load_feature_archive
from metrics import (
    calculate_binary_metrics,
    print_binary_metrics,
)
from models import (
    HybridQuantumModel,
    count_trainable_parameters,
)
from preprocessing import transform_features_to_angles
from training import (
    predict_probabilities,
    set_random_seed,
    train_binary_model,
)


seed = 42

print("Loading Camelyon seed 42...")

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
    train_size=2000,
    random_state=seed,
    stratify=train_labels,
)

validation_features, _, validation_labels, _ = (
    train_test_split(
        validation_features,
        validation_labels,
        train_size=1000,
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

print(
    "Trainable parameters:",
    count_trainable_parameters(model),
)
print(
    "Small training rows:",
    len(train_labels),
)
print(
    "Small validation rows:",
    len(validation_labels),
)

training_result = train_binary_model(
    model=model,
    train_features=train_features,
    train_labels=train_labels,
    validation_features=validation_features,
    validation_labels=validation_labels,
    seed=seed,
    learning_rate=0.01,
    batch_size=64,
    maximum_epochs=2,
    patience=2,
)

probabilities = predict_probabilities(
    model=training_result["model"],
    features=validation_features,
    batch_size=256,
)

metrics = calculate_binary_metrics(
    labels=validation_labels,
    probabilities=probabilities,
    threshold=0.5,
)

print_binary_metrics(
    title="Hybrid quantum smoke-test results",
    metrics=metrics,
)

print()
print(
    "Best epoch:",
    training_result["best_epoch"],
)
print("Quantum smoke test completed.")