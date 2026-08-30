import json
import time
from pathlib import Path

from data_loader import load_feature_archive
from metrics import (
    calculate_binary_metrics,
    print_binary_metrics,
)
from models import (
    ClassicalNeuralModel,
    HybridQuantumModel,
    count_trainable_parameters,
)
from preprocessing import transform_features_to_angles
from training import (
    predict_probabilities,
    set_random_seed,
    train_binary_model,
)


TASK_NAME = "seer"
SEED = 2026

LEARNING_RATE = 0.01
BATCH_SIZE = 1024
MAXIMUM_EPOCHS = 15
PATIENCE = 4

TASKS = {
    "camelyon": {
        "train_file": "image_train_8d.npz",
        "selection_file": "image_ood_val_8d.npz",
        "supporting_file": "image_id_val_8d.npz",
        "selection_name": "OOD validation",
        "supporting_name": "ID validation",
    },
    "seer": {
        "train_file": "tabular_train_8d.npz",
        "selection_file": "tabular_temporal_val_8d.npz",
        "supporting_file": "tabular_id_val_8d.npz",
        "selection_name": "temporal validation",
        "supporting_name": "ID validation",
    },
}

MODEL_FACTORIES = {
    "classical_neural": ClassicalNeuralModel,
    "hybrid_quantum": HybridQuantumModel,
}


if TASK_NAME not in TASKS:
    raise ValueError(
        f"Unknown task: {TASK_NAME}"
    )

task_config = TASKS[TASK_NAME]

print(
    f"Official development experiment: "
    f"{TASK_NAME}, seed {SEED}"
)
print("Locked-test data will not be loaded.")

train_features, train_labels = load_feature_archive(
    seed=SEED,
    filename=task_config["train_file"],
)

selection_features, selection_labels = (
    load_feature_archive(
        seed=SEED,
        filename=task_config["selection_file"],
    )
)

supporting_features, supporting_labels = (
    load_feature_archive(
        seed=SEED,
        filename=task_config["supporting_file"],
    )
)

train_features = transform_features_to_angles(
    train_features
)
selection_features = transform_features_to_angles(
    selection_features
)
supporting_features = transform_features_to_angles(
    supporting_features
)

print("Training rows:", len(train_labels))
print(
    "Selection-validation rows:",
    len(selection_labels),
)
print(
    "Supporting-validation rows:",
    len(supporting_labels),
)

experiment_results = {
    "task": TASK_NAME,
    "seed": SEED,
    "development_only": True,
    "locked_test_loaded": False,
    "training_configuration": {
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "maximum_epochs": MAXIMUM_EPOCHS,
        "patience": PATIENCE,
        "threshold": 0.5,
    },
    "models": {},
}


for model_name, model_factory in MODEL_FACTORIES.items():
    print()
    print("=" * 70)
    print(
        f"Training {model_name} for "
        f"{TASK_NAME}, seed {SEED}"
    )

    set_random_seed(SEED)
    model = model_factory()

    parameter_count = count_trainable_parameters(
        model
    )

    print(
        "Trainable parameters:",
        parameter_count,
    )

    start_time = time.perf_counter()

    training_result = train_binary_model(
        model=model,
        train_features=train_features,
        train_labels=train_labels,
        validation_features=selection_features,
        validation_labels=selection_labels,
        seed=SEED,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        maximum_epochs=MAXIMUM_EPOCHS,
        patience=PATIENCE,
    )

    elapsed_seconds = (
        time.perf_counter() - start_time
    )

    trained_model = training_result["model"]

    selection_probabilities = predict_probabilities(
        model=trained_model,
        features=selection_features,
        batch_size=4096,
    )

    supporting_probabilities = predict_probabilities(
        model=trained_model,
        features=supporting_features,
        batch_size=4096,
    )

    selection_metrics = calculate_binary_metrics(
        labels=selection_labels,
        probabilities=selection_probabilities,
        threshold=0.5,
    )

    supporting_metrics = calculate_binary_metrics(
        labels=supporting_labels,
        probabilities=supporting_probabilities,
        threshold=0.5,
    )

    print_binary_metrics(
        title=(
            f"{model_name} "
            f"{task_config['selection_name']}"
        ),
        metrics=selection_metrics,
    )

    print_binary_metrics(
        title=(
            f"{model_name} "
            f"{task_config['supporting_name']}"
        ),
        metrics=supporting_metrics,
    )

    print(
        "Best epoch:",
        training_result["best_epoch"],
    )
    print(
        "Elapsed seconds:",
        round(elapsed_seconds, 2),
    )

    experiment_results["models"][model_name] = {
        "trainable_parameters": parameter_count,
        "best_epoch": training_result[
            "best_epoch"
        ],
        "best_validation_roc_auc": (
            training_result[
                "best_validation_roc_auc"
            ]
        ),
        "elapsed_seconds": elapsed_seconds,
        "selection_metrics": selection_metrics,
        "supporting_metrics": supporting_metrics,
        "history": training_result["history"],
    }


results_directory = (
    Path(__file__).resolve().parent
    / "results"
    / "development"
)

results_directory.mkdir(
    parents=True,
    exist_ok=True,
)

results_file = results_directory / (
    f"{TASK_NAME}_seed{SEED}_"
    f"classical_vs_quantum.json"
)

with results_file.open(
    "w",
    encoding="utf-8",
) as output_file:
    json.dump(
        experiment_results,
        output_file,
        indent=2,
        default=float,
    )

print()
print("=" * 70)
print(
    "Development results saved to:",
    results_file,
)
print(
    "Official development experiment completed."
)