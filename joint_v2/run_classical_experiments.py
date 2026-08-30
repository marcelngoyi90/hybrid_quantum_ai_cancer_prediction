import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression

from data_loader import load_feature_archive
from metrics import (
    calculate_binary_metrics,
    print_binary_metrics,
)


SEEDS = [42, 123, 2026]

TASKS = {
    "camelyon": {
        "train_file": "image_train_8d.npz",
        "validation_file": "image_ood_val_8d.npz",
        "validation_name": "OOD validation",
    },
    "seer": {
        "train_file": "tabular_train_8d.npz",
        "validation_file": "tabular_temporal_val_8d.npz",
        "validation_name": "temporal validation",
    },
}


all_results = []


for task_name, task_config in TASKS.items():
    for seed in SEEDS:
        print()
        print("=" * 60)
        print(
            f"Running {task_name} seed {seed}"
        )

        train_features, train_labels = (
            load_feature_archive(
                seed=seed,
                filename=task_config["train_file"],
            )
        )

        validation_features, validation_labels = (
            load_feature_archive(
                seed=seed,
                filename=task_config[
                    "validation_file"
                ],
            )
        )

        print(
            "Training examples:",
            len(train_labels),
        )
        print(
            "Validation examples:",
            len(validation_labels),
        )
        print("Training logistic regression...")

        model = LogisticRegression(
            solver="lbfgs",
            C=np.inf,
            max_iter=2000,
            tol=1e-6,
        )

        model.fit(
            train_features,
            train_labels,
        )

        probabilities = model.predict_proba(
            validation_features,
        )[:, 1]

        metrics = calculate_binary_metrics(
            labels=validation_labels,
            probabilities=probabilities,
            threshold=0.5,
        )

        print_binary_metrics(
            title=(
                f"{task_name} seed {seed} "
                f"{task_config['validation_name']}"
            ),
            metrics=metrics,
        )

        all_results.append(
            {
                "task": task_name,
                "seed": seed,
                **metrics,
            }
        )


print()
print("=" * 60)
print("THREE-SEED SUMMARY")


for task_name in TASKS:
    task_results = [
        result
        for result in all_results
        if result["task"] == task_name
    ]

    print()
    print(task_name)

    for metric_name in [
        "roc_auc",
        "average_precision",
        "accuracy",
        "balanced_accuracy",
        "f1",
        "precision",
        "sensitivity",
        "specificity",
        "brier_score",
    ]:
        values = np.array(
            [
                result[metric_name]
                for result in task_results
            ],
            dtype=float,
        )

        mean = values.mean()
        sample_standard_deviation = values.std(
            ddof=1,
        )

        print(
            f"{metric_name}: "
            f"{mean:.6f} +/- "
            f"{sample_standard_deviation:.6f}"
        )
results_file = (
    Path(__file__).resolve().parent
    / "results"
    / "classical_logistic_results.json"
)

results_file.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with results_file.open(
    "w",
    encoding="utf-8",
) as output_file:
    json.dump(
        all_results,
        output_file,
        indent=2,
        default=float,
    )

print()
print(
    "Results saved to:",
    results_file,
)

print()
print("All classical experiments completed.")
