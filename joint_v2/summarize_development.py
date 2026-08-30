import json
from pathlib import Path

import numpy as np


SEEDS = [42, 123, 2026]
TASKS = ["camelyon", "seer"]

MODEL_NAMES = [
    "classical_neural",
    "hybrid_quantum",
]

METRIC_NAMES = [
    "roc_auc",
    "average_precision",
    "accuracy",
    "balanced_accuracy",
    "f1",
    "precision",
    "sensitivity",
    "specificity",
    "brier_score",
]

results_directory = (
    Path(__file__).resolve().parent
    / "results"
    / "development"
)

summary = {}


for task_name in TASKS:
    task_results = []

    for seed in SEEDS:
        results_file = results_directory / (
            f"{task_name}_seed{seed}_"
            f"classical_vs_quantum.json"
        )

        if not results_file.exists():
            raise FileNotFoundError(
                f"Missing result: {results_file}"
            )

        with results_file.open(
            "r",
            encoding="utf-8",
        ) as input_file:
            result = json.load(input_file)

        if result["locked_test_loaded"]:
            raise ValueError(
                "A result says locked-test data was loaded"
            )

        task_results.append(result)

    task_summary = {
        "individual_seeds": {},
        "model_summaries": {},
        "quantum_minus_classical": {},
    }

    for result in task_results:
        seed = str(result["seed"])

        task_summary["individual_seeds"][seed] = {
            model_name: result["models"][
                model_name
            ]["selection_metrics"]
            for model_name in MODEL_NAMES
        }

    for model_name in MODEL_NAMES:
        model_summary = {}

        for metric_name in METRIC_NAMES:
            values = np.array(
                [
                    result["models"][model_name][
                        "selection_metrics"
                    ][metric_name]
                    for result in task_results
                ],
                dtype=float,
            )

            model_summary[metric_name] = {
                "mean": float(values.mean()),
                "sample_standard_deviation": float(
                    values.std(ddof=1)
                ),
                "values": values.tolist(),
            }

        task_summary["model_summaries"][
            model_name
        ] = model_summary

    for metric_name in METRIC_NAMES:
        differences = np.array(
            [
                result["models"]["hybrid_quantum"][
                    "selection_metrics"
                ][metric_name]
                - result["models"]["classical_neural"][
                    "selection_metrics"
                ][metric_name]
                for result in task_results
            ],
            dtype=float,
        )

        task_summary[
            "quantum_minus_classical"
        ][metric_name] = {
            "mean": float(differences.mean()),
            "sample_standard_deviation": float(
                differences.std(ddof=1)
            ),
            "values": differences.tolist(),
        }

    summary[task_name] = task_summary


print("OFFICIAL DEVELOPMENT SUMMARY")
print("Locked-test data was not used.")


for task_name in TASKS:
    print()
    print("=" * 70)
    print(task_name.upper())

    for model_name in MODEL_NAMES:
        model_summary = summary[task_name][
            "model_summaries"
        ][model_name]

        print()
        print(model_name)

        for metric_name in METRIC_NAMES:
            metric_summary = model_summary[
                metric_name
            ]

            print(
                f"{metric_name}: "
                f"{metric_summary['mean']:.6f} "
                f"+/- "
                f"{metric_summary['sample_standard_deviation']:.6f}"
            )

    print()
    print("quantum_minus_classical")

    for metric_name in METRIC_NAMES:
        difference_summary = summary[task_name][
            "quantum_minus_classical"
        ][metric_name]

        print(
            f"{metric_name}: "
            f"{difference_summary['mean']:.6f} "
            f"+/- "
            f"{difference_summary['sample_standard_deviation']:.6f}"
        )


summary_file = (
    results_directory
    / "three_seed_development_summary.json"
)

with summary_file.open(
    "w",
    encoding="utf-8",
) as output_file:
    json.dump(
        summary,
        output_file,
        indent=2,
    )

print()
print("=" * 70)
print(
    "Summary saved to:",
    summary_file,
)
print("Development summary completed.")