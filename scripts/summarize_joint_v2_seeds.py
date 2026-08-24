from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "configs" / "joint_model_v2.json"
METRICS_DIRECTORY = (
    PROJECT_ROOT / "artifacts" / "metrics" / "joint_v2"
)

EXPECTED_SEEDS = (42, 123, 2026)

HISTORY_PATHS = {
    seed: METRICS_DIRECTORY / f"joint_seed{seed}_history.json"
    for seed in EXPECTED_SEEDS
}

PER_SEED_OUTPUT_PATH = (
    METRICS_DIRECTORY / "joint_v2_three_seed_results.csv"
)
SUMMARY_OUTPUT_PATH = (
    METRICS_DIRECTORY / "joint_v2_three_seed_summary.csv"
)
AUDIT_OUTPUT_PATH = (
    METRICS_DIRECTORY / "joint_v2_three_seed_audit.json"
)

VALID_STATUSES = {
    "completed",
    "early_stopped",
}

VALIDATION_CONTEXTS = (
    ("image", "id_val"),
    ("image", "ood_val"),
    ("tabular", "id_val"),
    ("tabular", "temporal_val"),
)

VALIDATION_METRICS = (
    "mean_loss",
    "accuracy",
    "balanced_accuracy",
    "roc_auc",
    "average_precision",
    "f1",
    "precision",
    "sensitivity",
    "specificity",
    "brier_score",
    "expected_calibration_error",
    "calibration_intercept",
    "calibration_slope",
)

SHARED_GRADIENT_METRICS = (
    "gradient_cosine_similarity_mean",
    "negative_cosine_fraction",
    "image_gradient_norm_mean",
    "tabular_gradient_norm_mean",
)

TEST_PROTECTION_FIELDS = (
    "camelyon_ood_test_evaluated",
    "seer_temporal_test_evaluated",
    "test_used_for_checkpoint_selection",
    "test_used_for_preprocessing",
    "test_used_for_tuning",
)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required file does not exist: {path}")

    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)

    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")

    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def require_mapping(
    value: Any,
    location: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{location} must be an object")

    return value


def require_sequence(
    value: Any,
    location: str,
) -> Sequence[Any]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{location} must be a sequence")

    if not isinstance(value, Sequence):
        raise TypeError(f"{location} must be a sequence")

    return value


def require_integer(
    value: Any,
    location: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{location} must be an integer")

    return value


def require_finite_number(
    value: Any,
    location: str,
) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{location} must be numeric")

    if not isinstance(value, (int, float)):
        raise TypeError(f"{location} must be numeric")

    numeric_value = float(value)

    if not math.isfinite(numeric_value):
        raise ValueError(f"{location} must be finite")

    return numeric_value


def require_false(
    value: Any,
    location: str,
) -> None:
    if value is not False:
        raise RuntimeError(f"{location} must be false")


def require_close(
    observed: float,
    expected: float,
    location: str,
) -> None:
    if not math.isclose(
        observed,
        expected,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            f"{location} mismatch: "
            f"observed={observed}, expected={expected}"
        )


def get_selected_epoch(
    history: Mapping[str, Any],
    seed: int,
) -> tuple[int, Mapping[str, Any]]:
    best = require_mapping(
        history.get("best"),
        f"seed {seed} best",
    )

    best_epoch = require_integer(
        best.get("epoch"),
        f"seed {seed} best epoch",
    )

    epochs = require_sequence(
        history.get("epochs"),
        f"seed {seed} epochs",
    )

    matching_epochs: list[Mapping[str, Any]] = []

    for epoch_index, raw_epoch in enumerate(epochs):
        epoch = require_mapping(
            raw_epoch,
            f"seed {seed} epochs[{epoch_index}]",
        )

        validation = require_mapping(
            epoch.get("validation"),
            f"seed {seed} epoch validation",
        )

        require_false(
            validation.get("locked_tests_evaluated"),
            (
                f"seed {seed} epoch "
                f"{epoch.get('epoch')} locked_tests_evaluated"
            ),
        )

        if epoch.get("epoch") == best_epoch:
            matching_epochs.append(epoch)

    if len(matching_epochs) != 1:
        raise RuntimeError(
            f"Seed {seed} must have exactly one record "
            f"for best epoch {best_epoch}"
        )

    return best_epoch, matching_epochs[0]


def validate_test_protection(
    history: Mapping[str, Any],
    seed: int,
) -> None:
    protection = require_mapping(
        history.get("test_protection"),
        f"seed {seed} test_protection",
    )

    for field in TEST_PROTECTION_FIELDS:
        require_false(
            protection.get(field),
            f"seed {seed} test_protection.{field}",
        )

    data_summary = require_mapping(
        history.get("data_summary"),
        f"seed {seed} data_summary",
    )

    require_false(
        data_summary.get("locked_tests_loaded"),
        f"seed {seed} data_summary.locked_tests_loaded",
    )

    require_false(
        data_summary.get("seer_patient_identifiers_loaded"),
        (
            f"seed {seed} "
            "data_summary.seer_patient_identifiers_loaded"
        ),
    )

    require_false(
        data_summary.get("camelyon_metadata_used_as_model_input"),
        (
            f"seed {seed} "
            "data_summary.camelyon_metadata_used_as_model_input"
        ),
    )


def extract_run(
    history_path: Path,
    expected_seed: int,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    history = load_json(history_path)

    observed_seed = require_integer(
        history.get("seed"),
        f"{history_path.name} seed",
    )

    if observed_seed != expected_seed:
        raise RuntimeError(
            f"{history_path.name} contains seed {observed_seed}, "
            f"expected {expected_seed}"
        )

    status = history.get("status")

    if status not in VALID_STATUSES:
        raise RuntimeError(
            f"Seed {expected_seed} has invalid status: {status}"
        )

    run_identity = require_mapping(
        history.get("run_identity"),
        f"seed {expected_seed} run_identity",
    )

    identity_seed = require_integer(
        run_identity.get("seed"),
        f"seed {expected_seed} run_identity.seed",
    )

    if identity_seed != expected_seed:
        raise RuntimeError(
            f"Seed {expected_seed} run identity has seed "
            f"{identity_seed}"
        )

    config_hash = sha256_file(CONFIG_PATH)

    if run_identity.get("joint_config_sha256") != config_hash:
        raise RuntimeError(
            f"Seed {expected_seed} configuration hash mismatch"
        )

    if run_identity.get("model_name") != config.get("model_name"):
        raise RuntimeError(
            f"Seed {expected_seed} model name mismatch"
        )

    validate_test_protection(history, expected_seed)

    best_epoch, selected_epoch = get_selected_epoch(
        history,
        expected_seed,
    )

    best = require_mapping(
        history.get("best"),
        f"seed {expected_seed} best",
    )

    checkpoint_score = require_finite_number(
        best.get("checkpoint_score"),
        f"seed {expected_seed} best checkpoint_score",
    )

    validation = require_mapping(
        selected_epoch.get("validation"),
        f"seed {expected_seed} selected validation",
    )

    selection = require_mapping(
        validation.get("selection"),
        f"seed {expected_seed} validation selection",
    )

    checkpoint_config = require_mapping(
        config.get("checkpoint_selection"),
        "checkpoint_selection configuration",
    )

    expected_image_split = config["tasks"]["image"][
        "selection_validation_split"
    ]
    expected_tabular_split = config["tasks"]["tabular"][
        "selection_validation_split"
    ]

    if selection.get("image_split") != expected_image_split:
        raise RuntimeError(
            f"Seed {expected_seed} image selection split mismatch"
        )

    if selection.get("tabular_split") != expected_tabular_split:
        raise RuntimeError(
            f"Seed {expected_seed} tabular selection split mismatch"
        )

    image_selection_auc = require_finite_number(
        selection.get("image_roc_auc"),
        f"seed {expected_seed} image selection AUC",
    )
    tabular_selection_auc = require_finite_number(
        selection.get("tabular_roc_auc"),
        f"seed {expected_seed} tabular selection AUC",
    )
    selected_checkpoint_score = require_finite_number(
        selection.get("checkpoint_score"),
        f"seed {expected_seed} selected checkpoint score",
    )

    image_weight = require_finite_number(
        checkpoint_config.get("image_metric_weight"),
        "checkpoint_selection.image_metric_weight",
    )
    tabular_weight = require_finite_number(
        checkpoint_config.get("tabular_metric_weight"),
        "checkpoint_selection.tabular_metric_weight",
    )

    require_close(
        image_weight + tabular_weight,
        1.0,
        "checkpoint selection weights",
    )

    calculated_checkpoint_score = (
        image_weight * image_selection_auc
        + tabular_weight * tabular_selection_auc
    )

    require_close(
        selected_checkpoint_score,
        calculated_checkpoint_score,
        f"seed {expected_seed} calculated checkpoint score",
    )
    require_close(
        checkpoint_score,
        selected_checkpoint_score,
        f"seed {expected_seed} stored best score",
    )

    checkpoint_decision = require_mapping(
        selected_epoch.get("checkpoint_decision"),
        f"seed {expected_seed} checkpoint_decision",
    )

    if checkpoint_decision.get("best_epoch") != best_epoch:
        raise RuntimeError(
            f"Seed {expected_seed} checkpoint best epoch mismatch"
        )

    decision_best_score = require_finite_number(
        checkpoint_decision.get("best_score"),
        f"seed {expected_seed} checkpoint decision best score",
    )

    require_close(
        decision_best_score,
        checkpoint_score,
        f"seed {expected_seed} checkpoint decision best score",
    )

    row: dict[str, Any] = {
        "seed": expected_seed,
        "status": status,
        "best_epoch": best_epoch,
        "checkpoint_score": checkpoint_score,
    }

    for task, split in VALIDATION_CONTEXTS:
        task_results = require_mapping(
            validation.get(task),
            f"seed {expected_seed} validation.{task}",
        )
        split_results = require_mapping(
            task_results.get(split),
            f"seed {expected_seed} validation.{task}.{split}",
        )
        overall = require_mapping(
            split_results.get("overall"),
            (
                f"seed {expected_seed} "
                f"validation.{task}.{split}.overall"
            ),
        )

        for metric in VALIDATION_METRICS:
            column_name = f"{task}_{split}_{metric}"
            row[column_name] = require_finite_number(
                overall.get(metric),
                (
                    f"seed {expected_seed} "
                    f"{task}.{split}.{metric}"
                ),
            )

    training = require_mapping(
        selected_epoch.get("training"),
        f"seed {expected_seed} selected training",
    )
    monitoring = require_mapping(
        training.get("shared_gradient_monitoring"),
        f"seed {expected_seed} shared gradient monitoring",
    )

    for metric in SHARED_GRADIENT_METRICS:
        row[f"shared_{metric}"] = require_finite_number(
            monitoring.get(metric),
            f"seed {expected_seed} shared {metric}",
        )

    return {
        "row": row,
        "run_identity": dict(run_identity),
        "data_summary": dict(
            require_mapping(
                history.get("data_summary"),
                f"seed {expected_seed} data_summary",
            )
        ),
        "history_sha256": sha256_file(history_path),
    }


def validate_cross_seed_consistency(
    extracted_runs: Sequence[Mapping[str, Any]],
) -> None:
    if len(extracted_runs) != len(EXPECTED_SEEDS):
        raise RuntimeError("Unexpected number of extracted runs")

    first_identity = dict(extracted_runs[0]["run_identity"])
    first_identity.pop("seed", None)

    first_data_summary = dict(extracted_runs[0]["data_summary"])
    first_data_summary.pop("seed", None)

    for run in extracted_runs[1:]:
        identity = dict(run["run_identity"])
        identity.pop("seed", None)

        if identity != first_identity:
            raise RuntimeError(
                "Model identity differs across random seeds"
            )

        data_summary = dict(run["data_summary"])
        data_summary.pop("seed", None)

        if data_summary != first_data_summary:
            raise RuntimeError(
                "Data summary differs across random seeds"
            )


def calculate_statistics(
    values: Sequence[float],
) -> dict[str, float | int]:
    if not values:
        raise ValueError("Cannot summarize an empty value sequence")

    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "sample_standard_deviation": (
            statistics.stdev(values)
            if len(values) > 1
            else 0.0
        ),
        "minimum": min(values),
        "maximum": max(values),
    }


def build_summary_rows(
    per_seed_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []

    def add_summary(
        task: str,
        split: str,
        metric: str,
        source_column: str,
    ) -> None:
        values = [
            require_finite_number(
                row[source_column],
                f"aggregate {source_column}",
            )
            for row in per_seed_rows
        ]

        summary_rows.append(
            {
                "task": task,
                "split": split,
                "metric": metric,
                **calculate_statistics(values),
            }
        )

    add_summary(
        task="joint",
        split="development_selection",
        metric="checkpoint_score",
        source_column="checkpoint_score",
    )

    for task, split in VALIDATION_CONTEXTS:
        for metric in VALIDATION_METRICS:
            add_summary(
                task=task,
                split=split,
                metric=metric,
                source_column=f"{task}_{split}_{metric}",
            )

    for metric in SHARED_GRADIENT_METRICS:
        add_summary(
            task="shared_trunk",
            split="selected_epoch_training",
            metric=metric,
            source_column=f"shared_{metric}",
        )

    return summary_rows


def write_csv_atomic(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="raise",
        )
        writer.writeheader()
        writer.writerows(rows)

    temporary_path.replace(path)


def write_json_atomic(
    path: Path,
    value: Mapping[str, Any],
) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(
            value,
            file,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        file.write("\n")

    temporary_path.replace(path)


def find_summary(
    summary_rows: Sequence[Mapping[str, Any]],
    task: str,
    split: str,
    metric: str,
) -> Mapping[str, Any]:
    matches = [
        row
        for row in summary_rows
        if row["task"] == task
        and row["split"] == split
        and row["metric"] == metric
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Could not uniquely locate summary for "
            f"{task}/{split}/{metric}"
        )

    return matches[0]


def main() -> None:
    config = load_json(CONFIG_PATH)

    configured_seeds = tuple(
        require_sequence(
            config.get("random_seeds"),
            "joint model random_seeds",
        )
    )

    if configured_seeds != EXPECTED_SEEDS:
        raise RuntimeError(
            f"Configured seeds are {configured_seeds}, "
            f"expected {EXPECTED_SEEDS}"
        )

    METRICS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    extracted_runs = [
        extract_run(
            history_path=HISTORY_PATHS[seed],
            expected_seed=seed,
            config=config,
        )
        for seed in EXPECTED_SEEDS
    ]

    validate_cross_seed_consistency(extracted_runs)

    per_seed_rows = [
        dict(run["row"])
        for run in extracted_runs
    ]

    summary_rows = build_summary_rows(per_seed_rows)

    per_seed_fieldnames = list(per_seed_rows[0].keys())
    summary_fieldnames = [
        "task",
        "split",
        "metric",
        "n",
        "mean",
        "sample_standard_deviation",
        "minimum",
        "maximum",
    ]

    write_csv_atomic(
        PER_SEED_OUTPUT_PATH,
        per_seed_rows,
        per_seed_fieldnames,
    )
    write_csv_atomic(
        SUMMARY_OUTPUT_PATH,
        summary_rows,
        summary_fieldnames,
    )

    common_identity = dict(extracted_runs[0]["run_identity"])
    common_identity.pop("seed", None)

    audit = {
        "version": "1.0",
        "purpose": (
            "Audited aggregation of the three predeclared "
            "joint model v2 random-seed runs."
        ),
        "expected_seeds": list(EXPECTED_SEEDS),
        "seed_count": len(EXPECTED_SEEDS),
        "model_identity": common_identity,
        "selection_rule": {
            "within_seed": dict(config["checkpoint_selection"]),
            "across_seeds": (
                "Report every seed and mean plus sample standard "
                "deviation; do not select the best-looking seed."
            ),
        },
        "selected_epochs": {
            str(row["seed"]): row["best_epoch"]
            for row in per_seed_rows
        },
        "input_histories": [
            {
                "seed": seed,
                "path": str(
                    HISTORY_PATHS[seed].relative_to(PROJECT_ROOT)
                ),
                "sha256": extracted_runs[index][
                    "history_sha256"
                ],
            }
            for index, seed in enumerate(EXPECTED_SEEDS)
        ],
        "test_protection": {
            "locked_tests_loaded": False,
            "locked_tests_evaluated": False,
            "locked_tests_used_for_selection": False,
            "locked_tests_used_for_tuning": False,
        },
        "privacy": {
            "patient_identifiers_loaded": False,
            "patient_level_rows_exported": False,
            "category_vocabularies_exported": False,
            "outputs_contain_aggregate_metrics_only": True,
        },
        "outputs": {
            "per_seed_results": {
                "path": str(
                    PER_SEED_OUTPUT_PATH.relative_to(PROJECT_ROOT)
                ),
                "sha256": sha256_file(PER_SEED_OUTPUT_PATH),
            },
            "three_seed_summary": {
                "path": str(
                    SUMMARY_OUTPUT_PATH.relative_to(PROJECT_ROOT)
                ),
                "sha256": sha256_file(SUMMARY_OUTPUT_PATH),
            },
        },
        "summary": summary_rows,
    }

    write_json_atomic(AUDIT_OUTPUT_PATH, audit)

    joint_score = find_summary(
        summary_rows,
        "joint",
        "development_selection",
        "checkpoint_score",
    )
    image_auc = find_summary(
        summary_rows,
        "image",
        "ood_val",
        "roc_auc",
    )
    image_accuracy = find_summary(
        summary_rows,
        "image",
        "ood_val",
        "accuracy",
    )
    tabular_auc = find_summary(
        summary_rows,
        "tabular",
        "temporal_val",
        "roc_auc",
    )
    tabular_accuracy = find_summary(
        summary_rows,
        "tabular",
        "temporal_val",
        "accuracy",
    )

    print(
        "seeds="
        + ",".join(str(seed) for seed in EXPECTED_SEEDS)
    )
    print(
        "selected_epochs="
        + ",".join(
            f"{row['seed']}:{row['best_epoch']}"
            for row in per_seed_rows
        )
    )
    print(
        "joint_checkpoint_score_mean="
        f"{joint_score['mean']:.6f}"
    )
    print(
        "joint_checkpoint_score_sd="
        f"{joint_score['sample_standard_deviation']:.6f}"
    )
    print(
        "image_ood_val_roc_auc_mean="
        f"{image_auc['mean']:.6f}"
    )
    print(
        "image_ood_val_accuracy_mean="
        f"{image_accuracy['mean']:.6f}"
    )
    print(
        "tabular_temporal_val_roc_auc_mean="
        f"{tabular_auc['mean']:.6f}"
    )
    print(
        "tabular_temporal_val_accuracy_mean="
        f"{tabular_accuracy['mean']:.6f}"
    )
    print("locked_tests_evaluated=False")
    print(f"per_seed_results={PER_SEED_OUTPUT_PATH}")
    print(f"summary={SUMMARY_OUTPUT_PATH}")
    print(f"audit={AUDIT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
