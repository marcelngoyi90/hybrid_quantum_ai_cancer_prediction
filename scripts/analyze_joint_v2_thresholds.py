from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "PYTORCH_ENABLE_MPS_FALLBACK",
    "1",
)

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.joint_evaluation import (
    binary_classification_metrics,
    validate_binary_inputs,
)
from src.joint_feature_extraction import (
    build_tabular_feature_loader,
    load_selected_feature_context,
)


JOINT_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "joint_model_v2.json"
)
FEATURE_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "seer_features_v2.json"
)
THRESHOLD_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "joint_v2_thresholds.json"
)
PREPROCESSOR_STATE_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "features"
    / "seer_v2_preprocessor.json"
)
SEER_MODEL_TABLE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "seer_v2_model_table.parquet"
)
CHECKPOINT_DIRECTORY = (
    PROJECT_ROOT
    / "artifacts"
    / "checkpoints"
    / "joint_v2"
)
METRICS_DIRECTORY = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "joint_v2"
)

RESULTS_OUTPUT_PATH = (
    METRICS_DIRECTORY
    / "joint_v2_threshold_results.csv"
)
SUMMARY_OUTPUT_PATH = (
    METRICS_DIRECTORY
    / "joint_v2_threshold_summary.csv"
)
AUDIT_OUTPUT_PATH = (
    METRICS_DIRECTORY
    / "joint_v2_threshold_audit.json"
)

METRIC_NAMES = (
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
)

EXPECTED_REPORTED_METRICS = (
    *METRIC_NAMES,
    "confusion_matrix",
)

THRESHOLD_RULES = (
    "default",
    "selected_on_id_val",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the pre-registered development-only "
            "SEER threshold analysis"
        )
    )

    parser.add_argument(
        "--device",
        choices=("mps", "cpu"),
        default=None,
    )

    return parser.parse_args()


def read_json_object(
    path: Path,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required JSON file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)

    if not isinstance(value, dict):
        raise TypeError(
            f"Expected a JSON object in {path}"
        )

    return value


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def require_mapping(
    value: Any,
    location: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(
            f"{location} must be an object"
        )

    return value


def require_sequence(
    value: Any,
    location: str,
) -> Sequence[Any]:
    if isinstance(value, (str, bytes)):
        raise TypeError(
            f"{location} must be a sequence"
        )

    if not isinstance(value, Sequence):
        raise TypeError(
            f"{location} must be a sequence"
        )

    return value


def require_finite_number(
    value: Any,
    location: str,
) -> float:
    if isinstance(value, bool):
        raise TypeError(
            f"{location} must be numeric"
        )

    if not isinstance(value, (int, float)):
        raise TypeError(
            f"{location} must be numeric"
        )

    numeric_value = float(value)

    if not math.isfinite(numeric_value):
        raise ValueError(
            f"{location} must be finite"
        )

    return numeric_value


def require_false(
    value: Any,
    location: str,
) -> None:
    if value is not False:
        raise RuntimeError(
            f"{location} must be false"
        )


def require_true(
    value: Any,
    location: str,
) -> None:
    if value is not True:
        raise RuntimeError(
            f"{location} must be true"
        )


def require_close(
    observed: float,
    expected: float,
    location: str,
    tolerance: float = 1e-6,
) -> None:
    if not math.isclose(
        observed,
        expected,
        rel_tol=tolerance,
        abs_tol=tolerance,
    ):
        raise RuntimeError(
            f"{location} mismatch: "
            f"observed={observed}, expected={expected}"
        )


def validate_configuration(
    threshold_config: Mapping[str, Any],
    joint_config: Mapping[str, Any],
) -> None:
    if threshold_config.get("model_name") != (
        joint_config.get("model_name")
    ):
        raise RuntimeError(
            "Threshold and joint model names differ"
        )

    if threshold_config.get("model_version") != (
        joint_config.get("version")
    ):
        raise RuntimeError(
            "Threshold and joint model versions differ"
        )

    if threshold_config.get("task") != "tabular":
        raise ValueError(
            "Threshold analysis must remain tabular"
        )

    configured_seeds = tuple(
        int(value)
        for value in require_sequence(
            threshold_config.get("random_seeds"),
            "threshold random_seeds",
        )
    )
    joint_seeds = tuple(
        int(value)
        for value in require_sequence(
            joint_config.get("random_seeds"),
            "joint random_seeds",
        )
    )

    if configured_seeds != joint_seeds:
        raise RuntimeError(
            "Threshold seeds differ from joint-model seeds"
        )

    selection = require_mapping(
        threshold_config.get("threshold_selection"),
        "threshold_selection",
    )
    tabular_task = require_mapping(
        joint_config["tasks"]["tabular"],
        "joint tabular task",
    )

    expected_split_values = {
        "fit_split": tabular_task[
            "internal_validation_split"
        ],
        "development_evaluation_split": (
            tabular_task[
                "selection_validation_split"
            ]
        ),
        "locked_test_split": tabular_task[
            "locked_test_split"
        ],
    }

    for field, expected_value in (
        expected_split_values.items()
    ):
        if selection.get(field) != expected_value:
            raise RuntimeError(
                f"Threshold {field} is inconsistent "
                "with the joint configuration"
            )

    if selection.get("selection_metric") != (
        "balanced_accuracy"
    ):
        raise ValueError(
            "Balanced accuracy must remain the "
            "threshold-selection metric"
        )

    if selection.get(
        "positive_prediction_rule"
    ) != (
        "probability_greater_than_or_equal_to_threshold"
    ):
        raise ValueError(
            "Unexpected positive prediction rule"
        )

    require_true(
        selection.get("select_separately_for_each_seed"),
        (
            "threshold_selection."
            "select_separately_for_each_seed"
        ),
    )

    expected_tie_breakers = (
        "minimum_absolute_distance_from_default_threshold",
        "lower_threshold",
    )
    observed_tie_breakers = tuple(
        require_sequence(
            selection.get("tie_breakers_in_order"),
            "threshold tie breakers",
        )
    )

    if observed_tie_breakers != expected_tie_breakers:
        raise RuntimeError(
            "Unexpected threshold tie-breaker order"
        )

    probability_processing = require_mapping(
        threshold_config.get(
            "probability_processing"
        ),
        "probability_processing",
    )

    for field, value in (
        probability_processing.items()
    ):
        require_false(
            value,
            f"probability_processing.{field}",
        )

    test_protection = require_mapping(
        threshold_config.get("test_protection"),
        "test_protection",
    )

    for field, value in test_protection.items():
        require_false(
            value,
            f"test_protection.{field}",
        )

    development_rules = require_mapping(
        threshold_config.get("development_rules"),
        "development_rules",
    )

    for field in (
        "model_retraining_allowed",
        "checkpoint_changes_allowed",
        "feature_changes_allowed",
        "preprocessing_changes_allowed",
        "temporal_val_used_for_threshold_selection",
    ):
        require_false(
            development_rules.get(field),
            f"development_rules.{field}",
        )

    for field in (
        "threshold_selected_using_id_val_only",
        "temporal_val_used_for_development_evaluation",
    ):
        require_true(
            development_rules.get(field),
            f"development_rules.{field}",
        )

    reported_metrics = tuple(
        require_sequence(
            threshold_config.get("reported_metrics"),
            "reported_metrics",
        )
    )

    if reported_metrics != EXPECTED_REPORTED_METRICS:
        raise RuntimeError(
            "Reported metrics differ from the "
            "pre-registered metric order"
        )

    if selection.get("locked_test_split") in {
        selection.get("fit_split"),
        selection.get(
            "development_evaluation_split"
        ),
    }:
        raise RuntimeError(
            "Locked temporal test entered development "
            "threshold analysis"
        )


def build_candidate_thresholds(
    threshold_config: Mapping[str, Any],
) -> np.ndarray:
    selection = require_mapping(
        threshold_config["threshold_selection"],
        "threshold_selection",
    )
    grid = require_mapping(
        selection.get("candidate_grid"),
        "threshold candidate_grid",
    )

    if grid.get("method") != "fixed_inclusive_grid":
        raise ValueError(
            "Only a fixed inclusive threshold grid "
            "is supported"
        )

    minimum = require_finite_number(
        grid.get("minimum"),
        "threshold grid minimum",
    )
    maximum = require_finite_number(
        grid.get("maximum"),
        "threshold grid maximum",
    )
    step = require_finite_number(
        grid.get("step"),
        "threshold grid step",
    )

    if not 0.0 <= minimum < maximum <= 1.0:
        raise ValueError(
            "Threshold grid bounds are invalid"
        )

    if step <= 0.0:
        raise ValueError(
            "Threshold grid step must be positive"
        )

    interval_count_float = (
        maximum - minimum
    ) / step
    interval_count = int(
        round(interval_count_float)
    )

    if not math.isclose(
        interval_count_float,
        interval_count,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "Threshold grid interval is not exactly "
            "divisible by the configured step"
        )

    thresholds = (
        minimum
        + step
        * np.arange(
            interval_count + 1,
            dtype=np.float64,
        )
    )

    thresholds = np.round(
        thresholds,
        decimals=12,
    )

    require_close(
        float(thresholds[0]),
        minimum,
        "first threshold",
        tolerance=1e-12,
    )
    require_close(
        float(thresholds[-1]),
        maximum,
        "last threshold",
        tolerance=1e-12,
    )

    return thresholds


def collect_tabular_probabilities(
    context,
    split: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    (
        loader,
        expected_rows,
        loader_audit,
    ) = build_tabular_feature_loader(
        context=context,
        seer_model_table_path=(
            SEER_MODEL_TABLE_PATH
        ),
        split=split,
    )

    context.model.eval()

    label_batches: list[torch.Tensor] = []
    probability_batches: list[torch.Tensor] = []

    with torch.inference_mode():
        for batch in loader:
            numeric_inputs = batch[
                "numeric"
            ].to(
                context.device
            )
            categorical_inputs = batch[
                "categorical"
            ].to(
                context.device
            )

            logits = context.model.forward_tabular(
                numeric_inputs,
                categorical_inputs,
            )
            probabilities = torch.sigmoid(
                logits
            )

            labels = batch["target"].to(
                dtype=torch.int64
            )

            if logits.shape != labels.shape:
                raise RuntimeError(
                    f"Logit and label shapes differ "
                    f"for {split}"
                )

            label_batches.append(
                labels.detach().to(
                    device="cpu",
                    dtype=torch.int64,
                )
            )
            probability_batches.append(
    probabilities.detach().to(
        device="cpu",
        dtype=torch.float32,
    )
)

    if not label_batches:
        raise RuntimeError(
            f"No SEER probabilities produced for "
            f"{split}"
        )

    labels = np.ascontiguousarray(
        torch.cat(
            label_batches,
            dim=0,
        ).numpy(),
        dtype=np.int64,
    )
    probabilities = np.ascontiguousarray(
        torch.cat(
            probability_batches,
            dim=0,
        ).numpy(),
        dtype=np.float64,
    )

    labels, probabilities = (
        validate_binary_inputs(
            labels,
            probabilities,
        )
    )

    if len(labels) != expected_rows:
        raise RuntimeError(
            f"Unexpected probability row count for "
            f"{split}: {len(labels)} versus "
            f"{expected_rows}"
        )

    audit = {
        "split": split,
        "rows": int(len(labels)),
        "negative_count": int(
            len(labels) - labels.sum()
        ),
        "positive_count": int(
            labels.sum()
        ),
        "probability_minimum": float(
            probabilities.min()
        ),
        "probability_maximum": float(
            probabilities.max()
        ),
        "probability_mean": float(
            probabilities.mean()
        ),
        "probabilities_saved": False,
        "labels_saved": False,
        "patient_identifiers_loaded": False,
        "patient_identifiers_saved": False,
        "locked_test_loaded": False,
        "loader": loader_audit,
    }

    return labels, probabilities, audit


def select_balanced_accuracy_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
    default_threshold: float,
    comparison_tolerance: float,
) -> tuple[float, float]:
    labels, probabilities = (
        validate_binary_inputs(
            labels,
            probabilities,
        )
    )

    positive_probabilities = np.sort(
        probabilities[labels == 1]
    )
    negative_probabilities = np.sort(
        probabilities[labels == 0]
    )

    positive_count = int(
        len(positive_probabilities)
    )
    negative_count = int(
        len(negative_probabilities)
    )

    if positive_count == 0 or negative_count == 0:
        raise ValueError(
            "Threshold selection requires both classes"
        )

    positive_below_threshold = np.searchsorted(
        positive_probabilities,
        thresholds,
        side="left",
    )
    negative_below_threshold = np.searchsorted(
        negative_probabilities,
        thresholds,
        side="left",
    )

    true_positive = (
        positive_count
        - positive_below_threshold
    )
    true_negative = (
        negative_below_threshold
    )

    sensitivity = (
        true_positive / positive_count
    )
    specificity = (
        true_negative / negative_count
    )
    balanced_accuracy = (
        sensitivity + specificity
    ) / 2.0

    maximum_score = float(
        balanced_accuracy.max()
    )

    eligible_indices = np.flatnonzero(
        balanced_accuracy
        >= maximum_score
        - comparison_tolerance
    )

    if len(eligible_indices) == 0:
        raise RuntimeError(
            "No threshold satisfied the selection rule"
        )

    eligible_thresholds = thresholds[
        eligible_indices
    ]
    distances = np.abs(
        eligible_thresholds
        - default_threshold
    )
    minimum_distance = float(
        distances.min()
    )

    distance_indices = eligible_indices[
        distances
        <= minimum_distance
        + comparison_tolerance
    ]

    selected_index = int(
        distance_indices[
            np.argmin(
                thresholds[distance_indices]
            )
        ]
    )

    selected_threshold = float(
        thresholds[selected_index]
    )
    selected_score = float(
        balanced_accuracy[selected_index]
    )

    return (
        selected_threshold,
        selected_score,
    )


def selected_history_epoch(
    history: Mapping[str, Any],
) -> Mapping[str, Any]:
    best = require_mapping(
        history.get("best"),
        "history best",
    )
    best_epoch = int(
        best["epoch"]
    )

    epochs = require_sequence(
        history.get("epochs"),
        "history epochs",
    )

    matches = [
        require_mapping(
            epoch,
            "history epoch",
        )
        for epoch in epochs
        if epoch.get("epoch") == best_epoch
    ]

    if len(matches) != 1:
        raise RuntimeError(
            "Could not uniquely locate the selected "
            "history epoch"
        )

    return matches[0]


def validate_default_metrics_against_history(
    history: Mapping[str, Any],
    split: str,
    observed_metrics: Mapping[str, Any],
) -> None:
    epoch = selected_history_epoch(
        history
    )
    expected_metrics = epoch[
        "validation"
    ][
        "tabular"
    ][
        split
    ][
        "overall"
    ]

    for metric_name in METRIC_NAMES:
        require_close(
            observed=require_finite_number(
                observed_metrics[metric_name],
                f"observed {split} {metric_name}",
            ),
            expected=require_finite_number(
                expected_metrics[metric_name],
                f"history {split} {metric_name}",
            ),
            location=(
                f"default metric reproduction "
                f"{split} {metric_name}"
            ),
            tolerance=1e-5,
        )

    observed_confusion = require_mapping(
        observed_metrics["confusion_matrix"],
        f"observed {split} confusion matrix",
    )
    expected_confusion = require_mapping(
        expected_metrics["confusion_matrix"],
        f"history {split} confusion matrix",
    )

    if dict(observed_confusion) != dict(
        expected_confusion
    ):
        raise RuntimeError(
            f"Default confusion matrix for {split} "
            "does not reproduce the history"
        )


def build_result_record(
    seed: int,
    selected_epoch: int,
    split: str,
    threshold_rule: str,
    threshold_source_split: str,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    if threshold_rule not in THRESHOLD_RULES:
        raise ValueError(
            f"Unknown threshold rule: {threshold_rule}"
        )

    confusion = require_mapping(
        metrics.get("confusion_matrix"),
        "confusion_matrix",
    )

    record = {
        "seed": int(seed),
        "selected_epoch": int(
            selected_epoch
        ),
        "split": split,
        "threshold_rule": threshold_rule,
        "threshold_source_split": (
            threshold_source_split
        ),
        "threshold": require_finite_number(
            metrics.get("threshold"),
            "result threshold",
        ),
    }

    for metric_name in METRIC_NAMES:
        record[metric_name] = (
            require_finite_number(
                metrics.get(metric_name),
                f"result {metric_name}",
            )
        )

    record.update(
        {
            "true_negative": int(
                confusion["true_negative"]
            ),
            "false_positive": int(
                confusion["false_positive"]
            ),
            "false_negative": int(
                confusion["false_negative"]
            ),
            "true_positive": int(
                confusion["true_positive"]
            ),
        }
    )

    return record


def calculate_statistics(
    values: Sequence[float],
) -> dict[str, float | int]:
    if not values:
        raise ValueError(
            "Cannot summarize an empty sequence"
        )

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
    result_rows: Sequence[Mapping[str, Any]],
    selected_thresholds: Mapping[int, float],
) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []

    threshold_values = [
        float(selected_thresholds[seed])
        for seed in sorted(selected_thresholds)
    ]

    summary_rows.append(
        {
            "scope": "threshold",
            "split": "id_val",
            "comparison": "selected_on_id_val",
            "metric": "threshold",
            **calculate_statistics(
                threshold_values
            ),
        }
    )

    for split in (
        "id_val",
        "temporal_val",
    ):
        for threshold_rule in THRESHOLD_RULES:
            matching_rows = [
                row
                for row in result_rows
                if row["split"] == split
                and row["threshold_rule"]
                == threshold_rule
            ]

            if len(matching_rows) != len(
                selected_thresholds
            ):
                raise RuntimeError(
                    "Unexpected result count during "
                    "threshold aggregation"
                )

            for metric_name in METRIC_NAMES:
                values = [
                    float(row[metric_name])
                    for row in matching_rows
                ]

                summary_rows.append(
                    {
                        "scope": "metric",
                        "split": split,
                        "comparison": (
                            threshold_rule
                        ),
                        "metric": metric_name,
                        **calculate_statistics(
                            values
                        ),
                    }
                )

        default_rows = {
            int(row["seed"]): row
            for row in result_rows
            if row["split"] == split
            and row["threshold_rule"]
            == "default"
        }
        selected_rows = {
            int(row["seed"]): row
            for row in result_rows
            if row["split"] == split
            and row["threshold_rule"]
            == "selected_on_id_val"
        }

        if set(default_rows) != set(
            selected_rows
        ):
            raise RuntimeError(
                "Default and selected threshold seed "
                "sets differ"
            )

        for metric_name in METRIC_NAMES:
            changes = [
                float(
                    selected_rows[seed][metric_name]
                )
                - float(
                    default_rows[seed][metric_name]
                )
                for seed in sorted(default_rows)
            ]

            summary_rows.append(
                {
                    "scope": "absolute_change",
                    "split": split,
                    "comparison": (
                        "selected_minus_default"
                    ),
                    "metric": metric_name,
                    **calculate_statistics(
                        changes
                    ),
                }
            )

    return summary_rows


def write_csv_atomic(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

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
    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
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
    scope: str,
    split: str,
    comparison: str,
    metric: str,
) -> Mapping[str, Any]:
    matches = [
        row
        for row in summary_rows
        if row["scope"] == scope
        and row["split"] == split
        and row["comparison"] == comparison
        and row["metric"] == metric
    ]

    if len(matches) != 1:
        raise RuntimeError(
            "Could not uniquely locate threshold "
            f"summary {scope}/{split}/"
            f"{comparison}/{metric}"
        )

    return matches[0]


def main() -> None:
    arguments = parse_arguments()

    joint_config = read_json_object(
        JOINT_CONFIG_PATH
    )
    threshold_config = read_json_object(
        THRESHOLD_CONFIG_PATH
    )

    validate_configuration(
        threshold_config,
        joint_config,
    )

    seeds = tuple(
        int(value)
        for value in threshold_config[
            "random_seeds"
        ]
    )

    selection_config = require_mapping(
        threshold_config[
            "threshold_selection"
        ],
        "threshold_selection",
    )

    fit_split = str(
        selection_config["fit_split"]
    )
    evaluation_split = str(
        selection_config[
            "development_evaluation_split"
        ]
    )
    locked_split = str(
        selection_config["locked_test_split"]
    )

    if locked_split in {
        fit_split,
        evaluation_split,
    }:
        raise RuntimeError(
            "Locked split entered threshold development"
        )

    default_threshold = (
        require_finite_number(
            selection_config[
                "default_threshold"
            ],
            "default threshold",
        )
    )
    comparison_tolerance = (
        require_finite_number(
            selection_config[
                "metric_comparison_tolerance"
            ],
            "threshold comparison tolerance",
        )
    )

    calibration_bins = int(
        joint_config[
            "evaluation"
        ][
            "calibration_bins"
        ]
    )

    candidate_thresholds = (
        build_candidate_thresholds(
            threshold_config
        )
    )

    METRICS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_rows: list[dict[str, Any]] = []
    selected_thresholds: dict[int, float] = {}
    per_seed_audits: list[dict[str, Any]] = []

    for seed in seeds:
        checkpoint_path = (
            CHECKPOINT_DIRECTORY
            / f"joint_seed{seed}_best.pt"
        )
        history_path = (
            METRICS_DIRECTORY
            / f"joint_seed{seed}_history.json"
        )

        context = load_selected_feature_context(
            joint_config_path=(
                JOINT_CONFIG_PATH
            ),
            feature_config_path=(
                FEATURE_CONFIG_PATH
            ),
            preprocessor_state_path=(
                PREPROCESSOR_STATE_PATH
            ),
            checkpoint_path=checkpoint_path,
            history_path=history_path,
            seed=seed,
            requested_device=arguments.device,
        )

        history = read_json_object(
            history_path
        )

        (
            fit_labels,
            fit_probabilities,
            fit_audit,
        ) = collect_tabular_probabilities(
            context,
            fit_split,
        )

        (
            evaluation_labels,
            evaluation_probabilities,
            evaluation_audit,
        ) = collect_tabular_probabilities(
            context,
            evaluation_split,
        )

        (
            selected_threshold,
            selected_fit_score,
        ) = select_balanced_accuracy_threshold(
            labels=fit_labels,
            probabilities=fit_probabilities,
            thresholds=candidate_thresholds,
            default_threshold=default_threshold,
            comparison_tolerance=(
                comparison_tolerance
            ),
        )

        selected_thresholds[seed] = (
            selected_threshold
        )

        split_arrays = {
            fit_split: (
                fit_labels,
                fit_probabilities,
            ),
            evaluation_split: (
                evaluation_labels,
                evaluation_probabilities,
            ),
        }

        seed_results: list[dict[str, Any]] = []

        for split, (
            labels,
            probabilities,
        ) in split_arrays.items():
            threshold_values = {
                "default": default_threshold,
                "selected_on_id_val": (
                    selected_threshold
                ),
            }

            for (
                threshold_rule,
                threshold,
            ) in threshold_values.items():
                metrics = (
                    binary_classification_metrics(
                        labels=labels,
                        probabilities=probabilities,
                        threshold=threshold,
                        calibration_bins=(
                            calibration_bins
                        ),
                    )
                )

                if threshold_rule == "default":
                    validate_default_metrics_against_history(
                        history=history,
                        split=split,
                        observed_metrics=metrics,
                    )

                result = build_result_record(
                    seed=seed,
                    selected_epoch=(
                        context.selected_epoch
                    ),
                    split=split,
                    threshold_rule=(
                        threshold_rule
                    ),
                    threshold_source_split=(
                        "joint_config"
                        if threshold_rule
                        == "default"
                        else fit_split
                    ),
                    metrics=metrics,
                )

                result_rows.append(result)
                seed_results.append(result)

        selected_fit_result = next(
            row
            for row in seed_results
            if row["split"] == fit_split
            and row["threshold_rule"]
            == "selected_on_id_val"
        )

        require_close(
            observed=float(
                selected_fit_result[
                    "balanced_accuracy"
                ]
            ),
            expected=selected_fit_score,
            location=(
                f"seed {seed} selected threshold "
                "balanced accuracy"
            ),
            tolerance=1e-12,
        )

        per_seed_audits.append(
            {
                "seed": seed,
                "selected_epoch": (
                    context.selected_epoch
                ),
                "selected_threshold": (
                    selected_threshold
                ),
                "selected_id_val_balanced_accuracy": (
                    selected_fit_score
                ),
                "checkpoint": (
                    context.checkpoint_audit
                ),
                "splits": {
                    fit_split: fit_audit,
                    evaluation_split: (
                        evaluation_audit
                    ),
                },
                "probabilities_saved": False,
                "labels_saved": False,
                "patient_identifiers_loaded": False,
                "locked_test_loaded": False,
                "locked_test_evaluated": False,
            }
        )

        del fit_labels
        del fit_probabilities
        del evaluation_labels
        del evaluation_probabilities
        del context

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    summary_rows = build_summary_rows(
        result_rows=result_rows,
        selected_thresholds=(
            selected_thresholds
        ),
    )

    results_fieldnames = [
        "seed",
        "selected_epoch",
        "split",
        "threshold_rule",
        "threshold_source_split",
        "threshold",
        *METRIC_NAMES,
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
    ]
    summary_fieldnames = [
        "scope",
        "split",
        "comparison",
        "metric",
        "n",
        "mean",
        "sample_standard_deviation",
        "minimum",
        "maximum",
    ]

    write_csv_atomic(
        RESULTS_OUTPUT_PATH,
        result_rows,
        results_fieldnames,
    )
    write_csv_atomic(
        SUMMARY_OUTPUT_PATH,
        summary_rows,
        summary_fieldnames,
    )

    audit = {
        "version": "1.0",
        "purpose": (
            "Development-only SEER threshold "
            "selection using id_val, with unchanged "
            "application to temporal_val."
        ),
        "random_seeds": list(seeds),
        "threshold_configuration": {
            "filename": (
                THRESHOLD_CONFIG_PATH.name
            ),
            "sha256": sha256_file(
                THRESHOLD_CONFIG_PATH
            ),
        },
        "joint_configuration": {
            "filename": JOINT_CONFIG_PATH.name,
            "sha256": sha256_file(
                JOINT_CONFIG_PATH
            ),
        },
        "feature_configuration": {
            "filename": FEATURE_CONFIG_PATH.name,
            "sha256": sha256_file(
                FEATURE_CONFIG_PATH
            ),
        },
        "candidate_grid": {
            "count": int(
                len(candidate_thresholds)
            ),
            "minimum": float(
                candidate_thresholds.min()
            ),
            "maximum": float(
                candidate_thresholds.max()
            ),
            "step": float(
                candidate_thresholds[1]
                - candidate_thresholds[0]
            ),
        },
        "fit_split": fit_split,
        "development_evaluation_split": (
            evaluation_split
        ),
        "locked_test_split": locked_split,
        "per_seed": per_seed_audits,
        "summary": summary_rows,
        "privacy": {
            "patient_identifiers_loaded": False,
            "patient_level_rows_saved": False,
            "probabilities_saved": False,
            "labels_saved": False,
            "aggregate_metrics_only": True,
        },
        "test_protection": {
            "locked_test_loaded": False,
            "locked_test_evaluated": False,
            "locked_test_used_for_threshold_selection": (
                False
            ),
            "locked_test_used_for_tuning": False,
        },
        "outputs": {
            "results": {
                "path": str(
                    RESULTS_OUTPUT_PATH.relative_to(
                        PROJECT_ROOT
                    )
                ),
                "sha256": sha256_file(
                    RESULTS_OUTPUT_PATH
                ),
            },
            "summary": {
                "path": str(
                    SUMMARY_OUTPUT_PATH.relative_to(
                        PROJECT_ROOT
                    )
                ),
                "sha256": sha256_file(
                    SUMMARY_OUTPUT_PATH
                ),
            },
        },
    }

    write_json_atomic(
        AUDIT_OUTPUT_PATH,
        audit,
    )

    for seed in seeds:
        selected_threshold = (
            selected_thresholds[seed]
        )
        seed_rows = [
            row
            for row in result_rows
            if int(row["seed"]) == seed
        ]
        id_selected = next(
            row
            for row in seed_rows
            if row["split"] == fit_split
            and row["threshold_rule"]
            == "selected_on_id_val"
        )
        temporal_default = next(
            row
            for row in seed_rows
            if row["split"]
            == evaluation_split
            and row["threshold_rule"]
            == "default"
        )
        temporal_selected = next(
            row
            for row in seed_rows
            if row["split"]
            == evaluation_split
            and row["threshold_rule"]
            == "selected_on_id_val"
        )

        print(
            f"seed={seed} "
            f"threshold={selected_threshold:.3f} "
            f"id_val_balanced_accuracy="
            f"{id_selected['balanced_accuracy']:.6f} "
            f"temporal_balanced_accuracy_default="
            f"{temporal_default['balanced_accuracy']:.6f} "
            f"temporal_balanced_accuracy_selected="
            f"{temporal_selected['balanced_accuracy']:.6f} "
            f"temporal_sensitivity_default="
            f"{temporal_default['sensitivity']:.6f} "
            f"temporal_sensitivity_selected="
            f"{temporal_selected['sensitivity']:.6f}"
        )

    threshold_summary = find_summary(
        summary_rows,
        scope="threshold",
        split="id_val",
        comparison="selected_on_id_val",
        metric="threshold",
    )
    balanced_default = find_summary(
        summary_rows,
        scope="metric",
        split=evaluation_split,
        comparison="default",
        metric="balanced_accuracy",
    )
    balanced_selected = find_summary(
        summary_rows,
        scope="metric",
        split=evaluation_split,
        comparison="selected_on_id_val",
        metric="balanced_accuracy",
    )
    sensitivity_default = find_summary(
        summary_rows,
        scope="metric",
        split=evaluation_split,
        comparison="default",
        metric="sensitivity",
    )
    sensitivity_selected = find_summary(
        summary_rows,
        scope="metric",
        split=evaluation_split,
        comparison="selected_on_id_val",
        metric="sensitivity",
    )
    accuracy_default = find_summary(
        summary_rows,
        scope="metric",
        split=evaluation_split,
        comparison="default",
        metric="accuracy",
    )
    accuracy_selected = find_summary(
        summary_rows,
        scope="metric",
        split=evaluation_split,
        comparison="selected_on_id_val",
        metric="accuracy",
    )

    print(
        "selected_threshold_mean="
        f"{threshold_summary['mean']:.6f}"
    )
    print(
        "temporal_balanced_accuracy_default_mean="
        f"{balanced_default['mean']:.6f}"
    )
    print(
        "temporal_balanced_accuracy_selected_mean="
        f"{balanced_selected['mean']:.6f}"
    )
    print(
        "temporal_sensitivity_default_mean="
        f"{sensitivity_default['mean']:.6f}"
    )
    print(
        "temporal_sensitivity_selected_mean="
        f"{sensitivity_selected['mean']:.6f}"
    )
    print(
        "temporal_accuracy_default_mean="
        f"{accuracy_default['mean']:.6f}"
    )
    print(
        "temporal_accuracy_selected_mean="
        f"{accuracy_selected['mean']:.6f}"
    )
    print("locked_temporal_test_loaded=False")
    print(f"results={RESULTS_OUTPUT_PATH}")
    print(f"summary={SUMMARY_OUTPUT_PATH}")
    print(f"audit={AUDIT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
