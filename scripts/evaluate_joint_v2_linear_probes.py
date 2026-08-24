from __future__ import annotations

import csv
import hashlib
import json
import platform
import statistics
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "joint_v2_8d_classical.json"
)
PRIVATE_FEATURE_DIRECTORY = (
    PROJECT_ROOT
    / "artifacts"
    / "features"
    / "joint_v2"
)
PRIVATE_MODEL_DIRECTORY = (
    PRIVATE_FEATURE_DIRECTORY
    / "linear_probes"
)
EXTRACTION_AUDIT_DIRECTORY = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "joint_v2"
    / "feature_extraction"
)
PCA_AUDIT_DIRECTORY = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "joint_v2"
    / "pca"
)
PUBLIC_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "joint_v2"
    / "linear_probe"
)

EXPECTED_SEEDS = (
    42,
    123,
    2026,
)
SUPPORTED_TASKS = (
    "image",
    "tabular",
)
REPRESENTATIONS = {
    "source_32d": 32,
    "pca_8d": 8,
}
SCALAR_METRICS = (
    "roc_auc",
    "average_precision",
    "accuracy",
    "balanced_accuracy",
    "f1",
    "precision",
    "sensitivity",
    "specificity",
    "brier_score",
    "expected_calibration_error",
)

RESULT_FIELDS = (
    "seed",
    "task",
    "representation",
    "dimension",
    "training_rows",
    "split",
    "rows",
    "positive_count",
    "positive_percent",
    "threshold",
    *SCALAR_METRICS,
    "true_negative",
    "false_positive",
    "false_negative",
    "true_positive",
    "iterations",
    "model_filename",
    "model_sha256",
)

COMPARISON_FIELDS = (
    "seed",
    "task",
    "split",
    "rows",
    *tuple(
        field
        for metric in SCALAR_METRICS
        for field in (
            f"{metric}_32d",
            f"{metric}_8d",
            f"{metric}_delta_8d_minus_32d",
        )
    ),
    "roc_auc_retention_ratio",
)

SUMMARY_FIELDS = (
    "summary_type",
    "task",
    "split",
    "representation",
    "metric",
    "seeds",
    "mean",
    "sample_standard_deviation",
)


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


def require_mapping(
    value: Any,
    location: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(
            f"{location} must be an object"
        )

    return value


def require_true(
    value: Any,
    location: str,
) -> None:
    if value is not True:
        raise RuntimeError(
            f"{location} must be true"
        )


def require_false(
    value: Any,
    location: str,
) -> None:
    if value is not False:
        raise RuntimeError(
            f"{location} must be false"
        )


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


def sha256_array(
    array: np.ndarray,
) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(
        memoryview(contiguous).cast("B")
    )

    return digest.hexdigest()


def write_json_atomic(
    path: Path,
    value: Mapping[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
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


def write_csv_atomic(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: tuple[str, ...],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
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


def write_joblib_atomic(
    path: Path,
    value: Mapping[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    joblib.dump(
        value,
        temporary_path,
        compress=3,
    )

    temporary_path.replace(path)


def validate_configuration(
    config: Mapping[str, Any],
) -> None:
    if config.get("version") != "1.0":
        raise RuntimeError(
            "Unexpected linear probe configuration version"
        )

    if config.get("model_name") != (
        "camelyon17_seer_joint_network_v2"
    ):
        raise RuntimeError(
            "Unexpected model name"
        )

    observed_seeds = tuple(
        int(value)
        for value in config.get(
            "random_seeds",
            [],
        )
    )

    if observed_seeds != EXPECTED_SEEDS:
        raise RuntimeError(
            "Linear probe seed contract changed"
        )

    tasks = require_mapping(
        config.get("tasks"),
        "tasks",
    )

    if tuple(tasks.keys()) != SUPPORTED_TASKS:
        raise RuntimeError(
            "Linear probe task contract changed"
        )

    expected_splits = {
        "image": {
            "training": "train",
            "evaluation": (
                "id_val",
                "ood_val",
            ),
            "locked": "ood_test",
        },
        "tabular": {
            "training": "train",
            "evaluation": (
                "id_val",
                "temporal_val",
            ),
            "locked": "temporal_test",
        },
    }

    for task in SUPPORTED_TASKS:
        task_config = require_mapping(
            tasks[task],
            f"tasks.{task}",
        )

        if task_config.get(
            "training_split"
        ) != expected_splits[task]["training"]:
            raise RuntimeError(
                f"{task} training split changed"
            )

        evaluation_splits = tuple(
            str(value)
            for value in task_config.get(
                "evaluation_splits",
                [],
            )
        )

        if evaluation_splits != (
            expected_splits[task]["evaluation"]
        ):
            raise RuntimeError(
                f"{task} evaluation splits changed"
            )

        locked_split = task_config.get(
            "locked_test_split"
        )

        if locked_split != (
            expected_splits[task]["locked"]
        ):
            raise RuntimeError(
                f"{task} locked split changed"
            )

        if locked_split in evaluation_splits:
            raise RuntimeError(
                f"{task} locked split entered evaluation"
            )

    representations = require_mapping(
        config.get("representations"),
        "representations",
    )

    if set(representations) != set(
        REPRESENTATIONS
    ):
        raise RuntimeError(
            "Representation contract changed"
        )

    for name, dimension in (
        REPRESENTATIONS.items()
    ):
        representation_config = require_mapping(
            representations[name],
            f"representations.{name}",
        )

        if int(
            representation_config.get("dimension")
        ) != dimension:
            raise RuntimeError(
                f"{name} dimension changed"
            )

    paired_design = require_mapping(
        config.get("paired_probe_design"),
        "paired_probe_design",
    )
    require_true(
        paired_design.get(
            "train_one_probe_per_seed_task_and_representation"
        ),
        "paired_probe_design.train_one_probe_per_seed_task_and_representation",
    )

    if int(
        paired_design.get(
            "number_of_fitted_probes"
        )
    ) != 12:
        raise RuntimeError(
            "Expected exactly 12 fitted probes"
        )

    require_true(
        paired_design.get(
            "use_identical_rows_and_labels_for_32d_and_8d"
        ),
        "paired_probe_design.use_identical_rows_and_labels_for_32d_and_8d",
    )
    require_true(
        paired_design.get(
            "use_identical_estimator_configuration"
        ),
        "paired_probe_design.use_identical_estimator_configuration",
    )


    if paired_design.get(
        "primary_comparison"
    ) != "pca_8d_minus_source_32d":
        raise RuntimeError(
            "Primary comparison changed"
        )

    require_true(
        paired_design.get(
            "comparison_is_descriptive"
        ),
        "paired_probe_design.comparison_is_descriptive",
    )
    require_true(
        paired_design.get(
            "comparison_is_not_model_selection"
        ),
        "paired_probe_design.comparison_is_not_model_selection",
    )

    estimator = require_mapping(
        config.get("estimator"),
        "estimator",
    )

    if estimator.get("implementation") != (
        "sklearn.linear_model.LogisticRegression"
    ):
        raise RuntimeError(
            "Estimator implementation changed"
        )

    if estimator.get("solver") != "lbfgs":
        raise RuntimeError(
            "Estimator solver changed"
        )

    if estimator.get("penalty") is not None:
        raise RuntimeError(
            "Estimator must remain unregularized"
        )

    if estimator.get("class_weight") is not None:
        raise RuntimeError(
            "Class weighting is not pre-registered"
        )

    require_true(
        estimator.get("fit_intercept"),
        "estimator.fit_intercept",
    )
    require_false(
        estimator.get(
            "additional_feature_scaling"
        ),
        "estimator.additional_feature_scaling",
    )
    require_false(
        estimator.get("hyperparameter_tuning"),
        "estimator.hyperparameter_tuning",
    )
    require_false(
        estimator.get(
            "validation_based_model_selection"
        ),
        "estimator.validation_based_model_selection",
    )

    classification = require_mapping(
        config.get("classification"),
        "classification",
    )

    if float(
        classification.get("default_threshold")
    ) != 0.5:
        raise RuntimeError(
            "Classification threshold changed"
        )

    require_false(
        classification.get("threshold_tuning"),
        "classification.threshold_tuning",
    )
    require_false(
        classification.get(
            "probability_calibration"
        ),
        "classification.probability_calibration",
    )
    require_false(
        classification.get(
            "probability_clipping"
        ),
        "classification.probability_clipping",
    )
    require_false(
        classification.get(
            "probability_rounding"
        ),
        "classification.probability_rounding",
    )

    calibration = require_mapping(
        config.get("calibration_metrics"),
        "calibration_metrics",
    )

    if int(
        calibration.get(
            "expected_calibration_error_bins"
        )
    ) != 15:
        raise RuntimeError(
            "ECE bin count changed"
        )

    if calibration.get(
        "expected_calibration_error_binning"
    ) != "uniform":
        raise RuntimeError(
            "ECE binning must remain uniform"
        )

    validation = require_mapping(
        config.get("validation"),
        "validation",
    )
    require_true(
        validation.get(
            "require_row_count_match_between_32d_and_8d"
        ),
        "validation.require_row_count_match_between_32d_and_8d",
    )
    require_true(
        validation.get(
            "require_label_array_match_between_32d_and_8d"
        ),
        "validation.require_label_array_match_between_32d_and_8d",
    )
    require_true(
        validation.get(
            "require_label_hash_match_between_32d_and_8d"
        ),
        "validation.require_label_hash_match_between_32d_and_8d",
    )
    require_true(
        validation.get(
            "fail_on_convergence_warning"
        ),
        "validation.fail_on_convergence_warning",
    )

    privacy = require_mapping(
        config.get("privacy"),
        "privacy",
    )

    for key in (
        "patient_identifiers_allowed_in_memory",
        "patient_identifiers_allowed_in_outputs",
        "individual_seer_rows_allowed_in_public_outputs",
    ):
        require_false(
            privacy.get(key),
            f"privacy.{key}",
        )

    protection = require_mapping(
        config.get("test_protection"),
        "test_protection",
    )

    for key in (
        "allow_camelyon_ood_test_loading",
        "allow_seer_temporal_test_loading",
        "allow_locked_test_evaluation_at_this_stage",
        "allow_locked_test_tuning",
    ):
        require_false(
            protection.get(key),
            f"test_protection.{key}",
        )


def split_contract(
    config: Mapping[str, Any],
    task: str,
) -> tuple[str, tuple[str, ...]]:
    task_config = require_mapping(
        config["tasks"][task],
        f"tasks.{task}",
    )
    training_split = str(
        task_config["training_split"]
    )
    evaluation_splits = tuple(
        str(value)
        for value in task_config[
            "evaluation_splits"
        ]
    )
    locked_split = str(
        task_config["locked_test_split"]
    )

    if locked_split in (
        training_split,
        *evaluation_splits,
    ):
        raise RuntimeError(
            f"{task} locked split entered the probe contract"
        )

    return training_split, evaluation_splits


def feature_path(
    seed: int,
    task: str,
    split: str,
    representation: str,
) -> Path:
    suffix = (
        "32d"
        if representation == "source_32d"
        else "8d"
    )

    return (
        PRIVATE_FEATURE_DIRECTORY
        / f"seed{seed}"
        / f"{task}_{split}_{suffix}.npz"
    )


def validate_archive_arrays(
    features: np.ndarray,
    labels: np.ndarray,
    expected_dimension: int,
    location: str,
) -> None:
    if features.ndim != 2:
        raise ValueError(
            f"{location} features must be two-dimensional"
        )

    if features.shape[1] != expected_dimension:
        raise ValueError(
            f"{location} has the wrong feature dimension"
        )

    if labels.ndim != 1:
        raise ValueError(
            f"{location} labels must be one-dimensional"
        )

    if features.shape[0] != labels.shape[0]:
        raise ValueError(
            f"{location} feature and label rows differ"
        )

    if features.shape[0] == 0:
        raise ValueError(
            f"{location} is empty"
        )

    if not np.isfinite(features).all():
        raise ValueError(
            f"{location} contains non-finite features"
        )

    observed_labels = set(
        int(value)
        for value in np.unique(labels)
    )

    if observed_labels != {0, 1}:
        raise ValueError(
            f"{location} must contain both binary classes"
        )


def load_source_32d(
    seed: int,
    task: str,
    split: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    path = feature_path(
        seed,
        task,
        split,
        "source_32d",
    )
    audit_path = (
        EXTRACTION_AUDIT_DIRECTORY
        / f"seed{seed}_{task}_{split}_32d_audit.json"
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing 32D archive: {path}"
        )

    audit = read_json_object(audit_path)

    if int(audit.get("seed")) != seed:
        raise RuntimeError(
            "32D audit seed mismatch"
        )

    if audit.get("task") != task:
        raise RuntimeError(
            "32D audit task mismatch"
        )

    if audit.get("split") != split:
        raise RuntimeError(
            "32D audit split mismatch"
        )

    privacy = require_mapping(
        audit.get("privacy"),
        "32D audit privacy",
    )
    require_false(
        privacy.get(
            "patient_identifiers_loaded"
        ),
        "32D audit patient_identifiers_loaded",
    )
    require_false(
        privacy.get(
            "patient_identifiers_in_output"
        ),
        "32D audit patient_identifiers_in_output",
    )

    protection = require_mapping(
        audit.get("test_protection"),
        "32D audit test_protection",
    )
    require_false(
        protection.get("locked_test_loaded"),
        "32D audit locked_test_loaded",
    )
    require_false(
        protection.get("locked_test_evaluated"),
        "32D audit locked_test_evaluated",
    )

    private_output = require_mapping(
        audit.get("private_output"),
        "32D audit private_output",
    )

    actual_file_hash = sha256_file(path)

    if actual_file_hash != (
        private_output.get("file_sha256")
    ):
        raise RuntimeError(
            "32D archive file hash mismatch"
        )

    with np.load(
        path,
        allow_pickle=False,
    ) as archive:
        if set(archive.files) != {
            "features",
            "labels",
        }:
            raise RuntimeError(
                "32D archive contains unexpected arrays"
            )

        features = np.ascontiguousarray(
            archive["features"],
            dtype=np.float32,
        )
        labels = np.ascontiguousarray(
            archive["labels"],
            dtype=np.uint8,
        )

    validate_archive_arrays(
        features,
        labels,
        32,
        f"seed={seed} task={task} split={split} 32D",
    )

    extraction = require_mapping(
        audit.get("extraction"),
        "32D audit extraction",
    )
    output = require_mapping(
        extraction.get("output"),
        "32D audit extraction.output",
    )

    features_hash = sha256_array(features)
    labels_hash = sha256_array(labels)

    if features_hash != output.get(
        "features_sha256"
    ):
        raise RuntimeError(
            "32D feature array hash mismatch"
        )

    if labels_hash != output.get(
        "labels_sha256"
    ):
        raise RuntimeError(
            "32D label array hash mismatch"
        )

    record = {
        "seed": seed,
        "task": task,
        "split": split,
        "representation": "source_32d",
        "rows": int(features.shape[0]),
        "dimension": 32,
        "archive_filename": path.name,
        "archive_sha256": actual_file_hash,
        "features_sha256": features_hash,
        "labels_sha256": labels_hash,
        "patient_identifiers_loaded": False,
        "locked_test_loaded": False,
    }

    return features, labels, record


def load_pca_8d(
    seed: int,
    task: str,
    split: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    path = feature_path(
        seed,
        task,
        split,
        "pca_8d",
    )
    audit_path = (
        PCA_AUDIT_DIRECTORY
        / f"seed{seed}_{task}_pca_audit.json"
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing 8D archive: {path}"
        )

    audit = read_json_object(audit_path)

    if int(audit.get("seed")) != seed:
        raise RuntimeError(
            "PCA audit seed mismatch"
        )

    if audit.get("task") != task:
        raise RuntimeError(
            "PCA audit task mismatch"
        )

    if int(audit.get("source_dimension")) != 32:
        raise RuntimeError(
            "PCA source dimension mismatch"
        )

    if int(audit.get("output_dimension")) != 8:
        raise RuntimeError(
            "PCA output dimension mismatch"
        )

    privacy = require_mapping(
        audit.get("privacy"),
        "PCA audit privacy",
    )
    require_false(
        privacy.get(
            "patient_identifiers_loaded"
        ),
        "PCA audit patient_identifiers_loaded",
    )
    require_false(
        privacy.get(
            "patient_identifiers_in_output"
        ),
        "PCA audit patient_identifiers_in_output",
    )

    protection = require_mapping(
        audit.get("test_protection"),
        "PCA audit test_protection",
    )

    for key in (
        "locked_tests_loaded",
        "locked_tests_transformed",
        "locked_tests_used_for_scaler_fit",
        "locked_tests_used_for_pca_fit",
    ):
        require_false(
            protection.get(key),
            f"PCA audit test_protection.{key}",
        )

    split_outputs = require_mapping(
        audit.get("split_outputs"),
        "PCA audit split_outputs",
    )

    if split not in split_outputs:
        raise RuntimeError(
            f"PCA audit does not contain split {split}"
        )

    split_output = require_mapping(
        split_outputs[split],
        f"PCA split_outputs.{split}",
    )
    summary = require_mapping(
        split_output.get("summary"),
        f"PCA split_outputs.{split}.summary",
    )

    require_true(
        summary.get("label_hash_preserved"),
        "PCA summary.label_hash_preserved",
    )
    require_true(
        summary.get("row_order_preserved"),
        "PCA summary.row_order_preserved",
    )
    require_false(
        summary.get(
            "patient_identifiers_in_output"
        ),
        "PCA summary.patient_identifiers_in_output",
    )
    require_false(
        summary.get("locked_test_in_output"),
        "PCA summary.locked_test_in_output",
    )

    actual_file_hash = sha256_file(path)

    if actual_file_hash != (
        split_output.get("archive_sha256")
    ):
        raise RuntimeError(
            "8D archive file hash mismatch"
        )

    with np.load(
        path,
        allow_pickle=False,
    ) as archive:
        if set(archive.files) != {
            "features",
            "labels",
        }:
            raise RuntimeError(
                "8D archive contains unexpected arrays"
            )

        features = np.ascontiguousarray(
            archive["features"],
            dtype=np.float32,
        )
        labels = np.ascontiguousarray(
            archive["labels"],
            dtype=np.uint8,
        )

    validate_archive_arrays(
        features,
        labels,
        8,
        f"seed={seed} task={task} split={split} 8D",
    )

    features_hash = sha256_array(features)
    labels_hash = sha256_array(labels)

    if features_hash != summary.get(
        "features_sha256"
    ):
        raise RuntimeError(
            "8D feature array hash mismatch"
        )

    if labels_hash != summary.get(
        "labels_sha256"
    ):
        raise RuntimeError(
            "8D label array hash mismatch"
        )

    record = {
        "seed": seed,
        "task": task,
        "split": split,
        "representation": "pca_8d",
        "rows": int(features.shape[0]),
        "dimension": 8,
        "archive_filename": path.name,
        "archive_sha256": actual_file_hash,
        "features_sha256": features_hash,
        "labels_sha256": labels_hash,
        "patient_identifiers_loaded": False,
        "locked_test_loaded": False,
    }

    return features, labels, record


def validate_representation_pair(
    features_32d: np.ndarray,
    labels_32d: np.ndarray,
    features_8d: np.ndarray,
    labels_8d: np.ndarray,
    seed: int,
    task: str,
    split: str,
) -> None:
    if features_32d.shape[0] != (
        features_8d.shape[0]
    ):
        raise RuntimeError(
            f"Row mismatch for seed={seed} "
            f"task={task} split={split}"
        )

    if labels_32d.shape != labels_8d.shape:
        raise RuntimeError(
            f"Label shape mismatch for seed={seed} "
            f"task={task} split={split}"
        )

    if not np.array_equal(
        labels_32d,
        labels_8d,
    ):
        raise RuntimeError(
            f"Label order mismatch for seed={seed} "
            f"task={task} split={split}"
        )

    if sha256_array(labels_32d) != (
        sha256_array(labels_8d)
    ):
        raise RuntimeError(
            f"Label hash mismatch for seed={seed} "
            f"task={task} split={split}"
        )


def build_estimator(
    config: Mapping[str, Any],
) -> LogisticRegression:
    estimator = require_mapping(
        config["estimator"],
        "estimator",
    )

    return LogisticRegression(
        solver=str(estimator["solver"]),
        penalty=None,
        fit_intercept=bool(
            estimator["fit_intercept"]
        ),
        class_weight=None,
        max_iter=int(
            estimator["max_iter"]
        ),
        tol=float(
            estimator["tolerance"]
        ),
        warm_start=bool(
            estimator["warm_start"]
        ),
    )


def fit_estimator(
    model: LogisticRegression,
    features: np.ndarray,
    labels: np.ndarray,
    location: str,
) -> int:
    with warnings.catch_warnings(
        record=True
    ) as caught_warnings:
        warnings.simplefilter(
            "always",
            ConvergenceWarning,
        )
        model.fit(features, labels)

    convergence_warnings = [
        warning
        for warning in caught_warnings
        if issubclass(
            warning.category,
            ConvergenceWarning,
        )
    ]

    if convergence_warnings:
        messages = "; ".join(
            str(warning.message)
            for warning in convergence_warnings
        )
        raise RuntimeError(
            f"Logistic regression did not converge "
            f"for {location}: {messages}"
        )

    if not np.array_equal(
        model.classes_,
        np.array([0, 1]),
    ):
        raise RuntimeError(
            f"Unexpected fitted classes for {location}"
        )

    if model.coef_.shape != (
        1,
        features.shape[1],
    ):
        raise RuntimeError(
            f"Unexpected coefficient shape for {location}"
        )

    if model.intercept_.shape != (1,):
        raise RuntimeError(
            f"Unexpected intercept shape for {location}"
        )

    if not np.isfinite(model.coef_).all():
        raise RuntimeError(
            f"Non-finite coefficients for {location}"
        )

    if not np.isfinite(
        model.intercept_
    ).all():
        raise RuntimeError(
            f"Non-finite intercept for {location}"
        )

    return int(model.n_iter_[0])


def uniform_expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    number_of_bins: int,
) -> float:
    edges = np.linspace(
        0.0,
        1.0,
        number_of_bins + 1,
        dtype=np.float64,
    )
    bin_indices = np.searchsorted(
        edges[1:-1],
        probabilities,
        side="right",
    )

    calibration_error = 0.0
    sample_count = len(labels)

    for bin_index in range(
        number_of_bins
    ):
        mask = bin_indices == bin_index
        count = int(mask.sum())

        if count == 0:
            continue

        mean_probability = float(
            probabilities[mask].mean()
        )
        observed_frequency = float(
            labels[mask].mean()
        )

        calibration_error += (
            count
            / sample_count
            * abs(
                observed_frequency
                - mean_probability
            )
        )

    return float(calibration_error)


def evaluate_estimator(
    model: LogisticRegression,
    features: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    calibration_bins: int,
) -> dict[str, Any]:
    positive_index_array = np.flatnonzero(
        model.classes_ == 1
    )

    if positive_index_array.shape != (1,):
        raise RuntimeError(
            "Unable to identify positive-class probability"
        )

    probabilities = np.asarray(
        model.predict_proba(features)[
            :,
            int(positive_index_array[0]),
        ],
        dtype=np.float64,
    )

    if probabilities.ndim != 1:
        raise RuntimeError(
            "Probabilities must be one-dimensional"
        )

    if len(probabilities) != len(labels):
        raise RuntimeError(
            "Probability and label rows differ"
        )

    if not np.isfinite(probabilities).all():
        raise RuntimeError(
            "Probabilities contain non-finite values"
        )

    if (
        probabilities.min() < 0.0
        or probabilities.max() > 1.0
    ):
        raise RuntimeError(
            "Probabilities are outside [0, 1]"
        )

    predictions = (
        probabilities >= threshold
    ).astype(np.uint8)

    (
        true_negative,
        false_positive,
        false_negative,
        true_positive,
    ) = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "rows": int(len(labels)),
        "positive_count": int(labels.sum()),
        "positive_percent": float(
            100.0 * labels.mean()
        ),
        "threshold": float(threshold),
        "roc_auc": float(
            roc_auc_score(
                labels,
                probabilities,
            )
        ),
        "average_precision": float(
            average_precision_score(
                labels,
                probabilities,
            )
        ),
        "accuracy": float(
            accuracy_score(
                labels,
                predictions,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                labels,
                predictions,
            )
        ),
        "f1": float(
            f1_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "precision": float(
            precision_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "sensitivity": float(
            recall_score(
                labels,
                predictions,
                pos_label=1,
                zero_division=0,
            )
        ),
        "specificity": float(
            recall_score(
                labels,
                predictions,
                pos_label=0,
                zero_division=0,
            )
        ),
        "brier_score": float(
            brier_score_loss(
                labels,
                probabilities,
            )
        ),
        "expected_calibration_error": (
            uniform_expected_calibration_error(
                labels,
                probabilities,
                calibration_bins,
            )
        ),
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "true_positive": int(true_positive),
    }


def save_fitted_model(
    model: LogisticRegression,
    seed: int,
    task: str,
    representation: str,
    dimension: int,
    training_rows: int,
    training_labels_sha256: str,
    configuration_sha256: str,
) -> tuple[Path, str]:
    path = (
        PRIVATE_MODEL_DIRECTORY
        / (
            f"seed{seed}_{task}_{representation}"
            "_logistic_probe.joblib"
        )
    )

    state = {
        "version": "1.0",
        "seed": seed,
        "task": task,
        "representation": representation,
        "dimension": dimension,
        "training_rows": training_rows,
        "training_labels_sha256": (
            training_labels_sha256
        ),
        "configuration_sha256": (
            configuration_sha256
        ),
        "model": model,
    }

    write_joblib_atomic(path, state)

    loaded = joblib.load(path)

    if not isinstance(loaded, dict):
        raise RuntimeError(
            "Saved probe state is not a dictionary"
        )

    if loaded.get("seed") != seed:
        raise RuntimeError(
            "Saved probe seed mismatch"
        )

    if loaded.get("task") != task:
        raise RuntimeError(
            "Saved probe task mismatch"
        )

    if loaded.get(
        "representation"
    ) != representation:
        raise RuntimeError(
            "Saved probe representation mismatch"
        )

    loaded_model = loaded.get("model")

    if not isinstance(
        loaded_model,
        LogisticRegression,
    ):
        raise RuntimeError(
            "Saved probe model type mismatch"
        )

    return path, sha256_file(path)


def build_comparisons(
    results: list[dict[str, Any]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    lookup = {
        (
            int(row["seed"]),
            str(row["task"]),
            str(row["split"]),
            str(row["representation"]),
        ): row
        for row in results
    }

    comparison_rows: list[
        dict[str, Any]
    ] = []

    for seed in EXPECTED_SEEDS:
        for task in SUPPORTED_TASKS:
            _, evaluation_splits = (
                split_contract(
                    config,
                    task,
                )
            )

            for split in evaluation_splits:
                result_32d = lookup[
                    (
                        seed,
                        task,
                        split,
                        "source_32d",
                    )
                ]
                result_8d = lookup[
                    (
                        seed,
                        task,
                        split,
                        "pca_8d",
                    )
                ]

                if int(
                    result_32d["rows"]
                ) != int(result_8d["rows"]):
                    raise RuntimeError(
                        "Comparison row count mismatch"
                    )

                row: dict[str, Any] = {
                    "seed": seed,
                    "task": task,
                    "split": split,
                    "rows": int(
                        result_32d["rows"]
                    ),
                }

                for metric in SCALAR_METRICS:
                    value_32d = float(
                        result_32d[metric]
                    )
                    value_8d = float(
                        result_8d[metric]
                    )

                    row[f"{metric}_32d"] = (
                        value_32d
                    )
                    row[f"{metric}_8d"] = (
                        value_8d
                    )
                    row[
                        f"{metric}_delta_8d_minus_32d"
                    ] = (
                        value_8d
                        - value_32d
                    )

                roc_auc_32d = float(
                    result_32d["roc_auc"]
                )

                if roc_auc_32d <= 0.0:
                    raise RuntimeError(
                        "Cannot calculate ROC-AUC retention"
                    )

                row[
                    "roc_auc_retention_ratio"
                ] = (
                    float(
                        result_8d["roc_auc"]
                    )
                    / roc_auc_32d
                )

                comparison_rows.append(row)

    return comparison_rows


def mean_and_sample_standard_deviation(
    values: list[float],
) -> tuple[float, float]:
    if len(values) != len(EXPECTED_SEEDS):
        raise RuntimeError(
            "Summary does not contain all three seeds"
        )

    return (
        float(statistics.mean(values)),
        float(statistics.stdev(values)),
    )


def build_summary(
    results: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    summary_rows: list[
        dict[str, Any]
    ] = []

    for task in SUPPORTED_TASKS:
        _, evaluation_splits = (
            split_contract(
                config,
                task,
            )
        )

        for split in evaluation_splits:
            for representation in (
                REPRESENTATIONS
            ):
                selected = [
                    row
                    for row in results
                    if (
                        row["task"] == task
                        and row["split"] == split
                        and row["representation"]
                        == representation
                    )
                ]

                for metric in SCALAR_METRICS:
                    values = [
                        float(row[metric])
                        for row in selected
                    ]
                    mean_value, standard_deviation = (
                        mean_and_sample_standard_deviation(
                            values
                        )
                    )

                    summary_rows.append(
                        {
                            "summary_type": (
                                "representation_metric"
                            ),
                            "task": task,
                            "split": split,
                            "representation": (
                                representation
                            ),
                            "metric": metric,
                            "seeds": len(values),
                            "mean": mean_value,
                            "sample_standard_deviation": (
                                standard_deviation
                            ),
                        }
                    )

            selected_comparisons = [
                row
                for row in comparisons
                if (
                    row["task"] == task
                    and row["split"] == split
                )
            ]

            for metric in SCALAR_METRICS:
                comparison_key = (
                    f"{metric}_delta_8d_minus_32d"
                )
                values = [
                    float(row[comparison_key])
                    for row in selected_comparisons
                ]
                mean_value, standard_deviation = (
                    mean_and_sample_standard_deviation(
                        values
                    )
                )

                summary_rows.append(
                    {
                        "summary_type": (
                            "comparison_delta"
                        ),
                        "task": task,
                        "split": split,
                        "representation": (
                            "pca_8d_minus_source_32d"
                        ),
                        "metric": metric,
                        "seeds": len(values),
                        "mean": mean_value,
                        "sample_standard_deviation": (
                            standard_deviation
                        ),
                    }
                )

            retention_values = [
                float(
                    row[
                        "roc_auc_retention_ratio"
                    ]
                )
                for row in selected_comparisons
            ]
            retention_mean, retention_sd = (
                mean_and_sample_standard_deviation(
                    retention_values
                )
            )

            summary_rows.append(
                {
                    "summary_type": (
                        "comparison_ratio"
                    ),
                    "task": task,
                    "split": split,
                    "representation": (
                        "pca_8d_divided_by_source_32d"
                    ),
                    "metric": (
                        "roc_auc_retention_ratio"
                    ),
                    "seeds": len(
                        retention_values
                    ),
                    "mean": retention_mean,
                    "sample_standard_deviation": (
                        retention_sd
                    ),
                }
            )

    return summary_rows


def find_summary_value(
    summary_rows: list[dict[str, Any]],
    summary_type: str,
    task: str,
    split: str,
    representation: str,
    metric: str,
) -> float:
    matches = [
        row
        for row in summary_rows
        if (
            row["summary_type"] == summary_type
            and row["task"] == task
            and row["split"] == split
            and row["representation"]
            == representation
            and row["metric"] == metric
        )
    ]

    if len(matches) != 1:
        raise RuntimeError(
            "Unable to identify one summary value"
        )

    return float(matches[0]["mean"])


def main() -> None:
    config = read_json_object(CONFIG_PATH)
    validate_configuration(config)

    configuration_sha256 = sha256_file(
        CONFIG_PATH
    )
    threshold = float(
        config["classification"][
            "default_threshold"
        ]
    )
    calibration_bins = int(
        config["calibration_metrics"][
            "expected_calibration_error_bins"
        ]
    )

    PRIVATE_MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )
    PUBLIC_OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    archive_records: list[
        dict[str, Any]
    ] = []
    fitted_model_records: list[
        dict[str, Any]
    ] = []
    result_rows: list[
        dict[str, Any]
    ] = []

    for seed in EXPECTED_SEEDS:
        for task in SUPPORTED_TASKS:
            training_split, evaluation_splits = (
                split_contract(
                    config,
                    task,
                )
            )
            splits = (
                training_split,
                *evaluation_splits,
            )

            features_by_representation: dict[
                str,
                dict[str, np.ndarray],
            ] = {
                "source_32d": {},
                "pca_8d": {},
            }
            labels_by_representation: dict[
                str,
                dict[str, np.ndarray],
            ] = {
                "source_32d": {},
                "pca_8d": {},
            }
            records_by_representation: dict[
                str,
                dict[str, dict[str, Any]],
            ] = {
                "source_32d": {},
                "pca_8d": {},
            }

            for split in splits:
                (
                    features_32d,
                    labels_32d,
                    record_32d,
                ) = load_source_32d(
                    seed,
                    task,
                    split,
                )
                (
                    features_8d,
                    labels_8d,
                    record_8d,
                ) = load_pca_8d(
                    seed,
                    task,
                    split,
                )

                validate_representation_pair(
                    features_32d,
                    labels_32d,
                    features_8d,
                    labels_8d,
                    seed,
                    task,
                    split,
                )

                features_by_representation[
                    "source_32d"
                ][split] = features_32d
                features_by_representation[
                    "pca_8d"
                ][split] = features_8d
                labels_by_representation[
                    "source_32d"
                ][split] = labels_32d
                labels_by_representation[
                    "pca_8d"
                ][split] = labels_8d
                records_by_representation[
                    "source_32d"
                ][split] = record_32d
                records_by_representation[
                    "pca_8d"
                ][split] = record_8d

                archive_records.extend(
                    [
                        record_32d,
                        record_8d,
                    ]
                )

            for (
                representation,
                dimension,
            ) in REPRESENTATIONS.items():
                training_features = (
                    features_by_representation[
                        representation
                    ][training_split]
                )
                training_labels = (
                    labels_by_representation[
                        representation
                    ][training_split]
                )

                model = build_estimator(config)
                location = (
                    f"seed={seed} task={task} "
                    f"representation={representation}"
                )
                iterations = fit_estimator(
                    model,
                    training_features,
                    training_labels,
                    location,
                )

                training_labels_sha256 = (
                    records_by_representation[
                        representation
                    ][training_split][
                        "labels_sha256"
                    ]
                )

                model_path, model_hash = (
                    save_fitted_model(
                        model=model,
                        seed=seed,
                        task=task,
                        representation=(
                            representation
                        ),
                        dimension=dimension,
                        training_rows=int(
                            training_features.shape[0]
                        ),
                        training_labels_sha256=(
                            training_labels_sha256
                        ),
                        configuration_sha256=(
                            configuration_sha256
                        ),
                    )
                )

                fitted_model_records.append(
                    {
                        "seed": seed,
                        "task": task,
                        "representation": (
                            representation
                        ),
                        "dimension": dimension,
                        "training_split": (
                            training_split
                        ),
                        "training_rows": int(
                            training_features.shape[0]
                        ),
                        "negative_count": int(
                            len(training_labels)
                            - training_labels.sum()
                        ),
                        "positive_count": int(
                            training_labels.sum()
                        ),
                        "iterations": iterations,
                        "model_filename": (
                            model_path.name
                        ),
                        "model_sha256": (
                            model_hash
                        ),
                        "model_committed": False,
                        "coefficients_in_public_output": (
                            False
                        ),
                    }
                )

                for split in evaluation_splits:
                    metrics = evaluate_estimator(
                        model=model,
                        features=(
                            features_by_representation[
                                representation
                            ][split]
                        ),
                        labels=(
                            labels_by_representation[
                                representation
                            ][split]
                        ),
                        threshold=threshold,
                        calibration_bins=(
                            calibration_bins
                        ),
                    )

                    result_rows.append(
                        {
                            "seed": seed,
                            "task": task,
                            "representation": (
                                representation
                            ),
                            "dimension": dimension,
                            "training_rows": int(
                                training_features.shape[0]
                            ),
                            "split": split,
                            **metrics,
                            "iterations": iterations,
                            "model_filename": (
                                model_path.name
                            ),
                            "model_sha256": (
                                model_hash
                            ),
                        }
                    )

                    print(
                        f"seed={seed} "
                        f"task={task} "
                        f"representation={representation} "
                        f"split={split} "
                        f"roc_auc={metrics['roc_auc']:.6f} "
                        f"accuracy={metrics['accuracy']:.6f}"
                    )

    if len(fitted_model_records) != 12:
        raise RuntimeError(
            "Expected 12 fitted probe records"
        )

    if len(result_rows) != 24:
        raise RuntimeError(
            "Expected 24 development result rows"
        )

    comparison_rows = build_comparisons(
        result_rows,
        config,
    )
    summary_rows = build_summary(
        result_rows,
        comparison_rows,
        config,
    )

    results_path = (
        PUBLIC_OUTPUT_DIRECTORY
        / "joint_v2_linear_probe_results.csv"
    )
    comparisons_path = (
        PUBLIC_OUTPUT_DIRECTORY
        / "joint_v2_linear_probe_comparisons.csv"
    )
    summary_path = (
        PUBLIC_OUTPUT_DIRECTORY
        / "joint_v2_linear_probe_summary.csv"
    )
    audit_path = (
        PUBLIC_OUTPUT_DIRECTORY
        / "joint_v2_linear_probe_audit.json"
    )

    write_csv_atomic(
        results_path,
        result_rows,
        RESULT_FIELDS,
    )
    write_csv_atomic(
        comparisons_path,
        comparison_rows,
        COMPARISON_FIELDS,
    )
    write_csv_atomic(
        summary_path,
        summary_rows,
        SUMMARY_FIELDS,
    )

    key_summary = {}

    for task, split in (
        ("image", "ood_val"),
        ("tabular", "temporal_val"),
    ):
        auc_32d = find_summary_value(
            summary_rows,
            "representation_metric",
            task,
            split,
            "source_32d",
            "roc_auc",
        )
        auc_8d = find_summary_value(
            summary_rows,
            "representation_metric",
            task,
            split,
            "pca_8d",
            "roc_auc",
        )
        auc_delta = find_summary_value(
            summary_rows,
            "comparison_delta",
            task,
            split,
            "pca_8d_minus_source_32d",
            "roc_auc",
        )
        retention_ratio = find_summary_value(
            summary_rows,
            "comparison_ratio",
            task,
            split,
            "pca_8d_divided_by_source_32d",
            "roc_auc_retention_ratio",
        )

        key_summary[f"{task}_{split}"] = {
            "source_32d_roc_auc_mean": auc_32d,
            "pca_8d_roc_auc_mean": auc_8d,
            "roc_auc_delta_8d_minus_32d_mean": (
                auc_delta
            ),
            "roc_auc_retention_ratio_mean": (
                retention_ratio
            ),
        }

    audit = {
        "version": "1.0",
        "experiment_name": (
            "joint_v2_paired_linear_probe"
        ),
        "configuration": {
            "filename": CONFIG_PATH.name,
            "sha256": configuration_sha256,
        },
        "design": {
            "random_seeds": list(
                EXPECTED_SEEDS
            ),
            "tasks": list(
                SUPPORTED_TASKS
            ),
            "representations": (
                REPRESENTATIONS
            ),
            "fitted_probes": len(
                fitted_model_records
            ),
            "development_result_rows": len(
                result_rows
            ),
            "comparison_rows": len(
                comparison_rows
            ),
            "hyperparameter_tuning_used": False,
            "threshold_tuning_used": False,
            "probability_calibration_used": False,
            "additional_feature_scaling_used": False,
        },
        "archive_validation": archive_records,
        "fitted_models": fitted_model_records,
        "key_summary": key_summary,
        "public_outputs": {
            "results": {
                "filename": results_path.name,
                "sha256": sha256_file(
                    results_path
                ),
            },
            "comparisons": {
                "filename": (
                    comparisons_path.name
                ),
                "sha256": sha256_file(
                    comparisons_path
                ),
            },
            "summary": {
                "filename": summary_path.name,
                "sha256": sha256_file(
                    summary_path
                ),
            },
        },
        "privacy": {
            "patient_identifiers_loaded": False,
            "patient_identifiers_in_output": False,
            "individual_seer_rows_in_output": False,
            "feature_values_in_public_output": False,
            "labels_in_public_output": False,
            "probabilities_in_public_output": False,
            "predictions_in_public_output": False,
            "coefficients_in_public_output": False,
            "fitted_models_committed": False,
        },
        "test_protection": {
            "camelyon_ood_test_loaded": False,
            "seer_temporal_test_loaded": False,
            "locked_tests_evaluated": False,
            "locked_tests_used_for_tuning": False,
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "interpretation": {
            "comparison_is_descriptive": True,
            "explained_variance_is_not_used_as_predictive_performance": True,
            "linear_probe_is_not_a_replacement_for_the_joint_model": True,
            "linear_probe_is_the_classical_8d_reference": True,
            "quantum_implementation_in_current_scope": False,
        },
    }

    write_json_atomic(
        audit_path,
        audit,
    )

    for task, split in (
        ("image", "ood_val"),
        ("tabular", "temporal_val"),
    ):
        summary = key_summary[
            f"{task}_{split}"
        ]
        print(
            f"task={task} "
            f"split={split} "
            f"source_32d_roc_auc_mean="
            f"{summary['source_32d_roc_auc_mean']:.6f} "
            f"pca_8d_roc_auc_mean="
            f"{summary['pca_8d_roc_auc_mean']:.6f} "
            f"delta="
            f"{summary['roc_auc_delta_8d_minus_32d_mean']:.6f} "
            f"retention_ratio="
            f"{summary['roc_auc_retention_ratio_mean']:.6f}"
        )

    print("fitted_probes=12")
    print("locked_tests_loaded=False")
    print("patient_identifiers_in_output=False")
    print(f"results={results_path}")
    print(f"comparisons={comparisons_path}")
    print(f"summary={summary_path}")
    print(f"audit={audit_path}")


if __name__ == "__main__":
    main()
