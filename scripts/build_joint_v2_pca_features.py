from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.joint_feature_extraction import (
    sha256_array,
)


PCA_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "joint_v2_pca.json"
)
JOINT_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "joint_model_v2.json"
)
PRIVATE_FEATURE_DIRECTORY = (
    PROJECT_ROOT
    / "artifacts"
    / "features"
    / "joint_v2"
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

EXPECTED_SEEDS = (
    42,
    123,
    2026,
)
SUPPORTED_TASKS = (
    "image",
    "tabular",
)
SOURCE_DIMENSION = 32
OUTPUT_DIMENSION = 8


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a protected training-only scaler "
            "and PCA pipeline for joint model v2"
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        choices=EXPECTED_SEEDS,
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=SUPPORTED_TASKS,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
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


def write_archive_atomic(
    path: Path,
    features: np.ndarray,
    labels: np.ndarray,
) -> None:
    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary_path.open("wb") as file:
        np.savez_compressed(
            file,
            features=features,
            labels=labels,
        )

    temporary_path.replace(path)


def write_joblib_atomic(
    path: Path,
    value: Mapping[str, Any],
) -> None:
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
    pca_config: Mapping[str, Any],
    joint_config: Mapping[str, Any],
) -> None:
    if pca_config.get("model_name") != (
        joint_config.get("model_name")
    ):
        raise RuntimeError(
            "PCA and joint model names differ"
        )

    if int(
        pca_config.get("source_dimension")
    ) != SOURCE_DIMENSION:
        raise RuntimeError(
            "Unexpected PCA source dimension"
        )

    if int(
        pca_config.get("output_dimension")
    ) != OUTPUT_DIMENSION:
        raise RuntimeError(
            "Unexpected PCA output dimension"
        )

    pca_seeds = tuple(
        int(value)
        for value in pca_config[
            "random_seeds"
        ]
    )
    joint_seeds = tuple(
        int(value)
        for value in joint_config[
            "random_seeds"
        ]
    )

    if pca_seeds != EXPECTED_SEEDS:
        raise RuntimeError(
            "PCA seed list differs from the "
            "pre-registered seed list"
        )

    if pca_seeds != joint_seeds:
        raise RuntimeError(
            "PCA and joint-model seed lists differ"
        )

    scope = require_mapping(
        pca_config.get("pipeline_scope"),
        "pipeline_scope",
    )

    for field in (
        "fit_separately_for_each_task",
        "fit_separately_for_each_seed",
        "use_full_training_split",
    ):
        require_true(
            scope.get(field),
            f"pipeline_scope.{field}",
        )

    for field in (
        "training_subsampling",
        "labels_used_during_fit",
    ):
        require_false(
            scope.get(field),
            f"pipeline_scope.{field}",
        )

    if int(
        scope.get("number_of_fitted_pipelines")
    ) != 6:
        raise RuntimeError(
            "Expected exactly six fitted pipelines"
        )

    scaler_config = require_mapping(
        pca_config.get("standard_scaler"),
        "standard_scaler",
    )

    if scaler_config.get("implementation") != (
        "sklearn.preprocessing.StandardScaler"
    ):
        raise RuntimeError(
            "Unexpected standard-scaler implementation"
        )

    for field in (
        "with_mean",
        "with_std",
        "copy",
    ):
        require_true(
            scaler_config.get(field),
            f"standard_scaler.{field}",
        )

    if scaler_config.get(
        "zero_variance_action"
    ) != "error":
        raise RuntimeError(
            "Unexpected zero-variance action"
        )

    pca_settings = require_mapping(
        pca_config.get("pca"),
        "pca",
    )

    if pca_settings.get("implementation") != (
        "sklearn.decomposition.PCA"
    ):
        raise RuntimeError(
            "Unexpected PCA implementation"
        )

    if int(
        pca_settings.get("n_components")
    ) != OUTPUT_DIMENSION:
        raise RuntimeError(
            "Unexpected PCA component count"
        )

    if pca_settings.get("svd_solver") != (
        "covariance_eigh"
    ):
        raise RuntimeError(
            "Unexpected PCA solver"
        )

    require_false(
        pca_settings.get("whiten"),
        "pca.whiten",
    )
    require_true(
        pca_settings.get("copy"),
        "pca.copy",
    )
    require_false(
        pca_settings.get("labels_used_during_fit"),
        "pca.labels_used_during_fit",
    )

    if pca_settings.get("random_state") is not None:
        raise RuntimeError(
            "Deterministic PCA solver must not use "
            "a random state"
        )

    fitting_rules = require_mapping(
        pca_config.get("fitting_rules"),
        "fitting_rules",
    )

    if fitting_rules.get("fit_split") != "train":
        raise RuntimeError(
            "PCA must be fitted on train"
        )

    for field in (
        "scaler_fit_on_training_features_only",
        "pca_fit_on_scaled_training_features_only",
        "output_dimension_fixed_before_fit",
    ):
        require_true(
            fitting_rules.get(field),
            f"fitting_rules.{field}",
        )

    for field in (
        "validation_features_used_for_fit",
        "locked_test_features_used_for_fit",
        "hyperparameter_tuning_using_explained_variance",
    ):
        require_false(
            fitting_rules.get(field),
            f"fitting_rules.{field}",
        )

    transformation_rules = require_mapping(
        pca_config.get("transformation_rules"),
        "transformation_rules",
    )

    for field in (
        "reuse_fitted_training_scaler",
        "reuse_fitted_training_pca",
        "preserve_binary_labels",
        "preserve_row_order",
    ):
        require_true(
            transformation_rules.get(field),
            f"transformation_rules.{field}",
        )

    require_false(
        transformation_rules.get(
            "refit_on_validation"
        ),
        "transformation_rules.refit_on_validation",
    )

    if transformation_rules.get(
        "output_dtype"
    ) != "float32":
        raise RuntimeError(
            "PCA output must use float32"
        )

    if transformation_rules.get(
        "label_dtype"
    ) != "uint8":
        raise RuntimeError(
            "PCA labels must use uint8"
        )

    test_protection = require_mapping(
        pca_config.get("test_protection"),
        "test_protection",
    )

    for field, value in test_protection.items():
        require_false(
            value,
            f"test_protection.{field}",
        )


def task_splits(
    pca_config: Mapping[str, Any],
    task: str,
) -> tuple[str, ...]:
    task_config = require_mapping(
        pca_config["tasks"][task],
        f"{task} PCA task configuration",
    )

    training_split = str(
        task_config["training_split"]
    )
    transform_splits = tuple(
        str(value)
        for value in task_config[
            "development_transform_splits"
        ]
    )
    locked_split = str(
        task_config["locked_test_split"]
    )

    if training_split != "train":
        raise RuntimeError(
            f"{task} PCA fit split must be train"
        )

    if not transform_splits:
        raise RuntimeError(
            f"{task} has no PCA transform splits"
        )

    if transform_splits[0] != training_split:
        raise RuntimeError(
            f"{task} train must be transformed first"
        )

    if len(set(transform_splits)) != len(
        transform_splits
    ):
        raise RuntimeError(
            f"{task} PCA transform splits are not unique"
        )

    if locked_split in transform_splits:
        raise RuntimeError(
            f"{task} locked test entered PCA transforms"
        )

    return transform_splits


def source_paths(
    seed: int,
    task: str,
    split: str,
) -> tuple[Path, Path]:
    archive_path = (
        PRIVATE_FEATURE_DIRECTORY
        / f"seed{seed}"
        / f"{task}_{split}_32d.npz"
    )
    audit_path = (
        EXTRACTION_AUDIT_DIRECTORY
        / f"seed{seed}_{task}_{split}_32d_audit.json"
    )

    return archive_path, audit_path


def output_paths(
    seed: int,
    task: str,
    splits: tuple[str, ...],
) -> tuple[
    dict[str, Path],
    Path,
    Path,
]:
    private_directory = (
        PRIVATE_FEATURE_DIRECTORY
        / f"seed{seed}"
    )
    private_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    PCA_AUDIT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    archives = {
        split: (
            private_directory
            / f"{task}_{split}_8d.npz"
        )
        for split in splits
    }
    state_path = (
        private_directory
        / f"{task}_32d_to_8d_pca.joblib"
    )
    audit_path = (
        PCA_AUDIT_DIRECTORY
        / f"seed{seed}_{task}_pca_audit.json"
    )

    return archives, state_path, audit_path


def load_source_archive(
    seed: int,
    task: str,
    split: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    archive_path, audit_path = (
        source_paths(
            seed,
            task,
            split,
        )
    )

    if not archive_path.is_file():
        raise FileNotFoundError(
            f"Source 32D archive not found: "
            f"{archive_path}"
        )

    audit = read_json_object(
        audit_path
    )

    if int(audit.get("seed")) != seed:
        raise RuntimeError(
            "Source audit seed mismatch"
        )

    if audit.get("task") != task:
        raise RuntimeError(
            "Source audit task mismatch"
        )

    if audit.get("split") != split:
        raise RuntimeError(
            "Source audit split mismatch"
        )

    private_output = require_mapping(
        audit.get("private_output"),
        "source private_output",
    )

    if sha256_file(archive_path) != (
        private_output.get("file_sha256")
    ):
        raise RuntimeError(
            "Source archive hash mismatch"
        )

    with np.load(
        archive_path,
        allow_pickle=False,
    ) as archive:
        if set(archive.files) != {
            "features",
            "labels",
        }:
            raise RuntimeError(
                "Source archive contains unexpected "
                "arrays"
            )

        features = np.ascontiguousarray(
            archive["features"],
            dtype=np.float32,
        )
        labels = np.ascontiguousarray(
            archive["labels"],
            dtype=np.uint8,
        )

    if features.ndim != 2:
        raise ValueError(
            "Source features must be two-dimensional"
        )

    if features.shape[1] != SOURCE_DIMENSION:
        raise ValueError(
            "Source feature dimension is not 32"
        )

    if labels.ndim != 1:
        raise ValueError(
            "Source labels must be one-dimensional"
        )

    if features.shape[0] != labels.shape[0]:
        raise ValueError(
            "Source feature and label rows differ"
        )

    if not np.isfinite(features).all():
        raise ValueError(
            "Source features contain non-finite values"
        )

    observed_labels = set(
        int(value)
        for value in np.unique(labels)
    )

    if not observed_labels.issubset({0, 1}):
        raise ValueError(
            "Source labels are not binary"
        )

    source_summary = require_mapping(
        audit["extraction"]["output"],
        "source extraction summary",
    )

    if sha256_array(features) != (
        source_summary["features_sha256"]
    ):
        raise RuntimeError(
            "Source feature array hash mismatch"
        )

    if sha256_array(labels) != (
        source_summary["labels_sha256"]
    ):
        raise RuntimeError(
            "Source label array hash mismatch"
        )

    source_record = {
        "split": split,
        "rows": int(features.shape[0]),
        "dimension": int(features.shape[1]),
        "archive_path": str(
            archive_path.relative_to(
                PROJECT_ROOT
            )
        ),
        "archive_sha256": sha256_file(
            archive_path
        ),
        "features_sha256": sha256_array(
            features
        ),
        "labels_sha256": sha256_array(
            labels
        ),
        "extraction_audit_filename": (
            audit_path.name
        ),
        "patient_identifiers_loaded": False,
        "locked_test_loaded": False,
    }

    return features, labels, source_record


def validate_training_variance(
    training_features: np.ndarray,
) -> dict[str, Any]:
    standard_deviations = training_features.std(
        axis=0,
        dtype=np.float64,
    )

    if not np.isfinite(
        standard_deviations
    ).all():
        raise ValueError(
            "Training standard deviations are "
            "not finite"
        )

    zero_variance_count = int(
        np.sum(
            standard_deviations <= 1e-12
        )
    )

    if zero_variance_count != 0:
        raise RuntimeError(
            "Training representation contains "
            f"{zero_variance_count} zero-variance "
            "dimensions"
        )

    return {
        "zero_variance_dimensions": (
            zero_variance_count
        ),
        "minimum_dimension_standard_deviation": (
            float(
                standard_deviations.min()
            )
        ),
        "maximum_dimension_standard_deviation": (
            float(
                standard_deviations.max()
            )
        ),
    }


def build_pipeline(
    pca_config: Mapping[str, Any],
) -> tuple[StandardScaler, PCA]:
    scaler_config = pca_config[
        "standard_scaler"
    ]
    pca_settings = pca_config["pca"]

    scaler = StandardScaler(
        copy=bool(
            scaler_config["copy"]
        ),
        with_mean=bool(
            scaler_config["with_mean"]
        ),
        with_std=bool(
            scaler_config["with_std"]
        ),
    )

    pca = PCA(
        n_components=int(
            pca_settings["n_components"]
        ),
        copy=bool(
            pca_settings["copy"]
        ),
        whiten=bool(
            pca_settings["whiten"]
        ),
        svd_solver=str(
            pca_settings["svd_solver"]
        ),
        random_state=(
            pca_settings["random_state"]
        ),
    )

    return scaler, pca


def validate_fitted_pipeline(
    scaler: StandardScaler,
    pca: PCA,
) -> None:
    if not hasattr(scaler, "mean_"):
        raise RuntimeError(
            "Standard scaler was not fitted"
        )

    if not hasattr(scaler, "scale_"):
        raise RuntimeError(
            "Standard scaler has no fitted scales"
        )

    if scaler.mean_.shape != (
        SOURCE_DIMENSION,
    ):
        raise RuntimeError(
            "Scaler mean has an unexpected shape"
        )

    if scaler.scale_.shape != (
        SOURCE_DIMENSION,
    ):
        raise RuntimeError(
            "Scaler scale has an unexpected shape"
        )

    if not np.isfinite(scaler.mean_).all():
        raise ValueError(
            "Scaler means are not finite"
        )

    if not np.isfinite(scaler.scale_).all():
        raise ValueError(
            "Scaler scales are not finite"
        )

    if np.any(scaler.scale_ <= 0.0):
        raise ValueError(
            "Scaler contains non-positive scales"
        )

    if not hasattr(pca, "components_"):
        raise RuntimeError(
            "PCA was not fitted"
        )

    if pca.components_.shape != (
        OUTPUT_DIMENSION,
        SOURCE_DIMENSION,
    ):
        raise RuntimeError(
            "PCA components have an unexpected shape"
        )

    if not np.isfinite(pca.components_).all():
        raise ValueError(
            "PCA components are not finite"
        )

    if pca.explained_variance_ratio_.shape != (
        OUTPUT_DIMENSION,
    ):
        raise RuntimeError(
            "PCA explained variance has an "
            "unexpected shape"
        )

    if not np.isfinite(
        pca.explained_variance_ratio_
    ).all():
        raise ValueError(
            "PCA explained variance is not finite"
        )

    if np.any(
        pca.explained_variance_ratio_ < 0.0
    ):
        raise ValueError(
            "PCA explained variance is negative"
        )


def transform_features(
    scaler: StandardScaler,
    pca: PCA,
    features: np.ndarray,
) -> np.ndarray:
    if features.ndim != 2:
        raise ValueError(
            "Features must be two-dimensional"
        )

    if features.shape[1] != SOURCE_DIMENSION:
        raise ValueError(
            "Features must have 32 dimensions"
        )

    scaled = scaler.transform(
        features
    )

    if not np.isfinite(scaled).all():
        raise ValueError(
            "Scaled features contain non-finite values"
        )

    transformed = pca.transform(
        scaled
    )

    transformed = np.ascontiguousarray(
        transformed,
        dtype=np.float32,
    )

    if transformed.shape != (
        features.shape[0],
        OUTPUT_DIMENSION,
    ):
        raise RuntimeError(
            "PCA returned an unexpected output shape"
        )

    if not np.isfinite(transformed).all():
        raise ValueError(
            "PCA features contain non-finite values"
        )

    return transformed


def output_summary(
    features: np.ndarray,
    labels: np.ndarray,
    source_labels_sha256: str,
) -> dict[str, Any]:
    if features.ndim != 2:
        raise ValueError(
            "Output features must be two-dimensional"
        )

    if features.shape[1] != OUTPUT_DIMENSION:
        raise ValueError(
            "Output features must have 8 dimensions"
        )

    if features.dtype != np.float32:
        raise TypeError(
            "Output features must use float32"
        )

    if labels.dtype != np.uint8:
        raise TypeError(
            "Output labels must use uint8"
        )

    if features.shape[0] != labels.shape[0]:
        raise ValueError(
            "Output feature and label rows differ"
        )

    if not np.isfinite(features).all():
        raise ValueError(
            "Output features contain non-finite values"
        )

    labels_hash = sha256_array(
        labels
    )

    if labels_hash != source_labels_sha256:
        raise RuntimeError(
            "PCA transformation changed label order "
            "or content"
        )

    return {
        "rows": int(features.shape[0]),
        "dimension": int(
            features.shape[1]
        ),
        "feature_dtype": str(
            features.dtype
        ),
        "label_dtype": str(
            labels.dtype
        ),
        "negative_count": int(
            len(labels) - labels.sum()
        ),
        "positive_count": int(
            labels.sum()
        ),
        "feature_minimum": float(
            features.min()
        ),
        "feature_maximum": float(
            features.max()
        ),
        "feature_mean": float(
            features.mean(
                dtype=np.float64
            )
        ),
        "feature_standard_deviation": float(
            features.std(
                dtype=np.float64
            )
        ),
        "features_sha256": sha256_array(
            features
        ),
        "labels_sha256": labels_hash,
        "label_hash_preserved": True,
        "row_order_preserved": True,
        "patient_identifiers_in_output": False,
        "locked_test_in_output": False,
    }


def validate_saved_output(
    path: Path,
    expected_summary: Mapping[str, Any],
) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Saved 8D archive not found: {path}"
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
                "Saved 8D archive contains "
                "unexpected arrays"
            )

        features = np.ascontiguousarray(
            archive["features"],
            dtype=np.float32,
        )
        labels = np.ascontiguousarray(
            archive["labels"],
            dtype=np.uint8,
        )

    if features.shape != (
        int(expected_summary["rows"]),
        OUTPUT_DIMENSION,
    ):
        raise RuntimeError(
            "Saved 8D feature shape mismatch"
        )

    if sha256_array(features) != (
        expected_summary["features_sha256"]
    ):
        raise RuntimeError(
            "Saved 8D feature hash mismatch"
        )

    if sha256_array(labels) != (
        expected_summary["labels_sha256"]
    ):
        raise RuntimeError(
            "Saved 8D label hash mismatch"
        )


def validate_completed_run(
    audit_path: Path,
    state_path: Path,
    archive_paths: Mapping[str, Path],
    expected_seed: int,
    expected_task: str,
) -> dict[str, Any]:
    audit = read_json_object(
        audit_path
    )

    if int(audit.get("seed")) != expected_seed:
        raise RuntimeError(
            "Existing PCA audit seed mismatch"
        )

    if audit.get("task") != expected_task:
        raise RuntimeError(
            "Existing PCA audit task mismatch"
        )

    if sha256_file(PCA_CONFIG_PATH) != (
        audit["pca_configuration"]["sha256"]
    ):
        raise RuntimeError(
            "Existing PCA audit configuration "
            "hash mismatch"
        )

    fitted_state = require_mapping(
        audit.get("fitted_state"),
        "fitted_state",
    )

    if not state_path.is_file():
        raise FileNotFoundError(
            "Fitted PCA state is missing"
        )

    if sha256_file(state_path) != (
        fitted_state["sha256"]
    ):
        raise RuntimeError(
            "Fitted PCA state hash mismatch"
        )

    loaded_state = joblib.load(
        state_path
    )

    if int(loaded_state["seed"]) != (
        expected_seed
    ):
        raise RuntimeError(
            "Loaded PCA state seed mismatch"
        )

    if loaded_state["task"] != expected_task:
        raise RuntimeError(
            "Loaded PCA state task mismatch"
        )

    split_outputs = require_mapping(
        audit.get("split_outputs"),
        "split_outputs",
    )

    for split, archive_path in (
        archive_paths.items()
    ):
        split_record = require_mapping(
            split_outputs.get(split),
            f"split_outputs.{split}",
        )

        if sha256_file(archive_path) != (
            split_record["archive_sha256"]
        ):
            raise RuntimeError(
                f"Saved {split} archive file "
                "hash mismatch"
            )

        validate_saved_output(
            archive_path,
            split_record["summary"],
        )

    require_false(
        audit["test_protection"][
            "locked_tests_loaded"
        ],
        "test_protection.locked_tests_loaded",
    )
    require_false(
        audit["privacy"][
            "patient_identifiers_in_output"
        ],
        "privacy.patient_identifiers_in_output",
    )

    return audit


def main() -> None:
    arguments = parse_arguments()

    seed = int(arguments.seed)
    task = str(arguments.task)

    pca_config = read_json_object(
        PCA_CONFIG_PATH
    )
    joint_config = read_json_object(
        JOINT_CONFIG_PATH
    )

    validate_configuration(
        pca_config,
        joint_config,
    )

    splits = task_splits(
        pca_config,
        task,
    )

    (
        archive_paths,
        state_path,
        audit_path,
    ) = output_paths(
        seed,
        task,
        splits,
    )

    all_output_paths = [
        *archive_paths.values(),
        state_path,
        audit_path,
    ]
    existing_outputs = [
        path
        for path in all_output_paths
        if path.exists()
    ]

    if existing_outputs and not arguments.resume:
        raise FileExistsError(
            "PCA outputs already exist. Use "
            "--resume only for the matching "
            f"seed/task: {existing_outputs}"
        )

    if audit_path.exists() and arguments.resume:
        audit = validate_completed_run(
            audit_path=audit_path,
            state_path=state_path,
            archive_paths=archive_paths,
            expected_seed=seed,
            expected_task=task,
        )

        print(
            f"seed={seed} "
            f"task={task} "
            "status=validated_existing"
        )
        print(
            "total_explained_variance_ratio="
            f"{audit['pca_fit']['total_explained_variance_ratio']:.6f}"
        )
        print("locked_tests_loaded=False")
        print(f"audit={audit_path}")
        return

    source_records: dict[
        str,
        dict[str, Any]
    ] = {}
    source_labels: dict[
        str,
        np.ndarray
    ] = {}
    source_features: dict[
        str,
        np.ndarray
    ] = {}

    for split in splits:
        (
            features,
            labels,
            source_record,
        ) = load_source_archive(
            seed,
            task,
            split,
        )

        source_features[split] = features
        source_labels[split] = labels
        source_records[split] = (
            source_record
        )

    training_split = str(
        pca_config[
            "tasks"
        ][
            task
        ][
            "training_split"
        ]
    )

    training_features = (
        source_features[
            training_split
        ]
    )

    variance_audit = (
        validate_training_variance(
            training_features
        )
    )

    scaler, pca = build_pipeline(
        pca_config
    )

    scaled_training = scaler.fit_transform(
        training_features
    )

    if scaled_training.shape != (
        training_features.shape
    ):
        raise RuntimeError(
            "Scaled training shape changed"
        )

    if not np.isfinite(
        scaled_training
    ).all():
        raise ValueError(
            "Scaled training features contain "
            "non-finite values"
        )

    training_8d = pca.fit_transform(
        scaled_training
    )
    training_8d = np.ascontiguousarray(
        training_8d,
        dtype=np.float32,
    )

    validate_fitted_pipeline(
        scaler,
        pca,
    )

    if training_8d.shape != (
        training_features.shape[0],
        OUTPUT_DIMENSION,
    ):
        raise RuntimeError(
            "Training PCA output shape is invalid"
        )

    if not np.isfinite(
        training_8d
    ).all():
        raise ValueError(
            "Training PCA output is not finite"
        )

    transformed_features = {
        training_split: training_8d
    }

    del scaled_training
    del training_8d

    for split in splits:
        if split == training_split:
            continue

        transformed_features[split] = (
            transform_features(
                scaler=scaler,
                pca=pca,
                features=source_features[split],
            )
        )

    fitted_state = {
        "state_version": "1.0",
        "seed": seed,
        "task": task,
        "source_dimension": (
            SOURCE_DIMENSION
        ),
        "output_dimension": (
            OUTPUT_DIMENSION
        ),
        "fit_split": training_split,
        "pca_configuration_sha256": (
            sha256_file(
                PCA_CONFIG_PATH
            )
        ),
        "training_source_archive_sha256": (
            source_records[
                training_split
            ][
                "archive_sha256"
            ]
        ),
        "training_source_features_sha256": (
            source_records[
                training_split
            ][
                "features_sha256"
            ]
        ),
        "scaler": scaler,
        "pca": pca,
    }

    write_joblib_atomic(
        state_path,
        fitted_state,
    )

    split_outputs: dict[
        str,
        dict[str, Any]
    ] = {}

    for split in splits:
        labels = source_labels[split]
        features_8d = (
            transformed_features[split]
        )

        summary = output_summary(
            features=features_8d,
            labels=labels,
            source_labels_sha256=(
                source_records[
                    split
                ][
                    "labels_sha256"
                ]
            ),
        )

        write_archive_atomic(
            archive_paths[split],
            features_8d,
            labels,
        )

        validate_saved_output(
            archive_paths[split],
            summary,
        )

        split_outputs[split] = {
            "archive_path": str(
                archive_paths[
                    split
                ].relative_to(
                    PROJECT_ROOT
                )
            ),
            "archive_filename": (
                archive_paths[
                    split
                ].name
            ),
            "archive_sha256": (
                sha256_file(
                    archive_paths[
                        split
                    ]
                )
            ),
            "size_bytes": int(
                archive_paths[
                    split
                ].stat().st_size
            ),
            "source": (
                source_records[split]
            ),
            "summary": summary,
        }

    explained_variance_ratios = [
        float(value)
        for value in (
            pca.explained_variance_ratio_
        )
    ]
    total_explained_variance = float(
        np.sum(
            pca.explained_variance_ratio_,
            dtype=np.float64,
        )
    )

    if not 0.0 < total_explained_variance <= 1.0:
        raise RuntimeError(
            "Total PCA explained variance is invalid"
        )

    audit = {
        "version": "1.0",
        "seed": seed,
        "task": task,
        "source_representation": (
            "joint_shared_trunk_output"
        ),
        "source_dimension": (
            SOURCE_DIMENSION
        ),
        "output_representation": (
            "training_only_standard_scaler_then_pca"
        ),
        "output_dimension": (
            OUTPUT_DIMENSION
        ),
        "pca_configuration": {
            "filename": (
                PCA_CONFIG_PATH.name
            ),
            "sha256": sha256_file(
                PCA_CONFIG_PATH
            ),
        },
        "joint_configuration": {
            "filename": (
                JOINT_CONFIG_PATH.name
            ),
            "sha256": sha256_file(
                JOINT_CONFIG_PATH
            ),
        },
        "fit": {
            "split": training_split,
            "rows": int(
                training_features.shape[0]
            ),
            "labels_used": False,
            "validation_features_used": False,
            "locked_test_features_used": False,
            "training_subsampling_used": False,
            "full_training_split_used": True,
        },
        "standard_scaler_fit": {
            "with_mean": True,
            "with_std": True,
            "input_dimension": (
                SOURCE_DIMENSION
            ),
            **variance_audit,
        },
        "pca_fit": {
            "n_components": (
                OUTPUT_DIMENSION
            ),
            "svd_solver": (
                "covariance_eigh"
            ),
            "whiten": False,
            "labels_used": False,
            "explained_variance_ratio": (
                explained_variance_ratios
            ),
            "total_explained_variance_ratio": (
                total_explained_variance
            ),
        },
        "fitted_state": {
            "path": str(
                state_path.relative_to(
                    PROJECT_ROOT
                )
            ),
            "filename": state_path.name,
            "sha256": sha256_file(
                state_path
            ),
            "size_bytes": int(
                state_path.stat().st_size
            ),
            "committed": False,
        },
        "split_outputs": split_outputs,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": (
                sklearn.__version__
            ),
            "joblib": joblib.__version__,
        },
        "privacy": {
            "patient_identifiers_loaded": False,
            "patient_identifiers_in_output": False,
            "patient_level_rows_in_public_audit": False,
            "feature_values_in_public_audit": False,
            "labels_in_public_audit": False,
            "private_archives_committed": False,
            "fitted_state_committed": False,
        },
        "test_protection": {
            "locked_tests_loaded": False,
            "locked_tests_transformed": False,
            "locked_tests_used_for_scaler_fit": False,
            "locked_tests_used_for_pca_fit": False,
        },
        "handoff": {
            "classical_and_quantum_receive_same_features": True,
            "classical_and_quantum_receive_same_labels": True,
            "quantum_implementation_in_current_scope": False,
        },
    }

    write_json_atomic(
        audit_path,
        audit,
    )

    validate_completed_run(
        audit_path=audit_path,
        state_path=state_path,
        archive_paths=archive_paths,
        expected_seed=seed,
        expected_task=task,
    )

    print(f"seed={seed}")
    print(f"task={task}")
    print(
        f"fit_rows="
        f"{training_features.shape[0]}"
    )
    print(
        f"source_dimension="
        f"{SOURCE_DIMENSION}"
    )
    print(
        f"output_dimension="
        f"{OUTPUT_DIMENSION}"
    )
    print(
        "explained_variance_ratios="
        + ",".join(
            f"{value:.6f}"
            for value in (
                explained_variance_ratios
            )
        )
    )
    print(
        "total_explained_variance_ratio="
        f"{total_explained_variance:.6f}"
    )

    for split in splits:
        print(
            f"split={split} "
            f"rows="
            f"{split_outputs[split]['summary']['rows']} "
            f"dimension="
            f"{split_outputs[split]['summary']['dimension']}"
        )

    print("labels_used_for_fit=False")
    print("locked_tests_loaded=False")
    print(
        "patient_identifiers_in_output=False"
    )
    print(f"audit={audit_path}")


if __name__ == "__main__":
    main()
