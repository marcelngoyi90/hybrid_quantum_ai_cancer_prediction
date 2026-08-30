from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "PYTORCH_ENABLE_MPS_FALLBACK",
    "1",
)

import joblib
import numpy as np
import sklearn
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.camelyon17_data import (
    Camelyon17ParquetDataset,
    build_transform,
)
from src.joint_feature_extraction import (
    EXPECTED_SHARED_DIMENSION,
    load_selected_feature_context,
    sha256_array,
    sha256_file,
)
from src.seer_preprocessing import read_seer_split


FINAL_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "joint_v2_locked_features.json"
)
PCA_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "joint_v2_pca.json"
)
PCA_AUDIT_DIRECTORY = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "joint_v2"
    / "pca"
)

EXPECTED_SEEDS = (42, 123, 2026)
SUPPORTED_TASKS = ("image", "tabular")
SOURCE_DIMENSION = 32
OUTPUT_DIMENSION = 8


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate frozen 32D and 8D representations "
            "for one final locked split"
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
        "--device",
        choices=("mps", "cpu"),
        default=None,
    )
    parser.add_argument(
        "--confirm-final",
        action="store_true",
        help=(
            "Confirm that the registered final locked-test "
            "phase has begun"
        ),
    )
    return parser.parse_args()


def read_json_object(path: Path) -> dict[str, Any]:
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


def resolve_project_path(value: Any) -> Path:
    relative_path = Path(str(value))

    if relative_path.is_absolute():
        raise ValueError(
            "Final configuration paths must be relative"
        )

    path = (PROJECT_ROOT / relative_path).resolve()

    if not path.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError(
            f"Configured path escapes project root: {value}"
        )

    return path


def formatted_project_path(
    template: Any,
    seed: int,
) -> Path:
    return resolve_project_path(
        str(template).format(seed=seed)
    )


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


def validate_final_configuration(
    config: Mapping[str, Any],
) -> None:
    if config.get("phase") != "final_locked_test":
        raise RuntimeError(
            "Configuration is not for the final locked phase"
        )

    if config.get("access_authorized") is not True:
        raise PermissionError(
            "Final locked-test access is not authorized"
        )

    configured_seeds = tuple(
        int(value)
        for value in config["random_seeds"]
    )

    if configured_seeds != EXPECTED_SEEDS:
        raise RuntimeError(
            "Frozen seed list has changed"
        )

    tasks = require_mapping(
        config.get("tasks"),
        "tasks",
    )

    if set(tasks) != set(SUPPORTED_TASKS):
        raise RuntimeError(
            "Frozen task list has changed"
        )

    representations = require_mapping(
        config.get("representations"),
        "representations",
    )

    if int(
        representations["source_dimension"]
    ) != SOURCE_DIMENSION:
        raise RuntimeError(
            "Frozen source dimension is not 32"
        )

    if int(
        representations["output_dimension"]
    ) != OUTPUT_DIMENSION:
        raise RuntimeError(
            "Frozen output dimension is not 8"
        )

    if representations["feature_dtype"] != "float32":
        raise RuntimeError(
            "Frozen feature dtype is not float32"
        )

    if representations["label_dtype"] != "uint8":
        raise RuntimeError(
            "Frozen label dtype is not uint8"
        )

    execution = require_mapping(
        config.get("execution"),
        "execution",
    )

    if execution.get(
        "required_confirmation_argument"
    ) != "--confirm-final":
        raise RuntimeError(
            "Final confirmation argument changed"
        )

    required_true_execution = (
        "one_seed_and_task_per_command",
        "model_evaluation_mode",
        "preserve_row_order",
    )

    for field in required_true_execution:
        if execution.get(field) is not True:
            raise RuntimeError(
                f"execution.{field} must be true"
            )

    required_false_execution = (
        "gradient_calculation",
        "shuffle",
        "training_augmentation",
    )

    for field in required_false_execution:
        if execution.get(field) is not False:
            raise RuntimeError(
                f"execution.{field} must be false"
            )

    pca_rules = require_mapping(
        config.get("pca_rules"),
        "pca_rules",
    )

    if pca_rules.get(
        "load_existing_fitted_state"
    ) is not True:
        raise RuntimeError(
            "Existing PCA state must be reused"
        )

    for field in (
        "allow_scaler_fit",
        "allow_pca_fit",
        "allow_validation_refit",
        "allow_locked_test_refit",
    ):
        if pca_rules.get(field) is not False:
            raise RuntimeError(
                f"pca_rules.{field} must be false"
            )

    prohibited = require_mapping(
        config.get("prohibited_actions"),
        "prohibited_actions",
    )

    for field, value in prohibited.items():
        if value is not False:
            raise RuntimeError(
                f"Prohibited action enabled: {field}"
            )

    protocol_path = resolve_project_path(
        config["protocol"]
    )

    if not protocol_path.is_file():
        raise FileNotFoundError(
            f"Frozen protocol not found: {protocol_path}"
        )


def task_paths(
    config: Mapping[str, Any],
    seed: int,
    task: str,
) -> dict[str, Path]:
    task_config = require_mapping(
        config["tasks"][task],
        f"tasks.{task}",
    )
    model_inputs = require_mapping(
        config["model_inputs"],
        "model_inputs",
    )

    paths = {
        "protocol": resolve_project_path(
            config["protocol"]
        ),
        "joint_config": resolve_project_path(
            model_inputs["joint_model_config"]
        ),
        "feature_config": resolve_project_path(
            model_inputs["seer_feature_config"]
        ),
        "preprocessor_state": resolve_project_path(
            model_inputs["seer_preprocessor_state"]
        ),
        "data_source": resolve_project_path(
            task_config["data_source"]
        ),
        "checkpoint": formatted_project_path(
            task_config["checkpoint_template"],
            seed,
        ),
        "history": formatted_project_path(
            task_config["history_template"],
            seed,
        ),
        "pca_state": formatted_project_path(
            task_config["pca_state_template"],
            seed,
        ),
        "output_32d": formatted_project_path(
            task_config["output_32d_template"],
            seed,
        ),
        "output_8d": formatted_project_path(
            task_config["output_8d_template"],
            seed,
        ),
    }

    audit_directory = resolve_project_path(
        config["public_audit_directory"]
    )
    paths["audit"] = (
        audit_directory
        / f"seed{seed}_{task}_locked_features_audit.json"
    )

    return paths


def validate_required_inputs(
    paths: Mapping[str, Path],
) -> None:
    required_files = (
        "protocol",
        "joint_config",
        "feature_config",
        "preprocessor_state",
        "checkpoint",
        "history",
        "pca_state",
    )

    for name in required_files:
        if not paths[name].is_file():
            raise FileNotFoundError(
                f"Required {name} file not found: "
                f"{paths[name]}"
            )

    if not paths["data_source"].exists():
        raise FileNotFoundError(
            "Locked data source not found: "
            f"{paths['data_source']}"
        )

    for name in (
        "output_32d",
        "output_8d",
        "audit",
    ):
        if paths[name].exists():
            raise FileExistsError(
                "Final output already exists and will not "
                f"be overwritten: {paths[name]}"
            )


def validate_task_contract(
    config: Mapping[str, Any],
    context: Any,
    seed: int,
    task: str,
) -> tuple[str, int]:
    task_config = require_mapping(
        config["tasks"][task],
        f"tasks.{task}",
    )

    locked_split = str(
        task_config["locked_split"]
    )
    expected_rows = int(
        task_config["expected_rows"]
    )
    selected_epoch = int(
        config["selected_epochs"][str(seed)]
    )

    joint_task = require_mapping(
        context.joint_config["tasks"][task],
        f"joint tasks.{task}",
    )

    if locked_split != joint_task[
        "locked_test_split"
    ]:
        raise RuntimeError(
            "Locked split differs from joint-model contract"
        )

    if context.selected_epoch != selected_epoch:
        raise RuntimeError(
            "Selected checkpoint epoch differs from "
            "the frozen final configuration"
        )

    if context.model.training:
        raise RuntimeError(
            "Joint model is not in evaluation mode"
        )

    if any(
        parameter.requires_grad
        for parameter in context.model.parameters()
    ):
        raise RuntimeError(
            "Joint model contains trainable parameters"
        )

    if int(
        context.model.shared_trunk.output_dimension
    ) != SOURCE_DIMENSION:
        raise RuntimeError(
            "Joint shared representation is not 32D"
        )

    return locked_split, expected_rows


def load_frozen_pca(
    seed: int,
    task: str,
    state_path: Path,
) -> tuple[
    StandardScaler,
    PCA,
    dict[str, Any],
]:
    audit_path = (
        PCA_AUDIT_DIRECTORY
        / f"seed{seed}_{task}_pca_audit.json"
    )

    if not audit_path.is_file():
        raise FileNotFoundError(
            f"Development PCA audit not found: {audit_path}"
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

    fitted_state_record = require_mapping(
        audit.get("fitted_state"),
        "PCA audit fitted_state",
    )

    state_hash = sha256_file(state_path)

    if state_hash != fitted_state_record.get(
        "sha256"
    ):
        raise RuntimeError(
            "Private PCA state hash differs from "
            "the committed development audit"
        )

    state = joblib.load(state_path)

    if not isinstance(state, Mapping):
        raise TypeError(
            "Fitted PCA state must be an object"
        )

    if int(state.get("seed")) != seed:
        raise RuntimeError(
            "Fitted PCA state seed mismatch"
        )

    if state.get("task") != task:
        raise RuntimeError(
            "Fitted PCA state task mismatch"
        )

    if int(
        state.get("source_dimension")
    ) != SOURCE_DIMENSION:
        raise RuntimeError(
            "Fitted PCA source dimension mismatch"
        )

    if int(
        state.get("output_dimension")
    ) != OUTPUT_DIMENSION:
        raise RuntimeError(
            "Fitted PCA output dimension mismatch"
        )

    if state.get("fit_split") != "train":
        raise RuntimeError(
            "PCA was not fitted on the training split"
        )

    if state.get(
        "pca_configuration_sha256"
    ) != sha256_file(PCA_CONFIG_PATH):
        raise RuntimeError(
            "Fitted PCA configuration hash mismatch"
        )

    scaler = state.get("scaler")
    pca = state.get("pca")

    if not isinstance(scaler, StandardScaler):
        raise TypeError(
            "Fitted scaler has an unexpected type"
        )

    if not isinstance(pca, PCA):
        raise TypeError(
            "Fitted PCA has an unexpected type"
        )

    if scaler.mean_.shape != (
        SOURCE_DIMENSION,
    ):
        raise RuntimeError(
            "Fitted scaler mean shape is invalid"
        )

    if scaler.scale_.shape != (
        SOURCE_DIMENSION,
    ):
        raise RuntimeError(
            "Fitted scaler scale shape is invalid"
        )

    if not np.isfinite(scaler.mean_).all():
        raise ValueError(
            "Fitted scaler means are not finite"
        )

    if not np.isfinite(scaler.scale_).all():
        raise ValueError(
            "Fitted scaler scales are not finite"
        )

    if np.any(scaler.scale_ <= 0.0):
        raise ValueError(
            "Fitted scaler contains non-positive scales"
        )

    if pca.components_.shape != (
        OUTPUT_DIMENSION,
        SOURCE_DIMENSION,
    ):
        raise RuntimeError(
            "Fitted PCA component shape is invalid"
        )

    if not np.isfinite(pca.components_).all():
        raise ValueError(
            "Fitted PCA components are not finite"
        )

    record = {
        "state_path": str(
            state_path.relative_to(PROJECT_ROOT)
        ),
        "state_sha256": state_hash,
        "development_audit_path": str(
            audit_path.relative_to(PROJECT_ROOT)
        ),
        "development_audit_sha256": sha256_file(
            audit_path
        ),
        "fit_split": "train",
        "scaler_refitted": False,
        "pca_refitted": False,
    }

    return scaler, pca, record


def build_locked_loader(
    config: Mapping[str, Any],
    context: Any,
    task: str,
    split: str,
    expected_rows: int,
    data_source: Path,
) -> tuple[DataLoader, dict[str, Any]]:
    number_of_workers = int(
        context.joint_config[
            "batching"
        ]["num_workers"]
    )

    if task == "image":
        dataset = Camelyon17ParquetDataset(
            data_root=data_source,
            split=split,
            transform=build_transform(
                training=False
            ),
            shuffle=False,
            seed=context.seed,
        )
        batch_size = int(
            context.joint_config[
                "batching"
            ]["image_evaluation_batch_size"]
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=number_of_workers,
            drop_last=False,
            pin_memory=False,
        )
        patient_identifiers_loaded = False
        raw_clinical_fields_loaded = False

    elif task == "tabular":
        frame = read_seer_split(
            model_table_path=data_source,
            split_name=split,
            preprocessor=context.preprocessor,
            allow_locked_test=True,
        )

        expected_columns = {
            context.preprocessor.split_column,
            *context.preprocessor.numeric_features,
            *context.preprocessor.categorical_features,
            context.preprocessor.target_column,
        }

        if set(frame.columns) != expected_columns:
            raise RuntimeError(
                "Locked SEER frame contains unexpected columns"
            )

        if "Patient ID" in frame.columns:
            raise RuntimeError(
                "Patient ID entered locked feature extraction"
            )

        dataset, preprocessing_audit = (
            context.preprocessor.transform(
                frame,
                split,
                allow_locked_test=True,
            )
        )
        del frame

        if preprocessing_audit.get(
            "patient_identifiers_in_tensors"
        ) is not False:
            raise RuntimeError(
                "Patient identifiers entered SEER tensors"
            )

        batch_size = int(
            context.joint_config[
                "batching"
            ]["tabular_evaluation_batch_size"]
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=number_of_workers,
            drop_last=False,
            pin_memory=False,
        )
        patient_identifiers_loaded = False
        raw_clinical_fields_loaded = False

    else:
        raise ValueError(
            f"Unsupported task: {task}"
        )

    observed_rows = int(len(dataset))

    if observed_rows != expected_rows:
        raise RuntimeError(
            "Locked split row count differs from the "
            f"frozen contract: {observed_rows} versus "
            f"{expected_rows}"
        )

    loader_audit = {
        "split": split,
        "expected_rows": expected_rows,
        "batch_size": batch_size,
        "shuffle": False,
        "drop_last": False,
        "training_augmentation": False,
        "patient_identifiers_loaded": (
            patient_identifiers_loaded
        ),
        "raw_clinical_fields_loaded": (
            raw_clinical_fields_loaded
        ),
        "locked_test_loaded": True,
    }

    return loader, loader_audit


def validate_shared_batch(
    shared_features: torch.Tensor,
    labels: torch.Tensor,
    task: str,
    split: str,
) -> None:
    if shared_features.ndim != 2:
        raise RuntimeError(
            f"{task} {split} features are not two-dimensional"
        )

    if shared_features.shape[1] != SOURCE_DIMENSION:
        raise RuntimeError(
            f"{task} {split} feature dimension is not 32"
        )

    if labels.ndim != 1:
        raise RuntimeError(
            f"{task} {split} labels are not one-dimensional"
        )

    if shared_features.shape[0] != labels.shape[0]:
        raise RuntimeError(
            f"{task} {split} batch row counts differ"
        )

    if not torch.isfinite(
        shared_features
    ).all().item():
        raise ValueError(
            f"{task} {split} contains non-finite features"
        )

    observed_labels = set(
        int(value)
        for value in torch.unique(
            labels.detach().to("cpu")
        ).tolist()
    )

    if not observed_labels.issubset({0, 1}):
        raise ValueError(
            f"{task} {split} labels are not binary"
        )


def validate_arrays(
    features: np.ndarray,
    labels: np.ndarray,
    expected_rows: int,
    expected_dimension: int,
) -> None:
    if features.shape != (
        expected_rows,
        expected_dimension,
    ):
        raise RuntimeError(
            "Feature array has an unexpected shape: "
            f"{features.shape}"
        )

    if labels.shape != (expected_rows,):
        raise RuntimeError(
            "Label array has an unexpected shape: "
            f"{labels.shape}"
        )

    if features.dtype != np.float32:
        raise TypeError(
            "Feature array must use float32"
        )

    if labels.dtype != np.uint8:
        raise TypeError(
            "Label array must use uint8"
        )

    if not features.flags.c_contiguous:
        raise ValueError(
            "Feature array must be contiguous"
        )

    if not labels.flags.c_contiguous:
        raise ValueError(
            "Label array must be contiguous"
        )

    if not np.isfinite(features).all():
        raise ValueError(
            "Feature array contains non-finite values"
        )

    observed_labels = set(
        int(value)
        for value in np.unique(labels)
    )

    if not observed_labels.issubset({0, 1}):
        raise ValueError(
            "Label array is not binary"
        )

    if not observed_labels:
        raise ValueError(
            "Locked feature set is empty"
        )


def extract_locked_features(
    context: Any,
    loader: DataLoader,
    task: str,
    split: str,
    expected_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    context.model.eval()

    feature_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    rows_completed = 0
    progress_frequency = (
        200 if task == "image" else 10
    )

    with torch.inference_mode():
        for batch_index, batch in enumerate(
            loader,
            start=1,
        ):
            if task == "image":
                images = batch["image"].to(
                    context.device
                )
                labels = batch["label"].to(
                    dtype=torch.int64
                )
                _, shared_features = (
                    context.model.encode_image(
                        images
                    )
                )
                del images

            else:
                numeric_inputs = batch[
                    "numeric"
                ].to(context.device)
                categorical_inputs = batch[
                    "categorical"
                ].to(context.device)
                labels = batch["target"].to(
                    dtype=torch.int64
                )
                _, shared_features = (
                    context.model.encode_tabular(
                        numeric_inputs,
                        categorical_inputs,
                    )
                )
                del numeric_inputs
                del categorical_inputs

            validate_shared_batch(
                shared_features,
                labels,
                task,
                split,
            )

            feature_batches.append(
                shared_features.detach().to(
                    device="cpu",
                    dtype=torch.float32,
                )
            )
            label_batches.append(
                labels.detach().to(
                    device="cpu",
                    dtype=torch.uint8,
                )
            )

            rows_completed += int(
                labels.shape[0]
            )

            if (
                batch_index
                % progress_frequency
                == 0
            ):
                print(
                    f"task={task} "
                    f"split={split} "
                    f"batch={batch_index} "
                    f"rows={rows_completed}"
                )

            del shared_features

    if not feature_batches:
        raise RuntimeError(
            "No locked features were extracted"
        )

    features = np.ascontiguousarray(
        torch.cat(
            feature_batches,
            dim=0,
        ).numpy(),
        dtype=np.float32,
    )
    labels = np.ascontiguousarray(
        torch.cat(
            label_batches,
            dim=0,
        ).numpy(),
        dtype=np.uint8,
    )

    validate_arrays(
        features,
        labels,
        expected_rows,
        SOURCE_DIMENSION,
    )

    return features, labels


def transform_to_8d(
    scaler: StandardScaler,
    pca: PCA,
    features_32d: np.ndarray,
    labels: np.ndarray,
    expected_rows: int,
) -> np.ndarray:
    original_label_hash = sha256_array(
        labels
    )

    scaled = scaler.transform(
        features_32d
    )

    if scaled.shape != features_32d.shape:
        raise RuntimeError(
            "Frozen scaler changed the feature shape"
        )

    if not np.isfinite(scaled).all():
        raise ValueError(
            "Frozen scaler produced non-finite values"
        )

    features_8d = pca.transform(
        scaled
    )
    features_8d = np.ascontiguousarray(
        features_8d,
        dtype=np.float32,
    )

    validate_arrays(
        features_8d,
        labels,
        expected_rows,
        OUTPUT_DIMENSION,
    )

    if sha256_array(labels) != original_label_hash:
        raise RuntimeError(
            "Dimensionality reduction changed label order"
        )

    return features_8d


def validate_saved_archive(
    path: Path,
    expected_features: np.ndarray,
    expected_labels: np.ndarray,
) -> None:
    with np.load(
        path,
        allow_pickle=False,
    ) as archive:
        if set(archive.files) != {
            "features",
            "labels",
        }:
            raise RuntimeError(
                "Saved archive contains unexpected arrays"
            )

        features = np.ascontiguousarray(
            archive["features"],
            dtype=np.float32,
        )
        labels = np.ascontiguousarray(
            archive["labels"],
            dtype=np.uint8,
        )

    if sha256_array(features) != sha256_array(
        expected_features
    ):
        raise RuntimeError(
            "Saved feature array hash mismatch"
        )

    if sha256_array(labels) != sha256_array(
        expected_labels
    ):
        raise RuntimeError(
            "Saved label array hash mismatch"
        )


def output_record(
    path: Path,
    features: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    return {
        "path": str(
            path.relative_to(PROJECT_ROOT)
        ),
        "filename": path.name,
        "file_sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
        "rows": int(features.shape[0]),
        "dimension": int(features.shape[1]),
        "feature_dtype": str(features.dtype),
        "label_dtype": str(labels.dtype),
        "features_sha256": sha256_array(
            features
        ),
        "labels_sha256": sha256_array(
            labels
        ),
        "archive_arrays": [
            "features",
            "labels",
        ],
        "patient_identifiers_in_output": False,
    }


def main() -> None:
    arguments = parse_arguments()

    if not arguments.confirm_final:
        raise PermissionError(
            "The final phase requires --confirm-final"
        )

    seed = int(arguments.seed)
    task = str(arguments.task)

    final_config = read_json_object(
        FINAL_CONFIG_PATH
    )
    validate_final_configuration(
        final_config
    )

    paths = task_paths(
        final_config,
        seed,
        task,
    )
    validate_required_inputs(paths)

    paths["output_32d"].parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    paths["output_8d"].parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    paths["audit"].parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    context = load_selected_feature_context(
        joint_config_path=paths["joint_config"],
        feature_config_path=paths["feature_config"],
        preprocessor_state_path=(
            paths["preprocessor_state"]
        ),
        checkpoint_path=paths["checkpoint"],
        history_path=paths["history"],
        seed=seed,
        requested_device=arguments.device,
    )

    (
        locked_split,
        expected_rows,
    ) = validate_task_contract(
        final_config,
        context,
        seed,
        task,
    )

    (
        scaler,
        pca,
        pca_record,
    ) = load_frozen_pca(
        seed,
        task,
        paths["pca_state"],
    )

    loader, loader_audit = (
        build_locked_loader(
            config=final_config,
            context=context,
            task=task,
            split=locked_split,
            expected_rows=expected_rows,
            data_source=paths["data_source"],
        )
    )

    features_32d, labels = (
        extract_locked_features(
            context=context,
            loader=loader,
            task=task,
            split=locked_split,
            expected_rows=expected_rows,
        )
    )

    features_8d = transform_to_8d(
        scaler=scaler,
        pca=pca,
        features_32d=features_32d,
        labels=labels,
        expected_rows=expected_rows,
    )

    label_hash = sha256_array(labels)

    write_archive_atomic(
        paths["output_32d"],
        features_32d,
        labels,
    )
    write_archive_atomic(
        paths["output_8d"],
        features_8d,
        labels,
    )

    validate_saved_archive(
        paths["output_32d"],
        features_32d,
        labels,
    )
    validate_saved_archive(
        paths["output_8d"],
        features_8d,
        labels,
    )

    output_32d_record = output_record(
        paths["output_32d"],
        features_32d,
        labels,
    )
    output_8d_record = output_record(
        paths["output_8d"],
        features_8d,
        labels,
    )

    if (
        output_32d_record["labels_sha256"]
        != label_hash
        or output_8d_record["labels_sha256"]
        != label_hash
    ):
        raise RuntimeError(
            "Locked label hash was not preserved"
        )

    audit = {
        "version": "1.0",
        "phase": "final_locked_test",
        "seed": seed,
        "task": task,
        "split": locked_split,
        "selected_epoch": (
            context.selected_epoch
        ),
        "expected_rows": expected_rows,
        "final_configuration": {
            "path": str(
                FINAL_CONFIG_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "sha256": sha256_file(
                FINAL_CONFIG_PATH
            ),
        },
        "protocol": {
            "path": str(
                paths["protocol"].relative_to(
                    PROJECT_ROOT
                )
            ),
            "sha256": sha256_file(
                paths["protocol"]
            ),
        },
        "frozen_model": {
            "checkpoint_path": str(
                paths["checkpoint"].relative_to(
                    PROJECT_ROOT
                )
            ),
            "checkpoint_sha256": sha256_file(
                paths["checkpoint"]
            ),
            "history_path": str(
                paths["history"].relative_to(
                    PROJECT_ROOT
                )
            ),
            "history_sha256": sha256_file(
                paths["history"]
            ),
            "joint_config_sha256": sha256_file(
                paths["joint_config"]
            ),
            "feature_config_sha256": sha256_file(
                paths["feature_config"]
            ),
            "preprocessor_state_sha256": sha256_file(
                paths["preprocessor_state"]
            ),
            "model_in_evaluation_mode": True,
            "model_parameters_trainable": False,
            "model_training_performed": False,
            "checkpoint_reselected": False,
        },
        "loader": loader_audit,
        "dimensionality_reduction": pca_record,
        "outputs": {
            "source_32d": output_32d_record,
            "reduced_8d": output_8d_record,
            "label_hash_preserved": True,
            "row_order_preserved": True,
        },
        "privacy": {
            "patient_identifiers_in_output": False,
            "raw_clinical_fields_in_output": False,
            "image_metadata_in_output": False,
            "private_archives_committed": False,
        },
        "test_protection": {
            "locked_test_loaded": True,
            "locked_test_evaluated": False,
            "locked_test_used_for_training": False,
            "locked_test_used_for_fitting": False,
            "locked_test_used_for_checkpoint_selection": False,
            "locked_test_used_for_threshold_selection": False,
            "locked_test_used_for_calibration": False,
            "locked_test_used_for_tuning": False,
        },
        "software": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "device": str(context.device),
        },
    }

    write_json_atomic(
        paths["audit"],
        audit,
    )

    print(f"seed={seed}")
    print(f"task={task}")
    print(f"split={locked_split}")
    print(f"rows={expected_rows}")
    print("source_dimension=32")
    print("output_dimension=8")
    print("locked_test_loaded=True")
    print("locked_test_evaluated=False")
    print(f"audit={paths['audit']}")

    del features_32d
    del features_8d
    del labels

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
