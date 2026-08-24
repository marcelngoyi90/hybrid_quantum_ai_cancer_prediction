from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "PYTORCH_ENABLE_MPS_FALLBACK",
    "1",
)

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.camelyon17_data import (
    Camelyon17ParquetDataset,
    build_transform,
)
from src.joint_checkpointing import (
    build_run_identity,
    validate_checkpoint_payload,
)
from src.joint_model import (
    build_joint_cancer_model,
)
from src.seer_preprocessing import (
    SeerPreprocessor,
    load_preprocessing_contract,
    read_seer_split,
)


EXPECTED_SHARED_DIMENSION = 32

SUPPORTED_TASKS = {
    "image",
    "tabular",
}

TEST_PROTECTION_FIELDS = (
    "camelyon_ood_test_evaluated",
    "seer_temporal_test_evaluated",
    "test_used_for_checkpoint_selection",
    "test_used_for_preprocessing",
    "test_used_for_tuning",
)


@dataclass(frozen=True)
class JointFeatureContext:
    model: torch.nn.Module
    preprocessor: SeerPreprocessor
    joint_config: dict[str, Any]
    seed: int
    device: torch.device
    selected_epoch: int
    checkpoint_audit: dict[str, Any]


@dataclass(frozen=True)
class ExtractedFeatureSet:
    task: str
    split: str
    seed: int
    features: np.ndarray
    labels: np.ndarray

    def validate(
        self,
        expected_rows: int,
        expected_dimension: int = EXPECTED_SHARED_DIMENSION,
    ) -> None:
        if self.task not in SUPPORTED_TASKS:
            raise ValueError(
                f"Unsupported feature task: {self.task}"
            )

        if not isinstance(self.seed, int):
            raise TypeError("Feature seed must be an integer")

        if self.features.ndim != 2:
            raise ValueError(
                "Extracted features must be two-dimensional"
            )

        if self.labels.ndim != 1:
            raise ValueError(
                "Extracted labels must be one-dimensional"
            )

        if self.features.shape[0] != self.labels.shape[0]:
            raise ValueError(
                "Feature and label row counts do not match"
            )

        if self.features.shape[0] != expected_rows:
            raise ValueError(
                "Unexpected extracted row count: "
                f"{self.features.shape[0]} versus {expected_rows}"
            )

        if self.features.shape[1] != expected_dimension:
            raise ValueError(
                "Unexpected shared feature dimension: "
                f"{self.features.shape[1]} versus "
                f"{expected_dimension}"
            )

        if self.features.dtype != np.float32:
            raise TypeError(
                "Extracted features must use float32"
            )

        if self.labels.dtype != np.uint8:
            raise TypeError(
                "Extracted labels must use uint8"
            )

        if not self.features.flags.c_contiguous:
            raise ValueError(
                "Extracted features must be contiguous"
            )

        if not self.labels.flags.c_contiguous:
            raise ValueError(
                "Extracted labels must be contiguous"
            )

        if not np.isfinite(self.features).all():
            raise ValueError(
                "Extracted features contain non-finite values"
            )

        observed_labels = set(
            int(value)
            for value in np.unique(self.labels)
        )

        if not observed_labels.issubset({0, 1}):
            raise ValueError(
                "Extracted labels are not binary"
            )

        if not observed_labels:
            raise ValueError(
                "Extracted feature set is empty"
            )

    def public_summary(self) -> dict[str, Any]:
        self.validate(
            expected_rows=int(self.features.shape[0]),
            expected_dimension=int(self.features.shape[1]),
        )

        positive_count = int(self.labels.sum())
        row_count = int(self.labels.shape[0])
        negative_count = row_count - positive_count

        return {
            "task": self.task,
            "split": self.split,
            "seed": self.seed,
            "rows": row_count,
            "feature_dimension": int(
                self.features.shape[1]
            ),
            "feature_dtype": str(self.features.dtype),
            "label_dtype": str(self.labels.dtype),
            "negative_count": negative_count,
            "positive_count": positive_count,
            "positive_percent": float(
                100.0 * positive_count / row_count
            ),
            "feature_minimum": float(
                self.features.min()
            ),
            "feature_maximum": float(
                self.features.max()
            ),
            "feature_mean": float(
                self.features.mean(dtype=np.float64)
            ),
            "feature_standard_deviation": float(
                self.features.std(dtype=np.float64)
            ),
            "features_sha256": sha256_array(
                self.features
            ),
            "labels_sha256": sha256_array(
                self.labels
            ),
            "patient_identifiers_in_output": False,
            "raw_clinical_fields_in_output": False,
            "image_metadata_in_output": False,
            "locked_test_loaded": False,
        }


def read_json_object(
    path: Path,
) -> dict[str, Any]:
    path = Path(path)

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
    path = Path(path)
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


def require_mapping(
    value: Any,
    location: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(
            f"{location} must be an object"
        )

    return value


def require_integer(
    value: Any,
    location: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"{location} must be an integer"
        )

    return value


def require_false(
    value: Any,
    location: str,
) -> None:
    if value is not False:
        raise RuntimeError(
            f"{location} must be false"
        )


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


def resolve_feature_device(
    requested_device: str | None = None,
) -> torch.device:
    if requested_device is None:
        if torch.backends.mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    normalized_device = requested_device.strip().lower()

    if normalized_device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS was requested but is not available"
            )

        return torch.device("mps")

    if normalized_device == "cpu":
        return torch.device("cpu")

    raise ValueError(
        "Feature extraction device must be mps or cpu"
    )


def validate_history_protection(
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
        data_summary.get(
            "seer_patient_identifiers_loaded"
        ),
        (
            f"seed {seed} data_summary."
            "seer_patient_identifiers_loaded"
        ),
    )

    require_false(
        data_summary.get(
            "camelyon_metadata_used_as_model_input"
        ),
        (
            f"seed {seed} data_summary."
            "camelyon_metadata_used_as_model_input"
        ),
    )

    epochs = history.get("epochs")

    if not isinstance(epochs, list) or not epochs:
        raise ValueError(
            f"Seed {seed} history has no epochs"
        )

    for epoch_index, epoch_value in enumerate(epochs):
        epoch = require_mapping(
            epoch_value,
            f"seed {seed} epochs[{epoch_index}]",
        )
        validation = require_mapping(
            epoch.get("validation"),
            (
                f"seed {seed} epochs[{epoch_index}]"
                ".validation"
            ),
        )

        require_false(
            validation.get("locked_tests_evaluated"),
            (
                f"seed {seed} epochs[{epoch_index}]"
                ".validation.locked_tests_evaluated"
            ),
        )


def validate_selected_checkpoint(
    checkpoint: Mapping[str, Any],
    history: Mapping[str, Any],
    seed: int,
) -> int:
    observed_seed = require_integer(
        history.get("seed"),
        f"seed {seed} history seed",
    )

    if observed_seed != seed:
        raise RuntimeError(
            f"History seed {observed_seed} does not "
            f"match requested seed {seed}"
        )

    if history.get("status") not in {
        "completed",
        "early_stopped",
    }:
        raise RuntimeError(
            f"Seed {seed} history is not complete"
        )

    best = require_mapping(
        history.get("best"),
        f"seed {seed} best",
    )

    selected_epoch = require_integer(
        best.get("epoch"),
        f"seed {seed} selected epoch",
    )

    best_score = require_finite_number(
        best.get("checkpoint_score"),
        f"seed {seed} best checkpoint score",
    )

    completed_epoch = require_integer(
        checkpoint.get("completed_epoch"),
        f"seed {seed} checkpoint completed epoch",
    )

    if completed_epoch != selected_epoch:
        raise RuntimeError(
            f"Seed {seed} best checkpoint contains "
            f"epoch {completed_epoch}, expected "
            f"epoch {selected_epoch}"
        )

    last_validation = require_mapping(
        checkpoint.get("last_validation"),
        f"seed {seed} checkpoint last_validation",
    )

    require_false(
        last_validation.get("locked_tests_evaluated"),
        (
            f"seed {seed} checkpoint "
            "last_validation.locked_tests_evaluated"
        ),
    )

    selection = require_mapping(
        last_validation.get("selection"),
        f"seed {seed} checkpoint selection",
    )

    checkpoint_score = require_finite_number(
        selection.get("checkpoint_score"),
        f"seed {seed} checkpoint selection score",
    )

    if not math.isclose(
        checkpoint_score,
        best_score,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            f"Seed {seed} checkpoint score does not "
            "match the public history"
        )

    return selected_epoch


def load_selected_feature_context(
    joint_config_path: Path,
    feature_config_path: Path,
    preprocessor_state_path: Path,
    checkpoint_path: Path,
    history_path: Path,
    seed: int,
    requested_device: str | None = None,
) -> JointFeatureContext:
    joint_config_path = Path(joint_config_path)
    feature_config_path = Path(feature_config_path)
    preprocessor_state_path = Path(
        preprocessor_state_path
    )
    checkpoint_path = Path(checkpoint_path)
    history_path = Path(history_path)
    seed = int(seed)

    (
        joint_config,
        feature_config,
    ) = load_preprocessing_contract(
        joint_config_path,
        feature_config_path,
    )

    configured_seeds = {
        int(value)
        for value in joint_config["random_seeds"]
    }

    if seed not in configured_seeds:
        raise ValueError(
            f"Seed {seed} is not in the frozen "
            "joint-model configuration"
        )

    history = read_json_object(history_path)
    validate_history_protection(
        history,
        seed,
    )

    preprocessor_state = read_json_object(
        preprocessor_state_path
    )

    preprocessor = SeerPreprocessor.load(
        preprocessor_state_path,
        joint_config,
        feature_config,
    )

    model = build_joint_cancer_model(
        joint_config,
        preprocessor_state,
    )

    expected_identity = build_run_identity(
        joint_config_path,
        preprocessor_state_path,
        seed,
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Selected checkpoint not found: "
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Checkpoint payload must be a dictionary"
        )

    validate_checkpoint_payload(
        checkpoint,
        expected_identity,
    )

    selected_epoch = validate_selected_checkpoint(
        checkpoint,
        history,
        seed,
    )

    model.load_state_dict(
        checkpoint["model_state"],
        strict=True,
    )

    if int(
        model.shared_trunk.output_dimension
    ) != EXPECTED_SHARED_DIMENSION:
        raise RuntimeError(
            "Joint model does not produce the expected "
            "32-dimensional shared representation"
        )

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    device = resolve_feature_device(
        requested_device
    )

    model = model.to(device)
    model.eval()

    if model.training:
        raise RuntimeError(
            "Joint model did not enter evaluation mode"
        )

    checkpoint_audit = {
        "seed": seed,
        "selected_epoch": selected_epoch,
        "checkpoint_filename": checkpoint_path.name,
        "checkpoint_sha256": sha256_file(
            checkpoint_path
        ),
        "history_filename": history_path.name,
        "history_sha256": sha256_file(
            history_path
        ),
        "joint_config_filename": (
            joint_config_path.name
        ),
        "joint_config_sha256": sha256_file(
            joint_config_path
        ),
        "preprocessor_state_filename": (
            preprocessor_state_path.name
        ),
        "preprocessor_state_sha256": sha256_file(
            preprocessor_state_path
        ),
        "shared_dimension": (
            EXPECTED_SHARED_DIMENSION
        ),
        "model_in_evaluation_mode": True,
        "model_parameters_trainable": False,
        "locked_tests_loaded": False,
        "locked_tests_evaluated": False,
        "patient_identifiers_loaded": False,
    }

    return JointFeatureContext(
        model=model,
        preprocessor=preprocessor,
        joint_config=joint_config,
        seed=seed,
        device=device,
        selected_epoch=selected_epoch,
        checkpoint_audit=checkpoint_audit,
    )


def allowed_development_splits(
    context: JointFeatureContext,
    task: str,
) -> set[str]:
    if task not in SUPPORTED_TASKS:
        raise ValueError(
            f"Unsupported feature task: {task}"
        )

    task_config = context.joint_config[
        "tasks"
    ][task]

    allowed = {
        task_config["training_split"],
        task_config["internal_validation_split"],
        task_config["selection_validation_split"],
    }

    locked_split = task_config[
        "locked_test_split"
    ]

    if locked_split in allowed:
        raise RuntimeError(
            f"{task} locked split entered the "
            "development split list"
        )

    return allowed


def validate_development_split(
    context: JointFeatureContext,
    task: str,
    split: str,
) -> None:
    allowed = allowed_development_splits(
        context,
        task,
    )

    locked_split = context.joint_config[
        "tasks"
    ][task][
        "locked_test_split"
    ]

    if split == locked_split:
        raise PermissionError(
            f"{task} locked test split cannot be "
            "loaded during feature development"
        )

    if split not in allowed:
        raise ValueError(
            f"Unexpected {task} development split: "
            f"{split}. Expected one of "
            f"{sorted(allowed)}"
        )


def build_image_feature_loader(
    context: JointFeatureContext,
    camelyon_data_root: Path,
    split: str,
) -> tuple[DataLoader, int, dict[str, Any]]:
    validate_development_split(
        context,
        "image",
        split,
    )

    camelyon_data_root = Path(
        camelyon_data_root
    )

    if not camelyon_data_root.is_dir():
        raise FileNotFoundError(
            "Camelyon data directory not found: "
            f"{camelyon_data_root}"
        )

    dataset = Camelyon17ParquetDataset(
        data_root=camelyon_data_root,
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
        ][
            "image_evaluation_batch_size"
        ]
    )
    number_of_workers = int(
        context.joint_config[
            "batching"
        ][
            "num_workers"
        ]
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=number_of_workers,
        drop_last=False,
        pin_memory=False,
    )

    expected_rows = int(len(dataset))

    audit = {
        "task": "image",
        "split": split,
        "expected_rows": expected_rows,
        "batch_size": batch_size,
        "shuffle": False,
        "training_augmentation": False,
        "drop_last": False,
        "patient_identifiers_retained": False,
        "image_metadata_used_as_model_input": False,
        "locked_test_loaded": False,
    }

    return loader, expected_rows, audit


def build_tabular_feature_loader(
    context: JointFeatureContext,
    seer_model_table_path: Path,
    split: str,
) -> tuple[DataLoader, int, dict[str, Any]]:
    validate_development_split(
        context,
        "tabular",
        split,
    )

    frame = read_seer_split(
        model_table_path=seer_model_table_path,
        split_name=split,
        preprocessor=context.preprocessor,
        allow_locked_test=False,
    )

    expected_columns = {
        context.preprocessor.split_column,
        *context.preprocessor.numeric_features,
        *context.preprocessor.categorical_features,
        context.preprocessor.target_column,
    }

    if set(frame.columns) != expected_columns:
        raise RuntimeError(
            "SEER split reader returned unexpected "
            "columns during feature extraction"
        )

    if "Patient ID" in frame.columns:
        raise RuntimeError(
            "SEER Patient ID entered feature extraction"
        )

    dataset, preprocessing_audit = (
        context.preprocessor.transform(
            frame,
            split,
        )
    )

    del frame

    require_false(
        preprocessing_audit.get(
            "patient_identifiers_in_tensors"
        ),
        (
            f"SEER {split} preprocessing audit "
            "patient_identifiers_in_tensors"
        ),
    )

    batch_size = int(
        context.joint_config[
            "batching"
        ][
            "tabular_evaluation_batch_size"
        ]
    )
    number_of_workers = int(
        context.joint_config[
            "batching"
        ][
            "num_workers"
        ]
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=number_of_workers,
        drop_last=False,
        pin_memory=False,
    )

    expected_rows = int(len(dataset))

    audit = {
        "task": "tabular",
        "split": split,
        "expected_rows": expected_rows,
        "batch_size": batch_size,
        "shuffle": False,
        "drop_last": False,
        "patient_identifiers_loaded": False,
        "patient_identifiers_in_tensors": False,
        "raw_clinical_fields_in_output": False,
        "locked_test_loaded": False,
        "preprocessing": preprocessing_audit,
    }

    return loader, expected_rows, audit


def validate_shared_batch(
    shared_features: torch.Tensor,
    labels: torch.Tensor,
    expected_dimension: int,
    task: str,
    split: str,
) -> None:
    if shared_features.ndim != 2:
        raise RuntimeError(
            f"{task} {split} shared features are not "
            "two-dimensional"
        )

    if shared_features.shape[1] != expected_dimension:
        raise RuntimeError(
            f"{task} {split} shared features have "
            "an unexpected dimension"
        )

    if labels.ndim != 1:
        raise RuntimeError(
            f"{task} {split} labels are not "
            "one-dimensional"
        )

    if shared_features.shape[0] != labels.shape[0]:
        raise RuntimeError(
            f"{task} {split} batch row counts differ"
        )

    if not shared_features.is_floating_point():
        raise TypeError(
            f"{task} {split} shared features must be "
            "floating point"
        )

    if not torch.isfinite(shared_features).all().item():
        raise ValueError(
            f"{task} {split} shared features contain "
            "non-finite values"
        )

    unique_labels = set(
        int(value)
        for value in torch.unique(
            labels.detach().to("cpu")
        ).tolist()
    )

    if not unique_labels.issubset({0, 1}):
        raise ValueError(
            f"{task} {split} labels are not binary"
        )


def extract_image_shared_features(
    context: JointFeatureContext,
    loader: DataLoader,
    split: str,
    expected_rows: int,
    progress_callback: (
        Callable[[int, int], None] | None
    ) = None,
) -> ExtractedFeatureSet:
    validate_development_split(
        context,
        "image",
        split,
    )

    context.model.eval()

    feature_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    rows_completed = 0

    with torch.inference_mode():
        for batch_index, batch in enumerate(
            loader,
            start=1,
        ):
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

            validate_shared_batch(
                shared_features=shared_features,
                labels=labels,
                expected_dimension=(
                    EXPECTED_SHARED_DIMENSION
                ),
                task="image",
                split=split,
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

            if progress_callback is not None:
                progress_callback(
                    batch_index,
                    rows_completed,
                )

            del images
            del shared_features

    if not feature_batches:
        raise RuntimeError(
            f"No image features were extracted for "
            f"{split}"
        )

    features_tensor = torch.cat(
        feature_batches,
        dim=0,
    )
    labels_tensor = torch.cat(
        label_batches,
        dim=0,
    )

    features = np.ascontiguousarray(
        features_tensor.numpy(),
        dtype=np.float32,
    )
    labels = np.ascontiguousarray(
        labels_tensor.numpy(),
        dtype=np.uint8,
    )

    extracted = ExtractedFeatureSet(
        task="image",
        split=split,
        seed=context.seed,
        features=features,
        labels=labels,
    )

    extracted.validate(
        expected_rows=expected_rows,
        expected_dimension=(
            EXPECTED_SHARED_DIMENSION
        ),
    )

    return extracted


def extract_tabular_shared_features(
    context: JointFeatureContext,
    loader: DataLoader,
    split: str,
    expected_rows: int,
    progress_callback: (
        Callable[[int, int], None] | None
    ) = None,
) -> ExtractedFeatureSet:
    validate_development_split(
        context,
        "tabular",
        split,
    )

    context.model.eval()

    feature_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    rows_completed = 0

    with torch.inference_mode():
        for batch_index, batch in enumerate(
            loader,
            start=1,
        ):
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
            labels = batch[
                "target"
            ].to(
                dtype=torch.int64
            )

            _, shared_features = (
                context.model.encode_tabular(
                    numeric_inputs,
                    categorical_inputs,
                )
            )

            validate_shared_batch(
                shared_features=shared_features,
                labels=labels,
                expected_dimension=(
                    EXPECTED_SHARED_DIMENSION
                ),
                task="tabular",
                split=split,
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

            if progress_callback is not None:
                progress_callback(
                    batch_index,
                    rows_completed,
                )

            del numeric_inputs
            del categorical_inputs
            del shared_features

    if not feature_batches:
        raise RuntimeError(
            f"No tabular features were extracted for "
            f"{split}"
        )

    features_tensor = torch.cat(
        feature_batches,
        dim=0,
    )
    labels_tensor = torch.cat(
        label_batches,
        dim=0,
    )

    features = np.ascontiguousarray(
        features_tensor.numpy(),
        dtype=np.float32,
    )
    labels = np.ascontiguousarray(
        labels_tensor.numpy(),
        dtype=np.uint8,
    )

    extracted = ExtractedFeatureSet(
        task="tabular",
        split=split,
        seed=context.seed,
        features=features,
        labels=labels,
    )

    extracted.validate(
        expected_rows=expected_rows,
        expected_dimension=(
            EXPECTED_SHARED_DIMENSION
        ),
    )

    return extracted


def extract_development_feature_set(
    context: JointFeatureContext,
    task: str,
    split: str,
    camelyon_data_root: Path | None = None,
    seer_model_table_path: Path | None = None,
    progress_callback: (
        Callable[[int, int], None] | None
    ) = None,
) -> tuple[
    ExtractedFeatureSet,
    dict[str, Any],
]:
    validate_development_split(
        context,
        task,
        split,
    )

    if task == "image":
        if camelyon_data_root is None:
            raise ValueError(
                "Camelyon data root is required for "
                "image feature extraction"
            )

        (
            loader,
            expected_rows,
            loader_audit,
        ) = build_image_feature_loader(
            context=context,
            camelyon_data_root=(
                camelyon_data_root
            ),
            split=split,
        )

        extracted = extract_image_shared_features(
            context=context,
            loader=loader,
            split=split,
            expected_rows=expected_rows,
            progress_callback=progress_callback,
        )

    elif task == "tabular":
        if seer_model_table_path is None:
            raise ValueError(
                "SEER model table path is required "
                "for tabular feature extraction"
            )

        (
            loader,
            expected_rows,
            loader_audit,
        ) = build_tabular_feature_loader(
            context=context,
            seer_model_table_path=(
                seer_model_table_path
            ),
            split=split,
        )

        extracted = (
            extract_tabular_shared_features(
                context=context,
                loader=loader,
                split=split,
                expected_rows=expected_rows,
                progress_callback=progress_callback,
            )
        )

    else:
        raise ValueError(
            f"Unsupported feature task: {task}"
        )

    audit = {
        "checkpoint": context.checkpoint_audit,
        "loader": loader_audit,
        "output": extracted.public_summary(),
        "locked_tests_loaded": False,
        "locked_tests_evaluated": False,
        "patient_identifiers_in_output": False,
    }

    return extracted, audit
