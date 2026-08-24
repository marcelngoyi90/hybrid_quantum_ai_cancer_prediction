from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping
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

from src.joint_feature_extraction import (
    EXPECTED_SHARED_DIMENSION,
    ExtractedFeatureSet,
    extract_development_feature_set,
    load_selected_feature_context,
    sha256_array,
)


JOINT_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "joint_model_v2.json"
)
FEATURE_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "seer_features_v2.json"
)
PREPROCESSOR_STATE_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "features"
    / "seer_v2_preprocessor.json"
)
CAMELYON_DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "camelyon17_hf"
    / "data"
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
HISTORY_DIRECTORY = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "joint_v2"
)
PRIVATE_FEATURE_DIRECTORY = (
    PROJECT_ROOT
    / "artifacts"
    / "features"
    / "joint_v2"
)
PUBLIC_AUDIT_DIRECTORY = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "joint_v2"
    / "feature_extraction"
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


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract protected 32-dimensional "
            "development features from joint model v2"
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


def write_feature_archive_atomic(
    path: Path,
    extracted: ExtractedFeatureSet,
) -> None:
    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary_path.open("wb") as file:
        np.savez_compressed(
            file,
            features=extracted.features,
            labels=extracted.labels,
        )

    temporary_path.replace(path)


def output_paths(
    seed: int,
    task: str,
    split: str,
) -> tuple[Path, Path]:
    private_directory = (
        PRIVATE_FEATURE_DIRECTORY
        / f"seed{seed}"
    )
    private_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    PUBLIC_AUDIT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_path = (
        private_directory
        / f"{task}_{split}_32d.npz"
    )
    audit_path = (
        PUBLIC_AUDIT_DIRECTORY
        / f"seed{seed}_{task}_{split}_32d_audit.json"
    )

    return feature_path, audit_path


def manifest_path(
    seed: int,
    task: str,
) -> Path:
    PUBLIC_AUDIT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        PUBLIC_AUDIT_DIRECTORY
        / f"seed{seed}_{task}_32d_manifest.json"
    )


def ordered_development_splits(
    joint_config: Mapping[str, Any],
    task: str,
) -> tuple[str, str, str]:
    task_config = require_mapping(
        joint_config["tasks"][task],
        f"{task} task configuration",
    )

    splits = (
        str(task_config["training_split"]),
        str(
            task_config[
                "internal_validation_split"
            ]
        ),
        str(
            task_config[
                "selection_validation_split"
            ]
        ),
    )

    if len(set(splits)) != len(splits):
        raise RuntimeError(
            f"{task} development splits are not unique"
        )

    locked_split = str(
        task_config["locked_test_split"]
    )

    if locked_split in splits:
        raise RuntimeError(
            f"{task} locked test entered development "
            "feature extraction"
        )

    return splits


def progress_callback(
    task: str,
    split: str,
):
    frequency = (
        200 if task == "image" else 10
    )

    def report(
        batch_index: int,
        rows_completed: int,
    ) -> None:
        if batch_index % frequency == 0:
            print(
                f"task={task} "
                f"split={split} "
                f"batch={batch_index} "
                f"rows={rows_completed}"
            )

    return report


def validate_archive(
    feature_path: Path,
    audit_path: Path,
    expected_seed: int,
    expected_task: str,
    expected_split: str,
) -> dict[str, Any]:
    if not feature_path.is_file():
        raise FileNotFoundError(
            f"Private feature file not found: "
            f"{feature_path}"
        )

    audit = read_json_object(
        audit_path
    )

    if int(audit.get("seed")) != expected_seed:
        raise RuntimeError(
            "Existing feature audit seed mismatch"
        )

    if audit.get("task") != expected_task:
        raise RuntimeError(
            "Existing feature audit task mismatch"
        )

    if audit.get("split") != expected_split:
        raise RuntimeError(
            "Existing feature audit split mismatch"
        )

    private_output = require_mapping(
        audit.get("private_output"),
        "private_output",
    )

    expected_file_hash = private_output.get(
        "file_sha256"
    )
    observed_file_hash = sha256_file(
        feature_path
    )

    if observed_file_hash != expected_file_hash:
        raise RuntimeError(
            "Existing private feature file hash "
            "does not match its audit"
        )

    with np.load(
        feature_path,
        allow_pickle=False,
    ) as archive:
        if set(archive.files) != {
            "features",
            "labels",
        }:
            raise RuntimeError(
                "Private feature archive contains "
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

    output_summary = require_mapping(
        audit["extraction"]["output"],
        "extraction output summary",
    )

    extracted = ExtractedFeatureSet(
        task=expected_task,
        split=expected_split,
        seed=expected_seed,
        features=features,
        labels=labels,
    )
    extracted.validate(
        expected_rows=int(
            output_summary["rows"]
        ),
        expected_dimension=(
            EXPECTED_SHARED_DIMENSION
        ),
    )

    if sha256_array(features) != (
        output_summary["features_sha256"]
    ):
        raise RuntimeError(
            "Existing feature array hash mismatch"
        )

    if sha256_array(labels) != (
        output_summary["labels_sha256"]
    ):
        raise RuntimeError(
            "Existing label array hash mismatch"
        )

    require_false(
        audit["privacy"][
            "patient_identifiers_in_output"
        ],
        "privacy.patient_identifiers_in_output",
    )
    require_false(
        audit["privacy"][
            "raw_clinical_fields_in_output"
        ],
        "privacy.raw_clinical_fields_in_output",
    )
    require_false(
        audit["test_protection"][
            "locked_test_loaded"
        ],
        "test_protection.locked_test_loaded",
    )
    require_false(
        audit["test_protection"][
            "locked_test_evaluated"
        ],
        "test_protection.locked_test_evaluated",
    )

    return audit


def save_extracted_split(
    extracted: ExtractedFeatureSet,
    extraction_audit: Mapping[str, Any],
    feature_path: Path,
    audit_path: Path,
) -> dict[str, Any]:
    if feature_path.exists():
        raise FileExistsError(
            f"Private feature output already exists: "
            f"{feature_path}"
        )

    if audit_path.exists():
        raise FileExistsError(
            f"Public feature audit already exists: "
            f"{audit_path}"
        )

    extracted.validate(
        expected_rows=int(
            extracted.features.shape[0]
        ),
        expected_dimension=(
            EXPECTED_SHARED_DIMENSION
        ),
    )

    write_feature_archive_atomic(
        feature_path,
        extracted,
    )

    audit = {
        "version": "1.0",
        "seed": extracted.seed,
        "task": extracted.task,
        "split": extracted.split,
        "representation": (
            "joint_shared_trunk_output"
        ),
        "feature_dimension": (
            EXPECTED_SHARED_DIMENSION
        ),
        "private_output": {
            "path": str(
                feature_path.relative_to(
                    PROJECT_ROOT
                )
            ),
            "filename": feature_path.name,
            "file_sha256": sha256_file(
                feature_path
            ),
            "size_bytes": int(
                feature_path.stat().st_size
            ),
            "archive_arrays": [
                "features",
                "labels"
            ],
        },
        "extraction": dict(
            extraction_audit
        ),
        "privacy": {
            "patient_identifiers_loaded": False,
            "patient_identifiers_in_output": False,
            "raw_clinical_fields_in_output": False,
            "image_metadata_in_output": False,
            "original_images_in_output": False,
            "private_feature_file_committed": False,
        },
        "test_protection": {
            "locked_test_loaded": False,
            "locked_test_evaluated": False,
            "locked_test_used_for_extraction": False,
        },
    }

    write_json_atomic(
        audit_path,
        audit,
    )

    return validate_archive(
        feature_path=feature_path,
        audit_path=audit_path,
        expected_seed=extracted.seed,
        expected_task=extracted.task,
        expected_split=extracted.split,
    )


def main() -> None:
    arguments = parse_arguments()

    seed = int(arguments.seed)
    task = str(arguments.task)

    joint_config = read_json_object(
        JOINT_CONFIG_PATH
    )

    configured_seeds = tuple(
        int(value)
        for value in joint_config[
            "random_seeds"
        ]
    )

    if configured_seeds != EXPECTED_SEEDS:
        raise RuntimeError(
            "Joint-model seed list differs from the "
            "frozen extraction seed list"
        )

    if task not in SUPPORTED_TASKS:
        raise ValueError(
            f"Unsupported extraction task: {task}"
        )

    checkpoint_path = (
        CHECKPOINT_DIRECTORY
        / f"joint_seed{seed}_best.pt"
    )
    history_path = (
        HISTORY_DIRECTORY
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

    splits = ordered_development_splits(
        joint_config=context.joint_config,
        task=task,
    )

    output_manifest_path = manifest_path(
        seed,
        task,
    )

    if (
        output_manifest_path.exists()
        and not arguments.resume
    ):
        raise FileExistsError(
            "Feature manifest already exists. Use "
            "--resume to validate and continue the "
            f"matching extraction: "
            f"{output_manifest_path}"
        )

    split_audits: list[dict[str, Any]] = []

    for split in splits:
        (
            feature_path,
            audit_path,
        ) = output_paths(
            seed,
            task,
            split,
        )

        feature_exists = (
            feature_path.exists()
        )
        audit_exists = (
            audit_path.exists()
        )

        if feature_exists != audit_exists:
            raise RuntimeError(
                "Feature and audit existence differ "
                f"for seed={seed}, task={task}, "
                f"split={split}"
            )

        if feature_exists:
            if not arguments.resume:
                raise FileExistsError(
                    "Feature output already exists. "
                    "Use --resume to validate and "
                    f"continue: {feature_path}"
                )

            audit = validate_archive(
                feature_path=feature_path,
                audit_path=audit_path,
                expected_seed=seed,
                expected_task=task,
                expected_split=split,
            )
            split_audits.append(audit)

            print(
                f"seed={seed} "
                f"task={task} "
                f"split={split} "
                "status=validated_existing"
            )
            continue

        (
            extracted,
            extraction_audit,
        ) = extract_development_feature_set(
            context=context,
            task=task,
            split=split,
            camelyon_data_root=(
                CAMELYON_DATA_ROOT
                if task == "image"
                else None
            ),
            seer_model_table_path=(
                SEER_MODEL_TABLE_PATH
                if task == "tabular"
                else None
            ),
            progress_callback=(
                progress_callback(
                    task,
                    split,
                )
            ),
        )

        audit = save_extracted_split(
            extracted=extracted,
            extraction_audit=(
                extraction_audit
            ),
            feature_path=feature_path,
            audit_path=audit_path,
        )
        split_audits.append(audit)

        output_summary = audit[
            "extraction"
        ][
            "output"
        ]

        print(
            f"seed={seed} "
            f"task={task} "
            f"split={split} "
            f"rows={output_summary['rows']} "
            f"dimension="
            f"{output_summary['feature_dimension']} "
            "status=extracted"
        )

        del extracted

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    manifest = {
        "version": "1.0",
        "seed": seed,
        "task": task,
        "selected_epoch": (
            context.selected_epoch
        ),
        "representation": (
            "joint_shared_trunk_output"
        ),
        "feature_dimension": (
            EXPECTED_SHARED_DIMENSION
        ),
        "development_splits": list(
            splits
        ),
        "split_audits": [
            {
                "split": audit["split"],
                "rows": audit[
                    "extraction"
                ][
                    "output"
                ][
                    "rows"
                ],
                "feature_dimension": audit[
                    "feature_dimension"
                ],
                "private_feature_path": audit[
                    "private_output"
                ][
                    "path"
                ],
                "private_feature_sha256": audit[
                    "private_output"
                ][
                    "file_sha256"
                ],
                "public_audit_filename": (
                    f"seed{seed}_{task}_"
                    f"{audit['split']}_32d_audit.json"
                ),
            }
            for audit in split_audits
        ],
        "checkpoint": (
            context.checkpoint_audit
        ),
        "privacy": {
            "patient_identifiers_loaded": False,
            "patient_identifiers_in_output": False,
            "patient_level_rows_in_public_outputs": False,
            "private_features_committed": False,
        },
        "test_protection": {
            "locked_test_loaded": False,
            "locked_test_evaluated": False,
            "locked_test_extracted": False,
        },
    }

    write_json_atomic(
        output_manifest_path,
        manifest,
    )

    print(f"seed={seed}")
    print(f"task={task}")
    print(
        f"selected_epoch="
        f"{context.selected_epoch}"
    )
    print(
        f"shared_dimension="
        f"{EXPECTED_SHARED_DIMENSION}"
    )
    print("locked_tests_loaded=False")
    print(
        "patient_identifiers_in_output=False"
    )
    print(f"manifest={output_manifest_path}")


if __name__ == "__main__":
    main()
