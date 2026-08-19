import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.seer_data import (
    PATIENT_ID_COLUMN,
    TARGET_COLUMN,
)
from src.seer_features import (
    build_feature_audit,
    build_seer_model_table,
    load_feature_config,
)


cohort_path = (
    project_root
    / "data"
    / "processed"
    / "seer_index_cohort_with_splits.parquet"
)

config_path = (
    project_root
    / "configs"
    / "seer_features.json"
)

private_output_directory = (
    project_root
    / "data"
    / "processed"
)

private_output_path = (
    private_output_directory
    / "seer_model_table.parquet"
)

metrics_directory = (
    project_root
    / "artifacts"
    / "metrics"
)

audit_path = (
    metrics_directory
    / "seer_feature_audit.json"
)


def calculate_sha256(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as file_handle:
        while True:
            chunk = file_handle.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def validate_persisted_model_table(
    original,
    persisted,
    config,
):
    expected_columns = (
        config["metadata_columns"]
        + config["model_numeric_features"]
        + config["model_categorical_features"]
        + [TARGET_COLUMN]
    )

    if len(original) != len(persisted):
        raise RuntimeError(
            "Persisted model-table row count changed"
        )

    if list(persisted.columns) != expected_columns:
        raise RuntimeError(
            "Persisted model-table columns changed"
        )

    if persisted[
        PATIENT_ID_COLUMN
    ].duplicated().any():
        raise RuntimeError(
            "Persisted model table contains "
            "duplicate Patient IDs"
        )

    if persisted[
        TARGET_COLUMN
    ].isna().any():
        raise RuntimeError(
            "Persisted model table contains "
            "missing targets"
        )

    split_column = config["split_column"]

    if persisted[
        split_column
    ].isna().any():
        raise RuntimeError(
            "Persisted model table contains "
            "missing split assignments"
        )

    for feature in config[
        "model_categorical_features"
    ]:
        if persisted[feature].isna().any():
            raise RuntimeError(
                "Persisted categorical feature "
                f"contains missing values: {feature}"
            )

    try:
        pd.testing.assert_frame_equal(
            original,
            persisted,
            check_dtype=False,
            check_exact=True,
        )
    except AssertionError as error:
        raise RuntimeError(
            "Persisted model-table values changed"
        ) from error


def main():
    if not cohort_path.is_file():
        raise FileNotFoundError(
            f"Split cohort not found: {cohort_path}"
        )

    private_output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    cohort = pd.read_parquet(
        cohort_path,
        engine="pyarrow",
    )

    config = load_feature_config(
        config_path
    )

    model_table, _ = build_seer_model_table(
        cohort,
        config,
    )

    model_table.to_parquet(
        private_output_path,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )

    persisted = pd.read_parquet(
        private_output_path,
        engine="pyarrow",
    )

    validate_persisted_model_table(
        model_table,
        persisted,
        config,
    )

    audit = build_feature_audit(
        persisted,
        config,
    )

    audit["files"] = {
        "input_split_cohort": {
            "file_name": cohort_path.name,
            "size_bytes": int(
                cohort_path.stat().st_size
            ),
            "sha256": calculate_sha256(
                cohort_path
            ),
        },
        "configuration": {
            "file_name": config_path.name,
            "size_bytes": int(
                config_path.stat().st_size
            ),
            "sha256": calculate_sha256(
                config_path
            ),
        },
        "private_model_table": {
            "file_name": private_output_path.name,
            "size_bytes": int(
                private_output_path.stat().st_size
            ),
            "sha256": calculate_sha256(
                private_output_path
            ),
            "rows": int(len(persisted)),
            "columns": int(
                persisted.shape[1]
            ),
        },
    }

    with audit_path.open(
        "w",
        encoding="utf-8",
    ) as file_handle:
        json.dump(
            audit,
            file_handle,
            indent=2,
            sort_keys=True,
        )
        file_handle.write("\n")

    print(f"patients={len(persisted)}")
    print(
        "model_features="
        f"{audit['model_feature_count']}"
    )
    print(
        "numeric_features="
        f"{len(audit['numeric_features'])}"
    )
    print(
        "categorical_features="
        f"{len(audit['categorical_features'])}"
    )
    print(
        f"private_model_table="
        f"{private_output_path}"
    )
    print(f"audit={audit_path}")


if __name__ == "__main__":
    main()