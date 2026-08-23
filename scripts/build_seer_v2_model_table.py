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
from src.seer_splits import (
    load_split_config,
)


v2_split_cohort_path = (
    project_root
    / "data"
    / "processed"
    / "seer_v2_index_cohort_with_splits.parquet"
)

baseline_model_table_path = (
    project_root
    / "data"
    / "processed"
    / "seer_model_table.parquet"
)

v2_feature_config_path = (
    project_root
    / "configs"
    / "seer_features_v2.json"
)

baseline_feature_config_path = (
    project_root
    / "configs"
    / "seer_features.json"
)

split_config_path = (
    project_root
    / "configs"
    / "seer_splits.json"
)

private_output_path = (
    project_root
    / "data"
    / "processed"
    / "seer_v2_model_table.parquet"
)

metrics_directory = (
    project_root
    / "artifacts"
    / "metrics"
)

audit_path = (
    metrics_directory
    / "seer_v2_feature_audit.json"
)


EXPECTED_ADDED_FEATURES = {
    "t_stage_harmonized",
    "m_stage_harmonized",
}


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


def expected_model_columns(config):
    return (
        config["metadata_columns"]
        + config["model_numeric_features"]
        + config["model_categorical_features"]
        + [TARGET_COLUMN]
    )


def validate_feature_amendment(
    baseline_config,
    v2_config,
):
    baseline_numeric = set(
        baseline_config[
            "model_numeric_features"
        ]
    )

    v2_numeric = set(
        v2_config[
            "model_numeric_features"
        ]
    )

    if baseline_numeric != v2_numeric:
        raise RuntimeError(
            "SEER v2 changed the numeric "
            "feature contract"
        )

    baseline_categorical = set(
        baseline_config[
            "model_categorical_features"
        ]
    )

    v2_categorical = set(
        v2_config[
            "model_categorical_features"
        ]
    )

    added_features = (
        v2_categorical
        - baseline_categorical
    )

    removed_features = (
        baseline_categorical
        - v2_categorical
    )

    if added_features != EXPECTED_ADDED_FEATURES:
        raise RuntimeError(
            "Unexpected SEER v2 feature "
            f"additions: {sorted(added_features)}"
        )

    if removed_features:
        raise RuntimeError(
            "SEER v2 removed baseline features: "
            f"{sorted(removed_features)}"
        )

    if (
        v2_config["target_column"]
        != baseline_config["target_column"]
    ):
        raise RuntimeError(
            "SEER v2 changed the prediction "
            "target"
        )

    if (
        v2_config["split_column"]
        != baseline_config["split_column"]
    ):
        raise RuntimeError(
            "SEER v2 changed the split column"
        )

    if (
        v2_config["metadata_columns"]
        != baseline_config["metadata_columns"]
    ):
        raise RuntimeError(
            "SEER v2 changed the metadata "
            "columns"
        )


def validate_model_table(
    model_table,
    config,
):
    expected_columns = (
        expected_model_columns(config)
    )

    if list(model_table.columns) != (
        expected_columns
    ):
        raise RuntimeError(
            "SEER v2 model-table columns "
            "do not match the feature contract"
        )

    if model_table.empty:
        raise ValueError(
            "SEER v2 model table is empty"
        )

    if model_table[
        PATIENT_ID_COLUMN
    ].isna().any():
        raise ValueError(
            "SEER v2 model table contains "
            "missing Patient IDs"
        )

    if model_table[
        PATIENT_ID_COLUMN
    ].duplicated().any():
        raise ValueError(
            "SEER v2 model table contains "
            "duplicate Patient IDs"
        )

    if model_table[
        TARGET_COLUMN
    ].isna().any():
        raise ValueError(
            "SEER v2 model table contains "
            "missing targets"
        )

    unexpected_targets = sorted(
        set(
            model_table[
                TARGET_COLUMN
            ].astype("int64").unique()
        )
        - {0, 1}
    )

    if unexpected_targets:
        raise ValueError(
            "SEER v2 model table contains "
            f"unexpected targets: "
            f"{unexpected_targets}"
        )

    split_column = config["split_column"]

    if model_table[
        split_column
    ].isna().any():
        raise ValueError(
            "SEER v2 model table contains "
            "missing split assignments"
        )

    for feature in config[
        "model_categorical_features"
    ]:
        if model_table[feature].isna().any():
            raise ValueError(
                "SEER v2 categorical feature "
                f"contains missing values: "
                f"{feature}"
            )

    forbidden_inputs = set(
        config["forbidden_model_inputs"]
    )

    actual_inputs = set(
        config["model_numeric_features"]
        + config[
            "model_categorical_features"
        ]
    )

    forbidden_overlap = sorted(
        actual_inputs & forbidden_inputs
    )

    if forbidden_overlap:
        raise RuntimeError(
            "Forbidden fields entered the "
            "SEER v2 model table: "
            f"{forbidden_overlap}"
        )


def validate_against_baseline(
    v2_model_table,
    baseline_model_table,
    baseline_config,
):
    baseline_columns = (
        expected_model_columns(
            baseline_config
        )
    )

    if list(
        baseline_model_table.columns
    ) != baseline_columns:
        raise RuntimeError(
            "Baseline model-table columns "
            "changed"
        )

    if len(v2_model_table) != len(
        baseline_model_table
    ):
        raise RuntimeError(
            "SEER v2 and baseline model tables "
            "have different row counts"
        )

    try:
        pd.testing.assert_frame_equal(
            v2_model_table[
                baseline_columns
            ].reset_index(drop=True),
            baseline_model_table[
                baseline_columns
            ].reset_index(drop=True),
            check_dtype=False,
            check_exact=True,
        )
    except AssertionError as error:
        raise RuntimeError(
            "A baseline SEER model value "
            "changed in v2"
        ) from error

    return {
        "record_count_equal": True,
        "patient_order_equal": True,
        "metadata_equal": True,
        "baseline_features_equal": True,
        "prediction_targets_equal": True,
        "split_assignments_equal": True,
        "only_added_features": sorted(
            EXPECTED_ADDED_FEATURES
        ),
    }


def validate_persisted_output(
    original,
    persisted,
    config,
):
    validate_model_table(
        persisted,
        config,
    )

    if len(original) != len(persisted):
        raise RuntimeError(
            "Persisted SEER v2 model-table "
            "row count changed"
        )

    if tuple(original.columns) != tuple(
        persisted.columns
    ):
        raise RuntimeError(
            "Persisted SEER v2 model-table "
            "columns changed"
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
            "Persisted SEER v2 model-table "
            "values changed"
        ) from error


def build_development_audit(
    model_table,
    feature_config,
    split_config,
):
    split_column = feature_config[
        "split_column"
    ]

    locked_split = split_config[
        "temporal_test_name"
    ]

    development = model_table.loc[
        ~model_table[split_column].eq(
            locked_split
        )
    ].copy()

    expected_development_splits = {
        split_config["training_name"],
        split_config[
            "internal_validation_name"
        ],
        split_config[
            "temporal_validation_name"
        ],
    }

    actual_development_splits = set(
        development[split_column].unique()
    )

    if (
        actual_development_splits
        != expected_development_splits
    ):
        raise RuntimeError(
            "Development audit split coverage "
            "is incorrect"
        )

    if development[
        split_column
    ].eq(locked_split).any():
        raise RuntimeError(
            "Locked temporal test entered the "
            "development feature audit"
        )

    audit = build_feature_audit(
        development,
        feature_config,
    )

    audit["audit_scope"] = (
        "train_id_val_temporal_val_only"
    )
    audit["locked_split_excluded"] = (
        locked_split
    )

    return audit


def main():
    required_paths = (
        v2_split_cohort_path,
        baseline_model_table_path,
        v2_feature_config_path,
        baseline_feature_config_path,
        split_config_path,
    )

    for required_path in required_paths:
        if not required_path.is_file():
            raise FileNotFoundError(
                f"Required file not found: "
                f"{required_path}"
            )

    metrics_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    v2_feature_config = (
        load_feature_config(
            v2_feature_config_path
        )
    )

    baseline_feature_config = (
        load_feature_config(
            baseline_feature_config_path
        )
    )

    split_config = load_split_config(
        split_config_path
    )

    validate_feature_amendment(
        baseline_feature_config,
        v2_feature_config,
    )

    v2_split_cohort = pd.read_parquet(
        v2_split_cohort_path,
        engine="pyarrow",
    )

    baseline_model_table = (
        pd.read_parquet(
            baseline_model_table_path,
            engine="pyarrow",
        )
    )

    model_table, _ = (
        build_seer_model_table(
            v2_split_cohort,
            v2_feature_config,
        )
    )

    validate_model_table(
        model_table,
        v2_feature_config,
    )

    baseline_equivalence = (
        validate_against_baseline(
            model_table,
            baseline_model_table,
            baseline_feature_config,
        )
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

    validate_persisted_output(
        model_table,
        persisted,
        v2_feature_config,
    )

    development_audit = (
        build_development_audit(
            persisted,
            v2_feature_config,
            split_config,
        )
    )

    audit = {
        "version": "2.0",
        "full_private_model_table": {
            "patients": int(
                len(persisted)
            ),
            "columns": int(
                len(persisted.columns)
            ),
            "model_feature_count": int(
                len(
                    v2_feature_config[
                        "model_numeric_features"
                    ]
                )
                + len(
                    v2_feature_config[
                        "model_categorical_features"
                    ]
                )
            ),
            "numeric_features": list(
                v2_feature_config[
                    "model_numeric_features"
                ]
            ),
            "categorical_features": list(
                v2_feature_config[
                    "model_categorical_features"
                ]
            ),
        },
        "baseline_equivalence": (
            baseline_equivalence
        ),
        "development_feature_audit": (
            development_audit
        ),
        "test_protection": {
            "temporal_test_locked": True,
            "test_rows_materialized_for_final_use": (
                True
            ),
            "test_feature_distributions_summarized": (
                False
            ),
            "test_used_for_feature_selection": False,
            "test_used_for_model_selection": False,
            "test_performance_evaluated": False,
        },
        "files": {
            "v2_private_split_cohort": {
                "file_name": (
                    v2_split_cohort_path.name
                ),
                "size_bytes": int(
                    v2_split_cohort_path
                    .stat()
                    .st_size
                ),
                "sha256": calculate_sha256(
                    v2_split_cohort_path
                ),
            },
            "baseline_private_model_table": {
                "file_name": (
                    baseline_model_table_path
                    .name
                ),
                "size_bytes": int(
                    baseline_model_table_path
                    .stat()
                    .st_size
                ),
                "sha256": calculate_sha256(
                    baseline_model_table_path
                ),
            },
            "v2_feature_configuration": {
                "file_name": (
                    v2_feature_config_path.name
                ),
                "size_bytes": int(
                    v2_feature_config_path
                    .stat()
                    .st_size
                ),
                "sha256": calculate_sha256(
                    v2_feature_config_path
                ),
            },
            "baseline_feature_configuration": {
                "file_name": (
                    baseline_feature_config_path
                    .name
                ),
                "size_bytes": int(
                    baseline_feature_config_path
                    .stat()
                    .st_size
                ),
                "sha256": calculate_sha256(
                    baseline_feature_config_path
                ),
            },
            "split_configuration": {
                "file_name": (
                    split_config_path.name
                ),
                "size_bytes": int(
                    split_config_path
                    .stat()
                    .st_size
                ),
                "sha256": calculate_sha256(
                    split_config_path
                ),
            },
            "private_v2_model_table": {
                "file_name": (
                    private_output_path.name
                ),
                "rows": int(
                    len(persisted)
                ),
                "columns": int(
                    len(persisted.columns)
                ),
                "size_bytes": int(
                    private_output_path
                    .stat()
                    .st_size
                ),
                "sha256": calculate_sha256(
                    private_output_path
                ),
            },
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
            allow_nan=False,
        )
        file_handle.write("\n")

    print(
        f"patients={len(persisted)}"
    )
    print(
        "model_features="
        f"{audit['full_private_model_table']['model_feature_count']}"
    )
    print(
        "baseline_features_equal=True"
    )
    print(
        "added_features="
        "t_stage_harmonized,"
        "m_stage_harmonized"
    )
    print(
        "locked_temporal_test_summarized=False"
    )
    print(
        f"private_model_table="
        f"{private_output_path}"
    )
    print(
        f"audit={audit_path}"
    )


if __name__ == "__main__":
    main()
