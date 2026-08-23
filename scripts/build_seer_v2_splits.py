import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.seer_data import (
    PARSED_YEAR_COLUMN,
    PATIENT_ID_COLUMN,
    TARGET_COLUMN,
)
from src.seer_splits import (
    load_split_config,
    summarize_splits,
)


v2_cohort_path = (
    project_root
    / "data"
    / "processed"
    / "seer_v2_index_cohort.parquet"
)

baseline_split_path = (
    project_root
    / "data"
    / "processed"
    / "seer_index_cohort_with_splits.parquet"
)

config_path = (
    project_root
    / "configs"
    / "seer_splits.json"
)

private_output_path = (
    project_root
    / "data"
    / "processed"
    / "seer_v2_index_cohort_with_splits.parquet"
)

metrics_directory = (
    project_root
    / "artifacts"
    / "metrics"
)

audit_path = (
    metrics_directory
    / "seer_v2_split_audit.json"
)

split_summary_path = (
    metrics_directory
    / "seer_v2_split_summary.csv"
)


def calculate_sha256(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as file_handle:
        while True:
            chunk = file_handle.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def assert_series_equal(
    left,
    right,
    description,
):
    try:
        pd.testing.assert_series_equal(
            left.reset_index(drop=True),
            right.reset_index(drop=True),
            check_dtype=False,
            check_names=False,
            check_exact=True,
        )
    except AssertionError as error:
        raise RuntimeError(
            f"SEER v2 differs from the frozen "
            f"baseline for {description}"
        ) from error


def validate_inputs(
    v2_cohort,
    baseline,
    config,
):
    split_column = config["split_column"]

    required_v2_columns = {
        PATIENT_ID_COLUMN,
        PARSED_YEAR_COLUMN,
        TARGET_COLUMN,
        "t_stage_harmonized",
        "m_stage_harmonized",
    }

    missing_v2_columns = sorted(
        required_v2_columns
        - set(v2_cohort.columns)
    )

    if missing_v2_columns:
        raise ValueError(
            "SEER v2 cohort is missing columns: "
            f"{missing_v2_columns}"
        )

    required_baseline_columns = {
        PATIENT_ID_COLUMN,
        PARSED_YEAR_COLUMN,
        TARGET_COLUMN,
        split_column,
    }

    missing_baseline_columns = sorted(
        required_baseline_columns
        - set(baseline.columns)
    )

    if missing_baseline_columns:
        raise ValueError(
            "Baseline split cohort is missing "
            f"columns: {missing_baseline_columns}"
        )

    if split_column in v2_cohort.columns:
        raise ValueError(
            "SEER v2 cohort already contains "
            "split assignments"
        )

    if len(v2_cohort) != len(baseline):
        raise RuntimeError(
            "SEER v2 and baseline split cohorts "
            "have different row counts"
        )

    for name, dataframe in (
        ("SEER v2 cohort", v2_cohort),
        ("baseline split cohort", baseline),
    ):
        if dataframe[
            PATIENT_ID_COLUMN
        ].isna().any():
            raise ValueError(
                f"{name} contains missing "
                "Patient IDs"
            )

        if dataframe[
            PATIENT_ID_COLUMN
        ].duplicated().any():
            raise ValueError(
                f"{name} contains duplicate "
                "Patient IDs"
            )

    v2_patient_ids = set(
        v2_cohort[
            PATIENT_ID_COLUMN
        ].tolist()
    )

    baseline_patient_ids = set(
        baseline[
            PATIENT_ID_COLUMN
        ].tolist()
    )

    if v2_patient_ids != baseline_patient_ids:
        raise RuntimeError(
            "SEER v2 patient membership differs "
            "from the frozen baseline"
        )

    assert_series_equal(
        v2_cohort[PATIENT_ID_COLUMN],
        baseline[PATIENT_ID_COLUMN],
        "patient ordering",
    )

    assert_series_equal(
        v2_cohort[TARGET_COLUMN],
        baseline[TARGET_COLUMN],
        "prediction targets",
    )

    assert_series_equal(
        v2_cohort[PARSED_YEAR_COLUMN],
        baseline[PARSED_YEAR_COLUMN],
        "diagnosis years",
    )

    if baseline[split_column].isna().any():
        raise ValueError(
            "Baseline contains missing split "
            "assignments"
        )

    expected_split_names = {
        config["training_name"],
        config["internal_validation_name"],
        config["temporal_validation_name"],
        config["temporal_test_name"],
    }

    actual_split_names = set(
        baseline[split_column].unique()
    )

    if actual_split_names != expected_split_names:
        raise RuntimeError(
            "Baseline split names do not match "
            "the frozen configuration"
        )


def attach_frozen_splits(
    v2_cohort,
    baseline,
    config,
):
    split_column = config["split_column"]

    split_mapping = (
        baseline[
            [
                PATIENT_ID_COLUMN,
                split_column,
            ]
        ]
        .set_index(PATIENT_ID_COLUMN)[
            split_column
        ]
    )

    assigned = v2_cohort.copy()

    assigned[split_column] = (
        assigned[PATIENT_ID_COLUMN]
        .map(split_mapping)
        .astype("string")
    )

    if assigned[split_column].isna().any():
        raise RuntimeError(
            "At least one SEER v2 patient did "
            "not receive a frozen split"
        )

    assert_series_equal(
        assigned[split_column],
        baseline[split_column],
        "split assignments",
    )

    return assigned


def validate_split_years(
    assigned,
    config,
):
    split_column = config["split_column"]

    expected_ranges = {
        config["training_name"]: (
            config["development_year_start"],
            config["development_year_end"],
        ),
        config["internal_validation_name"]: (
            config["development_year_start"],
            config["development_year_end"],
        ),
        config["temporal_validation_name"]: (
            config[
                "temporal_validation_year_start"
            ],
            config[
                "temporal_validation_year_end"
            ],
        ),
        config["temporal_test_name"]: (
            config["temporal_test_year_start"],
            config["temporal_test_year_end"],
        ),
    }

    for split_name, (
        minimum_year,
        maximum_year,
    ) in expected_ranges.items():
        subset = assigned.loc[
            assigned[split_column].eq(
                split_name
            )
        ]

        if subset.empty:
            raise RuntimeError(
                f"Split {split_name} is empty"
            )

        valid_years = subset[
            PARSED_YEAR_COLUMN
        ].between(
            minimum_year,
            maximum_year,
            inclusive="both",
        )

        if not valid_years.all():
            raise RuntimeError(
                f"Split {split_name} contains "
                "patients outside its frozen "
                "year range"
            )


def validate_persisted_output(
    assigned,
    persisted,
    baseline,
    config,
):
    split_column = config["split_column"]

    if len(assigned) != len(persisted):
        raise RuntimeError(
            "Persisted SEER v2 split row count "
            "changed"
        )

    if tuple(assigned.columns) != tuple(
        persisted.columns
    ):
        raise RuntimeError(
            "Persisted SEER v2 split columns "
            "changed"
        )

    try:
        pd.testing.assert_frame_equal(
            assigned,
            persisted,
            check_dtype=False,
            check_exact=True,
        )
    except AssertionError as error:
        raise RuntimeError(
            "Persisted SEER v2 split values "
            "changed"
        ) from error

    assert_series_equal(
        persisted[PATIENT_ID_COLUMN],
        baseline[PATIENT_ID_COLUMN],
        "persisted patient ordering",
    )

    assert_series_equal(
        persisted[TARGET_COLUMN],
        baseline[TARGET_COLUMN],
        "persisted prediction targets",
    )

    assert_series_equal(
        persisted[PARSED_YEAR_COLUMN],
        baseline[PARSED_YEAR_COLUMN],
        "persisted diagnosis years",
    )

    assert_series_equal(
        persisted[split_column],
        baseline[split_column],
        "persisted split assignments",
    )


def build_split_summary_table(
    assigned,
    config,
):
    summaries = summarize_splits(
        assigned,
        config,
    )

    ordered_split_names = [
        config["training_name"],
        config["internal_validation_name"],
        config["temporal_validation_name"],
        config["temporal_test_name"],
    ]

    rows = []

    for split_name in ordered_split_names:
        summary = summaries[split_name]

        rows.append(
            {
                "split": split_name,
                "patients": int(
                    summary["patients"]
                ),
                "n0": int(
                    summary["n0"]
                ),
                "n_positive": int(
                    summary["n_positive"]
                ),
                "n_positive_percent": float(
                    summary[
                        "n_positive_percent"
                    ]
                ),
                "minimum_year": int(
                    summary["minimum_year"]
                ),
                "maximum_year": int(
                    summary["maximum_year"]
                ),
            }
        )

    return pd.DataFrame(rows)


def main():
    for required_path in (
        v2_cohort_path,
        baseline_split_path,
        config_path,
    ):
        if not required_path.is_file():
            raise FileNotFoundError(
                f"Required file not found: "
                f"{required_path}"
            )

    metrics_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = load_split_config(
        config_path
    )

    v2_cohort = pd.read_parquet(
        v2_cohort_path,
        engine="pyarrow",
    )

    baseline = pd.read_parquet(
        baseline_split_path,
        engine="pyarrow",
    )

    validate_inputs(
        v2_cohort,
        baseline,
        config,
    )

    assigned = attach_frozen_splits(
        v2_cohort,
        baseline,
        config,
    )

    validate_split_years(
        assigned,
        config,
    )

    baseline_summaries = summarize_splits(
        baseline,
        config,
    )

    v2_summaries = summarize_splits(
        assigned,
        config,
    )

    if baseline_summaries != v2_summaries:
        raise RuntimeError(
            "SEER v2 split summaries differ "
            "from the frozen baseline"
        )

    assigned.to_parquet(
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
        assigned,
        persisted,
        baseline,
        config,
    )

    split_summary = (
        build_split_summary_table(
            persisted,
            config,
        )
    )

    split_summary.to_csv(
        split_summary_path,
        index=False,
    )

    audit = {
        "version": "2.0",
        "assignment": {
            "method": (
                "copy_frozen_v1_assignments_"
                "by_patient_id"
            ),
            "new_split_generation_performed": (
                False
            ),
            "split_configuration_changed": False,
        },
        "baseline_equivalence": {
            "record_count_equal": True,
            "patient_membership_equal": True,
            "patient_order_equal": True,
            "prediction_targets_equal": True,
            "diagnosis_years_equal": True,
            "split_assignments_equal": True,
            "split_summaries_equal": True,
        },
        "split_summary": (
            split_summary.to_dict(
                orient="records"
            )
        ),
        "test_protection": {
            "temporal_test_locked": True,
            "test_features_summarized": False,
            "test_used_for_model_selection": False,
            "test_performance_evaluated": False,
        },
        "files": {
            "v2_private_cohort": {
                "file_name": (
                    v2_cohort_path.name
                ),
                "size_bytes": int(
                    v2_cohort_path.stat().st_size
                ),
                "sha256": calculate_sha256(
                    v2_cohort_path
                ),
            },
            "baseline_private_split_cohort": {
                "file_name": (
                    baseline_split_path.name
                ),
                "size_bytes": int(
                    baseline_split_path
                    .stat()
                    .st_size
                ),
                "sha256": calculate_sha256(
                    baseline_split_path
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
            "private_v2_split_cohort": {
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

    for row in split_summary.itertuples(
        index=False
    ):
        print(
            f"split={row.split} "
            f"patients={row.patients} "
            f"n_positive_percent="
            f"{row.n_positive_percent}"
        )

    print(
        "baseline_split_assignments_equal=True"
    )
    print(
        f"private_v2_split_cohort="
        f"{private_output_path}"
    )
    print(
        f"audit={audit_path}"
    )


if __name__ == "__main__":
    main()
