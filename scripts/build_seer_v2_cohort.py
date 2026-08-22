import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


project_root = Path(__file__).resolve().parents[1]
sys.path.insert(
    0,
    str(project_root),
)

from src.seer_cohort import (
    build_index_cohort,
)
from src.seer_data import (
    PARSED_YEAR_COLUMN,
    PATIENT_ID_COLUMN,
    TARGET_COLUMN,
)
from src.seer_data_v2 import (
    M_STAGE_CATEGORIES,
    M_STAGE_COLUMN,
    T_STAGE_CATEGORIES,
    T_STAGE_COLUMN,
    load_raw_seer_v2,
)


raw_csv_path = (
    project_root
    / "data"
    / "seer_raw"
    / "seer_breast_nodes_2010_2023_v2.csv"
)

baseline_cohort_path = (
    project_root
    / "data"
    / "processed"
    / "seer_index_cohort.parquet"
)

private_output_directory = (
    project_root
    / "data"
    / "processed"
)

private_output_path = (
    private_output_directory
    / "seer_v2_index_cohort.parquet"
)

metrics_directory = (
    project_root
    / "artifacts"
    / "metrics"
)

audit_path = (
    metrics_directory
    / "seer_v2_cohort_audit.json"
)

development_staging_summary_path = (
    metrics_directory
    / "seer_v2_development_staging_summary.csv"
)


def calculate_sha256(
    path,
):
    digest = hashlib.sha256()

    with Path(path).open(
        "rb"
    ) as file_handle:
        while True:
            chunk = file_handle.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

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
            "SEER v2 differs from the baseline "
            f"for {description}"
        ) from error


def validate_against_baseline(
    cohort,
    baseline,
):
    if len(cohort) != len(
        baseline
    ):
        raise RuntimeError(
            "SEER v2 and baseline cohort sizes "
            "are different"
        )

    assert_series_equal(
        cohort[PATIENT_ID_COLUMN],
        baseline[PATIENT_ID_COLUMN],
        "patient membership and ordering",
    )

    assert_series_equal(
        cohort[TARGET_COLUMN],
        baseline[TARGET_COLUMN],
        "prediction targets",
    )

    assert_series_equal(
        cohort[PARSED_YEAR_COLUMN],
        baseline[PARSED_YEAR_COLUMN],
        "diagnosis years",
    )

    if cohort[
        PATIENT_ID_COLUMN
    ].duplicated().any():
        raise RuntimeError(
            "SEER v2 cohort contains duplicate "
            "Patient IDs"
        )

    if cohort[
        TARGET_COLUMN
    ].isna().any():
        raise RuntimeError(
            "SEER v2 cohort contains missing "
            "targets"
        )

    for feature in (
        T_STAGE_COLUMN,
        M_STAGE_COLUMN,
    ):
        if feature not in cohort.columns:
            raise RuntimeError(
                "SEER v2 cohort is missing "
                f"{feature}"
            )

        if cohort[feature].isna().any():
            raise RuntimeError(
                "SEER v2 cohort contains missing "
                f"values in {feature}"
            )

    actual_t_categories = set(
        cohort[
            T_STAGE_COLUMN
        ].unique().tolist()
    )
    actual_m_categories = set(
        cohort[
            M_STAGE_COLUMN
        ].unique().tolist()
    )

    if not actual_t_categories.issubset(
        T_STAGE_CATEGORIES
    ):
        raise RuntimeError(
            "SEER v2 contains an unexpected "
            "harmonized T category"
        )

    if not actual_m_categories.issubset(
        M_STAGE_CATEGORIES
    ):
        raise RuntimeError(
            "SEER v2 contains an unexpected "
            "harmonized M category"
        )

    return {
        "records_equal": True,
        "patient_membership_equal": True,
        "patient_order_equal": True,
        "targets_equal": True,
        "diagnosis_years_equal": True,
    }


def validate_persisted_cohort(
    original,
    persisted,
):
    if len(original) != len(
        persisted
    ):
        raise RuntimeError(
            "Persisted SEER v2 cohort row "
            "count changed"
        )

    if tuple(original.columns) != tuple(
        persisted.columns
    ):
        raise RuntimeError(
            "Persisted SEER v2 cohort columns "
            "changed"
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
            "Persisted SEER v2 cohort values "
            "changed"
        ) from error


def build_development_staging_summary(
    cohort,
):
    development = cohort.loc[
        cohort[
            PARSED_YEAR_COLUMN
        ].le(2021)
    ]

    if development.empty:
        raise ValueError(
            "SEER v2 development cohort is empty"
        )

    rows = []

    for feature in (
        T_STAGE_COLUMN,
        M_STAGE_COLUMN,
    ):
        counts = (
            development.groupby(
                [
                    feature,
                    TARGET_COLUMN,
                ],
                observed=True,
                dropna=False,
            )
            .size()
            .reset_index(
                name="patients"
            )
        )

        for record in counts.to_dict(
            orient="records"
        ):
            rows.append(
                {
                    "feature": feature,
                    "category": str(
                        record[feature]
                    ),
                    "target": int(
                        record[TARGET_COLUMN]
                    ),
                    "patients": int(
                        record["patients"]
                    ),
                }
            )

    summary = pd.DataFrame(
        rows
    )

    return summary.sort_values(
        [
            "feature",
            "category",
            "target",
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )


def main():
    if not baseline_cohort_path.is_file():
        raise FileNotFoundError(
            "Baseline cohort was not found: "
            f"{baseline_cohort_path}"
        )

    private_output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    metrics_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_dataframe = load_raw_seer_v2(
        raw_csv_path
    )

    cohort, cohort_audit = (
        build_index_cohort(
            raw_dataframe
        )
    )

    baseline = pd.read_parquet(
        baseline_cohort_path,
        engine="pyarrow",
    )

    equivalence_audit = (
        validate_against_baseline(
            cohort,
            baseline,
        )
    )

    cohort.to_parquet(
        private_output_path,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )

    persisted = pd.read_parquet(
        private_output_path,
        engine="pyarrow",
    )

    validate_persisted_cohort(
        cohort,
        persisted,
    )

    development_staging_summary = (
        build_development_staging_summary(
            persisted
        )
    )

    development_staging_summary.to_csv(
        development_staging_summary_path,
        index=False,
    )

    audit = {
        "version": "2.0",
        "cohort": cohort_audit,
        "baseline_equivalence": (
            equivalence_audit
        ),
        "harmonized_features": {
            T_STAGE_COLUMN: sorted(
                T_STAGE_CATEGORIES
            ),
            M_STAGE_COLUMN: sorted(
                M_STAGE_CATEGORIES
            ),
        },
        "development_staging_summary": {
            "maximum_year_included": 2021,
            "locked_test_years_excluded": [
                2022,
                2023,
            ],
            "rows": int(
                len(
                    development_staging_summary
                )
            ),
        },
        "test_protection": {
            "locked_test_used_for_model_selection": False,
            "locked_test_distribution_reported": False,
            "locked_test_performance_evaluated": False,
        },
        "files": {
            "raw_v2_csv": {
                "file_name": (
                    raw_csv_path.name
                ),
                "size_bytes": int(
                    raw_csv_path.stat().st_size
                ),
                "sha256": calculate_sha256(
                    raw_csv_path
                ),
            },
            "baseline_private_cohort": {
                "file_name": (
                    baseline_cohort_path.name
                ),
                "size_bytes": int(
                    baseline_cohort_path
                    .stat()
                    .st_size
                ),
                "sha256": calculate_sha256(
                    baseline_cohort_path
                ),
            },
            "private_v2_cohort": {
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
        file_handle.write(
            "\n"
        )

    print(
        f"final_index_records="
        f"{len(persisted)}"
    )
    print(
        "baseline_patient_membership_equal=True"
    )
    print(
        "baseline_targets_equal=True"
    )
    print(
        f"private_v2_cohort="
        f"{private_output_path}"
    )
    print(
        f"audit={audit_path}"
    )


if __name__ == "__main__":
    main()
