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
    TEMPORARY_HASH_COLUMN,
    assign_seer_splits,
    load_split_config,
)


cohort_path = (
    project_root
    / "data"
    / "processed"
    / "seer_index_cohort.parquet"
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
    / "seer_index_cohort_with_splits.parquet"
)

metrics_directory = (
    project_root
    / "artifacts"
    / "metrics"
)

audit_path = (
    metrics_directory
    / "seer_split_audit.json"
)

split_summary_path = (
    metrics_directory
    / "seer_split_summary.csv"
)

year_target_summary_path = (
    metrics_directory
    / "seer_split_year_target_summary.csv"
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


def build_split_summary(
    assigned,
    config,
):
    split_column = config["split_column"]

    split_order = [
        config["training_name"],
        config["internal_validation_name"],
        config["temporal_validation_name"],
        config["temporal_test_name"],
    ]

    summary_rows = []

    for split_name in split_order:
        subset = assigned.loc[
            assigned[split_column].eq(
                split_name
            )
        ]

        n0_count = int(
            subset[TARGET_COLUMN].eq(0).sum()
        )
        n_positive_count = int(
            subset[TARGET_COLUMN].eq(1).sum()
        )

        summary_rows.append(
            {
                "split": split_name,
                "patients": int(
                    len(subset)
                ),
                "n0": n0_count,
                "n_positive": n_positive_count,
                "n_positive_percent": round(
                    (
                        n_positive_count
                        / len(subset)
                        * 100
                    ),
                    4,
                ),
                "minimum_year": int(
                    subset[
                        PARSED_YEAR_COLUMN
                    ].min()
                ),
                "maximum_year": int(
                    subset[
                        PARSED_YEAR_COLUMN
                    ].max()
                ),
            }
        )

    return pd.DataFrame(summary_rows)


def build_year_target_summary(
    assigned,
    config,
):
    split_column = config["split_column"]

    summary = (
        assigned.groupby(
            [
                split_column,
                PARSED_YEAR_COLUMN,
            ],
            observed=True,
        )[TARGET_COLUMN]
        .agg(
            patients="size",
            n_positive="sum",
        )
        .reset_index()
        .rename(
            columns={
                split_column: "split",
                PARSED_YEAR_COLUMN: "year",
            }
        )
    )

    summary["patients"] = summary[
        "patients"
    ].astype("int64")

    summary["n_positive"] = summary[
        "n_positive"
    ].astype("int64")

    summary["n0"] = (
        summary["patients"]
        - summary["n_positive"]
    )

    summary["n_positive_percent"] = (
        summary["n_positive"]
        / summary["patients"]
        * 100
    ).round(4)

    split_order = {
        config["training_name"]: 0,
        config["internal_validation_name"]: 1,
        config["temporal_validation_name"]: 2,
        config["temporal_test_name"]: 3,
    }

    summary["_split_order"] = summary[
        "split"
    ].map(split_order)

    summary = summary.sort_values(
        [
            "_split_order",
            "year",
        ]
    ).drop(
        columns=["_split_order"]
    )

    return summary[
        [
            "split",
            "year",
            "patients",
            "n0",
            "n_positive",
            "n_positive_percent",
        ]
    ]


def validate_persisted_splits(
    assigned,
    persisted,
    config,
):
    split_column = config["split_column"]

    if len(assigned) != len(persisted):
        raise RuntimeError(
            "Persisted split row count changed"
        )

    if tuple(assigned.columns) != tuple(
        persisted.columns
    ):
        raise RuntimeError(
            "Persisted split columns changed"
        )

    if TEMPORARY_HASH_COLUMN in (
        persisted.columns
    ):
        raise RuntimeError(
            "Temporary split hash was persisted"
        )

    if persisted[
        PATIENT_ID_COLUMN
    ].duplicated().any():
        raise RuntimeError(
            "Persisted splits contain "
            "duplicate Patient IDs"
        )

    if persisted[split_column].isna().any():
        raise RuntimeError(
            "Persisted splits contain "
            "missing assignments"
        )

    comparison_columns = [
        PATIENT_ID_COLUMN,
        split_column,
        PARSED_YEAR_COLUMN,
        TARGET_COLUMN,
    ]

    original_comparison = (
        assigned[
            comparison_columns
        ]
        .sort_values(PATIENT_ID_COLUMN)
        .reset_index(drop=True)
    )

    persisted_comparison = (
        persisted[
            comparison_columns
        ]
        .sort_values(PATIENT_ID_COLUMN)
        .reset_index(drop=True)
    )

    try:
        pd.testing.assert_frame_equal(
            original_comparison,
            persisted_comparison,
            check_dtype=False,
            check_exact=True,
        )
    except AssertionError as error:
        raise RuntimeError(
            "Persisted patient split "
            "assignments changed"
        ) from error

    expected_split_names = {
        config["training_name"],
        config["internal_validation_name"],
        config["temporal_validation_name"],
        config["temporal_test_name"],
    }

    actual_split_names = set(
        persisted[split_column].unique()
    )

    if actual_split_names != (
        expected_split_names
    ):
        raise RuntimeError(
            "Persisted split names changed"
        )


def main():
    if not cohort_path.is_file():
        raise FileNotFoundError(
            f"Private cohort not found: "
            f"{cohort_path}"
        )

    metrics_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    cohort = pd.read_parquet(
        cohort_path,
        engine="pyarrow",
    )

    config = load_split_config(
        config_path
    )

    assigned, audit = assign_seer_splits(
        cohort,
        config,
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

    validate_persisted_splits(
        assigned,
        persisted,
        config,
    )

    split_summary = build_split_summary(
        persisted,
        config,
    )

    year_target_summary = (
        build_year_target_summary(
            persisted,
            config,
        )
    )

    audit["files"] = {
        "input_cohort": {
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
        "private_split_cohort": {
            "file_name": (
                private_output_path.name
            ),
            "size_bytes": int(
                private_output_path.stat().st_size
            ),
            "sha256": calculate_sha256(
                private_output_path
            ),
            "rows": int(
                len(persisted)
            ),
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

    split_summary.to_csv(
        split_summary_path,
        index=False,
    )

    year_target_summary.to_csv(
        year_target_summary_path,
        index=False,
    )

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
        f"private_split_cohort="
        f"{private_output_path}"
    )
    print(f"audit={audit_path}")


if __name__ == "__main__":
    main()