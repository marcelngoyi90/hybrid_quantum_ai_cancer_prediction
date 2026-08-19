import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.seer_cohort import build_index_cohort
from src.seer_data import (
    PARSED_AGE_COLUMN,
    PARSED_YEAR_COLUMN,
    PATIENT_ID_COLUMN,
    TARGET_COLUMN,
    load_raw_seer,
)


raw_csv_path = (
    project_root
    / "data"
    / "seer_raw"
    / "seer_breast_nodes_2010_2023_codes.csv"
)

private_output_directory = (
    project_root
    / "data"
    / "processed"
)

private_cohort_path = (
    private_output_directory
    / "seer_index_cohort.parquet"
)

metrics_directory = (
    project_root
    / "artifacts"
    / "metrics"
)

audit_path = (
    metrics_directory
    / "seer_cohort_audit.json"
)

year_summary_path = (
    metrics_directory
    / "seer_cohort_year_target_summary.csv"
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


def build_year_summary(cohort):
    summary = (
        cohort.groupby(
            PARSED_YEAR_COLUMN,
            observed=True,
        )[TARGET_COLUMN]
        .agg(
            patients="size",
            n_positive="sum",
        )
        .reset_index()
        .rename(
            columns={
                PARSED_YEAR_COLUMN: "year",
            }
        )
    )

    summary["year"] = summary[
        "year"
    ].astype("int64")

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

    return summary[
        [
            "year",
            "patients",
            "n0",
            "n_positive",
            "n_positive_percent",
        ]
    ]


def validate_persisted_cohort(
    original_cohort,
    persisted_cohort,
):
    if len(original_cohort) != len(
        persisted_cohort
    ):
        raise RuntimeError(
            "Persisted cohort row count changed"
        )

    if tuple(original_cohort.columns) != tuple(
        persisted_cohort.columns
    ):
        raise RuntimeError(
            "Persisted cohort columns changed"
        )

    if persisted_cohort[
        PATIENT_ID_COLUMN
    ].duplicated().any():
        raise RuntimeError(
            "Persisted cohort contains "
            "duplicate Patient IDs"
        )

    if persisted_cohort[
        PARSED_AGE_COLUMN
    ].lt(18).any():
        raise RuntimeError(
            "Persisted cohort contains "
            "under-age patients"
        )

    if persisted_cohort[
        TARGET_COLUMN
    ].isna().any():
        raise RuntimeError(
            "Persisted cohort contains "
            "missing targets"
        )

    original_target_counts = (
        original_cohort[
            TARGET_COLUMN
        ]
        .value_counts(dropna=False)
        .sort_index()
        .to_dict()
    )

    persisted_target_counts = (
        persisted_cohort[
            TARGET_COLUMN
        ]
        .value_counts(dropna=False)
        .sort_index()
        .to_dict()
    )

    if (
        original_target_counts
        != persisted_target_counts
    ):
        raise RuntimeError(
            "Persisted target counts changed"
        )


def main():
    private_output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    metrics_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_dataframe = load_raw_seer(
        raw_csv_path
    )

    cohort, audit = build_index_cohort(
        raw_dataframe
    )

    cohort.to_parquet(
        private_cohort_path,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )

    persisted_cohort = pd.read_parquet(
        private_cohort_path,
        engine="pyarrow",
    )

    validate_persisted_cohort(
        cohort,
        persisted_cohort,
    )

    year_summary = build_year_summary(
        persisted_cohort
    )

    audit["source_file"] = {
        "file_name": raw_csv_path.name,
        "size_bytes": int(
            raw_csv_path.stat().st_size
        ),
        "sha256": calculate_sha256(
            raw_csv_path
        ),
    }

    audit["private_output"] = {
        "file_name": private_cohort_path.name,
        "size_bytes": int(
            private_cohort_path.stat().st_size
        ),
        "sha256": calculate_sha256(
            private_cohort_path
        ),
        "rows": int(
            len(persisted_cohort)
        ),
        "columns": int(
            persisted_cohort.shape[1]
        ),
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

    year_summary.to_csv(
        year_summary_path,
        index=False,
    )

    print(
        "final_index_records="
        f"{audit['flow']['final_index_records']}"
    )
    print(
        "records_removed_as_non_index="
        f"{audit['flow']['records_removed_as_non_index']}"
    )
    print(
        "under_age_records="
        f"{audit['flow']['under_age_records']}"
    )
    print(
        "n0="
        f"{audit['index_cohort']['n0']}"
    )
    print(
        "n_positive="
        f"{audit['index_cohort']['n_positive']}"
    )
    print(
        "n_positive_percent="
        f"{audit['index_cohort']['n_positive_percent']}"
    )
    print(
        f"private_cohort={private_cohort_path}"
    )
    print(f"audit={audit_path}")


if __name__ == "__main__":
    main()