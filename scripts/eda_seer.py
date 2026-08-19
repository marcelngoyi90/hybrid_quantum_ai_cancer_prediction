import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.seer_data import (
    AGE_COLUMN,
    CONFIRMATION_COLUMN,
    ER_COLUMN,
    EXPECTED_COLUMNS,
    HER2_COLUMN,
    HISTOLOGY_COLUMN,
    LATERALITY_COLUMN,
    NEW_GRADE_COLUMN,
    NODES_POSITIVE_COLUMN,
    OLD_GRADE_COLUMN,
    ORIGIN_COLUMN,
    PARSED_AGE_COLUMN,
    PARSED_YEAR_COLUMN,
    PATIENT_ID_COLUMN,
    PR_COLUMN,
    PRIMARY_SITE_COLUMN,
    RACE_COLUMN,
    SUBTYPE_COLUMN,
    TARGET_COLUMN,
    TUMOR_SIZE_COLUMN,
    UNIFIED_GRADE_COLUMN,
    add_derived_fields,
    load_raw_seer,
)


raw_csv_path = (
    project_root
    / "data"
    / "seer_raw"
    / "seer_breast_nodes_2010_2023_codes.csv"
)

dictionary_path = raw_csv_path.with_suffix(".dic")

metrics_directory = (
    project_root
    / "artifacts"
    / "metrics"
)

figures_directory = (
    project_root
    / "artifacts"
    / "figures"
)


SUBGROUP_COLUMNS = (
    RACE_COLUMN,
    ORIGIN_COLUMN,
    PRIMARY_SITE_COLUMN,
    HISTOLOGY_COLUMN,
    LATERALITY_COLUMN,
    UNIFIED_GRADE_COLUMN,
    ER_COLUMN,
    PR_COLUMN,
    HER2_COLUMN,
    SUBTYPE_COLUMN,
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


def blank_mask(values):
    normalized = values.str.strip().str.casefold()

    return normalized.isin(
        {
            "",
            "blank(s)",
            "na",
            "n/a",
        }
    )


def unknown_mask(values):
    normalized = values.str.strip().str.casefold()

    return normalized.str.contains(
        (
            r"unknown|not available|not applicable|"
            r"not stated|no information"
        ),
        regex=True,
        na=False,
    )


def build_column_quality(dataframe):
    quality_rows = []

    for column in EXPECTED_COLUMNS:
        values = dataframe[column]

        value_counts = values.value_counts(
            dropna=False
        )

        most_common_value = str(
            value_counts.index[0]
        )
        most_common_count = int(
            value_counts.iloc[0]
        )

        if column == PATIENT_ID_COLUMN:
            most_common_value = "<suppressed>"

        current_blank_mask = blank_mask(values)
        current_unknown_mask = unknown_mask(values)
        missing_like_mask = (
            current_blank_mask
            | current_unknown_mask
        )

        quality_rows.append(
            {
                "column": column,
                "unique_values": int(
                    values.nunique(dropna=False)
                ),
                "blank_values": int(
                    current_blank_mask.sum()
                ),
                "unknown_values": int(
                    current_unknown_mask.sum()
                ),
                "missing_like_values": int(
                    missing_like_mask.sum()
                ),
                "missing_like_percent": round(
                    float(
                        missing_like_mask.mean()
                        * 100
                    ),
                    4,
                ),
                "most_common_value": (
                    most_common_value
                ),
                "most_common_count": (
                    most_common_count
                ),
                "most_common_percent": round(
                    (
                        most_common_count
                        / len(dataframe)
                        * 100
                    ),
                    4,
                ),
            }
        )

    return pd.DataFrame(quality_rows)


def build_year_target_summary(dataframe):
    valid_rows = dataframe.dropna(
        subset=[
            PARSED_YEAR_COLUMN,
            TARGET_COLUMN,
        ]
    )

    summary = (
        valid_rows.groupby(
            PARSED_YEAR_COLUMN,
            observed=True,
        )[TARGET_COLUMN]
        .agg(
            records="size",
            n_positive="sum",
        )
        .reset_index()
        .rename(
            columns={
                PARSED_YEAR_COLUMN: "year",
            }
        )
    )

    summary["year"] = summary["year"].astype(
        "int64"
    )
    summary["records"] = summary[
        "records"
    ].astype("int64")
    summary["n_positive"] = summary[
        "n_positive"
    ].astype("int64")

    summary["n0"] = (
        summary["records"]
        - summary["n_positive"]
    )

    summary["n_positive_percent"] = (
        summary["n_positive"]
        / summary["records"]
        * 100
    ).round(4)

    return summary[
        [
            "year",
            "records",
            "n0",
            "n_positive",
            "n_positive_percent",
        ]
    ]


def build_node_code_counts(dataframe):
    counts = (
        dataframe.groupby(
            [
                NODES_POSITIVE_COLUMN,
                TARGET_COLUMN,
            ],
            dropna=False,
            observed=True,
        )
        .size()
        .reset_index(name="records")
    )

    counts["target_label"] = (
        counts[TARGET_COLUMN]
        .map(
            {
                0: "N0",
                1: "N+",
            }
        )
        .fillna("excluded")
    )

    counts["record_percent"] = (
        counts["records"]
        / len(dataframe)
        * 100
    ).round(6)

    return counts[
        [
            NODES_POSITIVE_COLUMN,
            "target_label",
            "records",
            "record_percent",
        ]
    ].sort_values(
        NODES_POSITIVE_COLUMN
    )


def build_patient_multiplicity(dataframe):
    record_counts = dataframe.groupby(
        PATIENT_ID_COLUMN,
        sort=False,
    ).size()

    distribution = (
        record_counts.value_counts()
        .sort_index()
        .rename_axis("records_per_patient")
        .reset_index(name="patients")
    )

    distribution["patient_percent"] = (
        distribution["patients"]
        / record_counts.size
        * 100
    ).round(6)

    return distribution


def build_subgroup_target_summary(dataframe):
    subgroup_tables = []

    for column in SUBGROUP_COLUMNS:
        table = (
            dataframe.dropna(
                subset=[TARGET_COLUMN]
            )
            .groupby(
                column,
                dropna=False,
                observed=True,
            )[TARGET_COLUMN]
            .agg(
                records="size",
                n_positive="sum",
            )
            .reset_index()
            .rename(
                columns={
                    column: "category",
                }
            )
        )

        table["records"] = table[
            "records"
        ].astype("int64")
        table["n_positive"] = table[
            "n_positive"
        ].astype("int64")

        table["n0"] = (
            table["records"]
            - table["n_positive"]
        )

        table["n_positive_percent"] = (
            table["n_positive"]
            / table["records"]
            * 100
        ).round(4)

        table.insert(0, "variable", column)

        subgroup_tables.append(table)

    combined = pd.concat(
        subgroup_tables,
        ignore_index=True,
    )

    return combined[
        [
            "variable",
            "category",
            "records",
            "n0",
            "n_positive",
            "n_positive_percent",
        ]
    ].sort_values(
        [
            "variable",
            "records",
        ],
        ascending=[True, False],
    )


def build_overview(dataframe):
    patient_record_counts = dataframe.groupby(
        PATIENT_ID_COLUMN,
        sort=False,
    ).size()

    patient_target_counts = dataframe.groupby(
        PATIENT_ID_COLUMN,
        sort=False,
    )[TARGET_COLUMN].nunique(dropna=True)

    patient_year_ranges = dataframe.groupby(
        PATIENT_ID_COLUMN,
        sort=False,
    )[PARSED_YEAR_COLUMN].agg(
        first_year="min",
        last_year="max",
    )

    n0_count = int(
        dataframe[TARGET_COLUMN].eq(0).sum()
    )
    n_positive_count = int(
        dataframe[TARGET_COLUMN].eq(1).sum()
    )
    valid_target_count = (
        n0_count + n_positive_count
    )

    pre_2018_mask = dataframe[
        PARSED_YEAR_COLUMN
    ].le(2017)

    from_2018_mask = dataframe[
        PARSED_YEAR_COLUMN
    ].ge(2018)

    raw_columns = list(EXPECTED_COLUMNS)

    overview = {
        "source": {
            "csv_file": raw_csv_path.name,
            "csv_size_bytes": int(
                raw_csv_path.stat().st_size
            ),
            "csv_sha256": calculate_sha256(
                raw_csv_path
            ),
            "dictionary_file": (
                dictionary_path.name
                if dictionary_path.is_file()
                else None
            ),
            "dictionary_sha256": (
                calculate_sha256(dictionary_path)
                if dictionary_path.is_file()
                else None
            ),
        },
        "records": {
            "total_records": int(
                len(dataframe)
            ),
            "raw_columns": int(
                len(EXPECTED_COLUMNS)
            ),
            "exact_duplicate_records": int(
                dataframe[
                    raw_columns
                ].duplicated().sum()
            ),
        },
        "patients": {
            "unique_patients": int(
                patient_record_counts.size
            ),
            "patients_with_multiple_records": int(
                patient_record_counts.gt(1).sum()
            ),
            "maximum_records_per_patient": int(
                patient_record_counts.max()
            ),
            "patients_with_conflicting_targets": int(
                patient_target_counts.gt(1).sum()
            ),
            "patients_crossing_2019_2020": int(
                (
                    patient_year_ranges[
                        "first_year"
                    ].le(2019)
                    & patient_year_ranges[
                        "last_year"
                    ].ge(2020)
                ).sum()
            ),
            "patients_crossing_2021_2022": int(
                (
                    patient_year_ranges[
                        "first_year"
                    ].le(2021)
                    & patient_year_ranges[
                        "last_year"
                    ].ge(2022)
                ).sum()
            ),
        },
        "target": {
            "n0": n0_count,
            "n_positive": n_positive_count,
            "invalid_or_excluded": int(
                dataframe[
                    TARGET_COLUMN
                ].isna().sum()
            ),
            "n_positive_percent": round(
                (
                    n_positive_count
                    / valid_target_count
                    * 100
                ),
                4,
            ),
        },
        "year": {
            "minimum": int(
                dataframe[
                    PARSED_YEAR_COLUMN
                ].min()
            ),
            "maximum": int(
                dataframe[
                    PARSED_YEAR_COLUMN
                ].max()
            ),
            "unparseable": int(
                dataframe[
                    PARSED_YEAR_COLUMN
                ].isna().sum()
            ),
        },
        "age": {
            "minimum": int(
                dataframe[
                    PARSED_AGE_COLUMN
                ].min()
            ),
            "maximum": int(
                dataframe[
                    PARSED_AGE_COLUMN
                ].max()
            ),
            "median": float(
                dataframe[
                    PARSED_AGE_COLUMN
                ].median()
            ),
            "unparseable": int(
                dataframe[
                    PARSED_AGE_COLUMN
                ].isna().sum()
            ),
        },
        "grade": {
            "unified_grade_counts": {
                str(key): int(value)
                for key, value in (
                    dataframe[
                        UNIFIED_GRADE_COLUMN
                    ]
                    .value_counts(
                        dropna=False
                    )
                    .sort_index()
                    .items()
                )
            },
            "pre_2018_records": int(
                pre_2018_mask.sum()
            ),
            "pre_2018_old_grade_blank": int(
                (
                    pre_2018_mask
                    & blank_mask(
                        dataframe[
                            OLD_GRADE_COLUMN
                        ]
                    )
                ).sum()
            ),
            "pre_2018_new_grade_blank": int(
                (
                    pre_2018_mask
                    & blank_mask(
                        dataframe[
                            NEW_GRADE_COLUMN
                        ]
                    )
                ).sum()
            ),
            "from_2018_records": int(
                from_2018_mask.sum()
            ),
            "from_2018_old_grade_blank": int(
                (
                    from_2018_mask
                    & blank_mask(
                        dataframe[
                            OLD_GRADE_COLUMN
                        ]
                    )
                ).sum()
            ),
            "from_2018_new_grade_blank": int(
                (
                    from_2018_mask
                    & blank_mask(
                        dataframe[
                            NEW_GRADE_COLUMN
                        ]
                    )
                ).sum()
            ),
        },
        "selected_variable_checks": {
            "tumor_size_unique_values": int(
                dataframe[
                    TUMOR_SIZE_COLUMN
                ].nunique(dropna=False)
            ),
            "age_label_unique_values": int(
                dataframe[
                    AGE_COLUMN
                ].nunique(dropna=False)
            ),
            "diagnostic_confirmation_counts": {
                str(key): int(value)
                for key, value in (
                    dataframe[
                        CONFIRMATION_COLUMN
                    ]
                    .value_counts(
                        dropna=False
                    )
                    .items()
                )
            },
        },
    }

    return overview


def save_overview_figure(
    dataframe,
    year_summary,
    output_path,
):
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(13, 9),
    )

    n0_count = int(
        dataframe[TARGET_COLUMN].eq(0).sum()
    )
    n_positive_count = int(
        dataframe[TARGET_COLUMN].eq(1).sum()
    )

    target_bars = axes[0, 0].bar(
        ["N0", "N+"],
        [n0_count, n_positive_count],
        color=["#4C78A8", "#E45756"],
    )

    axes[0, 0].set_title(
        "Regional lymph-node target"
    )
    axes[0, 0].set_ylabel("Records")

    for bar in target_bars:
        axes[0, 0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(bar.get_height()):,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    axes[0, 1].plot(
        year_summary["year"],
        year_summary["records"],
        color="#4C78A8",
        marker="o",
        linewidth=2,
    )
    axes[0, 1].set_title(
        "Records by diagnosis year"
    )
    axes[0, 1].set_xlabel("Year")
    axes[0, 1].set_ylabel("Records")
    axes[0, 1].grid(alpha=0.25)

    axes[1, 0].plot(
        year_summary["year"],
        year_summary[
            "n_positive_percent"
        ],
        color="#E45756",
        marker="o",
        linewidth=2,
    )
    axes[1, 0].set_title(
        "N+ prevalence by diagnosis year"
    )
    axes[1, 0].set_xlabel("Year")
    axes[1, 0].set_ylabel("N+ (%)")
    axes[1, 0].grid(alpha=0.25)

    n0_ages = dataframe.loc[
        dataframe[TARGET_COLUMN].eq(0),
        PARSED_AGE_COLUMN,
    ].dropna()

    n_positive_ages = dataframe.loc[
        dataframe[TARGET_COLUMN].eq(1),
        PARSED_AGE_COLUMN,
    ].dropna()

    age_bins = np.arange(0, 101, 5)

    axes[1, 1].hist(
        [
            n0_ages,
            n_positive_ages,
        ],
        bins=age_bins,
        label=["N0", "N+"],
        color=["#4C78A8", "#E45756"],
        alpha=0.7,
    )
    axes[1, 1].set_title(
        "Age distribution by target"
    )
    axes[1, 1].set_xlabel(
        "Age (90 includes 90+)"
    )
    axes[1, 1].set_ylabel("Records")
    axes[1, 1].legend()

    figure.suptitle(
        "SEER Breast Cancer Dataset Overview",
        fontsize=15,
    )
    figure.tight_layout(
        rect=[0, 0, 1, 0.96]
    )
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def main():
    metrics_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    figures_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = add_derived_fields(
        load_raw_seer(raw_csv_path)
    )

    overview = build_overview(dataframe)
    column_quality = build_column_quality(
        dataframe
    )
    year_summary = build_year_target_summary(
        dataframe
    )
    node_code_counts = build_node_code_counts(
        dataframe
    )
    patient_multiplicity = (
        build_patient_multiplicity(dataframe)
    )
    subgroup_summary = (
        build_subgroup_target_summary(dataframe)
    )

    overview_path = (
        metrics_directory
        / "seer_eda_overview.json"
    )
    column_quality_path = (
        metrics_directory
        / "seer_column_quality.csv"
    )
    year_summary_path = (
        metrics_directory
        / "seer_year_target_summary.csv"
    )
    node_code_counts_path = (
        metrics_directory
        / "seer_node_code_counts.csv"
    )
    patient_multiplicity_path = (
        metrics_directory
        / "seer_patient_multiplicity.csv"
    )
    subgroup_summary_path = (
        metrics_directory
        / "seer_subgroup_target_summary.csv"
    )
    figure_path = (
        figures_directory
        / "seer_eda_overview.png"
    )

    with overview_path.open(
        "w",
        encoding="utf-8",
    ) as file_handle:
        json.dump(
            overview,
            file_handle,
            indent=2,
            sort_keys=True,
        )
        file_handle.write("\n")

    column_quality.to_csv(
        column_quality_path,
        index=False,
    )
    year_summary.to_csv(
        year_summary_path,
        index=False,
    )
    node_code_counts.to_csv(
        node_code_counts_path,
        index=False,
    )
    patient_multiplicity.to_csv(
        patient_multiplicity_path,
        index=False,
    )
    subgroup_summary.to_csv(
        subgroup_summary_path,
        index=False,
    )

    save_overview_figure(
        dataframe,
        year_summary,
        figure_path,
    )

    print(
        f"records={overview['records']['total_records']}"
    )
    print(
        f"patients={overview['patients']['unique_patients']}"
    )
    print(
        "patients_with_multiple_records="
        f"{overview['patients']['patients_with_multiple_records']}"
    )
    print(
        "patients_with_conflicting_targets="
        f"{overview['patients']['patients_with_conflicting_targets']}"
    )
    print(
        f"n0={overview['target']['n0']}"
    )
    print(
        f"n_positive={overview['target']['n_positive']}"
    )
    print(
        "n_positive_percent="
        f"{overview['target']['n_positive_percent']}"
    )
    print(f"overview={overview_path}")
    print(f"figure={figure_path}")


if __name__ == "__main__":
    main()