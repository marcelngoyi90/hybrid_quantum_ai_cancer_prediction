from typing import Final

import pandas as pd

from src.seer_data import (
    PARSED_AGE_COLUMN,
    PARSED_YEAR_COLUMN,
    PATIENT_ID_COLUMN,
    SEQUENCE_COLUMN,
    TARGET_COLUMN,
    add_derived_fields,
)


MINIMUM_AGE: Final = 18
SEQUENCE_RANK_COLUMN: Final = "sequence_rank"
SOURCE_ROW_COLUMN: Final = "source_row_number"


def derive_sequence_rank(sequence_values):
    extracted_rank = sequence_values.str.extract(
        r"^(\d+)",
        expand=False,
    )

    sequence_rank = pd.to_numeric(
        extracted_rank,
        errors="coerce",
    ).astype("Int16")

    one_primary_mask = sequence_values.eq(
        "One primary only"
    )
    sequence_rank.loc[one_primary_mask] = 1

    invalid_mask = (
        sequence_rank.isna()
        | sequence_rank.le(0)
    )

    if invalid_mask.any():
        invalid_labels = sorted(
            sequence_values.loc[
                invalid_mask
            ].unique().tolist()
        )

        raise ValueError(
            "Unrecognized SEER sequence labels: "
            f"{invalid_labels}"
        )

    sequence_rank.name = SEQUENCE_RANK_COLUMN

    return sequence_rank


def build_index_cohort(
    raw_dataframe,
    minimum_age=MINIMUM_AGE,
):
    if minimum_age < 0:
        raise ValueError(
            "minimum_age cannot be negative"
        )

    dataframe = add_derived_fields(
        raw_dataframe
    )

    dataframe[SOURCE_ROW_COLUMN] = pd.Series(
        range(len(dataframe)),
        index=dataframe.index,
        dtype="Int64",
    )

    valid_age_mask = dataframe[
        PARSED_AGE_COLUMN
    ].notna()

    under_age_mask = (
        valid_age_mask
        & dataframe[
            PARSED_AGE_COLUMN
        ].lt(minimum_age)
    )

    adult_mask = (
        valid_age_mask
        & dataframe[
            PARSED_AGE_COLUMN
        ].ge(minimum_age)
    )

    invalid_target_after_age_mask = (
        adult_mask
        & dataframe[
            TARGET_COLUMN
        ].isna()
    )

    eligible_mask = (
        adult_mask
        & dataframe[
            TARGET_COLUMN
        ].notna()
    )

    eligible = dataframe.loc[
        eligible_mask
    ].copy()

    if eligible.empty:
        raise ValueError(
            "No eligible SEER records remain"
        )

    eligible[SEQUENCE_RANK_COLUMN] = (
        derive_sequence_rank(
            eligible[SEQUENCE_COLUMN]
        )
    )

    selection_keys = [
        PATIENT_ID_COLUMN,
        PARSED_YEAR_COLUMN,
        SEQUENCE_RANK_COLUMN,
    ]

    ambiguous_tie_mask = eligible.duplicated(
        selection_keys,
        keep=False,
    )

    ambiguous_tie_records = int(
        ambiguous_tie_mask.sum()
    )

    if ambiguous_tie_records:
        ambiguous_patients = int(
            eligible.loc[
                ambiguous_tie_mask,
                PATIENT_ID_COLUMN,
            ].nunique()
        )

        raise ValueError(
            "Index-record selection is ambiguous "
            f"for {ambiguous_patients} patients "
            f"and {ambiguous_tie_records} records"
        )

    patient_record_counts = eligible.groupby(
        PATIENT_ID_COLUMN,
        sort=False,
    ).size()

    patient_target_counts = eligible.groupby(
        PATIENT_ID_COLUMN,
        sort=False,
    )[TARGET_COLUMN].nunique(dropna=True)

    sorted_eligible = eligible.sort_values(
        selection_keys,
        kind="stable",
    )

    cohort = (
        sorted_eligible.drop_duplicates(
            subset=[PATIENT_ID_COLUMN],
            keep="first",
        )
        .sort_values(
            PATIENT_ID_COLUMN,
            kind="stable",
        )
        .reset_index(drop=True)
    )

    if cohort[PATIENT_ID_COLUMN].duplicated().any():
        raise RuntimeError(
            "The index cohort contains "
            "duplicate Patient IDs"
        )

    if cohort[PARSED_AGE_COLUMN].lt(
        minimum_age
    ).any():
        raise RuntimeError(
            "The index cohort contains "
            "under-age records"
        )

    if cohort[TARGET_COLUMN].isna().any():
        raise RuntimeError(
            "The index cohort contains "
            "missing targets"
        )

    n0_count = int(
        cohort[TARGET_COLUMN].eq(0).sum()
    )
    n_positive_count = int(
        cohort[TARGET_COLUMN].eq(1).sum()
    )

    audit = {
        "selection_rule": {
            "minimum_age": int(minimum_age),
            "primary_sort": (
                "earliest diagnosis year"
            ),
            "secondary_sort": (
                "smallest sequence rank"
            ),
            "ambiguous_ties_allowed": False,
        },
        "flow": {
            "raw_records": int(
                len(dataframe)
            ),
            "unparseable_age_records": int(
                (
                    ~valid_age_mask
                ).sum()
            ),
            "under_age_records": int(
                under_age_mask.sum()
            ),
            "invalid_target_records_after_age_filter": int(
                invalid_target_after_age_mask.sum()
            ),
            "eligible_records": int(
                len(eligible)
            ),
            "records_removed_as_non_index": int(
                len(eligible) - len(cohort)
            ),
            "final_index_records": int(
                len(cohort)
            ),
        },
        "eligible_patient_structure": {
            "eligible_unique_patients": int(
                patient_record_counts.size
            ),
            "patients_with_multiple_records": int(
                patient_record_counts.gt(1).sum()
            ),
            "patients_with_conflicting_targets": int(
                patient_target_counts.gt(1).sum()
            ),
            "maximum_records_per_patient": int(
                patient_record_counts.max()
            ),
            "ambiguous_tie_records": (
                ambiguous_tie_records
            ),
        },
        "index_cohort": {
            "records": int(
                len(cohort)
            ),
            "unique_patients": int(
                cohort[
                    PATIENT_ID_COLUMN
                ].nunique()
            ),
            "n0": n0_count,
            "n_positive": n_positive_count,
            "n_positive_percent": round(
                (
                    n_positive_count
                    / len(cohort)
                    * 100
                ),
                4,
            ),
            "minimum_age": int(
                cohort[
                    PARSED_AGE_COLUMN
                ].min()
            ),
            "maximum_age": int(
                cohort[
                    PARSED_AGE_COLUMN
                ].max()
            ),
            "median_age": float(
                cohort[
                    PARSED_AGE_COLUMN
                ].median()
            ),
            "minimum_year": int(
                cohort[
                    PARSED_YEAR_COLUMN
                ].min()
            ),
            "maximum_year": int(
                cohort[
                    PARSED_YEAR_COLUMN
                ].max()
            ),
        },
    }

    return cohort, audit