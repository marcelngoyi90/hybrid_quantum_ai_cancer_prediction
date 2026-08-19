from pathlib import Path
from typing import Final

import pandas as pd


PATIENT_ID_COLUMN: Final = "Patient ID"
SEQUENCE_COLUMN: Final = "Sequence number"
YEAR_COLUMN: Final = "Year of diagnosis"
AGE_COLUMN: Final = "Age recode with single ages and 90+"
RACE_COLUMN: Final = "Race recode (W, B, AI, API)"
ORIGIN_COLUMN: Final = (
    "Origin recode NHIA (Hispanic, Non-Hisp)"
)
PRIMARY_SITE_COLUMN: Final = "Primary Site - labeled"
HISTOLOGY_COLUMN: Final = "Histologic Type ICD-O-3"
LATERALITY_COLUMN: Final = "Laterality"
TUMOR_SIZE_COLUMN: Final = (
    "Tumor Size Over Time Recode (1988+)"
)
OLD_GRADE_COLUMN: Final = "Grade Recode (thru 2017)"
NEW_GRADE_COLUMN: Final = "Grade Clinical (2018+)"
ER_COLUMN: Final = (
    "ER Status Recode Breast Cancer (1990+)"
)
PR_COLUMN: Final = (
    "PR Status Recode Breast Cancer (1990+)"
)
HER2_COLUMN: Final = "Derived HER2 Recode (2010+)"
SUBTYPE_COLUMN: Final = "Breast Subtype (2010+)"
NODES_POSITIVE_COLUMN: Final = (
    "Regional nodes positive (1988+)"
)
NODES_EXAMINED_COLUMN: Final = (
    "Regional nodes examined (1988+)"
)
CONFIRMATION_COLUMN: Final = "Diagnostic Confirmation"

TARGET_COLUMN: Final = "node_positive_target"
PARSED_YEAR_COLUMN: Final = "year_numeric"
PARSED_AGE_COLUMN: Final = "age_numeric"
UNIFIED_GRADE_COLUMN: Final = "grade_unified"


EXPECTED_COLUMNS: Final = (
    PATIENT_ID_COLUMN,
    SEQUENCE_COLUMN,
    YEAR_COLUMN,
    AGE_COLUMN,
    RACE_COLUMN,
    ORIGIN_COLUMN,
    PRIMARY_SITE_COLUMN,
    HISTOLOGY_COLUMN,
    LATERALITY_COLUMN,
    TUMOR_SIZE_COLUMN,
    OLD_GRADE_COLUMN,
    NEW_GRADE_COLUMN,
    ER_COLUMN,
    PR_COLUMN,
    HER2_COLUMN,
    SUBTYPE_COLUMN,
    NODES_POSITIVE_COLUMN,
    NODES_EXAMINED_COLUMN,
    CONFIRMATION_COLUMN,
)


def validate_raw_schema(dataframe):
    actual_columns = tuple(dataframe.columns)

    if actual_columns == EXPECTED_COLUMNS:
        return

    expected_set = set(EXPECTED_COLUMNS)
    actual_set = set(actual_columns)

    missing_columns = sorted(
        expected_set - actual_set
    )
    unexpected_columns = sorted(
        actual_set - expected_set
    )

    details = []

    if missing_columns:
        details.append(
            f"missing columns: {missing_columns}"
        )

    if unexpected_columns:
        details.append(
            f"unexpected columns: {unexpected_columns}"
        )

    if not missing_columns and not unexpected_columns:
        details.append(
            "the columns exist but their order is different"
        )

    raise ValueError(
        "SEER schema validation failed: "
        + "; ".join(details)
    )


def load_raw_seer(csv_path):
    csv_path = Path(csv_path)

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"SEER CSV was not found: {csv_path}"
        )

    dataframe = pd.read_csv(
        csv_path,
        dtype="string",
        keep_default_na=False,
        encoding="utf-8",
        low_memory=False,
    )

    if dataframe.empty:
        raise ValueError("The SEER CSV contains no rows")

    validate_raw_schema(dataframe)

    for column in dataframe.columns:
        dataframe[column] = (
            dataframe[column].str.strip()
        )

    return dataframe


def derive_node_target(node_codes):
    numeric_codes = pd.to_numeric(
        node_codes,
        errors="coerce",
    )

    target = pd.Series(
        pd.NA,
        index=node_codes.index,
        dtype="Int8",
        name=TARGET_COLUMN,
    )

    n0_mask = numeric_codes.eq(0)
    n_positive_mask = (
        numeric_codes.between(
            1,
            90,
            inclusive="both",
        )
        | numeric_codes.isin([95, 97])
    )

    target.loc[n0_mask] = 0
    target.loc[n_positive_mask] = 1

    return target


def derive_numeric_year(year_values):
    return pd.to_numeric(
        year_values,
        errors="coerce",
    ).astype("Int16")


def derive_numeric_age(age_values):
    extracted_age = age_values.str.extract(
        r"(\d+)",
        expand=False,
    )

    return pd.to_numeric(
        extracted_age,
        errors="coerce",
    ).astype("Int16")


def derive_unified_grade(dataframe):
    numeric_year = derive_numeric_year(
        dataframe[YEAR_COLUMN]
    )

    old_grade = dataframe[
        OLD_GRADE_COLUMN
    ].str.casefold()

    new_grade = dataframe[
        NEW_GRADE_COLUMN
    ].str.extract(
        r"([1-4])",
        expand=False,
    )

    unified_grade = pd.Series(
        "unknown",
        index=dataframe.index,
        dtype="string",
        name=UNIFIED_GRADE_COLUMN,
    )

    old_period = numeric_year.le(2017)
    new_period = numeric_year.ge(2018)

    old_grade_patterns = {
        "well differentiated": "1",
        "moderately differentiated": "2",
        "poorly differentiated": "3",
        "undifferentiated": "4",
    }

    for pattern, grade_value in (
        old_grade_patterns.items()
    ):
        grade_mask = (
            old_period
            & old_grade.str.startswith(
                pattern,
                na=False,
            )
        )
        unified_grade.loc[grade_mask] = grade_value

    valid_new_grade = (
        new_period
        & new_grade.isin(["1", "2", "3", "4"])
    )

    unified_grade.loc[valid_new_grade] = (
        new_grade.loc[valid_new_grade]
    )

    return unified_grade


def add_derived_fields(dataframe):
    validate_raw_schema(
        dataframe.loc[:, EXPECTED_COLUMNS]
    )

    derived = dataframe.copy()

    derived[TARGET_COLUMN] = derive_node_target(
        derived[NODES_POSITIVE_COLUMN]
    )
    derived[PARSED_YEAR_COLUMN] = (
        derive_numeric_year(
            derived[YEAR_COLUMN]
        )
    )
    derived[PARSED_AGE_COLUMN] = (
        derive_numeric_age(
            derived[AGE_COLUMN]
        )
    )
    derived[UNIFIED_GRADE_COLUMN] = (
        derive_unified_grade(derived)
    )

    return derived