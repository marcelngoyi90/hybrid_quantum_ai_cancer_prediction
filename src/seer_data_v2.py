from pathlib import Path
from typing import Final

import pandas as pd

from src.seer_data import (
    EXPECTED_COLUMNS,
    YEAR_COLUMN,
    derive_numeric_year,
)


AJCC7_T_COLUMN: Final = (
    "Derived AJCC T, 7th ed (2010-2015)"
)
AJCC7_M_COLUMN: Final = (
    "Derived AJCC M, 7th ed (2010-2015)"
)
COMBINED_T_COLUMN: Final = (
    "Derived SEER Combined T (2016-2017)"
)
COMBINED_M_COLUMN: Final = (
    "Derived SEER Combined M (2016-2017)"
)
EOD2018_T_COLUMN: Final = (
    "Derived EOD 2018 T Recode (2018+)"
)
EOD2018_M_COLUMN: Final = (
    "Derived EOD 2018 M Recode (2018+)"
)

T_STAGE_COLUMN: Final = "t_stage_harmonized"
M_STAGE_COLUMN: Final = "m_stage_harmonized"

V2_STAGING_SOURCE_COLUMNS: Final = (
    AJCC7_T_COLUMN,
    AJCC7_M_COLUMN,
    COMBINED_T_COLUMN,
    COMBINED_M_COLUMN,
    EOD2018_T_COLUMN,
    EOD2018_M_COLUMN,
)

V2_RAW_COLUMNS: Final = (
    EXPECTED_COLUMNS
    + V2_STAGING_SOURCE_COLUMNS
)

EXPECTED_V2_RAW_RECORDS: Final = 756356

UNKNOWN_SOURCE_LABELS: Final = {
    "",
    "na",
    "blank(s)",
    "not applicable",
    "88",
}

T_STAGE_CATEGORIES: Final = {
    "T0",
    "Tis",
    "T1",
    "T2",
    "T3",
    "T4",
    "TX",
    "unknown",
}

M_STAGE_CATEGORIES: Final = {
    "M0",
    "M1",
    "unknown",
}


def validate_v2_raw_schema(dataframe):
    actual_columns = tuple(
        dataframe.columns
    )

    if actual_columns == V2_RAW_COLUMNS:
        return

    expected_set = set(
        V2_RAW_COLUMNS
    )
    actual_set = set(
        actual_columns
    )

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
            "unexpected columns: "
            f"{unexpected_columns}"
        )

    if (
        not missing_columns
        and not unexpected_columns
    ):
        details.append(
            "the columns exist but their order "
            "is different"
        )

    raise ValueError(
        "SEER v2 schema validation failed: "
        + "; ".join(details)
    )


def validate_v2_years(dataframe):
    numeric_year = derive_numeric_year(
        dataframe[YEAR_COLUMN]
    )

    if numeric_year.isna().any():
        raise ValueError(
            "SEER v2 contains an invalid "
            "diagnosis year"
        )

    valid_year_mask = numeric_year.between(
        2010,
        2023,
        inclusive="both",
    )

    if not valid_year_mask.all():
        invalid_years = sorted(
            dataframe.loc[
                ~valid_year_mask,
                YEAR_COLUMN,
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "SEER v2 contains years outside "
            f"2010-2023: {invalid_years}"
        )

    return numeric_year


def select_year_specific_source(
    dataframe,
    numeric_year,
    early_column,
    middle_column,
    late_column,
):
    source = pd.Series(
        "",
        index=dataframe.index,
        dtype="string",
    )

    early_mask = numeric_year.between(
        2010,
        2015,
        inclusive="both",
    )
    middle_mask = numeric_year.between(
        2016,
        2017,
        inclusive="both",
    )
    late_mask = numeric_year.between(
        2018,
        2023,
        inclusive="both",
    )

    source.loc[early_mask] = dataframe.loc[
        early_mask,
        early_column,
    ]

    source.loc[middle_mask] = dataframe.loc[
        middle_mask,
        middle_column,
    ]

    source.loc[late_mask] = dataframe.loc[
        late_mask,
        late_column,
    ]

    return source


def normalize_stage_token(
    source,
    stage_letter,
):
    token = (
        source.str.casefold()
        .str.strip()
        .str.replace(
            r"^[cp]",
            "",
            regex=True,
        )
        .str.replace(
            rf"^{stage_letter}",
            "",
            regex=True,
        )
    )

    return token


def derive_harmonized_t_stage(
    dataframe,
):
    numeric_year = validate_v2_years(
        dataframe
    )

    source = select_year_specific_source(
        dataframe=dataframe,
        numeric_year=numeric_year,
        early_column=AJCC7_T_COLUMN,
        middle_column=COMBINED_T_COLUMN,
        late_column=EOD2018_T_COLUMN,
    )

    token = normalize_stage_token(
        source,
        "t",
    )

    stage = pd.Series(
        "unknown",
        index=dataframe.index,
        dtype="string",
        name=T_STAGE_COLUMN,
    )

    t0_mask = token.eq("0")
    tis_mask = token.str.startswith(
        "is"
    )
    t1_mask = token.str.startswith(
        "1"
    )
    t2_mask = token.str.startswith(
        "2"
    )
    t3_mask = token.str.startswith(
        "3"
    )
    t4_mask = token.str.startswith(
        "4"
    )
    tx_mask = token.eq("x")

    stage.loc[t0_mask] = "T0"
    stage.loc[tis_mask] = "Tis"
    stage.loc[t1_mask] = "T1"
    stage.loc[t2_mask] = "T2"
    stage.loc[t3_mask] = "T3"
    stage.loc[t4_mask] = "T4"
    stage.loc[tx_mask] = "TX"

    unknown_source_mask = (
        source.str.casefold()
        .isin(
            UNKNOWN_SOURCE_LABELS
        )
    )

    recognized_mask = (
        unknown_source_mask
        | t0_mask
        | tis_mask
        | t1_mask
        | t2_mask
        | t3_mask
        | t4_mask
        | tx_mask
    )

    if not recognized_mask.all():
        unrecognized_labels = sorted(
            source.loc[
                ~recognized_mask
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "Unrecognized SEER T-stage "
            f"labels: {unrecognized_labels}"
        )

    if not set(
        stage.unique().tolist()
    ).issubset(
        T_STAGE_CATEGORIES
    ):
        raise RuntimeError(
            "Unexpected harmonized T category"
        )

    return stage


def derive_harmonized_m_stage(
    dataframe,
):
    numeric_year = validate_v2_years(
        dataframe
    )

    source = select_year_specific_source(
        dataframe=dataframe,
        numeric_year=numeric_year,
        early_column=AJCC7_M_COLUMN,
        middle_column=COMBINED_M_COLUMN,
        late_column=EOD2018_M_COLUMN,
    )

    token = normalize_stage_token(
        source,
        "m",
    )

    stage = pd.Series(
        "unknown",
        index=dataframe.index,
        dtype="string",
        name=M_STAGE_COLUMN,
    )

    m0_mask = token.str.startswith(
        "0"
    )
    m1_mask = token.str.startswith(
        "1"
    )

    stage.loc[m0_mask] = "M0"
    stage.loc[m1_mask] = "M1"

    unknown_source_mask = (
        source.str.casefold()
        .isin(
            UNKNOWN_SOURCE_LABELS
        )
    )

    recognized_mask = (
        unknown_source_mask
        | m0_mask
        | m1_mask
    )

    if not recognized_mask.all():
        unrecognized_labels = sorted(
            source.loc[
                ~recognized_mask
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "Unrecognized SEER M-stage "
            f"labels: {unrecognized_labels}"
        )

    if not set(
        stage.unique().tolist()
    ).issubset(
        M_STAGE_CATEGORIES
    ):
        raise RuntimeError(
            "Unexpected harmonized M category"
        )

    return stage


def add_v2_staging_features(
    dataframe,
):
    validate_v2_raw_schema(
        dataframe.loc[
            :,
            V2_RAW_COLUMNS,
        ]
    )

    derived = dataframe.copy()

    derived[T_STAGE_COLUMN] = (
        derive_harmonized_t_stage(
            derived
        )
    )
    derived[M_STAGE_COLUMN] = (
        derive_harmonized_m_stage(
            derived
        )
    )

    return derived


def load_raw_seer_v2(
    csv_path,
):
    csv_path = Path(
        csv_path
    )

    if not csv_path.is_file():
        raise FileNotFoundError(
            "SEER v2 CSV was not found: "
            f"{csv_path}"
        )

    dataframe = pd.read_csv(
        csv_path,
        dtype="string",
        keep_default_na=False,
        encoding="utf-8",
        low_memory=False,
    )

    if dataframe.empty:
        raise ValueError(
            "The SEER v2 CSV contains no rows"
        )

    validate_v2_raw_schema(
        dataframe
    )

    if len(dataframe) != (
        EXPECTED_V2_RAW_RECORDS
    ):
        raise ValueError(
            "Unexpected SEER v2 record count: "
            f"{len(dataframe)}"
        )

    for column in dataframe.columns:
        dataframe[column] = (
            dataframe[column]
            .str.strip()
        )

    return add_v2_staging_features(
        dataframe
    )
