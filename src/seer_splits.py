import hashlib
import json
from pathlib import Path

import pandas as pd

from src.seer_data import (
    PARSED_YEAR_COLUMN,
    PATIENT_ID_COLUMN,
    TARGET_COLUMN,
)


TEMPORARY_HASH_COLUMN = "_split_hash"


REQUIRED_CONFIG_KEYS = {
    "version",
    "split_column",
    "development_year_start",
    "development_year_end",
    "internal_validation_fraction",
    "internal_validation_name",
    "training_name",
    "temporal_validation_year_start",
    "temporal_validation_year_end",
    "temporal_validation_name",
    "temporal_test_year_start",
    "temporal_test_year_end",
    "temporal_test_name",
    "internal_validation_strata",
    "assignment_method",
    "hash_salt",
    "temporal_test_locked",
}


def load_split_config(config_path):
    config_path = Path(config_path)

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Split configuration not found: "
            f"{config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file_handle:
        config = json.load(file_handle)

    validate_split_config(config)

    return config


def validate_split_config(config):
    missing_keys = sorted(
        REQUIRED_CONFIG_KEYS - set(config)
    )

    if missing_keys:
        raise ValueError(
            "Split configuration is missing: "
            f"{missing_keys}"
        )

    validation_fraction = config[
        "internal_validation_fraction"
    ]

    if not 0 < validation_fraction < 1:
        raise ValueError(
            "Internal validation fraction "
            "must be between 0 and 1"
        )

    development_start = config[
        "development_year_start"
    ]
    development_end = config[
        "development_year_end"
    ]
    temporal_validation_start = config[
        "temporal_validation_year_start"
    ]
    temporal_validation_end = config[
        "temporal_validation_year_end"
    ]
    temporal_test_start = config[
        "temporal_test_year_start"
    ]
    temporal_test_end = config[
        "temporal_test_year_end"
    ]

    if development_start > development_end:
        raise ValueError(
            "Development year range is invalid"
        )

    if (
        development_end + 1
        != temporal_validation_start
    ):
        raise ValueError(
            "Development and temporal validation "
            "years must be contiguous"
        )

    if (
        temporal_validation_start
        > temporal_validation_end
    ):
        raise ValueError(
            "Temporal validation range is invalid"
        )

    if (
        temporal_validation_end + 1
        != temporal_test_start
    ):
        raise ValueError(
            "Temporal validation and test years "
            "must be contiguous"
        )

    if temporal_test_start > temporal_test_end:
        raise ValueError(
            "Temporal test range is invalid"
        )

    split_names = [
        config["training_name"],
        config["internal_validation_name"],
        config["temporal_validation_name"],
        config["temporal_test_name"],
    ]

    if len(split_names) != len(
        set(split_names)
    ):
        raise ValueError(
            "Split names must be unique"
        )

    expected_strata = [
        PARSED_YEAR_COLUMN,
        TARGET_COLUMN,
    ]

    if (
        config["internal_validation_strata"]
        != expected_strata
    ):
        raise ValueError(
            "Internal validation must be "
            "stratified by year and target"
        )

    if (
        config["assignment_method"]
        != "sha256_rank_within_stratum"
    ):
        raise ValueError(
            "Unsupported split assignment method"
        )

    if not config["hash_salt"]:
        raise ValueError(
            "Split hash salt cannot be empty"
        )

    if not config["temporal_test_locked"]:
        raise ValueError(
            "Temporal test must remain locked"
        )


def stable_patient_hash(
    patient_id,
    hash_salt,
):
    value = (
        f"{hash_salt}|{patient_id}"
    ).encode("utf-8")

    return hashlib.sha256(value).hexdigest()


def summarize_splits(
    assigned,
    config,
):
    split_column = config["split_column"]

    ordered_split_names = [
        config["training_name"],
        config["internal_validation_name"],
        config["temporal_validation_name"],
        config["temporal_test_name"],
    ]

    split_summaries = {}

    for split_name in ordered_split_names:
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

        split_summaries[split_name] = {
            "patients": int(len(subset)),
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

    return split_summaries


def assign_seer_splits(
    cohort,
    config,
):
    validate_split_config(config)

    if cohort.empty:
        raise ValueError(
            "Cannot split an empty cohort"
        )

    if not cohort.index.is_unique:
        raise ValueError(
            "Cohort index must be unique"
        )

    if cohort[
        PATIENT_ID_COLUMN
    ].duplicated().any():
        raise ValueError(
            "Cohort contains duplicate Patient IDs"
        )

    if cohort[
        PATIENT_ID_COLUMN
    ].isna().any():
        raise ValueError(
            "Cohort contains missing Patient IDs"
        )

    if cohort[
        PARSED_YEAR_COLUMN
    ].isna().any():
        raise ValueError(
            "Cohort contains missing years"
        )

    if cohort[TARGET_COLUMN].isna().any():
        raise ValueError(
            "Cohort contains missing targets"
        )

    unexpected_targets = sorted(
        set(
            cohort[
                TARGET_COLUMN
            ].astype("int64").unique()
        )
        - {0, 1}
    )

    if unexpected_targets:
        raise ValueError(
            "Unexpected target values: "
            f"{unexpected_targets}"
        )

    assigned = cohort.copy()

    split_column = config["split_column"]

    assigned[split_column] = pd.Series(
        pd.NA,
        index=assigned.index,
        dtype="string",
    )

    year_values = assigned[
        PARSED_YEAR_COLUMN
    ]

    development_mask = year_values.between(
        config["development_year_start"],
        config["development_year_end"],
        inclusive="both",
    )

    temporal_validation_mask = (
        year_values.between(
            config[
                "temporal_validation_year_start"
            ],
            config[
                "temporal_validation_year_end"
            ],
            inclusive="both",
        )
    )

    temporal_test_mask = year_values.between(
        config["temporal_test_year_start"],
        config["temporal_test_year_end"],
        inclusive="both",
    )

    coverage_count = (
        development_mask.astype("int8")
        + temporal_validation_mask.astype(
            "int8"
        )
        + temporal_test_mask.astype("int8")
    )

    if not coverage_count.eq(1).all():
        invalid_years = sorted(
            assigned.loc[
                ~coverage_count.eq(1),
                PARSED_YEAR_COLUMN,
            ].unique().tolist()
        )

        raise ValueError(
            "Some years are missing or overlap "
            "between split periods: "
            f"{invalid_years}"
        )

    assigned.loc[
        temporal_validation_mask,
        split_column,
    ] = config["temporal_validation_name"]

    assigned.loc[
        temporal_test_mask,
        split_column,
    ] = config["temporal_test_name"]

    development = assigned.loc[
        development_mask
    ].copy()

    development[TEMPORARY_HASH_COLUMN] = (
        development[PATIENT_ID_COLUMN].map(
            lambda patient_id: (
                stable_patient_hash(
                    patient_id,
                    config["hash_salt"],
                )
            )
        )
    )

    if development[
        TEMPORARY_HASH_COLUMN
    ].duplicated().any():
        raise RuntimeError(
            "A SHA-256 split hash collision "
            "was detected"
        )

    validation_count = 0

    stratum_columns = config[
        "internal_validation_strata"
    ]

    for _, stratum in development.groupby(
        stratum_columns,
        sort=True,
        observed=True,
    ):
        ordered_indices = (
            stratum.sort_values(
                TEMPORARY_HASH_COLUMN,
                kind="stable",
            ).index
        )

        stratum_validation_count = int(
            len(ordered_indices)
            * config[
                "internal_validation_fraction"
            ]
        )

        if stratum_validation_count < 1:
            raise ValueError(
                "A development stratum is too "
                "small for internal validation"
            )

        validation_indices = (
            ordered_indices[
                :stratum_validation_count
            ]
        )
        training_indices = (
            ordered_indices[
                stratum_validation_count:
            ]
        )

        assigned.loc[
            validation_indices,
            split_column,
        ] = config[
            "internal_validation_name"
        ]

        assigned.loc[
            training_indices,
            split_column,
        ] = config["training_name"]

        validation_count += (
            stratum_validation_count
        )

    if assigned[split_column].isna().any():
        raise RuntimeError(
            "Some patients were not assigned "
            "to a split"
        )

    allowed_split_names = {
        config["training_name"],
        config["internal_validation_name"],
        config["temporal_validation_name"],
        config["temporal_test_name"],
    }

    actual_split_names = set(
        assigned[split_column].unique()
    )

    if actual_split_names != allowed_split_names:
        raise RuntimeError(
            "Unexpected final split names: "
            f"{sorted(actual_split_names)}"
        )

    expected_validation_count = int(
        sum(
            int(
                len(stratum)
                * config[
                    "internal_validation_fraction"
                ]
            )
            for _, stratum in (
                development.groupby(
                    stratum_columns,
                    sort=True,
                    observed=True,
                )
            )
        )
    )

    if validation_count != (
        expected_validation_count
    ):
        raise RuntimeError(
            "Internal validation count changed"
        )

    audit = {
        "configuration": dict(config),
        "total_patients": int(
            len(assigned)
        ),
        "development_patients": int(
            development_mask.sum()
        ),
        "internal_validation_patients": int(
            validation_count
        ),
        "split_summary": summarize_splits(
            assigned,
            config,
        ),
        "validation": {
            "missing_split_assignments": int(
                assigned[
                    split_column
                ].isna().sum()
            ),
            "duplicate_patient_ids": int(
                assigned[
                    PATIENT_ID_COLUMN
                ].duplicated().sum()
            ),
            "temporary_hash_stored": False,
            "all_periods_non_overlapping": bool(
                coverage_count.eq(1).all()
            ),
        },
    }

    return assigned, audit