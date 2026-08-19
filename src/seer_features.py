import json
from pathlib import Path

import pandas as pd

from src.seer_data import (
    PARSED_AGE_COLUMN,
    PATIENT_ID_COLUMN,
    TARGET_COLUMN,
    TUMOR_SIZE_COLUMN,
)


TUMOR_SIZE_MM_COLUMN = "tumor_size_mm"
TUMOR_SIZE_STATUS_COLUMN = "tumor_size_status"


REQUIRED_CONFIG_KEYS = {
    "version",
    "target_column",
    "split_column",
    "model_numeric_features",
    "model_categorical_features",
    "metadata_columns",
    "derived_features",
    "forbidden_model_inputs",
    "paired_bridge_feature_status",
    "paired_bridge_candidate_features",
}


def load_feature_config(config_path):
    config_path = Path(config_path)

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Feature configuration not found: "
            f"{config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file_handle:
        config = json.load(file_handle)

    validate_feature_config(config)

    return config


def validate_feature_config(config):
    missing_keys = sorted(
        REQUIRED_CONFIG_KEYS - set(config)
    )

    if missing_keys:
        raise ValueError(
            "Feature configuration is missing: "
            f"{missing_keys}"
        )

    if config["target_column"] != (
        TARGET_COLUMN
    ):
        raise ValueError(
            "Unexpected target column"
        )

    numeric_features = config[
        "model_numeric_features"
    ]
    categorical_features = config[
        "model_categorical_features"
    ]

    all_model_features = (
        numeric_features
        + categorical_features
    )

    if len(all_model_features) != len(
        set(all_model_features)
    ):
        raise ValueError(
            "Model feature names must be unique"
        )

    overlap = (
        set(numeric_features)
        & set(categorical_features)
    )

    if overlap:
        raise ValueError(
            "Features cannot be both numeric "
            f"and categorical: {sorted(overlap)}"
        )

    forbidden_features = set(
        config[
            "forbidden_model_inputs"
        ]
    )

    forbidden_overlap = (
        set(all_model_features)
        & forbidden_features
    )

    if forbidden_overlap:
        raise ValueError(
            "Forbidden variables were included "
            "as model features: "
            f"{sorted(forbidden_overlap)}"
        )

    expected_numeric_features = {
        PARSED_AGE_COLUMN,
        TUMOR_SIZE_MM_COLUMN,
    }

    if set(numeric_features) != (
        expected_numeric_features
    ):
        raise ValueError(
            "Numeric feature contract changed"
        )

    if TUMOR_SIZE_STATUS_COLUMN not in (
        categorical_features
    ):
        raise ValueError(
            "Tumor-size status must be "
            "categorical"
        )

    required_metadata = {
        PATIENT_ID_COLUMN,
        config["split_column"],
    }

    if not required_metadata.issubset(
        set(config["metadata_columns"])
    ):
        raise ValueError(
            "Required metadata columns "
            "are missing"
        )

    candidate_features = set(
        config[
            "paired_bridge_candidate_features"
        ]
    )

    if not candidate_features.issubset(
        set(all_model_features)
    ):
        raise ValueError(
            "Bridge candidates must be a "
            "subset of model features"
        )


def derive_tumor_size_features(
    tumor_size_values,
):
    normalized = (
        tumor_size_values
        .astype("string")
        .str.strip()
    )

    numeric_code = pd.to_numeric(
        normalized.str.extract(
            r"^(\d{3})",
            expand=False,
        ),
        errors="coerce",
    )

    tumor_size_mm = pd.Series(
        pd.NA,
        index=normalized.index,
        dtype="Float32",
        name=TUMOR_SIZE_MM_COLUMN,
    )

    tumor_size_status = pd.Series(
        pd.NA,
        index=normalized.index,
        dtype="string",
        name=TUMOR_SIZE_STATUS_COLUMN,
    )

    measured_mask = numeric_code.between(
        1,
        400,
        inclusive="both",
    )

    no_evidence_mask = numeric_code.eq(0)

    unknown_or_unreasonable_mask = (
        numeric_code.between(
            401,
            989,
            inclusive="both",
        )
        | numeric_code.eq(999)
        | normalized.str.casefold().str.startswith(
            "unknown",
            na=False,
        )
    )

    microscopic_focus_mask = (
        numeric_code.eq(990)
    )

    site_specific_mask = numeric_code.eq(998)

    inconsistent_mask = (
        normalized.eq("XX1")
        | normalized.str.casefold().str.contains(
            (
                "not consistent over time"
                "|not applicable for this site"
            ),
            regex=True,
            na=False,
        )
    )

    tumor_size_mm.loc[
        measured_mask
    ] = numeric_code.loc[
        measured_mask
    ].astype("float32")

    tumor_size_mm.loc[
        no_evidence_mask
    ] = 0.0

    tumor_size_status.loc[
        measured_mask
    ] = "measured"

    tumor_size_status.loc[
        no_evidence_mask
    ] = "no_evidence_of_primary_tumor"

    tumor_size_status.loc[
        unknown_or_unreasonable_mask
    ] = "unknown_or_unreasonable"

    tumor_size_status.loc[
        microscopic_focus_mask
    ] = "microscopic_focus"

    tumor_size_status.loc[
        site_specific_mask
    ] = "site_specific"

    tumor_size_status.loc[
        inconsistent_mask
    ] = "inconsistent_or_not_applicable"

    unrecognized_mask = (
        tumor_size_status.isna()
    )

    if unrecognized_mask.any():
        unrecognized_values = sorted(
            normalized.loc[
                unrecognized_mask
            ].unique().tolist()
        )

        raise ValueError(
            "Unrecognized tumor-size values: "
            f"{unrecognized_values}"
        )

    return tumor_size_mm, tumor_size_status


def normalize_categorical_feature(values):
    normalized = (
        values.astype("string").str.strip()
    )

    invalid_mask = (
        normalized.isna()
        | normalized.eq("")
    )

    if invalid_mask.any():
        raise ValueError(
            "A categorical model feature "
            "contains empty values"
        )

    return normalized


def build_feature_audit(
    model_table,
    config,
):
    numeric_summary = {}

    for feature in config[
        "model_numeric_features"
    ]:
        values = model_table[feature]

        numeric_summary[feature] = {
            "missing": int(
                values.isna().sum()
            ),
            "missing_percent": round(
                float(
                    values.isna().mean()
                    * 100
                ),
                4,
            ),
            "minimum": (
                float(values.min())
                if values.notna().any()
                else None
            ),
            "maximum": (
                float(values.max())
                if values.notna().any()
                else None
            ),
            "median": (
                float(values.median())
                if values.notna().any()
                else None
            ),
        }

    categorical_summary = {}

    for feature in config[
        "model_categorical_features"
    ]:
        values = model_table[feature]

        categorical_summary[feature] = {
            "unique_values": int(
                values.nunique(dropna=False)
            ),
            "missing": int(
                values.isna().sum()
            ),
        }

    split_column = config["split_column"]
    split_summary = {}

    for split_name, subset in (
        model_table.groupby(
            split_column,
            observed=True,
        )
    ):
        n_positive_count = int(
            subset[TARGET_COLUMN].eq(1).sum()
        )

        split_summary[str(split_name)] = {
            "patients": int(len(subset)),
            "n0": int(
                subset[
                    TARGET_COLUMN
                ].eq(0).sum()
            ),
            "n_positive": n_positive_count,
            "n_positive_percent": round(
                (
                    n_positive_count
                    / len(subset)
                    * 100
                ),
                4,
            ),
        }

    tumor_status_counts = {
        str(key): int(value)
        for key, value in (
            model_table[
                TUMOR_SIZE_STATUS_COLUMN
            ]
            .value_counts(dropna=False)
            .sort_index()
            .items()
        )
    }

    return {
        "feature_contract_version": (
            config["version"]
        ),
        "patients": int(
            len(model_table)
        ),
        "numeric_features": list(
            config[
                "model_numeric_features"
            ]
        ),
        "categorical_features": list(
            config[
                "model_categorical_features"
            ]
        ),
        "model_feature_count": int(
            len(
                config[
                    "model_numeric_features"
                ]
            )
            + len(
                config[
                    "model_categorical_features"
                ]
            )
        ),
        "numeric_summary": numeric_summary,
        "categorical_summary": (
            categorical_summary
        ),
        "tumor_size_status_counts": (
            tumor_status_counts
        ),
        "split_summary": split_summary,
        "validation": {
            "duplicate_patient_ids": int(
                model_table[
                    PATIENT_ID_COLUMN
                ].duplicated().sum()
            ),
            "missing_targets": int(
                model_table[
                    TARGET_COLUMN
                ].isna().sum()
            ),
            "forbidden_model_inputs_present": [],
        },
    }


def build_seer_model_table(
    cohort,
    config,
):
    validate_feature_config(config)

    if cohort.empty:
        raise ValueError(
            "Cannot prepare an empty cohort"
        )

    split_column = config["split_column"]

    required_source_columns = {
        PATIENT_ID_COLUMN,
        split_column,
        TARGET_COLUMN,
        PARSED_AGE_COLUMN,
        TUMOR_SIZE_COLUMN,
    }

    required_source_columns.update(
        feature
        for feature in config[
            "model_categorical_features"
        ]
        if feature
        != TUMOR_SIZE_STATUS_COLUMN
    )

    missing_source_columns = sorted(
        required_source_columns
        - set(cohort.columns)
    )

    if missing_source_columns:
        raise ValueError(
            "Cohort is missing required "
            f"columns: {missing_source_columns}"
        )

    if cohort[
        PATIENT_ID_COLUMN
    ].duplicated().any():
        raise ValueError(
            "Cohort contains duplicate Patient IDs"
        )

    if cohort[TARGET_COLUMN].isna().any():
        raise ValueError(
            "Cohort contains missing targets"
        )

    if cohort[split_column].isna().any():
        raise ValueError(
            "Cohort contains missing splits"
        )

    tumor_size_mm, tumor_size_status = (
        derive_tumor_size_features(
            cohort[TUMOR_SIZE_COLUMN]
        )
    )

    output_columns = {}

    for column in config[
        "metadata_columns"
    ]:
        output_columns[column] = cohort[
            column
        ]

    for feature in config[
        "model_numeric_features"
    ]:
        if feature == TUMOR_SIZE_MM_COLUMN:
            output_columns[feature] = (
                tumor_size_mm
            )
        else:
            output_columns[feature] = (
                pd.to_numeric(
                    cohort[feature],
                    errors="coerce",
                ).astype("Float32")
            )

    for feature in config[
        "model_categorical_features"
    ]:
        if (
            feature
            == TUMOR_SIZE_STATUS_COLUMN
        ):
            output_columns[feature] = (
                tumor_size_status
            )
        else:
            output_columns[feature] = (
                normalize_categorical_feature(
                    cohort[feature]
                )
            )

    output_columns[TARGET_COLUMN] = (
        cohort[TARGET_COLUMN].astype("Int8")
    )

    model_table = pd.DataFrame(
        output_columns
    ).reset_index(drop=True)

    expected_output_columns = (
        config["metadata_columns"]
        + config["model_numeric_features"]
        + config[
            "model_categorical_features"
        ]
        + [TARGET_COLUMN]
    )

    if list(model_table.columns) != (
        expected_output_columns
    ):
        raise RuntimeError(
            "Model-table column order changed"
        )

    forbidden_inputs = set(
        config[
            "forbidden_model_inputs"
        ]
    )

    actual_model_inputs = set(
        config["model_numeric_features"]
        + config[
            "model_categorical_features"
        ]
    )

    forbidden_present = sorted(
        actual_model_inputs
        & forbidden_inputs
    )

    if forbidden_present:
        raise RuntimeError(
            "Forbidden inputs entered the model "
            f"table: {forbidden_present}"
        )

    audit = build_feature_audit(
        model_table,
        config,
    )

    return model_table, audit