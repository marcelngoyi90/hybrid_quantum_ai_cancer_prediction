import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


project_root = Path(
    __file__
).resolve().parents[1]

sys.path.insert(
    0,
    str(project_root),
)

from src.seer_drift import (
    analyze_seer_drift,
    load_drift_config,
)
from src.seer_features import (
    load_feature_config,
)


model_table_path = (
    project_root
    / "data"
    / "processed"
    / "seer_model_table.parquet"
)

feature_config_path = (
    project_root
    / "configs"
    / "seer_features.json"
)

drift_config_path = (
    project_root
    / "configs"
    / "seer_drift.json"
)

metrics_directory = (
    project_root
    / "artifacts"
    / "metrics"
)

feature_drift_path = (
    metrics_directory
    / "seer_development_feature_drift.csv"
)

prevalence_path = (
    metrics_directory
    / "seer_development_prevalence_drift.csv"
)

summary_path = (
    metrics_directory
    / "seer_development_drift_summary.csv"
)

audit_path = (
    metrics_directory
    / "seer_development_drift_audit.json"
)


def calculate_sha256(path):
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

            digest.update(chunk)

    return digest.hexdigest()


def validate_persisted_report(
    original,
    persisted,
    report_name,
):
    if list(original.columns) != list(
        persisted.columns
    ):
        raise RuntimeError(
            f"Persisted {report_name} "
            "columns changed"
        )

    if len(original) != len(persisted):
        raise RuntimeError(
            f"Persisted {report_name} "
            "row count changed"
        )

    try:
        pd.testing.assert_frame_equal(
            original.reset_index(
                drop=True
            ),
            persisted.reset_index(
                drop=True
            ),
            check_dtype=False,
            check_exact=False,
            rtol=1e-9,
            atol=1e-12,
        )
    except AssertionError as error:
        raise RuntimeError(
            f"Persisted {report_name} "
            "values changed"
        ) from error


def file_audit(path):
    path = Path(path)

    return {
        "file_name": path.name,
        "size_bytes": int(
            path.stat().st_size
        ),
        "sha256": calculate_sha256(
            path
        ),
    }


def main():
    if not model_table_path.is_file():
        raise FileNotFoundError(
            "SEER model table not found: "
            f"{model_table_path}"
        )

    metrics_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_config = (
        load_feature_config(
            feature_config_path
        )
    )

    drift_config = load_drift_config(
        drift_config_path
    )

    split_column = feature_config[
        "split_column"
    ]

    target_column = feature_config[
        "target_column"
    ]

    reference_split = drift_config[
        "reference_split"
    ]

    comparisons = list(
        drift_config[
            "development_comparisons"
        ]
    )

    permitted_splits = [
        reference_split,
        *comparisons,
    ]

    locked_comparison = drift_config[
        "locked_comparison"
    ]

    if locked_comparison in (
        permitted_splits
    ):
        raise PermissionError(
            "Locked temporal test cannot be "
            "loaded during development"
        )

    model_columns = [
        split_column,
        target_column,
        *feature_config[
            "model_numeric_features"
        ],
        *feature_config[
            "model_categorical_features"
        ],
    ]

    model_table = pd.read_parquet(
        model_table_path,
        columns=model_columns,
        filters=[
            (
                split_column,
                "in",
                permitted_splits,
            )
        ],
        engine="pyarrow",
    )

    loaded_splits = set(
        model_table[
            split_column
        ].unique().tolist()
    )

    if loaded_splits != set(
        permitted_splits
    ):
        raise RuntimeError(
            "Development drift input splits "
            f"changed: {sorted(loaded_splits)}"
        )

    if locked_comparison in loaded_splits:
        raise RuntimeError(
            "Locked temporal test entered the "
            "development analysis"
        )

    results = analyze_seer_drift(
        model_table,
        feature_config,
        drift_config,
        comparisons=comparisons,
        allow_locked_comparison=False,
    )

    feature_drift = results[
        "feature_drift"
    ]

    prevalence = results[
        "prevalence"
    ]

    comparison_summary = results[
        "comparison_summary"
    ]

    feature_drift.to_csv(
        feature_drift_path,
        index=False,
    )

    prevalence.to_csv(
        prevalence_path,
        index=False,
    )

    comparison_summary.to_csv(
        summary_path,
        index=False,
    )

    persisted_feature_drift = (
        pd.read_csv(
            feature_drift_path
        )
    )

    persisted_prevalence = (
        pd.read_csv(
            prevalence_path
        )
    )

    persisted_summary = pd.read_csv(
        summary_path
    )

    validate_persisted_report(
        feature_drift,
        persisted_feature_drift,
        "feature-drift report",
    )

    validate_persisted_report(
        prevalence,
        persisted_prevalence,
        "prevalence report",
    )

    validate_persisted_report(
        comparison_summary,
        persisted_summary,
        "comparison summary",
    )

    audit = results["audit"]

    audit[
        "development_only_validation"
    ] = {
        "permitted_splits": (
            permitted_splits
        ),
        "loaded_splits": sorted(
            loaded_splits
        ),
        "loaded_patients": int(
            len(model_table)
        ),
        "locked_comparison_loaded": False,
        "patient_identifier_loaded": False,
        "patient_level_rows_exported": False,
    }

    audit[
        "interpretation_guardrails"
    ] = {
        "psi_thresholds_are_descriptive_only": bool(
            drift_config[
                "psi_interpretation"
            ][
                "thresholds_are_descriptive_only"
            ]
        ),
        "label_shift_assessment_is_univariate_only": True,
        "feature_dropping_performed": False,
        "feature_reweighting_performed": False,
        "calibration_analyzed": False,
    }

    audit["files"] = {
        "private_model_table": (
            file_audit(
                model_table_path
            )
        ),
        "feature_configuration": (
            file_audit(
                feature_config_path
            )
        ),
        "drift_configuration": (
            file_audit(
                drift_config_path
            )
        ),
        "feature_drift_report": (
            file_audit(
                feature_drift_path
            )
        ),
        "prevalence_report": (
            file_audit(
                prevalence_path
            )
        ),
        "comparison_summary": (
            file_audit(
                summary_path
            )
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

    print(
        "development_patients="
        f"{len(model_table)}"
    )

    print(
        "feature_drift_rows="
        f"{len(feature_drift)}"
    )

    for row in (
        comparison_summary.itertuples(
            index=False
        )
    ):
        print(
            f"comparison={row.comparison} "
            "overall_maximum_psi="
            f"{row.overall_maximum_psi:.6f} "
            "moderate_or_large_features="
            f"{row.overall_moderate_or_large_features}"
        )

    print(
        "locked_comparison_included=False"
    )

    print(f"audit={audit_path}")


if __name__ == "__main__":
    main()