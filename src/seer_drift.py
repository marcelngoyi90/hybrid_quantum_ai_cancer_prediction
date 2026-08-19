import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp


MISSING_CATEGORY = "__MISSING__"


REQUIRED_DRIFT_CONFIG_KEYS = {
    "version",
    "reference_split",
    "development_comparisons",
    "locked_comparison",
    "locked_comparison_stage",
    "numeric_features",
    "categorical_features",
    "psi_interpretation",
    "class_conditional_analysis",
    "label_prevalence_analysis",
    "calibration_analysis",
    "actions",
}


def load_drift_config(config_path):
    config_path = Path(config_path)

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Drift configuration not found: "
            f"{config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file_handle:
        config = json.load(file_handle)

    validate_drift_config(config)

    return config


def validate_drift_config(config):
    missing_keys = sorted(
        REQUIRED_DRIFT_CONFIG_KEYS
        - set(config)
    )

    if missing_keys:
        raise ValueError(
            "Drift configuration is missing: "
            f"{missing_keys}"
        )

    reference_split = config[
        "reference_split"
    ]

    development_comparisons = config[
        "development_comparisons"
    ]

    locked_comparison = config[
        "locked_comparison"
    ]

    if (
        not isinstance(
            development_comparisons,
            list,
        )
        or not development_comparisons
    ):
        raise ValueError(
            "Development comparisons must be "
            "a non-empty list"
        )

    if len(development_comparisons) != len(
        set(development_comparisons)
    ):
        raise ValueError(
            "Development comparisons must "
            "be unique"
        )

    if reference_split in (
        development_comparisons
    ):
        raise ValueError(
            "Reference split cannot also be "
            "a comparison split"
        )

    if locked_comparison in (
        development_comparisons
    ):
        raise ValueError(
            "Locked comparison cannot be used "
            "during development"
        )

    numeric_config = config[
        "numeric_features"
    ]

    categorical_config = config[
        "categorical_features"
    ]

    if numeric_config[
        "primary_metric"
    ] != "population_stability_index":
        raise ValueError(
            "Unexpected numeric primary metric"
        )

    if numeric_config[
        "secondary_metric"
    ] != "kolmogorov_smirnov_statistic":
        raise ValueError(
            "Unexpected numeric secondary metric"
        )

    if numeric_config[
        "psi_binning"
    ] != "reference_quantiles":
        raise ValueError(
            "Unexpected numeric PSI binning "
            "method"
        )

    if int(
        numeric_config["psi_bins"]
    ) < 2:
        raise ValueError(
            "Numeric PSI requires at least "
            "two bins"
        )

    if categorical_config[
        "primary_metric"
    ] != "population_stability_index":
        raise ValueError(
            "Unexpected categorical primary "
            "metric"
        )

    if categorical_config[
        "secondary_metric"
    ] != "jensen_shannon_distance":
        raise ValueError(
            "Unexpected categorical secondary "
            "metric"
        )

    for metric_config in (
        numeric_config,
        categorical_config,
    ):
        epsilon = float(
            metric_config["epsilon"]
        )

        if not 0 < epsilon < 1:
            raise ValueError(
                "Drift epsilon must be between "
                "zero and one"
            )

    unseen_category = (
        categorical_config[
            "unseen_category"
        ]
    )

    if (
        not isinstance(
            unseen_category,
            str,
        )
        or not unseen_category
    ):
        raise ValueError(
            "Unseen-category token must "
            "be a string"
        )

    if unseen_category == MISSING_CATEGORY:
        raise ValueError(
            "Unseen and missing category tokens "
            "must be different"
        )

    thresholds = config[
        "psi_interpretation"
    ]

    small_upper = float(
        thresholds[
            "small_upper_exclusive"
        ]
    )

    moderate_upper = float(
        thresholds[
            "moderate_upper_exclusive"
        ]
    )

    large_lower = float(
        thresholds[
            "large_lower_inclusive"
        ]
    )

    if not (
        0
        < small_upper
        < moderate_upper
    ):
        raise ValueError(
            "PSI thresholds are not ordered"
        )

    if moderate_upper != large_lower:
        raise ValueError(
            "Moderate and large PSI thresholds "
            "must meet"
        )

    classes = config[
        "class_conditional_analysis"
    ]["classes"]

    if (
        not classes
        or len(classes) != len(set(classes))
    ):
        raise ValueError(
            "Class-conditional classes must "
            "be unique"
        )


def _smoothed_distribution(
    counts,
    epsilon,
):
    counts = np.asarray(
        counts,
        dtype=np.float64,
    )

    if (
        counts.ndim != 1
        or counts.size == 0
    ):
        raise ValueError(
            "Distribution counts must be "
            "one-dimensional"
        )

    if (
        np.any(counts < 0)
        or counts.sum() <= 0
    ):
        raise ValueError(
            "Distribution counts must have "
            "positive mass"
        )

    proportions = (
        counts / counts.sum()
    )

    proportions = np.maximum(
        proportions,
        float(epsilon),
    )

    return (
        proportions
        / proportions.sum()
    )


def population_stability_index(
    reference_counts,
    comparison_counts,
    epsilon,
):
    reference_distribution = (
        _smoothed_distribution(
            reference_counts,
            epsilon,
        )
    )

    comparison_distribution = (
        _smoothed_distribution(
            comparison_counts,
            epsilon,
        )
    )

    if (
        reference_distribution.shape
        != comparison_distribution.shape
    ):
        raise ValueError(
            "PSI distributions must have "
            "matching shapes"
        )

    terms = (
        comparison_distribution
        - reference_distribution
    ) * np.log(
        comparison_distribution
        / reference_distribution
    )

    return float(np.sum(terms))


def interpret_psi(
    psi_value,
    config,
):
    thresholds = config[
        "psi_interpretation"
    ]

    if psi_value < float(
        thresholds[
            "small_upper_exclusive"
        ]
    ):
        return "small"

    if psi_value < float(
        thresholds[
            "moderate_upper_exclusive"
        ]
    ):
        return "moderate"

    return "large"


def _numeric_values(values):
    return pd.to_numeric(
        values,
        errors="coerce",
    )


def build_reference_quantile_edges(
    values,
    number_of_bins,
):
    numeric = (
        _numeric_values(values)
        .dropna()
        .to_numpy(dtype=np.float64)
    )

    if numeric.size == 0:
        raise ValueError(
            "Reference numeric feature has "
            "no values"
        )

    quantiles = np.linspace(
        0.0,
        1.0,
        int(number_of_bins) + 1,
    )

    edges = np.unique(
        np.quantile(
            numeric,
            quantiles,
        )
    )

    if edges.size == 1:
        return np.array(
            [
                -np.inf,
                np.inf,
            ],
            dtype=np.float64,
        )

    edges = edges.astype(
        np.float64,
        copy=True,
    )

    edges[0] = -np.inf
    edges[-1] = np.inf

    return edges


def _numeric_bin_counts(
    values,
    edges,
):
    numeric = _numeric_values(values)

    non_missing = (
        numeric
        .dropna()
        .to_numpy(dtype=np.float64)
    )

    histogram, _ = np.histogram(
        non_missing,
        bins=edges,
    )

    missing_count = int(
        numeric.isna().sum()
    )

    return np.append(
        histogram.astype(np.int64),
        missing_count,
    )


def _numeric_feature_metrics(
    reference_values,
    comparison_values,
    drift_config,
):
    numeric_config = drift_config[
        "numeric_features"
    ]

    edges = (
        build_reference_quantile_edges(
            reference_values,
            numeric_config["psi_bins"],
        )
    )

    reference_counts = (
        _numeric_bin_counts(
            reference_values,
            edges,
        )
    )

    comparison_counts = (
        _numeric_bin_counts(
            comparison_values,
            edges,
        )
    )

    psi_value = (
        population_stability_index(
            reference_counts,
            comparison_counts,
            numeric_config["epsilon"],
        )
    )

    reference_numeric = (
        _numeric_values(
            reference_values
        ).dropna()
    )

    comparison_numeric = (
        _numeric_values(
            comparison_values
        ).dropna()
    )

    if (
        reference_numeric.empty
        or comparison_numeric.empty
    ):
        ks_statistic = None
    else:
        ks_statistic = float(
            ks_2samp(
                reference_numeric.to_numpy(
                    dtype=np.float64
                ),
                comparison_numeric.to_numpy(
                    dtype=np.float64
                ),
                method="auto",
            ).statistic
        )

    return {
        "psi": psi_value,
        "secondary_metric_name": (
            numeric_config[
                "secondary_metric"
            ]
        ),
        "secondary_metric_value": (
            ks_statistic
        ),
        "effective_psi_bins": int(
            len(edges) - 1
        ),
        "unseen_category_percent": None,
    }


def _normalized_categories(
    values,
    unseen_category,
):
    normalized = (
        values
        .astype("string")
        .str.strip()
    )

    non_missing = normalized.dropna()

    reserved_tokens = {
        MISSING_CATEGORY,
        unseen_category,
    }

    collisions = sorted(
        set(
            non_missing
            .unique()
            .tolist()
        )
        & reserved_tokens
    )

    if collisions:
        raise ValueError(
            "Categorical values use reserved "
            f"tokens: {collisions}"
        )

    return normalized.fillna(
        MISSING_CATEGORY
    )


def _categorical_feature_metrics(
    reference_values,
    comparison_values,
    drift_config,
):
    categorical_config = drift_config[
        "categorical_features"
    ]

    unseen_category = (
        categorical_config[
            "unseen_category"
        ]
    )

    reference = _normalized_categories(
        reference_values,
        unseen_category,
    )

    comparison = _normalized_categories(
        comparison_values,
        unseen_category,
    )

    reference_categories = sorted(
        reference.unique().tolist()
    )

    if MISSING_CATEGORY not in (
        reference_categories
    ):
        reference_categories.append(
            MISSING_CATEGORY
        )
        reference_categories.sort()

    reference_category_set = set(
        reference_categories
    )

    comparison_mapped = (
        comparison.where(
            comparison.isin(
                reference_category_set
            ),
            unseen_category,
        )
    )

    categories = (
        reference_categories
        + [unseen_category]
    )

    reference_value_counts = (
        reference.value_counts(
            dropna=False
        )
    )

    comparison_value_counts = (
        comparison_mapped.value_counts(
            dropna=False
        )
    )

    reference_counts = np.array(
        [
            reference_value_counts.get(
                category,
                0,
            )
            for category in categories
        ],
        dtype=np.int64,
    )

    comparison_counts = np.array(
        [
            comparison_value_counts.get(
                category,
                0,
            )
            for category in categories
        ],
        dtype=np.int64,
    )

    epsilon = categorical_config[
        "epsilon"
    ]

    psi_value = (
        population_stability_index(
            reference_counts,
            comparison_counts,
            epsilon,
        )
    )

    reference_distribution = (
        _smoothed_distribution(
            reference_counts,
            epsilon,
        )
    )

    comparison_distribution = (
        _smoothed_distribution(
            comparison_counts,
            epsilon,
        )
    )

    js_distance = float(
        jensenshannon(
            reference_distribution,
            comparison_distribution,
            base=2.0,
        )
    )

    unseen_count = int(
        comparison_mapped
        .eq(unseen_category)
        .sum()
    )

    return {
        "psi": psi_value,
        "secondary_metric_name": (
            categorical_config[
                "secondary_metric"
            ]
        ),
        "secondary_metric_value": (
            js_distance
        ),
        "effective_psi_bins": int(
            len(categories)
        ),
        "unseen_category_percent": float(
            unseen_count
            / len(comparison_mapped)
            * 100
        ),
    }


def _build_feature_row(
    feature,
    feature_type,
    reference_values,
    comparison_values,
    comparison_name,
    scope,
    class_value,
    drift_config,
):
    if feature_type == "numeric":
        metrics = (
            _numeric_feature_metrics(
                reference_values,
                comparison_values,
                drift_config,
            )
        )
    elif feature_type == "categorical":
        metrics = (
            _categorical_feature_metrics(
                reference_values,
                comparison_values,
                drift_config,
            )
        )
    else:
        raise ValueError(
            "Unknown feature type: "
            f"{feature_type}"
        )

    psi_value = metrics["psi"]

    return {
        "comparison": comparison_name,
        "scope": scope,
        "class_value": class_value,
        "feature": feature,
        "feature_type": feature_type,
        "reference_count": int(
            len(reference_values)
        ),
        "comparison_count": int(
            len(comparison_values)
        ),
        "reference_missing_percent": float(
            reference_values
            .isna()
            .mean()
            * 100
        ),
        "comparison_missing_percent": float(
            comparison_values
            .isna()
            .mean()
            * 100
        ),
        "reference_unique_values": int(
            reference_values.nunique(
                dropna=False
            )
        ),
        "comparison_unique_values": int(
            comparison_values.nunique(
                dropna=False
            )
        ),
        "psi": psi_value,
        "psi_interpretation": (
            interpret_psi(
                psi_value,
                drift_config,
            )
        ),
        **metrics,
    }


def _build_prevalence_row(
    reference,
    comparison,
    target_column,
    comparison_name,
):
    reference_positive = int(
        reference[
            target_column
        ].eq(1).sum()
    )

    comparison_positive = int(
        comparison[
            target_column
        ].eq(1).sum()
    )

    reference_rate = (
        reference_positive
        / len(reference)
    )

    comparison_rate = (
        comparison_positive
        / len(comparison)
    )

    relative_change = (
        (
            comparison_rate
            / reference_rate
            - 1.0
        )
        * 100
        if reference_rate > 0
        else None
    )

    return {
        "comparison": comparison_name,
        "reference_patients": int(
            len(reference)
        ),
        "comparison_patients": int(
            len(comparison)
        ),
        "reference_n_positive": (
            reference_positive
        ),
        "comparison_n_positive": (
            comparison_positive
        ),
        "reference_positive_percent": float(
            reference_rate * 100
        ),
        "comparison_positive_percent": float(
            comparison_rate * 100
        ),
        "absolute_percentage_point_change": (
            float(
                (
                    comparison_rate
                    - reference_rate
                )
                * 100
            )
        ),
        "relative_percent_change": (
            float(relative_change)
            if relative_change is not None
            else None
        ),
    }


def _validate_analysis_inputs(
    model_table,
    feature_config,
    drift_config,
    comparisons,
    allow_locked_comparison,
):
    split_column = feature_config[
        "split_column"
    ]

    target_column = feature_config[
        "target_column"
    ]

    model_features = (
        feature_config[
            "model_numeric_features"
        ]
        + feature_config[
            "model_categorical_features"
        ]
    )

    required_columns = {
        split_column,
        target_column,
        *model_features,
    }

    missing_columns = sorted(
        required_columns
        - set(model_table.columns)
    )

    if missing_columns:
        raise ValueError(
            "Model table is missing drift "
            f"columns: {missing_columns}"
        )

    if model_table.empty:
        raise ValueError(
            "Cannot analyze drift in an empty "
            "model table"
        )

    if model_table[
        target_column
    ].isna().any():
        raise ValueError(
            "Drift input contains missing targets"
        )

    observed_targets = set(
        model_table[
            target_column
        ].unique().tolist()
    )

    configured_classes = set(
        drift_config[
            "class_conditional_analysis"
        ]["classes"]
    )

    if not observed_targets.issubset(
        configured_classes
    ):
        raise ValueError(
            "Drift input contains unexpected "
            "target values"
        )

    locked_comparison = drift_config[
        "locked_comparison"
    ]

    if (
        locked_comparison in comparisons
        and not allow_locked_comparison
    ):
        raise PermissionError(
            "Locked temporal comparison is "
            "forbidden during development"
        )

    if not allow_locked_comparison:
        allowed = set(
            drift_config[
                "development_comparisons"
            ]
        )

        unexpected = sorted(
            set(comparisons) - allowed
        )

        if unexpected:
            raise PermissionError(
                "Non-development comparisons "
                f"were requested: {unexpected}"
            )

    reference_split = drift_config[
        "reference_split"
    ]

    required_splits = {
        reference_split,
        *comparisons,
    }

    available_splits = set(
        model_table[
            split_column
        ].unique().tolist()
    )

    missing_splits = sorted(
        required_splits - available_splits
    )

    if missing_splits:
        raise ValueError(
            "Drift input is missing splits: "
            f"{missing_splits}"
        )


def analyze_seer_drift(
    model_table,
    feature_config,
    drift_config,
    comparisons=None,
    allow_locked_comparison=False,
):
    validate_drift_config(
        drift_config
    )

    if comparisons is None:
        comparisons = list(
            drift_config[
                "development_comparisons"
            ]
        )
    else:
        comparisons = list(comparisons)

    if (
        not comparisons
        or len(comparisons)
        != len(set(comparisons))
    ):
        raise ValueError(
            "Drift comparisons must be "
            "non-empty and unique"
        )

    _validate_analysis_inputs(
        model_table,
        feature_config,
        drift_config,
        comparisons,
        allow_locked_comparison,
    )

    split_column = feature_config[
        "split_column"
    ]

    target_column = feature_config[
        "target_column"
    ]

    reference_name = drift_config[
        "reference_split"
    ]

    reference = model_table.loc[
        model_table[
            split_column
        ].eq(reference_name)
    ]

    classes = drift_config[
        "class_conditional_analysis"
    ]["classes"]

    class_conditional_enabled = (
        drift_config[
            "class_conditional_analysis"
        ]["enabled"]
    )

    feature_definitions = [
        (
            feature,
            "numeric",
        )
        for feature in feature_config[
            "model_numeric_features"
        ]
    ] + [
        (
            feature,
            "categorical",
        )
        for feature in feature_config[
            "model_categorical_features"
        ]
    ]

    feature_rows = []
    prevalence_rows = []
    summary_rows = []

    for comparison_name in comparisons:
        comparison = model_table.loc[
            model_table[
                split_column
            ].eq(comparison_name)
        ]

        prevalence_rows.append(
            _build_prevalence_row(
                reference,
                comparison,
                target_column,
                comparison_name,
            )
        )

        for (
            feature,
            feature_type,
        ) in feature_definitions:
            feature_rows.append(
                _build_feature_row(
                    feature,
                    feature_type,
                    reference[feature],
                    comparison[feature],
                    comparison_name,
                    "overall",
                    "all",
                    drift_config,
                )
            )

        if class_conditional_enabled:
            for class_value in classes:
                reference_class = (
                    reference.loc[
                        reference[
                            target_column
                        ].eq(class_value)
                    ]
                )

                comparison_class = (
                    comparison.loc[
                        comparison[
                            target_column
                        ].eq(class_value)
                    ]
                )

                if (
                    reference_class.empty
                    or comparison_class.empty
                ):
                    raise ValueError(
                        "Class-conditional drift "
                        "subset is empty for "
                        f"comparison="
                        f"{comparison_name}, "
                        f"class={class_value}"
                    )

                for (
                    feature,
                    feature_type,
                ) in feature_definitions:
                    feature_rows.append(
                        _build_feature_row(
                            feature,
                            feature_type,
                            reference_class[
                                feature
                            ],
                            comparison_class[
                                feature
                            ],
                            comparison_name,
                            "class_conditional",
                            str(class_value),
                            drift_config,
                        )
                    )

        comparison_rows = [
            row
            for row in feature_rows
            if row["comparison"]
            == comparison_name
        ]

        overall_rows = [
            row
            for row in comparison_rows
            if row["scope"] == "overall"
        ]

        class_rows = [
            row
            for row in comparison_rows
            if row["scope"]
            == "class_conditional"
        ]

        small_upper = float(
            drift_config[
                "psi_interpretation"
            ][
                "small_upper_exclusive"
            ]
        )

        class_conditionals_stable = (
            bool(class_rows)
            and all(
                row["psi"] < small_upper
                for row in class_rows
            )
        )

        summary_rows.append(
            {
                "comparison": (
                    comparison_name
                ),
                "overall_maximum_psi": float(
                    max(
                        row["psi"]
                        for row
                        in overall_rows
                    )
                ),
                "overall_moderate_or_large_features": int(
                    sum(
                        row[
                            "psi_interpretation"
                        ]
                        in {
                            "moderate",
                            "large",
                        }
                        for row
                        in overall_rows
                    )
                ),
                "class_conditional_maximum_psi": (
                    float(
                        max(
                            row["psi"]
                            for row
                            in class_rows
                        )
                    )
                    if class_rows
                    else None
                ),
                "class_conditional_moderate_or_large_features": int(
                    sum(
                        row[
                            "psi_interpretation"
                        ]
                        in {
                            "moderate",
                            "large",
                        }
                        for row
                        in class_rows
                    )
                ),
                "all_univariate_class_conditionals_small": (
                    class_conditionals_stable
                ),
                "label_shift_assessment": (
                    "compatible_under_"
                    "univariate_checks"
                    if class_conditionals_stable
                    else
                    "not_compatible_with_"
                    "pure_label_shift_under_"
                    "univariate_checks"
                ),
            }
        )

    feature_drift = pd.DataFrame(
        feature_rows
    )

    prevalence = pd.DataFrame(
        prevalence_rows
    )

    comparison_summary = pd.DataFrame(
        summary_rows
    )

    audit = {
        "drift_config_version": (
            drift_config["version"]
        ),
        "feature_config_version": (
            feature_config["version"]
        ),
        "reference_split": reference_name,
        "development_comparisons": (
            comparisons
        ),
        "locked_comparison": (
            drift_config[
                "locked_comparison"
            ]
        ),
        "locked_comparison_included": bool(
            drift_config[
                "locked_comparison"
            ]
            in comparisons
        ),
        "class_conditional_analysis_enabled": bool(
            class_conditional_enabled
        ),
        "model_feature_count": int(
            len(feature_definitions)
        ),
        "feature_drift_rows": int(
            len(feature_drift)
        ),
        "split_patient_counts": {
            split_name: int(
                model_table[
                    split_column
                ]
                .eq(split_name)
                .sum()
            )
            for split_name in [
                reference_name,
                *comparisons,
            ]
        },
        "automatic_feature_dropping_performed": False,
        "automatic_feature_reweighting_performed": False,
    }

    return {
        "feature_drift": feature_drift,
        "prevalence": prevalence,
        "comparison_summary": (
            comparison_summary
        ),
        "audit": audit,
    }
    