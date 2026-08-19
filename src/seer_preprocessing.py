import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


PREPROCESSOR_STATE_VERSION = "1.0"


def load_json(path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file_handle:
        return json.load(file_handle)


def validate_preprocessing_contract(
    joint_config,
    feature_config,
):
    tabular_path = joint_config["tabular_path"]
    tabular_task = joint_config["tasks"]["tabular"]

    numeric_features = tabular_path[
        "numeric_features"
    ]
    categorical_features = tabular_path[
        "categorical_features"
    ]

    if numeric_features != feature_config[
        "model_numeric_features"
    ]:
        raise ValueError(
            "Joint-model numeric features do not "
            "match the SEER feature contract"
        )

    if categorical_features != feature_config[
        "model_categorical_features"
    ]:
        raise ValueError(
            "Joint-model categorical features do "
            "not match the SEER feature contract"
        )

    if tabular_task["target"] != feature_config[
        "target_column"
    ]:
        raise ValueError(
            "Joint-model target does not match the "
            "SEER feature contract"
        )

    all_features = (
        numeric_features
        + categorical_features
    )

    if len(all_features) != len(
        set(all_features)
    ):
        raise ValueError(
            "SEER model features must be unique"
        )

    forbidden_features = set(
        feature_config["forbidden_model_inputs"]
    )

    forbidden_overlap = (
        set(all_features)
        & forbidden_features
    )

    if forbidden_overlap:
        raise ValueError(
            "Forbidden variables were included as "
            f"model inputs: {sorted(forbidden_overlap)}"
        )

    numeric_preprocessing = tabular_path[
        "numeric_preprocessing"
    ]

    if numeric_preprocessing["fit_split"] != (
        tabular_task["training_split"]
    ):
        raise ValueError(
            "Numeric preprocessing must be fitted "
            "on the training split"
        )

    if numeric_preprocessing[
        "missing_value_strategy"
    ] != "median":
        raise ValueError(
            "Only median numeric imputation is "
            "permitted"
        )

    if numeric_preprocessing[
        "scaling"
    ] != "standard":
        raise ValueError(
            "Only standard numeric scaling is "
            "permitted"
        )

    if numeric_preprocessing[
        "add_missing_indicator"
    ]:
        raise ValueError(
            "Numeric missing indicators are not "
            "part of the frozen configuration"
        )

    categorical_preprocessing = tabular_path[
        "categorical_preprocessing"
    ]

    if categorical_preprocessing[
        "fit_split"
    ] != tabular_task["training_split"]:
        raise ValueError(
            "Categorical vocabularies must be fitted "
            "on the training split"
        )

    if categorical_preprocessing[
        "known_category_start_index"
    ] != 1:
        raise ValueError(
            "Known category indices must begin at 1"
        )

    if categorical_preprocessing[
        "unknown_category_index"
    ] != 0:
        raise ValueError(
            "Unknown category index must be 0"
        )

    if categorical_preprocessing[
        "missing_category_index"
    ] != 0:
        raise ValueError(
            "Missing category index must be 0"
        )

    expected_cardinalities = tabular_path[
        "training_category_cardinalities"
    ]

    if set(expected_cardinalities) != set(
        categorical_features
    ):
        raise ValueError(
            "Training category cardinalities do not "
            "cover the categorical feature list"
        )

    if any(
        int(cardinality) <= 0
        for cardinality
        in expected_cardinalities.values()
    ):
        raise ValueError(
            "Category cardinalities must be positive"
        )

    test_protection = joint_config[
        "test_protection"
    ]

    if not test_protection[
        "seer_temporal_test_locked"
    ]:
        raise ValueError(
            "SEER temporal test protection must "
            "remain enabled"
        )

    if test_protection[
        "allow_test_for_preprocessing"
    ]:
        raise ValueError(
            "The locked temporal test cannot be used "
            "for preprocessing"
        )


def load_preprocessing_contract(
    joint_config_path,
    feature_config_path,
):
    joint_config = load_json(
        joint_config_path
    )
    feature_config = load_json(
        feature_config_path
    )

    validate_preprocessing_contract(
        joint_config,
        feature_config,
    )

    return joint_config, feature_config


def normalize_categorical_series(series):
    normalized = (
        series
        .astype("string")
        .str.strip()
    )

    normalized = normalized.mask(
        normalized.isna()
        | normalized.eq("")
    )

    return normalized


def calculate_embedding_dimension(
    cardinality,
    minimum_dimension,
    maximum_dimension,
):
    dimension = round(
        math.sqrt(cardinality)
    )

    return int(
        min(
            maximum_dimension,
            max(
                minimum_dimension,
                dimension,
            ),
        )
    )


class SeerTensorDataset(Dataset):
    def __init__(
        self,
        numeric_features,
        categorical_features,
        targets,
    ):
        if numeric_features.ndim != 2:
            raise ValueError(
                "Numeric tensor must be two-dimensional"
            )

        if categorical_features.ndim != 2:
            raise ValueError(
                "Categorical tensor must be "
                "two-dimensional"
            )

        if targets.ndim != 1:
            raise ValueError(
                "Target tensor must be one-dimensional"
            )

        row_counts = {
            int(numeric_features.shape[0]),
            int(categorical_features.shape[0]),
            int(targets.shape[0]),
        }

        if len(row_counts) != 1:
            raise ValueError(
                "SEER tensors have inconsistent "
                "row counts"
            )

        if not numeric_features.is_floating_point():
            raise TypeError(
                "Numeric features must use a floating "
                "point dtype"
            )

        if categorical_features.dtype != (
            torch.int64
        ):
            raise TypeError(
                "Categorical features must use "
                "torch.int64"
            )

        if not targets.is_floating_point():
            raise TypeError(
                "Targets must use a floating point dtype"
            )

        self.numeric_features = (
            numeric_features.contiguous()
        )
        self.categorical_features = (
            categorical_features.contiguous()
        )
        self.targets = targets.contiguous()

    def __len__(self):
        return int(self.targets.shape[0])

    def __getitem__(self, index):
        return {
            "numeric": self.numeric_features[index],
            "categorical": (
                self.categorical_features[index]
            ),
            "target": self.targets[index],
        }


class SeerPreprocessor:
    def __init__(
        self,
        joint_config,
        feature_config,
    ):
        validate_preprocessing_contract(
            joint_config,
            feature_config,
        )

        self.joint_config_version = (
            joint_config["version"]
        )
        self.feature_config_version = (
            feature_config["version"]
        )

        self.tabular_task = joint_config[
            "tasks"
        ]["tabular"]

        self.tabular_config = joint_config[
            "tabular_path"
        ]

        self.split_column = feature_config[
            "split_column"
        ]
        self.target_column = feature_config[
            "target_column"
        ]

        self.numeric_features = list(
            self.tabular_config[
                "numeric_features"
            ]
        )
        self.categorical_features = list(
            self.tabular_config[
                "categorical_features"
            ]
        )

        self.expected_cardinalities = dict(
            self.tabular_config[
                "training_category_cardinalities"
            ]
        )

        categorical_preprocessing = (
            self.tabular_config[
                "categorical_preprocessing"
            ]
        )

        self.known_category_start_index = int(
            categorical_preprocessing[
                "known_category_start_index"
            ]
        )
        self.unknown_category_index = int(
            categorical_preprocessing[
                "unknown_category_index"
            ]
        )
        self.missing_category_index = int(
            categorical_preprocessing[
                "missing_category_index"
            ]
        )

        embedding_rule = self.tabular_config[
            "embedding_dimension_rule"
        ]

        self.minimum_embedding_dimension = int(
            embedding_rule["minimum_dimension"]
        )
        self.maximum_embedding_dimension = int(
            embedding_rule["maximum_dimension"]
        )

        self.numeric_statistics = {}
        self.category_vocabularies = {}
        self.embedding_dimensions = {}
        self.fitted = False

    @property
    def training_split(self):
        return self.tabular_task[
            "training_split"
        ]

    @property
    def locked_test_split(self):
        return self.tabular_task[
            "locked_test_split"
        ]

    @property
    def allowed_splits(self):
        return {
            self.tabular_task[
                "training_split"
            ],
            self.tabular_task[
                "internal_validation_split"
            ],
            self.tabular_task[
                "selection_validation_split"
            ],
            self.tabular_task[
                "locked_test_split"
            ],
        }

    @property
    def num_embeddings(self):
        if not self.fitted:
            raise RuntimeError(
                "Preprocessor has not been fitted"
            )

        return {
            feature: (
                len(
                    self.category_vocabularies[
                        feature
                    ]
                )
                + self.known_category_start_index
            )
            for feature
            in self.categorical_features
        }

    def _validate_frame_columns(
        self,
        frame,
    ):
        required_columns = {
            self.split_column,
            self.target_column,
            *self.numeric_features,
            *self.categorical_features,
        }

        missing_columns = sorted(
            required_columns
            - set(frame.columns)
        )

        if missing_columns:
            raise ValueError(
                "SEER frame is missing required "
                f"columns: {missing_columns}"
            )

    def _validate_single_split(
        self,
        frame,
        expected_split,
    ):
        observed_splits = set(
            frame[
                self.split_column
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        if observed_splits != {
            expected_split
        }:
            raise ValueError(
                "Expected only split "
                f"{expected_split}, found "
                f"{sorted(observed_splits)}"
            )

    def _validate_target(
        self,
        frame,
    ):
        target = pd.to_numeric(
            frame[self.target_column],
            errors="coerce",
        )

        if target.isna().any():
            raise ValueError(
                "SEER target contains missing or "
                "non-numeric values"
            )

        unique_targets = set(
            target.astype(int).unique().tolist()
        )

        if not unique_targets.issubset(
            {0, 1}
        ):
            raise ValueError(
                "SEER target must contain only 0 and 1"
            )

        return target.to_numpy(
            dtype=np.float32,
            copy=True,
        )

    def fit(
        self,
        train_frame,
    ):
        if len(train_frame) == 0:
            raise ValueError(
                "Cannot fit preprocessing on an "
                "empty frame"
            )

        self._validate_frame_columns(
            train_frame
        )
        self._validate_single_split(
            train_frame,
            self.training_split,
        )
        self._validate_target(
            train_frame
        )

        numeric_statistics = {}

        for feature in self.numeric_features:
            values = pd.to_numeric(
                train_frame[feature],
                errors="coerce",
            ).to_numpy(
                dtype=np.float64,
                na_value=np.nan,
            )

            finite_mask = np.isfinite(
                values
            )

            if not finite_mask.any():
                raise ValueError(
                    "Numeric feature contains no "
                    f"finite training values: {feature}"
                )

            median = float(
                np.median(
                    values[finite_mask]
                )
            )

            imputed = np.where(
                finite_mask,
                values,
                median,
            )

            mean = float(
                np.mean(imputed)
            )
            scale = float(
                np.std(
                    imputed,
                    ddof=0,
                )
            )

            if (
                not np.isfinite(scale)
                or scale <= 0.0
            ):
                raise ValueError(
                    "Numeric feature has invalid "
                    f"training scale: {feature}"
                )

            numeric_statistics[feature] = {
                "median": median,
                "mean": mean,
                "scale": scale,
                "training_missing_count": int(
                    (~finite_mask).sum()
                ),
            }

        category_vocabularies = {}
        embedding_dimensions = {}

        for feature in (
            self.categorical_features
        ):
            normalized = (
                normalize_categorical_series(
                    train_frame[feature]
                )
            )

            categories = sorted(
                normalized
                .dropna()
                .unique()
                .tolist()
            )

            expected_cardinality = int(
                self.expected_cardinalities[
                    feature
                ]
            )

            if len(categories) != (
                expected_cardinality
            ):
                raise ValueError(
                    "Training cardinality changed "
                    f"for {feature}: expected "
                    f"{expected_cardinality}, found "
                    f"{len(categories)}"
                )

            vocabulary = {
                category: index
                for index, category in enumerate(
                    categories,
                    start=(
                        self.known_category_start_index
                    ),
                )
            }

            category_vocabularies[
                feature
            ] = vocabulary

            embedding_dimensions[
                feature
            ] = calculate_embedding_dimension(
                cardinality=len(categories),
                minimum_dimension=(
                    self.minimum_embedding_dimension
                ),
                maximum_dimension=(
                    self.maximum_embedding_dimension
                ),
            )

        self.numeric_statistics = (
            numeric_statistics
        )
        self.category_vocabularies = (
            category_vocabularies
        )
        self.embedding_dimensions = (
            embedding_dimensions
        )
        self.fitted = True

        return self

    def transform(
        self,
        frame,
        split_name,
        allow_locked_test=False,
    ):
        if not self.fitted:
            raise RuntimeError(
                "Preprocessor must be fitted before "
                "transform"
            )

        if split_name not in self.allowed_splits:
            raise ValueError(
                f"Unknown SEER split: {split_name}"
            )

        if (
            split_name
            == self.locked_test_split
            and not allow_locked_test
        ):
            raise PermissionError(
                "SEER temporal_test is locked during "
                "development"
            )

        if len(frame) == 0:
            raise ValueError(
                f"SEER split is empty: {split_name}"
            )

        self._validate_frame_columns(
            frame
        )
        self._validate_single_split(
            frame,
            split_name,
        )

        row_count = len(frame)

        numeric_array = np.empty(
            (
                row_count,
                len(self.numeric_features),
            ),
            dtype=np.float32,
        )

        numeric_missing_counts = {}

        for column_index, feature in enumerate(
            self.numeric_features
        ):
            statistics = (
                self.numeric_statistics[
                    feature
                ]
            )

            values = pd.to_numeric(
                frame[feature],
                errors="coerce",
            ).to_numpy(
                dtype=np.float64,
                na_value=np.nan,
            )

            finite_mask = np.isfinite(
                values
            )

            numeric_missing_counts[
                feature
            ] = int(
                (~finite_mask).sum()
            )

            imputed = np.where(
                finite_mask,
                values,
                statistics["median"],
            )

            scaled = (
                imputed
                - statistics["mean"]
            ) / statistics["scale"]

            if not np.isfinite(
                scaled
            ).all():
                raise ValueError(
                    "Non-finite values remained after "
                    f"preprocessing: {feature}"
                )

            numeric_array[
                :,
                column_index,
            ] = scaled.astype(
                np.float32
            )

        categorical_array = np.empty(
            (
                row_count,
                len(
                    self.categorical_features
                ),
            ),
            dtype=np.int64,
        )

        categorical_missing_counts = {}
        categorical_unknown_counts = {}

        for column_index, feature in enumerate(
            self.categorical_features
        ):
            normalized = (
                normalize_categorical_series(
                    frame[feature]
                )
            )

            missing_mask = (
                normalized.isna()
            )

            mapped = normalized.map(
                self.category_vocabularies[
                    feature
                ]
            )

            unknown_mask = (
                mapped.isna()
                & ~missing_mask
            )

            categorical_missing_counts[
                feature
            ] = int(
                missing_mask.sum()
            )
            categorical_unknown_counts[
                feature
            ] = int(
                unknown_mask.sum()
            )

            indices = (
                mapped
                .fillna(
                    self.unknown_category_index
                )
                .astype(np.int64)
                .to_numpy()
            )

            categorical_array[
                :,
                column_index,
            ] = indices

        target_array = self._validate_target(
            frame
        )

        dataset = SeerTensorDataset(
            numeric_features=torch.from_numpy(
                numeric_array
            ),
            categorical_features=torch.from_numpy(
                categorical_array
            ),
            targets=torch.from_numpy(
                target_array
            ),
        )

        positive_count = int(
            target_array.sum()
        )
        negative_count = int(
            row_count - positive_count
        )

        audit = {
            "split": split_name,
            "rows": int(row_count),
            "negative_count": negative_count,
            "positive_count": positive_count,
            "positive_percent": float(
                100.0
                * positive_count
                / row_count
            ),
            "numeric_missing_counts": (
                numeric_missing_counts
            ),
            "categorical_missing_counts": (
                categorical_missing_counts
            ),
            "categorical_unknown_counts": (
                categorical_unknown_counts
            ),
            "patient_identifiers_in_tensors": False,
        }

        return dataset, audit

    def state_dict(self):
        if not self.fitted:
            raise RuntimeError(
                "Cannot serialize an unfitted "
                "preprocessor"
            )

        return {
            "state_version": (
                PREPROCESSOR_STATE_VERSION
            ),
            "joint_config_version": (
                self.joint_config_version
            ),
            "feature_config_version": (
                self.feature_config_version
            ),
            "numeric_features": (
                self.numeric_features
            ),
            "categorical_features": (
                self.categorical_features
            ),
            "numeric_statistics": (
                self.numeric_statistics
            ),
            "category_vocabularies": (
                self.category_vocabularies
            ),
            "embedding_dimensions": (
                self.embedding_dimensions
            ),
            "num_embeddings": (
                self.num_embeddings
            ),
            "unknown_category_index": (
                self.unknown_category_index
            ),
        }

    def save(
        self,
        output_path,
    ):
        output_path = Path(
            output_path
        )
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file_handle:
            json.dump(
                self.state_dict(),
                file_handle,
                indent=2,
                sort_keys=True,
            )
            file_handle.write("\n")

    @classmethod
    def load(
        cls,
        state_path,
        joint_config,
        feature_config,
    ):
        state_path = Path(
            state_path
        )

        if not state_path.is_file():
            raise FileNotFoundError(
                "Preprocessor state not found: "
                f"{state_path}"
            )

        with state_path.open(
            "r",
            encoding="utf-8",
        ) as file_handle:
            state = json.load(
                file_handle
            )

        if state["state_version"] != (
            PREPROCESSOR_STATE_VERSION
        ):
            raise ValueError(
                "Unsupported preprocessor state version"
            )

        preprocessor = cls(
            joint_config,
            feature_config,
        )

        if state[
            "numeric_features"
        ] != preprocessor.numeric_features:
            raise ValueError(
                "Saved numeric feature order changed"
            )

        if state[
            "categorical_features"
        ] != preprocessor.categorical_features:
            raise ValueError(
                "Saved categorical feature order changed"
            )

        preprocessor.numeric_statistics = (
            state["numeric_statistics"]
        )
        preprocessor.category_vocabularies = (
            state["category_vocabularies"]
        )
        preprocessor.embedding_dimensions = {
            feature: int(dimension)
            for feature, dimension
            in state[
                "embedding_dimensions"
            ].items()
        }
        preprocessor.fitted = True

        expected_num_embeddings = (
            preprocessor.num_embeddings
        )

        saved_num_embeddings = {
            feature: int(value)
            for feature, value
            in state[
                "num_embeddings"
            ].items()
        }

        if saved_num_embeddings != (
            expected_num_embeddings
        ):
            raise ValueError(
                "Saved category vocabulary sizes "
                "are inconsistent"
            )

        return preprocessor


def read_seer_split(
    model_table_path,
    split_name,
    preprocessor,
    allow_locked_test=False,
):
    if split_name not in (
        preprocessor.allowed_splits
    ):
        raise ValueError(
            f"Unknown SEER split: {split_name}"
        )

    if (
        split_name
        == preprocessor.locked_test_split
        and not allow_locked_test
    ):
        raise PermissionError(
            "SEER temporal_test is locked during "
            "development"
        )

    model_table_path = Path(
        model_table_path
    )

    if not model_table_path.is_file():
        raise FileNotFoundError(
            "SEER model table not found: "
            f"{model_table_path}"
        )

    columns = [
        preprocessor.split_column,
        *preprocessor.numeric_features,
        *preprocessor.categorical_features,
        preprocessor.target_column,
    ]

    frame = pd.read_parquet(
        model_table_path,
        columns=columns,
        filters=[
            (
                preprocessor.split_column,
                "==",
                split_name,
            )
        ],
        engine="pyarrow",
    )

    preprocessor._validate_frame_columns(
        frame
    )
    preprocessor._validate_single_split(
        frame,
        split_name,
    )

    return frame
