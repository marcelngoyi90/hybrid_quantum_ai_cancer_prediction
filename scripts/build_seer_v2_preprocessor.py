import hashlib
import json
import sys
from pathlib import Path


project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.seer_preprocessing import (
    SeerPreprocessor,
    load_preprocessing_contract,
    read_seer_split,
)


model_table_path = (
    project_root
    / "data"
    / "processed"
    / "seer_v2_model_table.parquet"
)

joint_config_path = (
    project_root
    / "configs"
    / "joint_model_v2.json"
)

feature_config_path = (
    project_root
    / "configs"
    / "seer_features_v2.json"
)

baseline_state_path = (
    project_root
    / "artifacts"
    / "features"
    / "seer_preprocessor.json"
)

private_state_directory = (
    project_root
    / "artifacts"
    / "features"
)

private_state_path = (
    private_state_directory
    / "seer_v2_preprocessor.json"
)

metrics_directory = (
    project_root
    / "artifacts"
    / "metrics"
)

audit_path = (
    metrics_directory
    / "seer_v2_preprocessing_audit.json"
)


EXPECTED_SPLIT_ROWS = {
    "train": 461292,
    "id_val": 51241,
    "temporal_val": 107249,
}

EXPECTED_ADDED_CATEGORICAL_FEATURES = [
    "t_stage_harmonized",
    "m_stage_harmonized",
]


def calculate_sha256(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as file_handle:
        while True:
            chunk = file_handle.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def build_file_audit(path):
    path = Path(path)

    return {
        "file_name": path.name,
        "size_bytes": int(
            path.stat().st_size
        ),
        "sha256": calculate_sha256(path),
    }


def load_private_state(path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            "Private preprocessing state was "
            f"not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file_handle:
        return json.load(file_handle)


def validate_against_baseline(
    preprocessor,
    baseline_state,
):
    v2_state = preprocessor.state_dict()

    baseline_numeric_features = (
        baseline_state["numeric_features"]
    )

    baseline_categorical_features = (
        baseline_state[
            "categorical_features"
        ]
    )

    if (
        v2_state["numeric_features"]
        != baseline_numeric_features
    ):
        raise RuntimeError(
            "SEER v2 changed the baseline "
            "numeric feature order"
        )

    expected_v2_categorical_features = (
        baseline_categorical_features
        + EXPECTED_ADDED_CATEGORICAL_FEATURES
    )

    if (
        v2_state["categorical_features"]
        != expected_v2_categorical_features
    ):
        raise RuntimeError(
            "SEER v2 categorical feature "
            "amendment is incorrect"
        )

    if (
        v2_state["numeric_statistics"]
        != baseline_state[
            "numeric_statistics"
        ]
    ):
        raise RuntimeError(
            "SEER v2 changed baseline numeric "
            "preprocessing statistics"
        )

    for feature in (
        baseline_categorical_features
    ):
        if (
            v2_state[
                "category_vocabularies"
            ][feature]
            != baseline_state[
                "category_vocabularies"
            ][feature]
        ):
            raise RuntimeError(
                "SEER v2 changed the baseline "
                f"vocabulary for {feature}"
            )

        if (
            v2_state[
                "embedding_dimensions"
            ][feature]
            != baseline_state[
                "embedding_dimensions"
            ][feature]
        ):
            raise RuntimeError(
                "SEER v2 changed the baseline "
                f"embedding dimension for "
                f"{feature}"
            )

        if (
            v2_state[
                "num_embeddings"
            ][feature]
            != baseline_state[
                "num_embeddings"
            ][feature]
        ):
            raise RuntimeError(
                "SEER v2 changed the baseline "
                f"embedding count for {feature}"
            )

    added_feature_summary = {}

    for feature in (
        EXPECTED_ADDED_CATEGORICAL_FEATURES
    ):
        if feature not in (
            v2_state[
                "category_vocabularies"
            ]
        ):
            raise RuntimeError(
                "SEER v2 preprocessor is missing "
                f"{feature}"
            )

        added_feature_summary[feature] = {
            "training_cardinality": int(
                len(
                    v2_state[
                        "category_vocabularies"
                    ][feature]
                )
            ),
            "num_embeddings": int(
                v2_state[
                    "num_embeddings"
                ][feature]
            ),
            "embedding_dimension": int(
                v2_state[
                    "embedding_dimensions"
                ][feature]
            ),
        }

    return {
        "numeric_statistics_equal": True,
        "baseline_categorical_mappings_equal": (
            True
        ),
        "baseline_embedding_dimensions_equal": (
            True
        ),
        "baseline_num_embeddings_equal": True,
        "baseline_feature_order_preserved": True,
        "only_added_categorical_features": list(
            EXPECTED_ADDED_CATEGORICAL_FEATURES
        ),
        "added_feature_summary": (
            added_feature_summary
        ),
    }


def validate_public_audit(audit):
    serialized = json.dumps(
        audit,
        sort_keys=True,
    )

    forbidden_content = [
        "Patient ID",
        "category_vocabularies",
    ]

    found = [
        item
        for item in forbidden_content
        if item in serialized
    ]

    if found:
        raise RuntimeError(
            "Public preprocessing audit contains "
            f"private content: {found}"
        )


def main():
    required_paths = (
        model_table_path,
        joint_config_path,
        feature_config_path,
        baseline_state_path,
    )

    for required_path in required_paths:
        if not required_path.is_file():
            raise FileNotFoundError(
                f"Required file not found: "
                f"{required_path}"
            )

    private_state_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        joint_config,
        feature_config,
    ) = load_preprocessing_contract(
        joint_config_path,
        feature_config_path,
    )

    preprocessor = SeerPreprocessor(
        joint_config,
        feature_config,
    )

    development_splits = (
        preprocessor.training_split,
        joint_config[
            "tasks"
        ][
            "tabular"
        ][
            "internal_validation_split"
        ],
        joint_config[
            "tasks"
        ][
            "tabular"
        ][
            "selection_validation_split"
        ],
    )

    if (
        preprocessor.locked_test_split
        in development_splits
    ):
        raise RuntimeError(
            "Locked temporal test was included "
            "in development preprocessing"
        )

    if set(development_splits) != set(
        EXPECTED_SPLIT_ROWS
    ):
        raise RuntimeError(
            "Unexpected development split "
            "contract"
        )

    train_frame = read_seer_split(
        model_table_path,
        preprocessor.training_split,
        preprocessor,
    )

    expected_train_rows = (
        EXPECTED_SPLIT_ROWS[
            preprocessor.training_split
        ]
    )

    if len(train_frame) != (
        expected_train_rows
    ):
        raise RuntimeError(
            "Unexpected SEER v2 training row "
            f"count: {len(train_frame)}"
        )

    preprocessor.fit(train_frame)

    baseline_state = load_private_state(
        baseline_state_path
    )

    baseline_equivalence = (
        validate_against_baseline(
            preprocessor,
            baseline_state,
        )
    )

    preprocessor.save(
        private_state_path
    )

    restored_preprocessor = (
        SeerPreprocessor.load(
            private_state_path,
            joint_config,
            feature_config,
        )
    )

    if (
        restored_preprocessor.state_dict()
        != preprocessor.state_dict()
    ):
        raise RuntimeError(
            "Saved SEER v2 preprocessor state "
            "changed during round-trip validation"
        )

    split_audits = {}

    for split_name in development_splits:
        if split_name == (
            preprocessor.training_split
        ):
            frame = train_frame
        else:
            frame = read_seer_split(
                model_table_path,
                split_name,
                preprocessor,
            )

        expected_rows = (
            EXPECTED_SPLIT_ROWS[
                split_name
            ]
        )

        if len(frame) != expected_rows:
            raise RuntimeError(
                "Unexpected row count for "
                f"{split_name}: {len(frame)}"
            )

        dataset, split_audit = (
            restored_preprocessor.transform(
                frame,
                split_name,
            )
        )

        if len(dataset) != expected_rows:
            raise RuntimeError(
                "Transformed dataset row count "
                f"changed for {split_name}"
            )

        split_audits[
            split_name
        ] = split_audit

        del dataset

        if split_name != (
            preprocessor.training_split
        ):
            del frame

    del train_frame

    categorical_summary = {}

    for feature in (
        restored_preprocessor
        .categorical_features
    ):
        categorical_summary[
            feature
        ] = {
            "training_cardinality": int(
                len(
                    restored_preprocessor
                    .category_vocabularies[
                        feature
                    ]
                )
            ),
            "num_embeddings": int(
                restored_preprocessor
                .num_embeddings[
                    feature
                ]
            ),
            "embedding_dimension": int(
                restored_preprocessor
                .embedding_dimensions[
                    feature
                ]
            ),
        }

    audit = {
        "version": "2.0",
        "fit_split": (
            restored_preprocessor
            .training_split
        ),
        "target": (
            restored_preprocessor
            .target_column
        ),
        "numeric_features": (
            restored_preprocessor
            .numeric_features
        ),
        "categorical_features": (
            restored_preprocessor
            .categorical_features
        ),
        "numeric_statistics": (
            restored_preprocessor
            .numeric_statistics
        ),
        "categorical_summary": (
            categorical_summary
        ),
        "baseline_equivalence": (
            baseline_equivalence
        ),
        "development_split_audits": (
            split_audits
        ),
        "test_protection": {
            "locked_split": (
                restored_preprocessor
                .locked_test_split
            ),
            "locked_split_loaded": False,
            "locked_split_transformed": False,
            "locked_split_used_for_fit": False,
        },
        "privacy": {
            "patient_identifiers_loaded": False,
            "patient_identifiers_in_tensors": False,
            "patient_records_in_public_audit": False,
            "category_vocabulary_values_public": (
                False
            ),
        },
        "files": {
            "private_v2_model_table": (
                build_file_audit(
                    model_table_path
                )
            ),
            "v2_joint_configuration": (
                build_file_audit(
                    joint_config_path
                )
            ),
            "v2_feature_configuration": (
                build_file_audit(
                    feature_config_path
                )
            ),
            "baseline_private_preprocessor": (
                build_file_audit(
                    baseline_state_path
                )
            ),
            "private_v2_preprocessor_state": (
                build_file_audit(
                    private_state_path
                )
            ),
        },
    }

    validate_public_audit(audit)

    with audit_path.open(
        "w",
        encoding="utf-8",
    ) as file_handle:
        json.dump(
            audit,
            file_handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        file_handle.write("\n")

    print(
        f"fit_split="
        f"{restored_preprocessor.training_split}"
    )

    for split_name in (
        development_splits
    ):
        split_audit = split_audits[
            split_name
        ]

        unknown_total = sum(
            split_audit[
                "categorical_unknown_counts"
            ].values()
        )

        print(
            f"split={split_name} "
            f"rows={split_audit['rows']} "
            f"unknown_categories="
            f"{unknown_total}"
        )

    print(
        "baseline_preprocessing_equal=True"
    )

    for feature in (
        EXPECTED_ADDED_CATEGORICAL_FEATURES
    ):
        summary = (
            baseline_equivalence[
                "added_feature_summary"
            ][feature]
        )

        print(
            f"feature={feature} "
            f"training_cardinality="
            f"{summary['training_cardinality']} "
            f"embedding_dimension="
            f"{summary['embedding_dimension']}"
        )

    print(
        "locked_temporal_test_loaded=False"
    )
    print(
        f"private_state="
        f"{private_state_path}"
    )
    print(
        f"audit={audit_path}"
    )


if __name__ == "__main__":
    main()
