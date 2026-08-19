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
    / "seer_model_table.parquet"
)

joint_config_path = (
    project_root
    / "configs"
    / "joint_model.json"
)

feature_config_path = (
    project_root
    / "configs"
    / "seer_features.json"
)

private_state_directory = (
    project_root
    / "artifacts"
    / "features"
)

private_state_path = (
    private_state_directory
    / "seer_preprocessor.json"
)

metrics_directory = (
    project_root
    / "artifacts"
    / "metrics"
)

audit_path = (
    metrics_directory
    / "seer_preprocessing_audit.json"
)


EXPECTED_SPLIT_ROWS = {
    "train": 461292,
    "id_val": 51241,
    "temporal_val": 107249,
}


def calculate_sha256(path):
    digest = hashlib.sha256()

    with Path(path).open(
        "rb",
    ) as file_handle:
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
        "sha256": calculate_sha256(
            path
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
    if not model_table_path.is_file():
        raise FileNotFoundError(
            "SEER model table not found: "
            f"{model_table_path}"
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
            "Locked temporal test was included in "
            "development preprocessing"
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
            "Unexpected SEER training row count: "
            f"{len(train_frame)}"
        )

    preprocessor.fit(
        train_frame
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
            "Saved SEER preprocessor state changed "
            "during round-trip validation"
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
            preprocessor.transform(
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
        preprocessor.categorical_features
    ):
        categorical_summary[
            feature
        ] = {
            "training_cardinality": int(
                len(
                    preprocessor
                    .category_vocabularies[
                        feature
                    ]
                )
            ),
            "num_embeddings": int(
                preprocessor.num_embeddings[
                    feature
                ]
            ),
            "embedding_dimension": int(
                preprocessor
                .embedding_dimensions[
                    feature
                ]
            ),
        }

    audit = {
        "version": "1.0",
        "fit_split": (
            preprocessor.training_split
        ),
        "target": (
            preprocessor.target_column
        ),
        "numeric_features": (
            preprocessor.numeric_features
        ),
        "categorical_features": (
            preprocessor
            .categorical_features
        ),
        "numeric_statistics": (
            preprocessor
            .numeric_statistics
        ),
        "categorical_summary": (
            categorical_summary
        ),
        "development_split_audits": (
            split_audits
        ),
        "test_protection": {
            "locked_split": (
                preprocessor
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
            "category_vocabulary_values_public": False,
        },
        "files": {
            "private_model_table": (
                build_file_audit(
                    model_table_path
                )
            ),
            "joint_configuration": (
                build_file_audit(
                    joint_config_path
                )
            ),
            "feature_configuration": (
                build_file_audit(
                    feature_config_path
                )
            ),
            "private_preprocessor_state": (
                build_file_audit(
                    private_state_path
                )
            ),
        },
    }

    validate_public_audit(
        audit
    )

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
        f"fit_split="
        f"{preprocessor.training_split}"
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
