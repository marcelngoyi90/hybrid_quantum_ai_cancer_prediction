from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.camelyon17_data import (
    Camelyon17ParquetDataset,
    build_transform,
)
from src.seer_preprocessing import (
    read_seer_split,
)


class JointDevelopmentData:
    def __init__(
        self,
        image_train_dataset,
        image_evaluation_datasets,
        tabular_train_dataset,
        tabular_evaluation_datasets,
        tabular_split_audits,
        joint_config,
        seed,
    ):
        self.image_train_dataset = (
            image_train_dataset
        )
        self.image_evaluation_datasets = (
            image_evaluation_datasets
        )
        self.tabular_train_dataset = (
            tabular_train_dataset
        )
        self.tabular_evaluation_datasets = (
            tabular_evaluation_datasets
        )
        self.tabular_split_audits = (
            tabular_split_audits
        )
        self.joint_config = joint_config
        self.seed = int(
            seed
        )

        self.batching_config = (
            joint_config[
                "batching"
            ]
        )

        self.image_task = joint_config[
            "tasks"
        ][
            "image"
        ]
        self.tabular_task = joint_config[
            "tasks"
        ][
            "tabular"
        ]

        self._validate_development_splits()
        self._validate_training_lengths()

    def _validate_development_splits(
        self,
    ):
        loaded_image_splits = {
            self.image_task[
                "training_split"
            ],
            *self.image_evaluation_datasets.keys(),
        }

        loaded_tabular_splits = {
            self.tabular_task[
                "training_split"
            ],
            *self.tabular_evaluation_datasets.keys(),
        }

        if self.image_task[
            "locked_test_split"
        ] in loaded_image_splits:
            raise RuntimeError(
                "Camelyon OOD test was loaded during "
                "development"
            )

        if self.tabular_task[
            "locked_test_split"
        ] in loaded_tabular_splits:
            raise RuntimeError(
                "SEER temporal test was loaded during "
                "development"
            )

        expected_image_evaluation_splits = {
            self.image_task[
                "internal_validation_split"
            ],
            self.image_task[
                "selection_validation_split"
            ],
        }

        if set(
            self.image_evaluation_datasets
        ) != expected_image_evaluation_splits:
            raise ValueError(
                "Unexpected Camelyon development "
                "evaluation splits"
            )

        expected_tabular_evaluation_splits = {
            self.tabular_task[
                "internal_validation_split"
            ],
            self.tabular_task[
                "selection_validation_split"
            ],
        }

        if set(
            self.tabular_evaluation_datasets
        ) != expected_tabular_evaluation_splits:
            raise ValueError(
                "Unexpected SEER development "
                "evaluation splits"
            )

    def _training_loader_lengths(
        self,
    ):
        image_batch_size = int(
            self.batching_config[
                "image_training_batch_size"
            ]
        )
        tabular_batch_size = int(
            self.batching_config[
                "tabular_training_batch_size"
            ]
        )

        image_batches = (
            len(
                self.image_train_dataset
            )
            // image_batch_size
        )
        tabular_batches = (
            len(
                self.tabular_train_dataset
            )
            // tabular_batch_size
        )

        return (
            image_batches,
            tabular_batches,
        )

    def _validate_training_lengths(
        self,
    ):
        (
            image_batches,
            tabular_batches,
        ) = self._training_loader_lengths()

        expected_steps = int(
            self.batching_config[
                "expected_steps_per_epoch"
            ]
        )

        observed_steps = min(
            image_batches,
            tabular_batches,
        )

        if observed_steps != expected_steps:
            raise ValueError(
                "Observed joint steps do not match "
                f"the configuration: "
                f"{observed_steps} versus "
                f"{expected_steps}"
            )

        expected_image_samples = int(
            self.batching_config[
                "expected_image_samples_per_epoch"
            ]
        )
        expected_tabular_samples = int(
            self.batching_config[
                "expected_tabular_samples_per_epoch"
            ]
        )

        image_batch_size = int(
            self.batching_config[
                "image_training_batch_size"
            ]
        )
        tabular_batch_size = int(
            self.batching_config[
                "tabular_training_batch_size"
            ]
        )

        if (
            observed_steps
            * image_batch_size
            != expected_image_samples
        ):
            raise ValueError(
                "Configured image samples per epoch "
                "are inconsistent"
            )

        if (
            observed_steps
            * tabular_batch_size
            != expected_tabular_samples
        ):
            raise ValueError(
                "Configured tabular samples per epoch "
                "are inconsistent"
            )

    def build_training_loaders(
        self,
        epoch,
    ):
        epoch = int(
            epoch
        )

        if epoch <= 0:
            raise ValueError(
                "Epoch numbers must begin at one"
            )

        self.image_train_dataset.set_epoch(
            epoch
        )

        tabular_generator = (
            torch.Generator()
        )
        tabular_generator.manual_seed(
            self.seed + epoch
        )

        image_loader = DataLoader(
            self.image_train_dataset,
            batch_size=int(
                self.batching_config[
                    "image_training_batch_size"
                ]
            ),
            num_workers=int(
                self.batching_config[
                    "num_workers"
                ]
            ),
            drop_last=bool(
                self.batching_config[
                    "drop_last_training_batch"
                ]
            ),
            pin_memory=False,
        )

        tabular_loader = DataLoader(
            self.tabular_train_dataset,
            batch_size=int(
                self.batching_config[
                    "tabular_training_batch_size"
                ]
            ),
            shuffle=bool(
                self.batching_config[
                    "shuffle_each_epoch"
                ]
            ),
            generator=tabular_generator,
            num_workers=int(
                self.batching_config[
                    "num_workers"
                ]
            ),
            drop_last=bool(
                self.batching_config[
                    "drop_last_training_batch"
                ]
            ),
            pin_memory=False,
        )

        expected_steps = int(
            self.batching_config[
                "expected_steps_per_epoch"
            ]
        )

        if min(
            len(image_loader),
            len(tabular_loader),
        ) != expected_steps:
            raise RuntimeError(
                "Constructed training loaders have "
                "an unexpected length"
            )

        return (
            image_loader,
            tabular_loader,
        )

    def build_evaluation_loaders(
        self,
    ):
        number_of_workers = int(
            self.batching_config[
                "num_workers"
            ]
        )

        image_batch_size = int(
            self.batching_config[
                "image_evaluation_batch_size"
            ]
        )
        tabular_batch_size = int(
            self.batching_config[
                "tabular_evaluation_batch_size"
            ]
        )

        image_loaders = {
            split_name: DataLoader(
                dataset,
                batch_size=image_batch_size,
                num_workers=(
                    number_of_workers
                ),
                drop_last=False,
                pin_memory=False,
            )
            for split_name, dataset
            in self.image_evaluation_datasets.items()
        }

        tabular_loaders = {
            split_name: DataLoader(
                dataset,
                batch_size=(
                    tabular_batch_size
                ),
                shuffle=False,
                num_workers=(
                    number_of_workers
                ),
                drop_last=False,
                pin_memory=False,
            )
            for split_name, dataset
            in self.tabular_evaluation_datasets.items()
        }

        return {
            "image": image_loaders,
            "tabular": tabular_loaders,
        }

    def summary(
        self,
    ):
        (
            image_batches,
            tabular_batches,
        ) = self._training_loader_lengths()

        return {
            "seed": self.seed,
            "image_training_rows": int(
                len(
                    self.image_train_dataset
                )
            ),
            "tabular_training_rows": int(
                len(
                    self.tabular_train_dataset
                )
            ),
            "image_full_training_batches": int(
                image_batches
            ),
            "tabular_full_training_batches": int(
                tabular_batches
            ),
            "joint_steps_per_epoch": int(
                min(
                    image_batches,
                    tabular_batches,
                )
            ),
            "image_evaluation_rows": {
                split_name: int(
                    len(dataset)
                )
                for split_name, dataset
                in self
                .image_evaluation_datasets
                .items()
            },
            "tabular_evaluation_rows": {
                split_name: int(
                    len(dataset)
                )
                for split_name, dataset
                in self
                .tabular_evaluation_datasets
                .items()
            },
            "locked_tests_loaded": False,
            "seer_patient_identifiers_loaded": False,
            "camelyon_metadata_retained_in_batches": True,
            "camelyon_metadata_used_as_model_input": False,
        }


def build_joint_development_data(
    camelyon_data_root,
    seer_model_table_path,
    preprocessor,
    joint_config,
    seed,
):
    camelyon_data_root = Path(
        camelyon_data_root
    )
    seer_model_table_path = Path(
        seer_model_table_path
    )

    if not camelyon_data_root.is_dir():
        raise FileNotFoundError(
            "Camelyon data directory not found: "
            f"{camelyon_data_root}"
        )

    if not seer_model_table_path.is_file():
        raise FileNotFoundError(
            "SEER model table not found: "
            f"{seer_model_table_path}"
        )

    image_task = joint_config[
        "tasks"
    ][
        "image"
    ]
    tabular_task = joint_config[
        "tasks"
    ][
        "tabular"
    ]

    image_training_split = (
        image_task[
            "training_split"
        ]
    )
    image_internal_validation_split = (
        image_task[
            "internal_validation_split"
        ]
    )
    image_selection_validation_split = (
        image_task[
            "selection_validation_split"
        ]
    )

    if image_task[
        "locked_test_split"
    ] in {
        image_training_split,
        image_internal_validation_split,
        image_selection_validation_split,
    }:
        raise RuntimeError(
            "Camelyon locked test split entered "
            "development data construction"
        )

    image_train_dataset = (
        Camelyon17ParquetDataset(
            data_root=(
                camelyon_data_root
            ),
            split=(
                image_training_split
            ),
            transform=build_transform(
                training=True
            ),
            shuffle=True,
            seed=int(seed),
        )
    )

    image_evaluation_datasets = {
        image_internal_validation_split: (
            Camelyon17ParquetDataset(
                data_root=(
                    camelyon_data_root
                ),
                split=(
                    image_internal_validation_split
                ),
                transform=build_transform(
                    training=False
                ),
                shuffle=False,
                seed=int(seed),
            )
        ),
        image_selection_validation_split: (
            Camelyon17ParquetDataset(
                data_root=(
                    camelyon_data_root
                ),
                split=(
                    image_selection_validation_split
                ),
                transform=build_transform(
                    training=False
                ),
                shuffle=False,
                seed=int(seed),
            )
        ),
    }

    tabular_training_split = (
        tabular_task[
            "training_split"
        ]
    )
    tabular_internal_validation_split = (
        tabular_task[
            "internal_validation_split"
        ]
    )
    tabular_selection_validation_split = (
        tabular_task[
            "selection_validation_split"
        ]
    )

    if tabular_task[
        "locked_test_split"
    ] in {
        tabular_training_split,
        tabular_internal_validation_split,
        tabular_selection_validation_split,
    }:
        raise RuntimeError(
            "SEER locked test split entered "
            "development data construction"
        )

    tabular_datasets = {}
    tabular_split_audits = {}

    for split_name in (
        tabular_training_split,
        tabular_internal_validation_split,
        tabular_selection_validation_split,
    ):
        frame = read_seer_split(
            seer_model_table_path,
            split_name,
            preprocessor,
        )

        dataset, audit = (
            preprocessor.transform(
                frame,
                split_name,
            )
        )

        tabular_datasets[
            split_name
        ] = dataset
        tabular_split_audits[
            split_name
        ] = audit

        del frame

    tabular_train_dataset = (
        tabular_datasets.pop(
            tabular_training_split
        )
    )

    return JointDevelopmentData(
        image_train_dataset=(
            image_train_dataset
        ),
        image_evaluation_datasets=(
            image_evaluation_datasets
        ),
        tabular_train_dataset=(
            tabular_train_dataset
        ),
        tabular_evaluation_datasets=(
            tabular_datasets
        ),
        tabular_split_audits=(
            tabular_split_audits
        ),
        joint_config=joint_config,
        seed=seed,
    )
