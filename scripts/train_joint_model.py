import os

os.environ.setdefault(
    "PYTORCH_ENABLE_MPS_FALLBACK",
    "1",
)

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.joint_checkpointing import (
    build_run_identity,
    load_training_checkpoint,
    save_training_checkpoint,
)
from src.joint_data import (
    build_joint_development_data,
)
from src.joint_evaluation import (
    evaluate_image_task,
    evaluate_tabular_task,
    synchronize_device,
)
from src.joint_model import (
    build_joint_cancer_model,
)
from src.joint_training import (
    JointCheckpointTracker,
    build_joint_optimizer,
    build_joint_scheduler,
    calculate_checkpoint_score,
    calculate_shared_gradient_metrics,
    clip_joint_gradients,
    compute_joint_losses,
)
from src.seer_preprocessing import (
    SeerPreprocessor,
    load_preprocessing_contract,
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

preprocessor_state_path = (
    project_root
    / "artifacts"
    / "features"
    / "seer_preprocessor.json"
)

camelyon_data_root = (
    project_root
    / "data"
    / "camelyon17_hf"
    / "data"
)

seer_model_table_path = (
    project_root
    / "data"
    / "processed"
    / "seer_model_table.parquet"
)

checkpoint_directory = (
    project_root
    / "artifacts"
    / "checkpoints"
)

metrics_directory = (
    project_root
    / "artifacts"
    / "metrics"
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Train the joint Camelyon17 and SEER "
            "hard-parameter-sharing network"
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    return parser.parse_args()


def current_utc_time():
    return (
        datetime.now(
            timezone.utc
        )
        .isoformat()
    )


def set_random_seed(
    seed,
):
    seed = int(
        seed
    )

    random.seed(
        seed
    )
    np.random.seed(
        seed
    )
    torch.manual_seed(
        seed
    )

    if (
        torch.backends.mps.is_available()
        and hasattr(
            torch.mps,
            "manual_seed",
        )
    ):
        torch.mps.manual_seed(
            seed
        )


def resolve_device(
    joint_config,
):
    priorities = joint_config[
        "training"
    ][
        "device_priority"
    ]

    for device_name in priorities:
        if (
            device_name == "mps"
            and torch.backends.mps.is_available()
        ):
            if (
                joint_config[
                    "training"
                ][
                    "mps_fallback_enabled"
                ]
                and os.environ.get(
                    "PYTORCH_ENABLE_MPS_FALLBACK"
                )
                != "1"
            ):
                raise RuntimeError(
                    "MPS fallback is required but "
                    "not enabled"
                )

            return torch.device(
                "mps"
            )

        if device_name == "cpu":
            return torch.device(
                "cpu"
            )

    raise RuntimeError(
        "No configured training device is available"
    )


def atomic_json_save(
    payload,
    output_path,
):
    output_path = Path(
        output_path
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        output_path.with_name(
            output_path.name
            + ".tmp"
        )
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file_handle:
        json.dump(
            payload,
            file_handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        file_handle.write(
            "\n"
        )

    temporary_path.replace(
        output_path
    )


def optimizer_learning_rates(
    optimizer,
):
    return {
        group[
            "name"
        ]: float(
            group[
                "lr"
            ]
        )
        for group in (
            optimizer.param_groups
        )
    }


def aggregate_gradient_records(
    records,
):
    if not records:
        return {
            "measurement_count": 0,
            "image_gradient_norm_mean": None,
            "tabular_gradient_norm_mean": None,
            "gradient_cosine_similarity_mean": None,
            "gradient_cosine_similarity_minimum": None,
            "gradient_cosine_similarity_maximum": None,
            "negative_cosine_fraction": None,
            "measurements": [],
        }

    image_norms = np.asarray(
        [
            record[
                "image_gradient_norm"
            ]
            for record in records
        ],
        dtype=np.float64,
    )

    tabular_norms = np.asarray(
        [
            record[
                "tabular_gradient_norm"
            ]
            for record in records
        ],
        dtype=np.float64,
    )

    cosine_values = np.asarray(
        [
            record[
                "gradient_cosine_similarity"
            ]
            for record in records
        ],
        dtype=np.float64,
    )

    return {
        "measurement_count": int(
            len(records)
        ),
        "image_gradient_norm_mean": float(
            image_norms.mean()
        ),
        "tabular_gradient_norm_mean": float(
            tabular_norms.mean()
        ),
        "gradient_cosine_similarity_mean": float(
            cosine_values.mean()
        ),
        "gradient_cosine_similarity_minimum": float(
            cosine_values.min()
        ),
        "gradient_cosine_similarity_maximum": float(
            cosine_values.max()
        ),
        "negative_cosine_fraction": float(
            (
                cosine_values < 0.0
            ).mean()
        ),
        "measurements": records,
    }


def train_one_epoch(
    model,
    development_data,
    optimizer,
    scheduler,
    device,
    joint_config,
    epoch,
    global_step,
    seed,
):
    epoch_seed = int(
        seed + epoch
    )

    random.seed(
        epoch_seed
    )
    np.random.seed(
        epoch_seed
    )
    torch.manual_seed(
        epoch_seed
    )

    if (
        device.type == "mps"
        and hasattr(
            torch.mps,
            "manual_seed",
        )
    ):
        torch.mps.manual_seed(
            epoch_seed
        )

    (
        image_loader,
        tabular_loader,
    ) = (
        development_data
        .build_training_loaders(
            epoch
        )
    )

    expected_steps = int(
        joint_config[
            "batching"
        ][
            "expected_steps_per_epoch"
        ]
    )

    monitoring_config = joint_config[
        "gradient_monitoring"
    ]

    monitoring_enabled = bool(
        monitoring_config[
            "enabled"
        ]
    )

    if (
        monitoring_enabled
        and not monitoring_config[
            "shared_parameters_only"
        ]
    ):
        raise ValueError(
            "Gradient monitoring must remain "
            "restricted to shared parameters"
        )

    monitoring_frequency = int(
        monitoring_config[
            "measurement_frequency_steps"
        ]
    )

    if (
        monitoring_enabled
        and monitoring_frequency <= 0
    ):
        raise ValueError(
            "Gradient-monitoring frequency must be "
            "positive"
        )

    model.train()

    image_loss_sum = torch.zeros(
        (),
        device=device,
    )
    tabular_loss_sum = torch.zeros(
        (),
        device=device,
    )
    combined_loss_sum = torch.zeros(
        (),
        device=device,
    )
    image_correct_sum = torch.zeros(
        (),
        device=device,
        dtype=torch.int64,
    )
    tabular_correct_sum = torch.zeros(
        (),
        device=device,
        dtype=torch.int64,
    )

    gradient_records = []
    clipping_norm_sum = 0.0
    clipping_norm_maximum = 0.0
    steps_completed = 0

    synchronize_device(
        device
    )
    start_time = time.perf_counter()

    for (
        step_in_epoch,
        (
            image_batch,
            tabular_batch,
        ),
    ) in enumerate(
        zip(
            image_loader,
            tabular_loader,
        ),
        start=1,
    ):
        images = image_batch[
            "image"
        ].to(
            device
        )
        image_targets = image_batch[
            "label"
        ].to(
            device=device,
            dtype=torch.float32,
        )

        numeric_inputs = tabular_batch[
            "numeric"
        ].to(
            device
        )
        categorical_inputs = tabular_batch[
            "categorical"
        ].to(
            device
        )
        tabular_targets = tabular_batch[
            "target"
        ].to(
            device=device,
            dtype=torch.float32,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        image_logits = (
            model.forward_image(
                images
            )
        )
        tabular_logits = (
            model.forward_tabular(
                numeric_inputs,
                categorical_inputs,
            )
        )

        losses = compute_joint_losses(
            image_logits=image_logits,
            image_targets=(
                image_targets
            ),
            tabular_logits=(
                tabular_logits
            ),
            tabular_targets=(
                tabular_targets
            ),
            joint_config=(
                joint_config
            ),
        )

        next_global_step = (
            global_step + 1
        )

        if (
            monitoring_enabled
            and next_global_step
            % monitoring_frequency
            == 0
        ):
            gradient_metrics = (
                calculate_shared_gradient_metrics(
                    image_loss=(
                        losses[
                            "image"
                        ]
                    ),
                    tabular_loss=(
                        losses[
                            "tabular"
                        ]
                    ),
                    shared_parameters=(
                        model
                        .shared_parameters()
                    ),
                )
            )

            gradient_records.append(
                {
                    "epoch": int(
                        epoch
                    ),
                    "step_in_epoch": int(
                        step_in_epoch
                    ),
                    "global_step": int(
                        next_global_step
                    ),
                    **gradient_metrics,
                }
            )

        losses[
            "combined"
        ].backward()

        pre_clip_norm = (
            clip_joint_gradients(
                model,
                joint_config,
            )
        )

        clipping_norm_sum += (
            pre_clip_norm
        )
        clipping_norm_maximum = max(
            clipping_norm_maximum,
            pre_clip_norm,
        )

        optimizer.step()
        scheduler.step()

        global_step = (
            next_global_step
        )
        steps_completed = (
            step_in_epoch
        )

        with torch.no_grad():
            image_loss_sum.add_(
                losses[
                    "image"
                ].detach()
            )
            tabular_loss_sum.add_(
                losses[
                    "tabular"
                ].detach()
            )
            combined_loss_sum.add_(
                losses[
                    "combined"
                ].detach()
            )

            image_correct_sum.add_(
                (
                    (
                        image_logits.detach()
                        >= 0.0
                    )
                    == (
                        image_targets
                        >= 0.5
                    )
                )
                .sum()
            )

            tabular_correct_sum.add_(
                (
                    (
                        tabular_logits.detach()
                        >= 0.0
                    )
                    == (
                        tabular_targets
                        >= 0.5
                    )
                )
                .sum()
            )

        if (
            step_in_epoch
            % monitoring_frequency
            == 0
            or step_in_epoch
            == expected_steps
        ):
            elapsed_seconds = (
                time.perf_counter()
                - start_time
            )

            print(
                f"epoch={epoch} "
                f"step={step_in_epoch}/"
                f"{expected_steps} "
                f"image_loss="
                f"{image_loss_sum.item() / step_in_epoch:.6f} "
                f"tabular_loss="
                f"{tabular_loss_sum.item() / step_in_epoch:.6f} "
                f"combined_loss="
                f"{combined_loss_sum.item() / step_in_epoch:.6f} "
                f"elapsed_seconds="
                f"{elapsed_seconds:.2f}"
            )

    synchronize_device(
        device
    )
    elapsed_seconds = (
        time.perf_counter()
        - start_time
    )

    if steps_completed != expected_steps:
        raise RuntimeError(
            "Joint epoch ended after an unexpected "
            f"number of steps: {steps_completed}"
        )

    image_batch_size = int(
        joint_config[
            "batching"
        ][
            "image_training_batch_size"
        ]
    )
    tabular_batch_size = int(
        joint_config[
            "batching"
        ][
            "tabular_training_batch_size"
        ]
    )

    image_samples = (
        steps_completed
        * image_batch_size
    )
    tabular_samples = (
        steps_completed
        * tabular_batch_size
    )

    return {
        "global_step": int(
            global_step
        ),
        "metrics": {
            "steps": int(
                steps_completed
            ),
            "image_samples": int(
                image_samples
            ),
            "tabular_samples": int(
                tabular_samples
            ),
            "image_mean_loss": float(
                image_loss_sum.item()
                / steps_completed
            ),
            "tabular_mean_loss": float(
                tabular_loss_sum.item()
                / steps_completed
            ),
            "combined_mean_loss": float(
                combined_loss_sum.item()
                / steps_completed
            ),
            "image_accuracy": float(
                image_correct_sum.item()
                / image_samples
            ),
            "tabular_accuracy": float(
                tabular_correct_sum.item()
                / tabular_samples
            ),
            "elapsed_seconds": float(
                elapsed_seconds
            ),
            "image_examples_per_second": float(
                image_samples
                / elapsed_seconds
            ),
            "tabular_examples_per_second": float(
                tabular_samples
                / elapsed_seconds
            ),
            "pre_clip_gradient_norm_mean": float(
                clipping_norm_sum
                / steps_completed
            ),
            "pre_clip_gradient_norm_maximum": float(
                clipping_norm_maximum
            ),
            "learning_rates": (
                optimizer_learning_rates(
                    optimizer
                )
            ),
            "shared_gradient_monitoring": (
                aggregate_gradient_records(
                    gradient_records
                )
            ),
        },
    }


def validate_development_tasks(
    model,
    evaluation_loaders,
    device,
    joint_config,
):
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

    image_splits = (
        image_task[
            "internal_validation_split"
        ],
        image_task[
            "selection_validation_split"
        ],
    )
    tabular_splits = (
        tabular_task[
            "internal_validation_split"
        ],
        tabular_task[
            "selection_validation_split"
        ],
    )

    if image_task[
        "locked_test_split"
    ] in evaluation_loaders[
        "image"
    ]:
        raise RuntimeError(
            "Camelyon locked test loader entered "
            "development validation"
        )

    if tabular_task[
        "locked_test_split"
    ] in evaluation_loaders[
        "tabular"
    ]:
        raise RuntimeError(
            "SEER locked test loader entered "
            "development validation"
        )

    image_results = {}

    for split_name in image_splits:
        result = evaluate_image_task(
            model=model,
            loader=(
                evaluation_loaders[
                    "image"
                ][
                    split_name
                ]
            ),
            device=device,
            joint_config=(
                joint_config
            ),
        )

        image_results[
            split_name
        ] = result

        print(
            f"validation_task=image "
            f"split={split_name} "
            f"roc_auc="
            f"{result['overall']['roc_auc']:.6f} "
            f"accuracy="
            f"{result['overall']['accuracy']:.6f}"
        )

    tabular_results = {}

    for split_name in tabular_splits:
        result = (
            evaluate_tabular_task(
                model=model,
                loader=(
                    evaluation_loaders[
                        "tabular"
                    ][
                        split_name
                    ]
                ),
                device=device,
                joint_config=(
                    joint_config
                ),
            )
        )

        tabular_results[
            split_name
        ] = result

        print(
            f"validation_task=tabular "
            f"split={split_name} "
            f"roc_auc="
            f"{result['overall']['roc_auc']:.6f} "
            f"accuracy="
            f"{result['overall']['accuracy']:.6f}"
        )

    image_selection_split = (
        image_task[
            "selection_validation_split"
        ]
    )
    tabular_selection_split = (
        tabular_task[
            "selection_validation_split"
        ]
    )

    expected_image_metric = (
        f"{image_selection_split}"
        f"_roc_auc"
    )
    expected_tabular_metric = (
        f"{tabular_selection_split}"
        f"_roc_auc"
    )

    selection_config = joint_config[
        "checkpoint_selection"
    ]

    if selection_config[
        "image_metric"
    ] != expected_image_metric:
        raise ValueError(
            "Configured image checkpoint metric "
            "does not match the selection split"
        )

    if selection_config[
        "tabular_metric"
    ] != expected_tabular_metric:
        raise ValueError(
            "Configured tabular checkpoint metric "
            "does not match the selection split"
        )

    image_selection_auc = (
        image_results[
            image_selection_split
        ][
            "overall"
        ][
            "roc_auc"
        ]
    )
    tabular_selection_auc = (
        tabular_results[
            tabular_selection_split
        ][
            "overall"
        ][
            "roc_auc"
        ]
    )

    checkpoint_score = (
        calculate_checkpoint_score(
            image_roc_auc=(
                image_selection_auc
            ),
            tabular_roc_auc=(
                tabular_selection_auc
            ),
            joint_config=(
                joint_config
            ),
        )
    )

    return {
        "image": image_results,
        "tabular": tabular_results,
        "selection": {
            "image_split": (
                image_selection_split
            ),
            "tabular_split": (
                tabular_selection_split
            ),
            "image_roc_auc": float(
                image_selection_auc
            ),
            "tabular_roc_auc": float(
                tabular_selection_auc
            ),
            "checkpoint_score": float(
                checkpoint_score
            ),
        },
        "locked_tests_evaluated": False,
    }


def build_public_summary(
    joint_config,
    run_identity,
    seed,
    device,
    data_summary,
    schedule_summary,
    history,
    tracker,
    status,
    started_at,
    best_checkpoint_path,
    latest_checkpoint_path,
):
    return {
        "version": "1.0",
        "model_name": (
            joint_config[
                "model_name"
            ]
        ),
        "seed": int(
            seed
        ),
        "device": (
            device.type
        ),
        "started_at": (
            started_at
        ),
        "updated_at": (
            current_utc_time()
        ),
        "status": status,
        "run_identity": (
            run_identity
        ),
        "data_summary": (
            data_summary
        ),
        "schedule_summary": (
            schedule_summary
        ),
        "epochs": history,
        "best": {
            "epoch": (
                tracker.best_epoch
            ),
            "checkpoint_score": (
                tracker.best_score
            ),
        },
        "checkpoint_files": {
            "best": (
                best_checkpoint_path.name
            ),
            "latest": (
                latest_checkpoint_path.name
            ),
        },
        "test_protection": {
            "camelyon_ood_test_evaluated": False,
            "seer_temporal_test_evaluated": False,
            "test_used_for_preprocessing": False,
            "test_used_for_tuning": False,
            "test_used_for_checkpoint_selection": False,
        },
    }


def main():
    arguments = parse_arguments()

    if not joint_config_path.is_file():
        raise FileNotFoundError(
            "Joint model configuration not found: "
            f"{joint_config_path}"
        )

    if not feature_config_path.is_file():
        raise FileNotFoundError(
            "SEER feature configuration not found: "
            f"{feature_config_path}"
        )

    if not preprocessor_state_path.is_file():
        raise FileNotFoundError(
            "SEER preprocessor state not found: "
            f"{preprocessor_state_path}"
        )

    (
        joint_config,
        feature_config,
    ) = load_preprocessing_contract(
        joint_config_path,
        feature_config_path,
    )

    seed = int(
        arguments.seed
    )

    if seed not in {
        int(value)
        for value in joint_config[
            "random_seeds"
        ]
    }:
        raise ValueError(
            "Seed is not in the frozen random-seed "
            f"list: {seed}"
        )

    checkpoint_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    metrics_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_checkpoint_path = (
        checkpoint_directory
        / f"joint_seed{seed}_best.pt"
    )
    latest_checkpoint_path = (
        checkpoint_directory
        / f"joint_seed{seed}_latest.pt"
    )
    history_path = (
        metrics_directory
        / f"joint_seed{seed}_history.json"
    )

    protected_existing_paths = [
        path
        for path in (
            best_checkpoint_path,
            latest_checkpoint_path,
            history_path,
        )
        if path.exists()
    ]

    if (
        protected_existing_paths
        and not arguments.resume
    ):
        raise FileExistsError(
            "Run outputs already exist. Use "
            "--resume only when continuing the "
            "matching run: "
            f"{protected_existing_paths}"
        )

    if (
        arguments.resume
        and not latest_checkpoint_path.is_file()
    ):
        raise FileNotFoundError(
            "Cannot resume because the latest "
            "checkpoint does not exist: "
            f"{latest_checkpoint_path}"
        )

    set_random_seed(
        seed
    )

    device = resolve_device(
        joint_config
    )

    with preprocessor_state_path.open(
        "r",
        encoding="utf-8",
    ) as file_handle:
        preprocessor_state = (
            json.load(
                file_handle
            )
        )

    preprocessor = (
        SeerPreprocessor.load(
            preprocessor_state_path,
            joint_config,
            feature_config,
        )
    )

    model = build_joint_cancer_model(
        joint_config,
        preprocessor_state,
    )
    model = model.to(
        device
    )

    optimizer = build_joint_optimizer(
        model,
        joint_config,
    )
    (
        scheduler,
        schedule_summary,
    ) = build_joint_scheduler(
        optimizer,
        joint_config,
    )

    tracker = JointCheckpointTracker(
        joint_config
    )

    run_identity = build_run_identity(
        joint_config_path,
        preprocessor_state_path,
        seed,
    )

    history = []
    global_step = 0
    start_epoch = 1
    started_at = (
        current_utc_time()
    )

    if arguments.resume:
        restored = (
            load_training_checkpoint(
                checkpoint_path=(
                    latest_checkpoint_path
                ),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                tracker=tracker,
                expected_run_identity=(
                    run_identity
                ),
                restore_rng=True,
            )
        )

        history = restored[
            "history"
        ]
        global_step = int(
            restored[
                "global_step"
            ]
        )
        start_epoch = (
            int(
                restored[
                    "completed_epoch"
                ]
            )
            + 1
        )

        if history_path.is_file():
            with history_path.open(
                "r",
                encoding="utf-8",
            ) as file_handle:
                previous_summary = (
                    json.load(
                        file_handle
                    )
                )

            started_at = (
                previous_summary.get(
                    "started_at",
                    started_at,
                )
            )

    development_data = (
        build_joint_development_data(
            camelyon_data_root=(
                camelyon_data_root
            ),
            seer_model_table_path=(
                seer_model_table_path
            ),
            preprocessor=(
                preprocessor
            ),
            joint_config=(
                joint_config
            ),
            seed=seed,
        )
    )

    data_summary = (
        development_data.summary()
    )
    evaluation_loaders = (
        development_data
        .build_evaluation_loaders()
    )

    expected_steps_per_epoch = int(
        joint_config[
            "batching"
        ][
            "expected_steps_per_epoch"
        ]
    )

    expected_global_step = (
        (
            start_epoch - 1
        )
        * expected_steps_per_epoch
    )

    if global_step != (
        expected_global_step
    ):
        raise RuntimeError(
            "Checkpoint global step does not match "
            "its completed epoch"
        )

    if scheduler.last_epoch != (
        global_step
    ):
        raise RuntimeError(
            "Scheduler step does not match the "
            "training global step"
        )

    maximum_epochs = int(
        joint_config[
            "training"
        ][
            "maximum_epochs"
        ]
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"device={device.type}"
    )
    print(
        f"seed={seed}"
    )
    print(
        f"resume={arguments.resume}"
    )
    print(
        f"start_epoch={start_epoch}"
    )
    print(
        f"maximum_epochs={maximum_epochs}"
    )
    print(
        f"joint_steps_per_epoch="
        f"{expected_steps_per_epoch}"
    )
    print(
        f"trainable_parameters="
        f"{trainable_parameters}"
    )
    print(
        "locked_tests_loaded=False"
    )

    if start_epoch > maximum_epochs:
        summary = build_public_summary(
            joint_config=joint_config,
            run_identity=run_identity,
            seed=seed,
            device=device,
            data_summary=data_summary,
            schedule_summary=(
                schedule_summary
            ),
            history=history,
            tracker=tracker,
            status=(
                "already_completed"
            ),
            started_at=started_at,
            best_checkpoint_path=(
                best_checkpoint_path
            ),
            latest_checkpoint_path=(
                latest_checkpoint_path
            ),
        )

        atomic_json_save(
            summary,
            history_path,
        )

        print(
            "training_status=already_completed"
        )
        return

    final_status = "running"

    for epoch in range(
        start_epoch,
        maximum_epochs + 1,
    ):
        training_result = train_one_epoch(
            model=model,
            development_data=(
                development_data
            ),
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            joint_config=joint_config,
            epoch=epoch,
            global_step=(
                global_step
            ),
            seed=seed,
        )

        global_step = int(
            training_result[
                "global_step"
            ]
        )

        validation_result = (
            validate_development_tasks(
                model=model,
                evaluation_loaders=(
                    evaluation_loaders
                ),
                device=device,
                joint_config=(
                    joint_config
                ),
            )
        )

        checkpoint_score = float(
            validation_result[
                "selection"
            ][
                "checkpoint_score"
            ]
        )

        decision = tracker.update(
            epoch=epoch,
            score=checkpoint_score,
        )

        epoch_record = {
            "epoch": int(
                epoch
            ),
            "global_step": int(
                global_step
            ),
            "training": (
                training_result[
                    "metrics"
                ]
            ),
            "validation": (
                validation_result
            ),
            "checkpoint_decision": (
                decision
            ),
        }

        history.append(
            epoch_record
        )

        if (
            decision[
                "improved"
            ]
            and joint_config[
                "training"
            ][
                "save_best_checkpoint"
            ]
        ):
            best_file = (
                save_training_checkpoint(
                    output_path=(
                        best_checkpoint_path
                    ),
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    tracker=tracker,
                    completed_epoch=epoch,
                    global_step=(
                        global_step
                    ),
                    seed=seed,
                    run_identity=(
                        run_identity
                    ),
                    history=history,
                    last_validation=(
                        validation_result
                    ),
                )
            )

            print(
                f"best_checkpoint="
                f"{best_checkpoint_path} "
                f"size_bytes="
                f"{best_file['size_bytes']}"
            )

        if joint_config[
            "training"
        ][
            "save_latest_recovery_checkpoint"
        ]:
            latest_file = (
                save_training_checkpoint(
                    output_path=(
                        latest_checkpoint_path
                    ),
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    tracker=tracker,
                    completed_epoch=epoch,
                    global_step=(
                        global_step
                    ),
                    seed=seed,
                    run_identity=(
                        run_identity
                    ),
                    history=history,
                    last_validation=(
                        validation_result
                    ),
                )
            )

            print(
                f"latest_checkpoint="
                f"{latest_checkpoint_path} "
                f"size_bytes="
                f"{latest_file['size_bytes']}"
            )

        if decision[
            "should_stop"
        ]:
            final_status = (
                "early_stopped"
            )
        elif epoch == maximum_epochs:
            final_status = (
                "maximum_epochs_completed"
            )
        else:
            final_status = "running"

        summary = build_public_summary(
            joint_config=joint_config,
            run_identity=run_identity,
            seed=seed,
            device=device,
            data_summary=data_summary,
            schedule_summary=(
                schedule_summary
            ),
            history=history,
            tracker=tracker,
            status=final_status,
            started_at=started_at,
            best_checkpoint_path=(
                best_checkpoint_path
            ),
            latest_checkpoint_path=(
                latest_checkpoint_path
            ),
        )

        atomic_json_save(
            summary,
            history_path,
        )

        print(
            f"epoch={epoch} "
            f"checkpoint_score="
            f"{checkpoint_score:.6f} "
            f"best_epoch="
            f"{tracker.best_epoch} "
            f"best_score="
            f"{tracker.best_score:.6f} "
            f"epochs_without_improvement="
            f"{tracker.epochs_without_improvement}"
        )

        if decision[
            "should_stop"
        ]:
            break

    print(
        f"training_status="
        f"{final_status}"
    )
    print(
        f"best_epoch="
        f"{tracker.best_epoch}"
    )
    print(
        f"best_checkpoint_score="
        f"{tracker.best_score:.6f}"
    )
    print(
        f"history={history_path}"
    )
    print(
        "locked_tests_evaluated=False"
    )


if __name__ == "__main__":
    main()
