import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch


CHECKPOINT_STATE_VERSION = "1.0"


def calculate_sha256(
    path,
):
    path = Path(
        path
    )
    digest = hashlib.sha256()

    with path.open(
        "rb",
    ) as file_handle:
        while True:
            chunk = file_handle.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def calculate_json_sha256(
    path,
):
    """Hash JSON text consistently across Windows and Unix checkouts."""
    path = Path(
        path
    )
    text = path.read_text(
        encoding="utf-8",
    )
    normalized = text.replace(
        "\r\n",
        "\n",
    ).replace(
        "\r",
        "\n",
    )
    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


def build_run_identity(
    joint_config_path,
    preprocessor_state_path,
    seed,
):
    joint_config_path = Path(
        joint_config_path
    )
    preprocessor_state_path = Path(
        preprocessor_state_path
    )

    if not joint_config_path.is_file():
        raise FileNotFoundError(
            "Joint configuration not found: "
            f"{joint_config_path}"
        )

    if not preprocessor_state_path.is_file():
        raise FileNotFoundError(
            "SEER preprocessor state not found: "
            f"{preprocessor_state_path}"
        )

    with joint_config_path.open(
        "r",
        encoding="utf-8",
    ) as file_handle:
        joint_config = json.load(
            file_handle
        )

    return {
        "model_name": (
            joint_config[
                "model_name"
            ]
        ),
        "joint_config_version": (
            joint_config[
                "version"
            ]
        ),
        "joint_config_sha256": (
            calculate_json_sha256(
                joint_config_path
            )
        ),
        "preprocessor_state_sha256": (
            calculate_json_sha256(
                preprocessor_state_path
            )
        ),
        "seed": int(
            seed
        ),
    }


def capture_random_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": (
            torch.random.get_rng_state()
        ),
    }

    if (
        torch.backends.mps.is_available()
        and hasattr(
            torch.mps,
            "get_rng_state",
        )
    ):
        state[
            "torch_mps"
        ] = torch.mps.get_rng_state()

    return state


def restore_random_state(
    state,
):
    random.setstate(
        state["python"]
    )
    np.random.set_state(
        state["numpy"]
    )
    torch.random.set_rng_state(
        state["torch_cpu"]
    )

    if (
        "torch_mps" in state
        and torch.backends.mps.is_available()
        and hasattr(
            torch.mps,
            "set_rng_state",
        )
    ):
        torch.mps.set_rng_state(
            state[
                "torch_mps"
            ]
        )


def tensors_to_cpu(
    value,
):
    if torch.is_tensor(
        value
    ):
        return (
            value
            .detach()
            .cpu()
        )

    if isinstance(
        value,
        dict,
    ):
        return {
            key: tensors_to_cpu(
                item
            )
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        list,
    ):
        return [
            tensors_to_cpu(
                item
            )
            for item in value
        ]

    if isinstance(
        value,
        tuple,
    ):
        return tuple(
            tensors_to_cpu(
                item
            )
            for item in value
        )

    return value


def atomic_torch_save(
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

    torch.save(
        tensors_to_cpu(
            payload
        ),
        temporary_path,
    )

    temporary_path.replace(
        output_path
    )

    return {
        "file_name": (
            output_path.name
        ),
        "size_bytes": int(
            output_path.stat().st_size
        ),
        "sha256": (
            calculate_sha256(
                output_path
            )
        ),
    }


def save_training_checkpoint(
    output_path,
    model,
    optimizer,
    scheduler,
    tracker,
    completed_epoch,
    global_step,
    seed,
    run_identity,
    history,
    last_validation,
):
    completed_epoch = int(
        completed_epoch
    )
    global_step = int(
        global_step
    )
    seed = int(
        seed
    )

    if completed_epoch <= 0:
        raise ValueError(
            "Completed epoch must be positive"
        )

    if global_step <= 0:
        raise ValueError(
            "Global step must be positive"
        )

    if seed != int(
        run_identity[
            "seed"
        ]
    ):
        raise ValueError(
            "Checkpoint seed does not match the "
            "run identity"
        )

    if not isinstance(
        history,
        list,
    ):
        raise TypeError(
            "Training history must be a list"
        )

    if not isinstance(
        last_validation,
        dict,
    ):
        raise TypeError(
            "Last validation must be a dictionary"
        )

    payload = {
        "state_version": (
            CHECKPOINT_STATE_VERSION
        ),
        "completed_epoch": (
            completed_epoch
        ),
        "global_step": (
            global_step
        ),
        "seed": seed,
        "run_identity": dict(
            run_identity
        ),
        "model_state": (
            model.state_dict()
        ),
        "optimizer_state": (
            optimizer.state_dict()
        ),
        "scheduler_state": (
            scheduler.state_dict()
        ),
        "tracker_state": (
            tracker.state_dict()
        ),
        "random_state": (
            capture_random_state()
        ),
        "history": history,
        "last_validation": (
            last_validation
        ),
    }

    return atomic_torch_save(
        payload,
        output_path,
    )


def move_optimizer_state_to_device(
    optimizer,
):
    for parameter, state in (
        optimizer.state.items()
    ):
        parameter_device = (
            parameter.device
        )

        for key, value in (
            state.items()
        ):
            if (
                torch.is_tensor(
                    value
                )
                and key != "step"
            ):
                state[key] = value.to(
                    parameter_device
                )


def validate_checkpoint_payload(
    checkpoint,
    expected_run_identity,
):
    required_keys = {
        "state_version",
        "completed_epoch",
        "global_step",
        "seed",
        "run_identity",
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "tracker_state",
        "random_state",
        "history",
        "last_validation",
    }

    missing_keys = sorted(
        required_keys
        - set(checkpoint)
    )

    if missing_keys:
        raise ValueError(
            "Checkpoint is missing required keys: "
            f"{missing_keys}"
        )

    if checkpoint[
        "state_version"
    ] != CHECKPOINT_STATE_VERSION:
        raise ValueError(
            "Unsupported checkpoint state version"
        )

    if checkpoint[
        "run_identity"
    ] != expected_run_identity:
        raise ValueError(
            "Checkpoint identity does not match the "
            "current configuration, preprocessing "
            "state, and seed"
        )

    if int(
        checkpoint[
            "seed"
        ]
    ) != int(
        expected_run_identity[
            "seed"
        ]
    ):
        raise ValueError(
            "Checkpoint seed does not match the "
            "expected seed"
        )

    if int(
        checkpoint[
            "completed_epoch"
        ]
    ) <= 0:
        raise ValueError(
            "Checkpoint epoch must be positive"
        )

    if int(
        checkpoint[
            "global_step"
        ]
    ) <= 0:
        raise ValueError(
            "Checkpoint global step must be positive"
        )

    if not isinstance(
        checkpoint[
            "history"
        ],
        list,
    ):
        raise TypeError(
            "Checkpoint history must be a list"
        )

    if not isinstance(
        checkpoint[
            "last_validation"
        ],
        dict,
    ):
        raise TypeError(
            "Checkpoint validation must be a "
            "dictionary"
        )


def load_training_checkpoint(
    checkpoint_path,
    model,
    optimizer,
    scheduler,
    tracker,
    expected_run_identity,
    restore_rng=True,
):
    checkpoint_path = Path(
        checkpoint_path
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "Training checkpoint not found: "
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    validate_checkpoint_payload(
        checkpoint,
        expected_run_identity,
    )

    model.load_state_dict(
        checkpoint[
            "model_state"
        ],
        strict=True,
    )

    optimizer.load_state_dict(
        checkpoint[
            "optimizer_state"
        ]
    )

    move_optimizer_state_to_device(
        optimizer
    )

    scheduler.load_state_dict(
        checkpoint[
            "scheduler_state"
        ]
    )

    tracker.load_state_dict(
        checkpoint[
            "tracker_state"
        ]
    )

    if restore_rng:
        restore_random_state(
            checkpoint[
                "random_state"
            ]
        )

    return {
        "completed_epoch": int(
            checkpoint[
                "completed_epoch"
            ]
        ),
        "global_step": int(
            checkpoint[
                "global_step"
            ]
        ),
        "seed": int(
            checkpoint[
                "seed"
            ]
        ),
        "history": (
            checkpoint[
                "history"
            ]
        ),
        "last_validation": (
            checkpoint[
                "last_validation"
            ]
        ),
    }
