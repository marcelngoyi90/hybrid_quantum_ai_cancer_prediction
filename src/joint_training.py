import math

import torch
from torch.nn import functional
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def build_joint_optimizer(
    model,
    joint_config,
):
    optimizer_config = joint_config[
        "optimizer"
    ]

    if optimizer_config[
        "name"
    ] != "adamw":
        raise ValueError(
            "Only AdamW optimization is supported"
        )

    model_groups = (
        model.optimizer_parameter_groups()
    )

    configured_learning_rates = (
        optimizer_config[
            "learning_rates"
        ]
    )

    if set(model_groups) != set(
        configured_learning_rates
    ):
        raise ValueError(
            "Optimizer learning-rate groups do not "
            "match the model parameter groups"
        )

    optimizer_groups = []

    for group_name, parameters in (
        model_groups.items()
    ):
        learning_rate = float(
            configured_learning_rates[
                group_name
            ]
        )

        if learning_rate <= 0.0:
            raise ValueError(
                "Learning rate must be positive for "
                f"{group_name}"
            )

        optimizer_groups.append(
            {
                "name": group_name,
                "params": parameters,
                "lr": learning_rate,
                "initial_lr": learning_rate,
            }
        )

    betas = tuple(
        float(value)
        for value
        in optimizer_config[
            "betas"
        ]
    )

    if (
        len(betas) != 2
        or not 0.0 <= betas[0] < 1.0
        or not 0.0 <= betas[1] < 1.0
    ):
        raise ValueError(
            "AdamW betas must contain two values "
            "in [0, 1)"
        )

    weight_decay = float(
        optimizer_config[
            "weight_decay"
        ]
    )

    if weight_decay < 0.0:
        raise ValueError(
            "Weight decay cannot be negative"
        )

    return AdamW(
        optimizer_groups,
        betas=betas,
        weight_decay=weight_decay,
    )


def learning_rate_multiplier(
    step_index,
    total_steps,
    warmup_steps,
    minimum_ratio,
):
    step_index = int(
        step_index
    )
    total_steps = int(
        total_steps
    )
    warmup_steps = int(
        warmup_steps
    )
    minimum_ratio = float(
        minimum_ratio
    )

    if step_index < 0:
        raise ValueError(
            "Step index cannot be negative"
        )

    if total_steps <= 0:
        raise ValueError(
            "Total scheduler steps must be positive"
        )

    if not 0 <= warmup_steps < total_steps:
        raise ValueError(
            "Warmup steps must be in "
            "[0, total_steps)"
        )

    if not 0.0 <= minimum_ratio <= 1.0:
        raise ValueError(
            "Minimum learning-rate ratio must be "
            "in [0, 1]"
        )

    if warmup_steps > 0 and (
        step_index < warmup_steps
    ):
        return float(
            step_index + 1
        ) / float(
            warmup_steps
        )

    decay_steps = max(
        1,
        total_steps - warmup_steps - 1,
    )

    decay_progress = (
        step_index - warmup_steps
    ) / decay_steps

    decay_progress = min(
        1.0,
        max(
            0.0,
            decay_progress,
        ),
    )

    cosine_value = 0.5 * (
        1.0
        + math.cos(
            math.pi * decay_progress
        )
    )

    return (
        minimum_ratio
        + (
            1.0 - minimum_ratio
        )
        * cosine_value
    )


def build_joint_scheduler(
    optimizer,
    joint_config,
):
    scheduler_config = joint_config[
        "scheduler"
    ]
    training_config = joint_config[
        "training"
    ]
    batching_config = joint_config[
        "batching"
    ]

    if scheduler_config[
        "name"
    ] != "linear_warmup_cosine_decay":
        raise ValueError(
            "Only linear warmup with cosine decay "
            "is supported"
        )

    if scheduler_config[
        "update_frequency"
    ] != "training_step":
        raise ValueError(
            "The scheduler must update every "
            "training step"
        )

    steps_per_epoch = int(
        batching_config[
            "expected_steps_per_epoch"
        ]
    )
    maximum_epochs = int(
        training_config[
            "maximum_epochs"
        ]
    )
    warmup_epochs = int(
        scheduler_config[
            "warmup_epochs"
        ]
    )

    total_steps = (
        steps_per_epoch
        * maximum_epochs
    )
    warmup_steps = (
        steps_per_epoch
        * warmup_epochs
    )

    if total_steps <= 0:
        raise ValueError(
            "Total training steps must be positive"
        )

    if not 0 <= warmup_steps < total_steps:
        raise ValueError(
            "Warmup duration must be shorter than "
            "the complete schedule"
        )

    minimum_ratio = float(
        scheduler_config[
            "minimum_learning_rate_ratio"
        ]
    )

    def multiplier(step_index):
        return learning_rate_multiplier(
            step_index=step_index,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            minimum_ratio=minimum_ratio,
        )

    scheduler = LambdaLR(
        optimizer,
        lr_lambda=[
            multiplier
            for _ in optimizer.param_groups
        ],
    )

    schedule_summary = {
        "steps_per_epoch": (
            steps_per_epoch
        ),
        "maximum_epochs": (
            maximum_epochs
        ),
        "total_steps": total_steps,
        "warmup_epochs": warmup_epochs,
        "warmup_steps": warmup_steps,
        "minimum_learning_rate_ratio": (
            minimum_ratio
        ),
    }

    return scheduler, schedule_summary


def compute_joint_losses(
    image_logits,
    image_targets,
    tabular_logits,
    tabular_targets,
    joint_config,
):
    loss_config = joint_config[
        "loss"
    ]

    if loss_config[
        "image"
    ] != "binary_cross_entropy_with_logits":
        raise ValueError(
            "Unexpected image loss configuration"
        )

    if loss_config[
        "tabular"
    ] != "binary_cross_entropy_with_logits":
        raise ValueError(
            "Unexpected tabular loss configuration"
        )

    if loss_config[
        "class_weighting"
    ] != "none":
        raise ValueError(
            "Class weighting is not part of the "
            "frozen joint model"
        )

    if loss_config[
        "reduction"
    ] != "mean":
        raise ValueError(
            "Joint task losses must use mean "
            "reduction"
        )

    if image_logits.ndim != 1:
        raise ValueError(
            "Image logits must be one-dimensional"
        )

    if tabular_logits.ndim != 1:
        raise ValueError(
            "Tabular logits must be "
            "one-dimensional"
        )

    if image_targets.shape != (
        image_logits.shape
    ):
        raise ValueError(
            "Image targets and logits have "
            "different shapes"
        )

    if tabular_targets.shape != (
        tabular_logits.shape
    ):
        raise ValueError(
            "Tabular targets and logits have "
            "different shapes"
        )

    image_targets = image_targets.to(
        dtype=image_logits.dtype
    )
    tabular_targets = tabular_targets.to(
        dtype=tabular_logits.dtype
    )

    image_loss = (
        functional
        .binary_cross_entropy_with_logits(
            image_logits,
            image_targets,
            reduction="mean",
        )
    )

    tabular_loss = (
        functional
        .binary_cross_entropy_with_logits(
            tabular_logits,
            tabular_targets,
            reduction="mean",
        )
    )

    image_weight = float(
        loss_config[
            "image_weight"
        ]
    )
    tabular_weight = float(
        loss_config[
            "tabular_weight"
        ]
    )

    if (
        image_weight < 0.0
        or tabular_weight < 0.0
    ):
        raise ValueError(
            "Task loss weights cannot be negative"
        )

    if not math.isclose(
        image_weight + tabular_weight,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "Task loss weights must sum to one"
        )

    combined_loss = (
        image_weight * image_loss
        + tabular_weight * tabular_loss
    )

    return {
        "image": image_loss,
        "tabular": tabular_loss,
        "combined": combined_loss,
    }


def calculate_shared_gradient_metrics(
    image_loss,
    tabular_loss,
    shared_parameters,
):
    shared_parameters = list(
        shared_parameters
    )

    if not shared_parameters:
        raise ValueError(
            "Shared parameter list is empty"
        )

    image_gradients = torch.autograd.grad(
        image_loss,
        shared_parameters,
        retain_graph=True,
        create_graph=False,
        allow_unused=False,
    )

    tabular_gradients = torch.autograd.grad(
        tabular_loss,
        shared_parameters,
        retain_graph=True,
        create_graph=False,
        allow_unused=False,
    )

    image_squared_norm = sum(
        gradient.detach().square().sum()
        for gradient in image_gradients
    )

    tabular_squared_norm = sum(
        gradient.detach().square().sum()
        for gradient in tabular_gradients
    )

    gradient_dot_product = sum(
        image_gradient.detach()
        .mul(
            tabular_gradient.detach()
        )
        .sum()
        for image_gradient, tabular_gradient
        in zip(
            image_gradients,
            tabular_gradients,
        )
    )

    image_gradient_norm = (
        image_squared_norm.sqrt()
    )
    tabular_gradient_norm = (
        tabular_squared_norm.sqrt()
    )

    denominator = (
        image_gradient_norm
        * tabular_gradient_norm
    )

    if denominator.detach().item() > 0.0:
        cosine_similarity = (
            gradient_dot_product
            / denominator
        )
    else:
        cosine_similarity = torch.zeros(
            (),
            device=gradient_dot_product.device,
            dtype=gradient_dot_product.dtype,
        )

    diagnostic_values = torch.stack(
        [
            image_gradient_norm,
            tabular_gradient_norm,
            cosine_similarity,
        ]
    )

    if not bool(
        torch.isfinite(
            diagnostic_values
        ).all()
    ):
        raise RuntimeError(
            "Non-finite shared-gradient metric "
            "detected"
        )

    return {
        "image_gradient_norm": float(
            image_gradient_norm
            .detach()
            .item()
        ),
        "tabular_gradient_norm": float(
            tabular_gradient_norm
            .detach()
            .item()
        ),
        "gradient_cosine_similarity": float(
            cosine_similarity
            .detach()
            .item()
        ),
    }


def clip_joint_gradients(
    model,
    joint_config,
):
    maximum_norm = float(
        joint_config[
            "optimizer"
        ][
            "gradient_clipping_max_norm"
        ]
    )

    if maximum_norm <= 0.0:
        raise ValueError(
            "Maximum gradient norm must be positive"
        )

    total_norm = (
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=maximum_norm,
            error_if_nonfinite=True,
        )
    )

    return float(
        total_norm.detach().item()
    )


def calculate_checkpoint_score(
    image_roc_auc,
    tabular_roc_auc,
    joint_config,
):
    selection_config = joint_config[
        "checkpoint_selection"
    ]

    if selection_config[
        "selection_rule"
    ] != "weighted_mean":
        raise ValueError(
            "Only weighted-mean checkpoint "
            "selection is supported"
        )

    image_roc_auc = float(
        image_roc_auc
    )
    tabular_roc_auc = float(
        tabular_roc_auc
    )

    if not math.isfinite(
        image_roc_auc
    ):
        raise ValueError(
            "Image ROC-AUC must be finite"
        )

    if not math.isfinite(
        tabular_roc_auc
    ):
        raise ValueError(
            "Tabular ROC-AUC must be finite"
        )

    if not 0.0 <= image_roc_auc <= 1.0:
        raise ValueError(
            "Image ROC-AUC must be in [0, 1]"
        )

    if not 0.0 <= tabular_roc_auc <= 1.0:
        raise ValueError(
            "Tabular ROC-AUC must be in [0, 1]"
        )

    image_weight = float(
        selection_config[
            "image_metric_weight"
        ]
    )
    tabular_weight = float(
        selection_config[
            "tabular_metric_weight"
        ]
    )

    if not math.isclose(
        image_weight + tabular_weight,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "Checkpoint metric weights must sum "
            "to one"
        )

    return (
        image_weight * image_roc_auc
        + tabular_weight * tabular_roc_auc
    )


class JointCheckpointTracker:
    def __init__(
        self,
        joint_config,
    ):
        training_config = joint_config[
            "training"
        ]
        selection_config = joint_config[
            "checkpoint_selection"
        ]

        self.minimum_epochs = int(
            training_config[
                "minimum_epochs_before_early_stopping"
            ]
        )
        self.patience = int(
            training_config[
                "early_stopping_patience"
            ]
        )
        self.minimum_improvement = float(
            training_config[
                "early_stopping_minimum_improvement"
            ]
        )
        self.tie_tolerance = float(
            selection_config[
                "tie_tolerance"
            ]
        )

        if self.minimum_epochs <= 0:
            raise ValueError(
                "Minimum epochs must be positive"
            )

        if self.patience <= 0:
            raise ValueError(
                "Early-stopping patience must be "
                "positive"
            )

        if self.minimum_improvement < 0.0:
            raise ValueError(
                "Minimum improvement cannot be "
                "negative"
            )

        if self.tie_tolerance < 0.0:
            raise ValueError(
                "Tie tolerance cannot be negative"
            )

        self.required_improvement = max(
            self.minimum_improvement,
            self.tie_tolerance,
        )

        self.best_score = None
        self.best_epoch = None
        self.epochs_without_improvement = 0

    def update(
        self,
        epoch,
        score,
    ):
        epoch = int(
            epoch
        )
        score = float(
            score
        )

        if epoch <= 0:
            raise ValueError(
                "Epoch numbers must begin at one"
            )

        if not math.isfinite(score):
            raise ValueError(
                "Checkpoint score must be finite"
            )

        improved = (
            self.best_score is None
            or score
            > (
                self.best_score
                + self.required_improvement
            )
        )

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        should_stop = (
            epoch >= self.minimum_epochs
            and self.epochs_without_improvement
            >= self.patience
        )

        return {
            "improved": improved,
            "should_stop": should_stop,
            "best_score": (
                self.best_score
            ),
            "best_epoch": (
                self.best_epoch
            ),
            "epochs_without_improvement": (
                self.epochs_without_improvement
            ),
        }

    def state_dict(
        self,
    ):
        return {
            "minimum_epochs": (
                self.minimum_epochs
            ),
            "patience": self.patience,
            "minimum_improvement": (
                self.minimum_improvement
            ),
            "tie_tolerance": (
                self.tie_tolerance
            ),
            "required_improvement": (
                self.required_improvement
            ),
            "best_score": (
                self.best_score
            ),
            "best_epoch": (
                self.best_epoch
            ),
            "epochs_without_improvement": (
                self.epochs_without_improvement
            ),
        }

    def load_state_dict(
        self,
        state,
    ):
        fixed_fields = {
            "minimum_epochs": (
                self.minimum_epochs
            ),
            "patience": self.patience,
            "minimum_improvement": (
                self.minimum_improvement
            ),
            "tie_tolerance": (
                self.tie_tolerance
            ),
            "required_improvement": (
                self.required_improvement
            ),
        }

        for field, expected_value in (
            fixed_fields.items()
        ):
            if state[field] != expected_value:
                raise ValueError(
                    "Checkpoint tracker configuration "
                    f"changed for {field}"
                )

        best_score = state[
            "best_score"
        ]
        best_epoch = state[
            "best_epoch"
        ]

        self.best_score = (
            None
            if best_score is None
            else float(best_score)
        )
        self.best_epoch = (
            None
            if best_epoch is None
            else int(best_epoch)
        )
        self.epochs_without_improvement = int(
            state[
                "epochs_without_improvement"
            ]
        )
