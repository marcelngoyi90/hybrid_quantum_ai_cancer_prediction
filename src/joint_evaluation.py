import time

import numpy as np
import torch
from scipy.optimize import minimize
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.nn import functional


def synchronize_device(device):
    if device.type == "mps":
        torch.mps.synchronize()

    if device.type == "cuda":
        torch.cuda.synchronize(
            device
        )


def validate_binary_inputs(
    labels,
    probabilities,
):
    labels = np.asarray(
        labels,
        dtype=np.int64,
    )
    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    if labels.ndim != 1:
        raise ValueError(
            "Labels must be one-dimensional"
        )

    if probabilities.ndim != 1:
        raise ValueError(
            "Probabilities must be "
            "one-dimensional"
        )

    if len(labels) != len(
        probabilities
    ):
        raise ValueError(
            "Labels and probabilities must have "
            "equal lengths"
        )

    if len(labels) == 0:
        raise ValueError(
            "Cannot evaluate an empty dataset"
        )

    if not np.isfinite(
        probabilities
    ).all():
        raise ValueError(
            "Probabilities contain non-finite values"
        )

    if (
        probabilities.min() < 0.0
        or probabilities.max() > 1.0
    ):
        raise ValueError(
            "Probabilities must be in [0, 1]"
        )

    unique_labels = set(
        np.unique(labels).tolist()
    )

    if not unique_labels.issubset(
        {0, 1}
    ):
        raise ValueError(
            "Labels must contain only 0 and 1"
        )

    return labels, probabilities


def expected_calibration_error(
    labels,
    probabilities,
    number_of_bins,
):
    labels, probabilities = (
        validate_binary_inputs(
            labels,
            probabilities,
        )
    )

    number_of_bins = int(
        number_of_bins
    )

    if number_of_bins <= 0:
        raise ValueError(
            "Calibration bins must be positive"
        )

    number_of_bins = min(
        number_of_bins,
        len(labels),
    )

    sorted_indices = np.argsort(
        probabilities,
        kind="stable",
    )

    bin_indices = np.array_split(
        sorted_indices,
        number_of_bins,
    )

    calibration_error = 0.0

    for indices in bin_indices:
        if len(indices) == 0:
            continue

        bin_confidence = float(
            probabilities[
                indices
            ].mean()
        )
        bin_frequency = float(
            labels[
                indices
            ].mean()
        )
        bin_weight = (
            len(indices)
            / len(labels)
        )

        calibration_error += (
            bin_weight
            * abs(
                bin_frequency
                - bin_confidence
            )
        )

    return float(
        calibration_error
    )


def calibration_intercept_and_slope(
    labels,
    probabilities,
    probability_epsilon=1e-6,
):
    labels, probabilities = (
        validate_binary_inputs(
            labels,
            probabilities,
        )
    )

    if len(np.unique(labels)) < 2:
        return {
            "calibration_intercept": None,
            "calibration_slope": None,
            "calibration_converged": False,
        }

    clipped_probabilities = np.clip(
        probabilities,
        probability_epsilon,
        1.0 - probability_epsilon,
    )

    prediction_logits = np.log(
        clipped_probabilities
        / (
            1.0
            - clipped_probabilities
        )
    )

    if np.std(
        prediction_logits
    ) <= 1e-12:
        return {
            "calibration_intercept": None,
            "calibration_slope": None,
            "calibration_converged": False,
        }

    def objective(parameters):
        intercept, slope = parameters

        linear_predictor = (
            intercept
            + slope
            * prediction_logits
        )

        negative_log_likelihood = (
            np.logaddexp(
                0.0,
                linear_predictor,
            )
            - labels
            * linear_predictor
        ).mean()

        fitted_probabilities = (
            1.0
            / (
                1.0
                + np.exp(
                    -np.clip(
                        linear_predictor,
                        -40.0,
                        40.0,
                    )
                )
            )
        )

        residuals = (
            fitted_probabilities
            - labels
        )

        gradient = np.array(
            [
                residuals.mean(),
                (
                    residuals
                    * prediction_logits
                ).mean(),
            ],
            dtype=np.float64,
        )

        return (
            float(
                negative_log_likelihood
            ),
            gradient,
        )

    result = minimize(
        fun=objective,
        x0=np.array(
            [0.0, 1.0],
            dtype=np.float64,
        ),
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": 200,
            "ftol": 1e-12,
            "gtol": 1e-8,
        },
    )

    if (
        not result.success
        or not np.isfinite(
            result.x
        ).all()
    ):
        return {
            "calibration_intercept": None,
            "calibration_slope": None,
            "calibration_converged": False,
        }

    return {
        "calibration_intercept": float(
            result.x[0]
        ),
        "calibration_slope": float(
            result.x[1]
        ),
        "calibration_converged": True,
    }


def binary_classification_metrics(
    labels,
    probabilities,
    threshold,
    calibration_bins,
):
    labels, probabilities = (
        validate_binary_inputs(
            labels,
            probabilities,
        )
    )

    threshold = float(
        threshold
    )

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "Classification threshold must be "
            "in [0, 1]"
        )

    predictions = (
        probabilities >= threshold
    ).astype(
        np.int64
    )

    (
        true_negative,
        false_positive,
        false_negative,
        true_positive,
    ) = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    sensitivity_denominator = (
        true_positive
        + false_negative
    )
    specificity_denominator = (
        true_negative
        + false_positive
    )

    sensitivity = (
        true_positive
        / sensitivity_denominator
        if sensitivity_denominator
        else 0.0
    )

    specificity = (
        true_negative
        / specificity_denominator
        if specificity_denominator
        else 0.0
    )

    both_classes_present = (
        len(
            np.unique(labels)
        )
        == 2
    )

    if both_classes_present:
        roc_auc = float(
            roc_auc_score(
                labels,
                probabilities,
            )
        )
        average_precision = float(
            average_precision_score(
                labels,
                probabilities,
            )
        )
        balanced_accuracy = float(
            balanced_accuracy_score(
                labels,
                predictions,
            )
        )
        calibration = (
            calibration_intercept_and_slope(
                labels,
                probabilities,
            )
        )
    else:
        roc_auc = None
        average_precision = None
        balanced_accuracy = None
        calibration = {
            "calibration_intercept": None,
            "calibration_slope": None,
            "calibration_converged": False,
        }

    metrics = {
        "samples": int(
            len(labels)
        ),
        "positive_count": int(
            labels.sum()
        ),
        "positive_percent": float(
            100.0
            * labels.mean()
        ),
        "threshold": threshold,
        "roc_auc": roc_auc,
        "average_precision": (
            average_precision
        ),
        "balanced_accuracy": (
            balanced_accuracy
        ),
        "accuracy": float(
            accuracy_score(
                labels,
                predictions,
            )
        ),
        "f1": float(
            f1_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "precision": float(
            precision_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "sensitivity": float(
            sensitivity
        ),
        "specificity": float(
            specificity
        ),
        "brier_score": float(
            brier_score_loss(
                labels,
                probabilities,
            )
        ),
        "expected_calibration_error": (
            expected_calibration_error(
                labels,
                probabilities,
                calibration_bins,
            )
        ),
        "confusion_matrix": {
            "true_negative": int(
                true_negative
            ),
            "false_positive": int(
                false_positive
            ),
            "false_negative": int(
                false_negative
            ),
            "true_positive": int(
                true_positive
            ),
        },
    }

    metrics.update(
        calibration
    )

    return metrics


def evaluation_settings(
    joint_config,
):
    evaluation_config = joint_config[
        "evaluation"
    ]

    if evaluation_config[
        "primary_metric"
    ] != "roc_auc":
        raise ValueError(
            "ROC-AUC must remain the primary metric"
        )

    if evaluation_config[
        "calibration_binning"
    ] != "equal_frequency":
        raise ValueError(
            "Only equal-frequency calibration bins "
            "are supported"
        )

    threshold = float(
        evaluation_config[
            "probability_threshold"
        ]
    )
    calibration_bins = int(
        evaluation_config[
            "calibration_bins"
        ]
    )

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "Probability threshold must be in [0, 1]"
        )

    if calibration_bins <= 0:
        raise ValueError(
            "Calibration bins must be positive"
        )

    return threshold, calibration_bins


def evaluate_image_task(
    model,
    loader,
    device,
    joint_config,
):
    threshold, calibration_bins = (
        evaluation_settings(
            joint_config
        )
    )

    was_training = model.training
    model.eval()

    labels = []
    probabilities = []
    centers = []
    total_loss = 0.0
    total_samples = 0

    synchronize_device(
        device
    )
    start_time = time.perf_counter()

    try:
        with torch.inference_mode():
            for batch in loader:
                images = batch[
                    "image"
                ].to(
                    device
                )
                targets = batch[
                    "label"
                ].to(
                    device=device,
                    dtype=torch.float32,
                )

                logits = model.forward_image(
                    images
                )

                if logits.shape != (
                    targets.shape
                ):
                    raise RuntimeError(
                        "Image logits and targets "
                        "have different shapes"
                    )

                loss = (
                    functional
                    .binary_cross_entropy_with_logits(
                        logits,
                        targets,
                        reduction="sum",
                    )
                )

                batch_probabilities = (
                    torch.sigmoid(
                        logits
                    )
                )

                batch_size = int(
                    targets.shape[0]
                )

                total_loss += float(
                    loss.detach().item()
                )
                total_samples += (
                    batch_size
                )

                labels.extend(
                    targets
                    .detach()
                    .cpu()
                    .to(
                        dtype=torch.int64
                    )
                    .tolist()
                )

                probabilities.extend(
                    batch_probabilities
                    .detach()
                    .cpu()
                    .tolist()
                )

                centers.extend(
                    batch[
                        "center"
                    ]
                    .detach()
                    .cpu()
                    .tolist()
                )
    finally:
        if was_training:
            model.train()

    synchronize_device(
        device
    )
    elapsed_seconds = (
        time.perf_counter()
        - start_time
    )

    if total_samples == 0:
        raise ValueError(
            "Image evaluation loader returned no "
            "samples"
        )

    overall = (
        binary_classification_metrics(
            labels,
            probabilities,
            threshold=threshold,
            calibration_bins=(
                calibration_bins
            ),
        )
    )

    overall[
        "mean_loss"
    ] = float(
        total_loss
        / total_samples
    )
    overall[
        "elapsed_seconds"
    ] = float(
        elapsed_seconds
    )
    overall[
        "examples_per_second"
    ] = float(
        total_samples
        / elapsed_seconds
    )

    labels_array = np.asarray(
        labels,
        dtype=np.int64,
    )
    probabilities_array = np.asarray(
        probabilities,
        dtype=np.float64,
    )
    centers_array = np.asarray(
        centers,
        dtype=np.int64,
    )

    by_center = {}

    for center in sorted(
        np.unique(
            centers_array
        )
    ):
        center_mask = (
            centers_array == center
        )

        by_center[
            str(
                int(center)
            )
        ] = (
            binary_classification_metrics(
                labels_array[
                    center_mask
                ],
                probabilities_array[
                    center_mask
                ],
                threshold=threshold,
                calibration_bins=(
                    calibration_bins
                ),
            )
        )

    return {
        "overall": overall,
        "by_center": by_center,
    }


def evaluate_tabular_task(
    model,
    loader,
    device,
    joint_config,
):
    threshold, calibration_bins = (
        evaluation_settings(
            joint_config
        )
    )

    was_training = model.training
    model.eval()

    labels = []
    probabilities = []
    total_loss = 0.0
    total_samples = 0

    synchronize_device(
        device
    )
    start_time = time.perf_counter()

    try:
        with torch.inference_mode():
            for batch in loader:
                numeric_inputs = batch[
                    "numeric"
                ].to(
                    device
                )
                categorical_inputs = batch[
                    "categorical"
                ].to(
                    device
                )
                targets = batch[
                    "target"
                ].to(
                    device=device,
                    dtype=torch.float32,
                )

                logits = model.forward_tabular(
                    numeric_inputs,
                    categorical_inputs,
                )

                if logits.shape != (
                    targets.shape
                ):
                    raise RuntimeError(
                        "Tabular logits and targets "
                        "have different shapes"
                    )

                loss = (
                    functional
                    .binary_cross_entropy_with_logits(
                        logits,
                        targets,
                        reduction="sum",
                    )
                )

                batch_probabilities = (
                    torch.sigmoid(
                        logits
                    )
                )

                batch_size = int(
                    targets.shape[0]
                )

                total_loss += float(
                    loss.detach().item()
                )
                total_samples += (
                    batch_size
                )

                labels.extend(
                    targets
                    .detach()
                    .cpu()
                    .to(
                        dtype=torch.int64
                    )
                    .tolist()
                )

                probabilities.extend(
                    batch_probabilities
                    .detach()
                    .cpu()
                    .tolist()
                )
    finally:
        if was_training:
            model.train()

    synchronize_device(
        device
    )
    elapsed_seconds = (
        time.perf_counter()
        - start_time
    )

    if total_samples == 0:
        raise ValueError(
            "Tabular evaluation loader returned no "
            "samples"
        )

    overall = (
        binary_classification_metrics(
            labels,
            probabilities,
            threshold=threshold,
            calibration_bins=(
                calibration_bins
            ),
        )
    )

    overall[
        "mean_loss"
    ] = float(
        total_loss
        / total_samples
    )
    overall[
        "elapsed_seconds"
    ] = float(
        elapsed_seconds
    )
    overall[
        "examples_per_second"
    ] = float(
        total_samples
        / elapsed_seconds
    )

    return {
        "overall": overall,
    }
