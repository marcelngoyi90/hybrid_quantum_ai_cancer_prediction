import time

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def synchronize_device(device):
    if device.type == "mps":
        torch.mps.synchronize()

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def binary_classification_metrics(
    labels,
    probabilities,
    threshold=0.5,
):
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    if labels.ndim != 1 or probabilities.ndim != 1:
        raise ValueError(
            "labels and probabilities must be one-dimensional"
        )

    if len(labels) != len(probabilities):
        raise ValueError(
            "labels and probabilities must have equal length"
        )

    if len(labels) == 0:
        raise ValueError("Cannot evaluate an empty dataset")

    if not np.isfinite(probabilities).all():
        raise ValueError("Non-finite probabilities detected")

    predictions = (
        probabilities >= threshold
    ).astype(np.int64)

    true_negative, false_positive, false_negative, true_positive = (
        confusion_matrix(
            labels,
            predictions,
            labels=[0, 1],
        ).ravel()
    )

    specificity_denominator = (
        true_negative + false_positive
    )

    specificity = (
        true_negative / specificity_denominator
        if specificity_denominator
        else 0.0
    )

    unique_labels = np.unique(labels)

    roc_auc = (
        roc_auc_score(labels, probabilities)
        if len(unique_labels) == 2
        else None
    )

    average_precision = (
        average_precision_score(labels, probabilities)
        if len(unique_labels) == 2
        else None
    )

    return {
        "samples": int(len(labels)),
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(labels, predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                labels,
                predictions,
            )
        ),
        "roc_auc": (
            float(roc_auc)
            if roc_auc is not None
            else None
        ),
        "average_precision": (
            float(average_precision)
            if average_precision is not None
            else None
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
            recall_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "specificity": float(specificity),
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "true_positive": int(true_positive),
    }


def evaluate_classifier(
    model,
    loader,
    device,
    threshold=0.5,
):
    was_training = model.training
    model.eval()

    labels = []
    probabilities = []
    centers = []
    total_loss = 0.0
    total_samples = 0

    synchronize_device(device)
    start_time = time.perf_counter()

    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device)
            batch_labels = batch["label"].to(device)

            logits = model(images)

            if logits.ndim != 2 or logits.shape[1] != 2:
                raise ValueError(
                    f"Expected logits shaped (batch, 2), "
                    f"found {tuple(logits.shape)}"
                )

            loss = torch.nn.functional.cross_entropy(
                logits,
                batch_labels,
                reduction="sum",
            )

            batch_probabilities = torch.softmax(
                logits,
                dim=1,
            )[:, 1]

            batch_size = batch_labels.size(0)
            total_loss += loss.item()
            total_samples += batch_size

            labels.extend(
                batch_labels.detach().cpu().tolist()
            )
            probabilities.extend(
                batch_probabilities.detach().cpu().tolist()
            )
            centers.extend(batch["center"].tolist())

    synchronize_device(device)
    elapsed_seconds = time.perf_counter() - start_time

    if total_samples == 0:
        raise ValueError("Evaluation loader returned no samples")

    overall = binary_classification_metrics(
        labels,
        probabilities,
        threshold=threshold,
    )
    overall["mean_loss"] = (
        total_loss / total_samples
    )
    overall["elapsed_seconds"] = elapsed_seconds
    overall["images_per_second"] = (
        total_samples / elapsed_seconds
    )

    labels_array = np.asarray(labels, dtype=np.int64)
    probabilities_array = np.asarray(
        probabilities,
        dtype=np.float64,
    )
    centers_array = np.asarray(
        centers,
        dtype=np.int64,
    )

    by_center = {}

    for center in sorted(np.unique(centers_array)):
        center_mask = centers_array == center

        by_center[str(int(center))] = (
            binary_classification_metrics(
                labels_array[center_mask],
                probabilities_array[center_mask],
                threshold=threshold,
            )
        )

    if was_training:
        model.train()

    return {
        "overall": overall,
        "by_center": by_center,
    }