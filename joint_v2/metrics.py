import numpy as np
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


def calculate_binary_metrics(
    labels,
    probabilities,
    threshold=0.5,
):
    predictions = (
        probabilities >= threshold
    ).astype(np.uint8)

    true_negative, false_positive, false_negative, true_positive = (
        confusion_matrix(
            labels,
            predictions,
            labels=[0, 1],
        ).ravel()
    )

    specificity = true_negative / (
        true_negative + false_positive
    )

    return {
        "roc_auc": roc_auc_score(
            labels,
            probabilities,
        ),
        "average_precision": average_precision_score(
            labels,
            probabilities,
        ),
        "accuracy": accuracy_score(
            labels,
            predictions,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            labels,
            predictions,
        ),
        "f1": f1_score(
            labels,
            predictions,
        ),
        "precision": precision_score(
            labels,
            predictions,
            zero_division=0,
        ),
        "sensitivity": recall_score(
            labels,
            predictions,
        ),
        "specificity": specificity,
        "brier_score": brier_score_loss(
            labels,
            probabilities,
        ),
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "true_positive": int(true_positive),
    }


def print_binary_metrics(title, metrics):
    print()
    print(title)

    metric_names = [
        "roc_auc",
        "average_precision",
        "accuracy",
        "balanced_accuracy",
        "f1",
        "precision",
        "sensitivity",
        "specificity",
        "brier_score",
    ]

    for name in metric_names:
        print(
            f"{name}: {metrics[name]:.6f}"
        )

    print(
        "confusion_matrix:",
        [
            [
                metrics["true_negative"],
                metrics["false_positive"],
            ],
            [
                metrics["false_negative"],
                metrics["true_positive"],
            ],
        ],
    )