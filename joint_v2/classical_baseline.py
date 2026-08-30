from sklearn.linear_model import LogisticRegression
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

from data_loader import load_feature_archive


print("Loading Camelyon seed 42 data...")

train_features, train_labels = load_feature_archive(
    seed=42,
    filename="image_train_8d.npz",
)

validation_features, validation_labels = load_feature_archive(
    seed=42,
    filename="image_ood_val_8d.npz",
)

print("Training logistic regression...")

model = LogisticRegression(
    solver="lbfgs",
    penalty=None,
    max_iter=2000,
    tol=1e-6,
)

model.fit(
    train_features,
    train_labels,
)

probabilities = model.predict_proba(
    validation_features,
)[:, 1]

predictions = (
    probabilities >= 0.5
).astype("uint8")

true_negative, false_positive, false_negative, true_positive = (
    confusion_matrix(
        validation_labels,
        predictions,
        labels=[0, 1],
    ).ravel()
)

specificity = true_negative / (
    true_negative + false_positive
)

print()
print("Camelyon seed 42 OOD validation results")
print(
    "ROC-AUC:",
    round(
        roc_auc_score(
            validation_labels,
            probabilities,
        ),
        6,
    ),
)
print(
    "Average precision:",
    round(
        average_precision_score(
            validation_labels,
            probabilities,
        ),
        6,
    ),
)
print(
    "Accuracy:",
    round(
        accuracy_score(
            validation_labels,
            predictions,
        ),
        6,
    ),
)
print(
    "Balanced accuracy:",
    round(
        balanced_accuracy_score(
            validation_labels,
            predictions,
        ),
        6,
    ),
)
print(
    "F1:",
    round(
        f1_score(
            validation_labels,
            predictions,
        ),
        6,
    ),
)
print(
    "Precision:",
    round(
        precision_score(
            validation_labels,
            predictions,
        ),
        6,
    ),
)
print(
    "Sensitivity:",
    round(
        recall_score(
            validation_labels,
            predictions,
        ),
        6,
    ),
)
print(
    "Specificity:",
    round(
        specificity,
        6,
    ),
)
print(
    "Brier score:",
    round(
        brier_score_loss(
            validation_labels,
            probabilities,
        ),
        6,
    ),
)
print(
    "Confusion matrix:",
    [
        [int(true_negative), int(false_positive)],
        [int(false_negative), int(true_positive)],
    ],
)