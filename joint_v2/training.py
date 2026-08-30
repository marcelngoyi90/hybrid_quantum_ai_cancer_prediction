import copy
import random

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_data_loader(
    features,
    labels,
    batch_size,
    shuffle,
    seed,
):
    feature_tensor = torch.as_tensor(
        features,
        dtype=torch.float32,
    )

    label_tensor = torch.as_tensor(
        labels,
        dtype=torch.float32,
    )

    dataset = TensorDataset(
        feature_tensor,
        label_tensor,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )
def predict_probabilities(
    model,
    features,
    batch_size=4096,
):
    model.eval()

    feature_tensor = torch.as_tensor(
        features,
        dtype=torch.float32,
    )

    probabilities = []

    with torch.no_grad():
        for start in range(
            0,
            len(feature_tensor),
            batch_size,
        ):
            end = start + batch_size
            batch = feature_tensor[start:end]

            logits = model(batch)

            batch_probabilities = torch.sigmoid(
                logits
            )

            probabilities.append(
                batch_probabilities.cpu().numpy()
            )

    return np.concatenate(probabilities)
def train_binary_model(
    model,
    train_features,
    train_labels,
    validation_features,
    validation_labels,
    seed,
    learning_rate=0.001,
    batch_size=1024,
    maximum_epochs=30,
    patience=5,
):
    set_random_seed(seed)

    training_loader = create_data_loader(
        features=train_features,
        labels=train_labels,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )

    negative_count = int(
        (train_labels == 0).sum()
    )
    positive_count = int(
        (train_labels == 1).sum()
    )

    positive_weight = torch.tensor(
        negative_count / positive_count,
        dtype=torch.float32,
    )

    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=positive_weight,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    best_validation_auc = float("-inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(
        1,
        maximum_epochs + 1,
    ):
        model.train()
        total_loss = 0.0
        total_examples = 0

        for batch_features, batch_labels in training_loader:
            optimizer.zero_grad()

            logits = model(batch_features)

            loss = loss_function(
                logits,
                batch_labels,
            )

            loss.backward()
            optimizer.step()

            batch_size_actual = len(batch_labels)

            total_loss += (
                loss.item() * batch_size_actual
            )
            total_examples += batch_size_actual

        average_loss = total_loss / total_examples

        validation_probabilities = (
            predict_probabilities(
                model=model,
                features=validation_features,
            )
        )

        validation_auc = roc_auc_score(
            validation_labels,
            validation_probabilities,
        )

        history.append(
            {
                "epoch": epoch,
                "training_loss": average_loss,
                "validation_roc_auc": validation_auc,
            }
        )

        print(
            f"Epoch {epoch:02d} | "
            f"loss {average_loss:.6f} | "
            f"validation ROC-AUC "
            f"{validation_auc:.6f}"
        )

        if validation_auc > best_validation_auc:
            best_validation_auc = validation_auc
            best_epoch = epoch
            best_state = copy.deepcopy(
                model.state_dict()
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping")
            break

    if best_state is None:
        raise RuntimeError(
            "Training did not produce a valid model"
        )

    model.load_state_dict(best_state)

    return {
        "model": model,
        "best_epoch": best_epoch,
        "best_validation_roc_auc": (
            best_validation_auc
        ),
        "history": history,
    }