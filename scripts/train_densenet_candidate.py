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
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.camelyon17_data import (
    Camelyon17ParquetDataset,
    build_transform,
)
from src.densenet_model import (
    TRAINING_MODES,
    DenseNet121Classifier,
    count_parameters,
)
from src.evaluation import (
    evaluate_classifier,
    synchronize_device,
)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        required=True,
        choices=sorted(TRAINING_MODES),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--train-limit",
        type=int,
        default=20000,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    arguments = parser.parse_args()

    if arguments.epochs <= 0:
        parser.error("--epochs must be positive")

    if arguments.train_limit <= 0:
        parser.error("--train-limit must be positive")

    if arguments.batch_size <= 0:
        parser.error("--batch-size must be positive")

    if arguments.eval_batch_size <= 0:
        parser.error(
            "--eval-batch-size must be positive"
        )

    return arguments


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_optimizer(model, mode):
    backbone_parameters = [
        parameter
        for parameter in model.backbone.parameters()
        if parameter.requires_grad
    ]
    classifier_parameters = list(
        model.classifier.parameters()
    )

    learning_rates = {
        "scratch": {
            "backbone": 1e-4,
            "classifier": 1e-4,
        },
        "frozen": {
            "classifier": 1e-3,
        },
        "last_block": {
            "backbone": 1e-4,
            "classifier": 1e-3,
        },
        "full": {
            "backbone": 5e-5,
            "classifier": 5e-4,
        },
    }

    parameter_groups = []

    if backbone_parameters:
        parameter_groups.append(
            {
                "params": backbone_parameters,
                "lr": learning_rates[mode]["backbone"],
                "name": "backbone",
            }
        )

    parameter_groups.append(
        {
            "params": classifier_parameters,
            "lr": learning_rates[mode]["classifier"],
            "name": "classifier",
        }
    )

    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=1e-4,
    )

    return optimizer, learning_rates[mode]


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    epoch,
):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_images = 0

    criterion = nn.CrossEntropyLoss()

    synchronize_device(device)
    start_time = time.perf_counter()

    for batch_index, batch in enumerate(
        loader,
        start=1,
    ):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss at epoch {epoch}, "
                f"batch {batch_index}"
            )

        loss.backward()
        optimizer.step()

        current_batch_size = labels.size(0)
        total_images += current_batch_size
        total_loss += loss.item() * current_batch_size
        total_correct += (
            logits.argmax(dim=1) == labels
        ).sum().item()

        if batch_index % 200 == 0:
            synchronize_device(device)
            elapsed = time.perf_counter() - start_time

            print(
                f"epoch={epoch} "
                f"batch={batch_index}/{len(loader)} "
                f"images={total_images} "
                f"elapsed_seconds={elapsed:.2f}"
            )

    synchronize_device(device)
    elapsed_seconds = time.perf_counter() - start_time

    return {
        "samples": total_images,
        "mean_loss": total_loss / total_images,
        "accuracy": total_correct / total_images,
        "elapsed_seconds": elapsed_seconds,
        "images_per_second": (
            total_images / elapsed_seconds
        ),
    }


def copy_state_dict_to_cpu(model):
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def main():
    arguments = parse_arguments()
    seed_everything(arguments.seed)

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")

    device = torch.device("mps")

    data_root = (
        project_root
        / "data"
        / "camelyon17_hf"
        / "data"
    )
    checkpoints_root = (
        project_root
        / "artifacts"
        / "checkpoints"
    )
    metrics_root = (
        project_root
        / "artifacts"
        / "metrics"
    )

    checkpoints_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    metrics_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_dataset = Camelyon17ParquetDataset(
        data_root=data_root,
        split="train",
        transform=build_transform(training=True),
        shuffle=True,
        seed=arguments.seed,
        limit=arguments.train_limit,
    )

    ood_validation_dataset = (
        Camelyon17ParquetDataset(
            data_root=data_root,
            split="ood_val",
            transform=build_transform(training=False),
            shuffle=False,
            seed=arguments.seed,
            limit=None,
        )
    )

    id_validation_dataset = (
        Camelyon17ParquetDataset(
            data_root=data_root,
            split="id_val",
            transform=build_transform(training=False),
            shuffle=False,
            seed=arguments.seed,
            limit=None,
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=arguments.batch_size,
        num_workers=0,
        drop_last=False,
    )

    ood_validation_loader = DataLoader(
        ood_validation_dataset,
        batch_size=arguments.eval_batch_size,
        num_workers=0,
        drop_last=False,
    )

    id_validation_loader = DataLoader(
        id_validation_dataset,
        batch_size=arguments.eval_batch_size,
        num_workers=0,
        drop_last=False,
    )

    model = DenseNet121Classifier(
        training_mode=arguments.mode,
    )
    model = model.to(device)

    parameter_counts = count_parameters(model)

    optimizer, initial_learning_rates = (
        build_optimizer(
            model,
            arguments.mode,
        )
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=arguments.epochs,
        )
    )

    run_name = (
        f"{arguments.mode}"
        f"_n{arguments.train_limit}"
        f"_e{arguments.epochs}"
        f"_seed{arguments.seed}"
    )

    checkpoint_path = (
        checkpoints_root
        / f"candidate_{run_name}.pt"
    )

    metrics_path = (
        metrics_root
        / f"candidate_{run_name}.json"
    )

    history = []
    best_score = (-1.0, -1.0)
    best_epoch = None
    best_state_dict = None
    best_ood_validation = None

    for epoch in range(1, arguments.epochs + 1):
        train_dataset.set_epoch(epoch - 1)

        training_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
        )

        ood_validation_metrics = evaluate_classifier(
            model=model,
            loader=ood_validation_loader,
            device=device,
            threshold=0.5,
        )

        ood_overall = (
            ood_validation_metrics["overall"]
        )

        current_score = (
            ood_overall["accuracy"],
            ood_overall["roc_auc"],
        )

        learning_rates = {
            group["name"]: group["lr"]
            for group in optimizer.param_groups
        }

        epoch_record = {
            "epoch": epoch,
            "learning_rates": learning_rates,
            "training": training_metrics,
            "ood_validation": (
                ood_validation_metrics
            ),
        }
        history.append(epoch_record)

        print(
            f"epoch={epoch} "
            f"train_loss="
            f"{training_metrics['mean_loss']:.6f} "
            f"train_accuracy="
            f"{training_metrics['accuracy']:.4f} "
            f"ood_val_accuracy="
            f"{ood_overall['accuracy']:.4f} "
            f"ood_val_roc_auc="
            f"{ood_overall['roc_auc']:.4f} "
            f"ood_val_f1="
            f"{ood_overall['f1']:.4f}"
        )

        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch
            best_state_dict = (
                copy_state_dict_to_cpu(model)
            )
            best_ood_validation = (
                ood_validation_metrics
            )

            torch.save(
                {
                    "model_state_dict": (
                        best_state_dict
                    ),
                    "training_mode": arguments.mode,
                    "seed": arguments.seed,
                    "epoch": best_epoch,
                    "ood_validation": (
                        best_ood_validation
                    ),
                },
                checkpoint_path,
            )

        scheduler.step()

    if best_state_dict is None:
        raise RuntimeError(
            "No candidate checkpoint was selected"
        )

    model.load_state_dict(best_state_dict)

    id_validation_metrics = evaluate_classifier(
        model=model,
        loader=id_validation_loader,
        device=device,
        threshold=0.5,
    )

    final_ood_validation_metrics = (
        evaluate_classifier(
            model=model,
            loader=ood_validation_loader,
            device=device,
            threshold=0.5,
        )
    )

    results = {
        "configuration": {
            "training_mode": arguments.mode,
            "seed": arguments.seed,
            "epochs": arguments.epochs,
            "train_limit": arguments.train_limit,
            "batch_size": arguments.batch_size,
            "eval_batch_size": (
                arguments.eval_batch_size
            ),
            "selection_metric": (
                "ood_validation_accuracy"
            ),
            "threshold": 0.5,
            "initial_learning_rates": (
                initial_learning_rates
            ),
            "weight_decay": 1e-4,
        },
        "parameters": parameter_counts,
        "best_epoch": best_epoch,
        "best_id_validation": (
            id_validation_metrics
        ),
        "best_ood_validation": (
            final_ood_validation_metrics
        ),
        "history": history,
        "checkpoint": str(checkpoint_path),
    }

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    id_overall = id_validation_metrics["overall"]
    ood_overall = (
        final_ood_validation_metrics["overall"]
    )

    print(f"best_epoch={best_epoch}")
    print(
        f"id_val_accuracy="
        f"{id_overall['accuracy']:.4f}"
    )
    print(
        f"id_val_roc_auc="
        f"{id_overall['roc_auc']:.4f}"
    )
    print(
        f"ood_val_accuracy="
        f"{ood_overall['accuracy']:.4f}"
    )
    print(
        f"ood_val_roc_auc="
        f"{ood_overall['roc_auc']:.4f}"
    )
    print(
        f"ood_val_f1="
        f"{ood_overall['f1']:.4f}"
    )
    print(f"checkpoint={checkpoint_path}")
    print(f"metrics_file={metrics_path}")


if __name__ == "__main__":
    main()