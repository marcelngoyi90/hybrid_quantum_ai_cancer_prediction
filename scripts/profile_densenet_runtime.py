import os

os.environ.setdefault(
    "PYTORCH_ENABLE_MPS_FALLBACK",
    "1",
)

import json
import random
import sys
import time
from collections import Counter
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
    DenseNet121Classifier,
    count_parameters,
)


seed = 42
sample_limit = 20000
batch_size = 32

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if not torch.backends.mps.is_available():
    raise RuntimeError("MPS is not available")

device = torch.device("mps")

data_root = (
    project_root
    / "data"
    / "camelyon17_hf"
    / "data"
)
metrics_root = project_root / "artifacts" / "metrics"
metrics_root.mkdir(parents=True, exist_ok=True)

dataset = Camelyon17ParquetDataset(
    data_root=data_root,
    split="train",
    transform=build_transform(training=True),
    shuffle=True,
    seed=seed,
    limit=sample_limit,
)

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=0,
    drop_last=False,
)

model = DenseNet121Classifier(
    training_mode="scratch",
)
model = model.to(device)

parameter_counts = count_parameters(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4,
)

model.train()

total_loss = 0.0
total_correct = 0
total_images = 0
label_counts = Counter()
maximum_allocated_bytes = 0
maximum_driver_bytes = 0

torch.mps.synchronize()
start_time = time.perf_counter()

for batch_index, batch in enumerate(loader, start=1):
    images = batch["image"].to(device)
    labels = batch["label"].to(device)

    optimizer.zero_grad(set_to_none=True)

    logits = model(images)
    loss = criterion(logits, labels)

    if not torch.isfinite(loss):
        raise RuntimeError(
            f"Non-finite loss at batch {batch_index}"
        )

    loss.backward()
    optimizer.step()

    current_batch_size = labels.size(0)
    total_images += current_batch_size
    total_loss += loss.item() * current_batch_size
    total_correct += (
        logits.argmax(dim=1) == labels
    ).sum().item()
    label_counts.update(labels.detach().cpu().tolist())

    maximum_allocated_bytes = max(
        maximum_allocated_bytes,
        torch.mps.current_allocated_memory(),
    )
    maximum_driver_bytes = max(
        maximum_driver_bytes,
        torch.mps.driver_allocated_memory(),
    )

    if batch_index % 100 == 0:
        torch.mps.synchronize()
        elapsed = time.perf_counter() - start_time
        print(
            f"batch={batch_index}/{len(loader)} "
            f"images={total_images} "
            f"elapsed_seconds={elapsed:.2f}"
        )

torch.mps.synchronize()
elapsed_seconds = time.perf_counter() - start_time

if total_images != sample_limit:
    raise RuntimeError(
        f"Expected {sample_limit} images, "
        f"processed {total_images}"
    )

images_per_second = total_images / elapsed_seconds
full_epoch_minutes = (
    302436 / images_per_second / 60
)

results = {
    "device": str(device),
    "training_mode": "scratch",
    "sample_limit": sample_limit,
    "batch_size": batch_size,
    "num_workers": 0,
    "batches": len(loader),
    "label_counts": dict(sorted(label_counts.items())),
    "mean_training_loss": total_loss / total_images,
    "training_accuracy": total_correct / total_images,
    "elapsed_seconds": elapsed_seconds,
    "images_per_second": images_per_second,
    "estimated_full_epoch_minutes": full_epoch_minutes,
    "maximum_allocated_memory_gib": (
        maximum_allocated_bytes / 1024**3
    ),
    "maximum_driver_memory_gib": (
        maximum_driver_bytes / 1024**3
    ),
    "total_parameters": parameter_counts["total"],
    "trainable_parameters": (
        parameter_counts["trainable"]
    ),
}

output_path = (
    metrics_root / "densenet_runtime_profile.json"
)

with output_path.open("w", encoding="utf-8") as file:
    json.dump(results, file, indent=2)

for key, value in results.items():
    print(f"{key}={value}")

print(f"metrics_file={output_path}")