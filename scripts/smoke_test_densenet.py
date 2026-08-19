import os

os.environ.setdefault(
    "PYTORCH_ENABLE_MPS_FALLBACK",
    "1",
)

import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import densenet121


project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.camelyon17_data import (
    Camelyon17ParquetDataset,
    build_transform,
)


seed = 42
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

dataset = Camelyon17ParquetDataset(
    data_root=data_root,
    split="train",
    transform=build_transform(training=True),
    shuffle=True,
    seed=seed,
    limit=1024,
)

loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=0,
    drop_last=False,
)

model = densenet121(weights=None)
model.classifier = nn.Linear(
    model.classifier.in_features,
    2,
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4,
)

classifier_before = (
    model.classifier.weight
    .detach()
    .cpu()
    .clone()
)

model.train()

total_loss = 0.0
total_correct = 0
total_images = 0

torch.mps.synchronize()
start_time = time.perf_counter()

for batch in loader:
    images = batch["image"].to(device)
    labels = batch["label"].to(device)

    optimizer.zero_grad(set_to_none=True)

    logits = model(images)
    loss = criterion(logits, labels)

    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite loss detected")

    loss.backward()
    optimizer.step()

    batch_size = labels.size(0)
    total_images += batch_size
    total_loss += loss.item() * batch_size
    total_correct += (
        logits.argmax(dim=1) == labels
    ).sum().item()

torch.mps.synchronize()
elapsed_seconds = time.perf_counter() - start_time

classifier_after = (
    model.classifier.weight
    .detach()
    .cpu()
)

parameter_change = torch.linalg.vector_norm(
    classifier_after - classifier_before
).item()

if total_images != 1024:
    raise RuntimeError(
        f"Expected 1024 images, processed {total_images}"
    )

if parameter_change == 0:
    raise RuntimeError("Model parameters did not update")

print(f"device={device}")
print(f"images={total_images}")
print(f"batches={len(loader)}")
print(f"mean_loss={total_loss / total_images:.6f}")
print(f"smoke_accuracy={total_correct / total_images:.4f}")
print(f"elapsed_seconds={elapsed_seconds:.2f}")
print(
    f"images_per_second="
    f"{total_images / elapsed_seconds:.2f}"
)
print(f"classifier_parameter_change={parameter_change:.6f}")