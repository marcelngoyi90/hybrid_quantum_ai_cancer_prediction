from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as parquet
from PIL import Image


project_root = Path(__file__).resolve().parents[1]
data_root = project_root / "data" / "camelyon17_hf" / "data"
metrics_root = project_root / "artifacts" / "metrics"
figures_root = project_root / "artifacts" / "figures"

metrics_root.mkdir(parents=True, exist_ok=True)
figures_root.mkdir(parents=True, exist_ok=True)

split_order = ["train", "id_val", "ood_val", "ood_test"]
samples_per_class = 64
random_generator = random.Random(42)

sample_counts = Counter()
mode_counts = defaultdict(Counter)
size_counts = defaultdict(Counter)
alpha_pixels = Counter()
non_opaque_alpha_pixels = Counter()

pixel_sums = defaultdict(lambda: np.zeros(3, dtype=np.float64))
pixel_squared_sums = defaultdict(
    lambda: np.zeros(3, dtype=np.float64)
)
pixel_counts = Counter()

display_samples = defaultdict(list)


def get_logical_split(source_split, center):
    if source_split == "train":
        return "train"

    if source_split == "test":
        return "ood_test"

    if center == 1:
        return "ood_val"

    return "id_val"


parquet_files = []
row_groups = []

for source_split in ("train", "validation", "test"):
    paths = sorted(data_root.glob(f"{source_split}-*.parquet"))

    if not paths:
        raise FileNotFoundError(
            f"No Parquet files found for {source_split}"
        )

    for path in paths:
        parquet_file = parquet.ParquetFile(path)
        parquet_files.append(parquet_file)

        for row_group_index in range(parquet_file.num_row_groups):
            row_groups.append(
                (
                    source_split,
                    parquet_file,
                    row_group_index,
                )
            )

random_generator.shuffle(row_groups)

for source_split, parquet_file, row_group_index in row_groups:
    table = parquet_file.read_row_group(
        row_group_index,
        columns=["image", "label", "center"],
    )

    for row in table.to_pylist():
        label = int(row["label"])
        center = int(row["center"])
        split_name = get_logical_split(source_split, center)
        sample_key = (split_name, label)

        if sample_counts[sample_key] >= samples_per_class:
            continue

        image_bytes = row["image"]["bytes"]

        if image_bytes is None:
            raise ValueError("Image bytes are missing")

        image = Image.open(BytesIO(image_bytes))
        image.load()

        mode_counts[split_name][image.mode] += 1
        size_counts[split_name][image.size] += 1

        if "A" in image.getbands():
            alpha = np.asarray(
                image.getchannel("A"),
                dtype=np.uint8,
            )
            alpha_pixels[split_name] += alpha.size
            non_opaque_alpha_pixels[split_name] += int(
                np.count_nonzero(alpha != 255)
            )

        rgb_image = image.convert("RGB")
        pixels = (
            np.asarray(rgb_image, dtype=np.float32) / 255.0
        )

        pixel_sums[split_name] += pixels.sum(axis=(0, 1))
        pixel_squared_sums[split_name] += (
            pixels.astype(np.float64) ** 2
        ).sum(axis=(0, 1))
        pixel_counts[split_name] += (
            pixels.shape[0] * pixels.shape[1]
        )

        if len(display_samples[sample_key]) < 2:
            display_samples[sample_key].append(rgb_image.copy())

        sample_counts[sample_key] += 1

    if all(
        sample_counts[(split_name, label)]
        >= samples_per_class
        for split_name in split_order
        for label in (0, 1)
    ):
        break

for split_name in split_order:
    for label in (0, 1):
        count = sample_counts[(split_name, label)]

        if count != samples_per_class:
            raise ValueError(
                f"{split_name}, label {label}: "
                f"expected {samples_per_class}, found {count}"
            )

statistics_rows = []

for split_name in split_order:
    count = pixel_counts[split_name]
    channel_mean = pixel_sums[split_name] / count
    channel_variance = (
        pixel_squared_sums[split_name] / count
        - channel_mean**2
    )
    channel_std = np.sqrt(
        np.maximum(channel_variance, 0)
    )

    statistics_rows.append(
        {
            "split": split_name,
            "sampled_images": (
                sample_counts[(split_name, 0)]
                + sample_counts[(split_name, 1)]
            ),
            "source_modes": dict(mode_counts[split_name]),
            "image_sizes": dict(size_counts[split_name]),
            "non_opaque_alpha_pixels": (
                non_opaque_alpha_pixels[split_name]
            ),
            "red_mean": round(channel_mean[0], 6),
            "green_mean": round(channel_mean[1], 6),
            "blue_mean": round(channel_mean[2], 6),
            "red_std": round(channel_std[0], 6),
            "green_std": round(channel_std[1], 6),
            "blue_std": round(channel_std[2], 6),
        }
    )

statistics = pd.DataFrame(statistics_rows)
statistics_path = (
    metrics_root / "eda_image_sample_statistics.csv"
)
statistics.to_csv(statistics_path, index=False)

figure, axes = plt.subplots(
    nrows=4,
    ncols=4,
    figsize=(10, 10),
)

for row_index, split_name in enumerate(split_order):
    column_index = 0

    for label in (0, 1):
        label_name = (
            "non-tumor" if label == 0 else "metastatic tumor"
        )

        for image in display_samples[(split_name, label)]:
            axis = axes[row_index, column_index]
            axis.imshow(image)
            axis.set_title(f"label={label}: {label_name}")
            axis.axis("off")
            column_index += 1

    axes[row_index, 0].set_ylabel(
        split_name,
        rotation=90,
        size=11,
    )

figure.tight_layout()

figure_path = figures_root / "eda_sample_patches.png"
figure.savefig(
    figure_path,
    dpi=180,
    bbox_inches="tight",
)
plt.close(figure)

print(statistics.to_string(index=False))
print(f"statistics_csv={statistics_path}")
print(f"sample_figure={figure_path}")