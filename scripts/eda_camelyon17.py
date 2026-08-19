from itertools import combinations
from pathlib import Path

import pandas as pd
import pyarrow.dataset as pyarrow_dataset


project_root = Path(__file__).resolve().parents[1]
data_root = project_root / "data" / "camelyon17_hf" / "data"
metrics_root = project_root / "artifacts" / "metrics"
metrics_root.mkdir(parents=True, exist_ok=True)

columns = [
    "label",
    "center",
    "image_id",
    "patient",
    "node",
    "slide",
    "x_coord",
    "y_coord",
]


def load_metadata(source_split):
    files = sorted(data_root.glob(f"{source_split}-*.parquet"))

    if not files:
        raise FileNotFoundError(f"No files found for {source_split}")

    table = pyarrow_dataset.dataset(
        files,
        format="parquet",
    ).to_table(columns=columns)

    return table.to_pandas()


train = load_metadata("train")
validation = load_metadata("validation")
test = load_metadata("test")

train["split"] = "train"
validation["split"] = validation["center"].map(
    lambda center: "ood_val" if center == 1 else "id_val"
)
test["split"] = "ood_test"

metadata = pd.concat(
    [train, validation, test],
    ignore_index=True,
)

split_order = ["train", "id_val", "ood_val", "ood_test"]

expected_rows = {
    "train": 302436,
    "id_val": 33560,
    "ood_val": 34904,
    "ood_test": 85054,
}

expected_centers = {
    "train": {0, 3, 4},
    "id_val": {0, 3, 4},
    "ood_val": {1},
    "ood_test": {2},
}

summary_rows = []

for split_name in split_order:
    subset = metadata[metadata["split"] == split_name]
    centers = set(subset["center"].unique())

    if len(subset) != expected_rows[split_name]:
        raise ValueError(
            f"{split_name}: expected {expected_rows[split_name]} rows, "
            f"found {len(subset)}"
        )

    if centers != expected_centers[split_name]:
        raise ValueError(
            f"{split_name}: expected centers "
            f"{expected_centers[split_name]}, found {centers}"
        )

    tumor_count = int((subset["label"] == 1).sum())
    non_tumor_count = int((subset["label"] == 0).sum())

    summary_rows.append(
        {
            "split": split_name,
            "patches": len(subset),
            "non_tumor": non_tumor_count,
            "tumor": tumor_count,
            "tumor_percent": round(100 * tumor_count / len(subset), 2),
            "centers": ",".join(
                str(value) for value in sorted(centers)
            ),
            "patients": subset["patient"].nunique(),
            "slides": subset["slide"].nunique(),
            "unique_image_ids": subset["image_id"].nunique(),
        }
    )

summary = pd.DataFrame(summary_rows)

center_label_counts = (
    metadata.groupby(["split", "center", "label"], observed=True)
    .size()
    .rename("patches")
    .reset_index()
)

overlap_rows = []

for left_name, right_name in combinations(split_order, 2):
    left = metadata[metadata["split"] == left_name]
    right = metadata[metadata["split"] == right_name]

    overlap_rows.append(
        {
            "left_split": left_name,
            "right_split": right_name,
            "shared_patients": len(
                set(left["patient"]) & set(right["patient"])
            ),
            "shared_slides": len(
                set(left["slide"]) & set(right["slide"])
            ),
            "shared_image_ids": len(
                set(left["image_id"]) & set(right["image_id"])
            ),
        }
    )

overlaps = pd.DataFrame(overlap_rows)

if set(metadata["label"].unique()) != {0, 1}:
    raise ValueError("Labels must contain only 0 and 1")

if metadata["image_id"].duplicated().any():
    raise ValueError("Duplicate image_id values found")

ood_pairs = overlaps[
    overlaps["left_split"].isin(["ood_val", "ood_test"])
    | overlaps["right_split"].isin(["ood_val", "ood_test"])
]

if (
    ood_pairs[
        ["shared_patients", "shared_slides", "shared_image_ids"]
    ]
    .to_numpy()
    .any()
):
    raise ValueError("Unexpected overlap involving an OOD split")

summary.to_csv(
    metrics_root / "eda_split_summary.csv",
    index=False,
)
center_label_counts.to_csv(
    metrics_root / "eda_center_label_counts.csv",
    index=False,
)
overlaps.to_csv(
    metrics_root / "eda_split_overlaps.csv",
    index=False,
)

print(summary.to_string(index=False))
print()
print(overlaps.to_string(index=False))