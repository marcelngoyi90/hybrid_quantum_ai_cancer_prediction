from pathlib import Path

from wilds import get_dataset


project_root = Path(__file__).resolve().parents[1]
data_root = project_root / "data"

dataset = get_dataset(
    dataset="camelyon17",
    root_dir=str(data_root),
    download=True,
)

print(f"total_labeled_patches={len(dataset)}")

for split_name in ("train", "id_val", "val", "test"):
    print(f"{split_name}={len(dataset.get_subset(split_name))}")

print(f"metadata_fields={dataset.metadata_fields}")