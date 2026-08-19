from io import BytesIO
from pathlib import Path
import random

import pyarrow.parquet as parquet
import torch
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info
from torchvision import transforms


SPLIT_CONFIGURATION = {
    "train": {
        "file_prefix": "train",
        "centers": {0, 3, 4},
        "size": 302436,
    },
    "id_val": {
        "file_prefix": "validation",
        "centers": {0, 3, 4},
        "size": 33560,
    },
    "ood_val": {
        "file_prefix": "validation",
        "centers": {1},
        "size": 34904,
    },
    "ood_test": {
        "file_prefix": "test",
        "centers": {2},
        "size": 85054,
    },
}


class RandomRightAngleRotation:
    def __call__(self, image):
        angle = random.choice((0, 90, 180, 270))
        return image.rotate(angle)


def build_transform(training):
    operations = [
        transforms.Resize(
            (96, 96),
            antialias=True,
        ),
    ]

    if training:
        operations.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                RandomRightAngleRotation(),
                transforms.ColorJitter(
                    brightness=0.20,
                    contrast=0.20,
                    saturation=0.15,
                    hue=0.04,
                ),
            ]
        )

    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )

    return transforms.Compose(operations)


class Camelyon17ParquetDataset(IterableDataset):
    def __init__(
        self,
        data_root,
        split,
        transform=None,
        shuffle=False,
        seed=42,
        limit=None,
    ):
        super().__init__()

        if split not in SPLIT_CONFIGURATION:
            allowed = ", ".join(SPLIT_CONFIGURATION)
            raise ValueError(
                f"Unknown split {split}. Expected one of: {allowed}"
            )

        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive or None")

        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.shuffle = shuffle
        self.seed = seed
        self.limit = limit
        self.epoch = 0

        configuration = SPLIT_CONFIGURATION[split]
        self.allowed_centers = configuration["centers"]
        self.full_size = configuration["size"]

        file_prefix = configuration["file_prefix"]
        self.files = sorted(
            self.data_root.glob(f"{file_prefix}-*.parquet")
        )

        if not self.files:
            raise FileNotFoundError(
                f"No Parquet files found in {self.data_root} "
                f"for split {split}"
            )

        self.row_groups = []

        for path in self.files:
            parquet_file = parquet.ParquetFile(path)
            center_column_index = (
                parquet_file.schema.names.index("center")
            )

            for row_group_index in range(
                parquet_file.num_row_groups
            ):
                center_column = (
                    parquet_file.metadata
                    .row_group(row_group_index)
                    .column(center_column_index)
                )
                statistics = center_column.statistics

                if statistics is not None:
                    contains_allowed_center = any(
                        statistics.min <= center
                        <= statistics.max
                        for center in self.allowed_centers
                    )

                    if not contains_allowed_center:
                        continue

                self.row_groups.append(
                    (path, row_group_index)
                )

    def __len__(self):
        if self.limit is None:
            return self.full_size

        return min(self.limit, self.full_size)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _worker_row_groups(self):
        row_groups = list(self.row_groups)
        random_generator = random.Random(
            self.seed + self.epoch
        )

        if self.shuffle:
            random_generator.shuffle(row_groups)

        worker = get_worker_info()

        if worker is None:
            return row_groups, random_generator, self.limit

        row_groups = row_groups[
            worker.id :: worker.num_workers
        ]

        worker_limit = None

        if self.limit is not None:
            base_limit = self.limit // worker.num_workers
            remainder = self.limit % worker.num_workers
            worker_limit = base_limit + (
                1 if worker.id < remainder else 0
            )

        return row_groups, random_generator, worker_limit

    def _decode_row(self, row):
        image_bytes = row["image"]["bytes"]

        if image_bytes is None:
            raise ValueError(
                f"Missing image bytes for "
                f"image_id={row['image_id']}"
            )

        with Image.open(BytesIO(image_bytes)) as source:
            image = source.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "label": torch.tensor(
                int(row["label"]),
                dtype=torch.long,
            ),
            "center": int(row["center"]),
            "image_id": int(row["image_id"]),
            "patient": int(row["patient"]),
            "node": int(row["node"]),
            "slide": int(row["slide"]),
            "x_coord": int(row["x_coord"]),
            "y_coord": int(row["y_coord"]),
        }

    def __iter__(self):
        (
            row_groups,
            random_generator,
            worker_limit,
        ) = self._worker_row_groups()

        parquet_cache = {}
        buffered_rows = []
        shuffle_buffer_size = 1000
        yielded = 0

        columns = [
            "image",
            "label",
            "center",
            "image_id",
            "patient",
            "node",
            "slide",
            "x_coord",
            "y_coord",
        ]

        def produce(rows):
            nonlocal yielded

            for row in rows:
                if (
                    worker_limit is not None
                    and yielded >= worker_limit
                ):
                    return

                yielded += 1
                yield self._decode_row(row)

        for path, row_group_index in row_groups:
            if worker_limit is not None and yielded >= worker_limit:
                break

            if path not in parquet_cache:
                parquet_cache[path] = parquet.ParquetFile(path)

            table = parquet_cache[path].read_row_group(
                row_group_index,
                columns=columns,
            )

            rows = [
                row
                for row in table.to_pylist()
                if row["center"] in self.allowed_centers
            ]

            if self.shuffle:
                buffered_rows.extend(rows)

                if len(buffered_rows) < shuffle_buffer_size:
                    continue

                random_generator.shuffle(buffered_rows)
                rows_to_produce = buffered_rows
                buffered_rows = []
            else:
                rows_to_produce = rows

            yield from produce(rows_to_produce)

        if self.shuffle and buffered_rows:
            random_generator.shuffle(buffered_rows)
            yield from produce(buffered_rows)