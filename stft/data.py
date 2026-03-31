import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

@dataclass
class Dataset5D:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    channels: int
    img_size: tuple[int, int]


def load_dataset(dataset_dir, mmap_mode="r"):
    dataset_dir = Path(dataset_dir)

    splits = {}
    for name in ("train", "val", "test"):
        path = dataset_dir / f"{name}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        arr = np.load(path, mmap_mode=mmap_mode)
        if arr.ndim != 5:
            raise ValueError(
                f"{name}.npy has {arr.ndim} dimensions, expected 5 (N, T, C, H, W)"
            )
        splits[name] = arr

    channels = splits["train"].shape[2]
    img_size = (splits["train"].shape[3], splits["train"].shape[4])

    for name, arr in splits.items():
        if arr.shape[2] != channels:
            raise ValueError(
                f"Channel mismatch: train has {channels}, {name} has {arr.shape[2]}"
            )
        if (arr.shape[3], arr.shape[4]) != img_size:
            raise ValueError(
                f"Spatial size mismatch: train has {img_size}, "
                f"{name} has {(arr.shape[3], arr.shape[4])}"
            )

    metadata_path = dataset_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        if "channels" in metadata and metadata["channels"] != channels:
            raise ValueError(
                f"metadata.json channels={metadata['channels']} "
                f"but data has {channels}"
            )
        if "img_size" in metadata and tuple(metadata["img_size"]) != img_size:
            raise ValueError(
                f"metadata.json img_size={metadata['img_size']} "
                f"but data has {list(img_size)}"
            )

    return Dataset5D(
        train=splits["train"],
        val=splits["val"],
        test=splits["test"],
        channels=channels,
        img_size=img_size,
    )


class TemporalDataset(Dataset):
    def __init__(self, data, snapshot_length=20, mean=None, std=None):
        self.data = data
        self.N, self.T, self.C, self.H, self.W = data.shape
        self.snapshot_length = snapshot_length
        self.mean = mean
        self.std = std
        self.indices = [
            (n, t)
            for n in range(self.N)
            for t in range(self.T - snapshot_length + 1)
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        n, start = self.indices[idx]
        x = torch.tensor(
            np.array(self.data[n, start : start + self.snapshot_length]),
            dtype=torch.float32,
        )
        if self.mean is not None:
            x = (x - self.mean) / self.std
        return x


def get_grid(H, W):
    x = np.linspace(0, 1, H)
    y = np.linspace(0, 1, W)

    x, y = np.meshgrid(x, y)
    x = x.T
    y = y.T

    grid = torch.tensor(np.concatenate((x[None], y[None]), axis=0), dtype=torch.float32)

    return grid
