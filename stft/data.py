import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class HDF5Dataset:
    path: Path
    channels: int
    img_size: tuple
    shapes: dict
    mean: np.ndarray
    std: np.ndarray
    metadata: dict = field(default_factory=dict)


def load_dataset(dataset_path, mmap_mode=None):
    dataset_path = Path(dataset_path)

    if dataset_path.suffix != ".h5":
        raise ValueError(f"dataset_path must point to an .h5 file, got: {dataset_path}")
    h5_path = dataset_path
    if not h5_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        shapes = {}
        for name in ("train", "val", "test"):
            if name not in f:
                raise KeyError(f"Missing split '{name}' in {h5_path}")
            shapes[name] = tuple(f[name].shape)

        channels = int(f.attrs["channels"])
        img_size = tuple(int(x) for x in f.attrs["img_size"])

        mean = np.array(f["stats/mean"], dtype=np.float64)
        std  = np.array(f["stats/std"],  dtype=np.float64)

        metadata = {k: v for k, v in f.attrs.items()
                    if k not in ("channels", "img_size")}

    return HDF5Dataset(
        path=h5_path,
        channels=channels,
        img_size=img_size,
        shapes=shapes,
        mean=mean,
        std=std,
        metadata=metadata,
    )


class _H5Dataset(Dataset):
    """Base class with lazy h5py open pattern."""

    def __init__(self):
        self._h5file = None
        self._h5dataset = None

    def _open_h5(self, split):
        self._h5file = h5py.File(self.hdf5_path, "r", libver='latest')
        self._h5dataset = self._h5file[split]

    def _ensure_open(self):
        if self._h5file is None:
            self._open_h5(self.split)

    def close(self):
        if self._h5file is not None:
            self._h5file.close()
            self._h5file = None
            self._h5dataset = None

    def __del__(self):
        self.close()


class TrainingDataset(_H5Dataset):
    def __init__(self, hdf5_path, cond_time, split="train", mean=None, std=None):
        super().__init__()
        self.hdf5_path = Path(hdf5_path)
        self.split = split
        self.cond_time = cond_time
        self.mean = mean
        self.std = std

        with h5py.File(self.hdf5_path, "r") as f:
            N, T = f[split].shape[:2]
        self.indices = [(n, t) for n in range(N) for t in range(T - cond_time)]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        self._ensure_open()
        n, t = self.indices[idx]
        x = torch.tensor(
            np.array(self._h5dataset[n, t : t + self.cond_time]), dtype=torch.float32
        )
        y = torch.tensor(
            np.array(self._h5dataset[n, t + self.cond_time]), dtype=torch.float32
        )
        if self.mean is not None:
            x = (x - self.mean) / self.std
            y = (y - self.mean) / self.std
        return x, y


class SnapshotDataset(_H5Dataset):
    def __init__(self, hdf5_path, snapshot_length, split="train", mean=None, std=None):
        super().__init__()
        self.hdf5_path = Path(hdf5_path)
        self.split = split
        self.snapshot_length = snapshot_length
        self.mean = mean
        self.std = std

        with h5py.File(self.hdf5_path, "r") as f:
            N, T = f[split].shape[:2]
        if snapshot_length > T:
            raise ValueError(
                f"snapshot_length={snapshot_length} exceeds trajectory length T={T}"
            )
        self.indices = [
            (n, start)
            for n in range(N)
            for start in range(T - snapshot_length + 1)
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        self._ensure_open()
        n, start = self.indices[idx]
        snap = torch.tensor(
            np.array(self._h5dataset[n, start : start + self.snapshot_length]),
            dtype=torch.float32,
        )
        if self.mean is not None:
            snap = (snap - self.mean) / self.std
        return snap


class LegacySnapshotDataset(_H5Dataset):
    def __init__(self, hdf5_path, snapshot_length, split="train", mean=None, std=None):
        super().__init__()
        self.hdf5_path = Path(hdf5_path)
        self.split = split
        self.snapshot_length = snapshot_length
        self.mean = mean
        self.std = std

        with h5py.File(self.hdf5_path, "r") as f:
            self.N, self.T = f[split].shape[:2]
        if snapshot_length > self.T:
            raise ValueError(
                f"snapshot_length={snapshot_length} exceeds trajectory length T={self.T}"
            )

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        self._ensure_open()
        start = random.randint(0, self.T - self.snapshot_length)
        snap = torch.tensor(
            np.array(self._h5dataset[idx, start : start + self.snapshot_length]),
            dtype=torch.float32,
        )
        if self.mean is not None:
            snap = (snap - self.mean) / self.std
        return snap


class RolloutDataset(_H5Dataset):
    def __init__(self, hdf5_path, split, mean=None, std=None):
        super().__init__()
        self.hdf5_path = Path(hdf5_path)
        self.split = split
        self.mean = mean
        self.std = std

        with h5py.File(self.hdf5_path, "r") as f:
            self.N = f[split].shape[0]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        self._ensure_open()
        x = torch.tensor(np.array(self._h5dataset[idx]), dtype=torch.float32)
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
