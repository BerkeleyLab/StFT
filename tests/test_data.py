"""
Unit tests for stft/data.py: load_dataset, dataset classes, get_grid,
and TrainingDataset → StFT interface.
"""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

import h5py

from stft import StFT
from stft.data import (
    HDF5Dataset,
    TrainingDataset,
    SnapshotDataset,
    RolloutDataset,
    get_grid,
    load_dataset,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

N, T, C, H, W = 3, 10, 1, 8, 8
COND_TIME = 3
SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_array(seed=SEED, n=N, t=T, c=C, h=H, w=W):
    rng = np.random.default_rng(seed)
    return rng.random((n, t, c, h, w)).astype(np.float32)


def make_hdf5(tmp_path, seed=SEED, n=N, t=T, c=C, h=H, w=W,
              with_metadata=False, channels_attr=None, img_size_attr=None):
    """Write a dataset.h5 file to tmp_path. Returns the path to the file."""
    h5_path = tmp_path / "dataset.h5"
    with h5py.File(h5_path, "w") as f:
        arrays = {}
        for i, name in enumerate(("train", "val", "test")):
            arr = make_array(seed=seed + i, n=n, t=t, c=c, h=h, w=w)
            arrays[name] = arr
            f.create_dataset(name, data=arr, dtype=np.float32,
                             chunks=(1, t, c, h, w))

        train = arrays["train"].astype(np.float64)
        mean = train.mean(axis=(0, 1, 3, 4))   # shape (C,)
        std  = train.std(axis=(0, 1, 3, 4))

        stats = f.create_group("stats")
        stats.create_dataset("mean", data=mean, dtype=np.float64)
        stats.create_dataset("std",  data=std,  dtype=np.float64)

        f.attrs["channels"] = channels_attr if channels_attr is not None else c
        f.attrs["img_size"] = img_size_attr if img_size_attr is not None else [h, w]
        if with_metadata:
            f.attrs["extra_key"] = "extra_value"

    return h5_path


# ---------------------------------------------------------------------------
# 1. load_dataset
# ---------------------------------------------------------------------------


def test_load_dataset_returns_correct_shapes(tmp_path):
    h5_path = make_hdf5(tmp_path)
    ds = load_dataset(h5_path)

    assert isinstance(ds, HDF5Dataset)
    assert ds.channels == C
    assert ds.img_size == (H, W)
    assert ds.shapes["train"] == (N, T, C, H, W)
    assert ds.shapes["val"]   == (N, T, C, H, W)
    assert ds.shapes["test"]  == (N, T, C, H, W)


def test_load_dataset_directory_path_raises(tmp_path):
    make_hdf5(tmp_path)
    with pytest.raises(ValueError, match=r"\.h5 file"):
        load_dataset(tmp_path)


def test_load_dataset_missing_h5_file_raises(tmp_path):
    h5_path = tmp_path / "missing.h5"
    with pytest.raises(FileNotFoundError, match="missing.h5"):
        load_dataset(h5_path)


def test_load_dataset_missing_split_raises(tmp_path):
    h5_path = make_hdf5(tmp_path)
    with h5py.File(h5_path, "a") as f:
        del f["val"]
    with pytest.raises(KeyError, match="val"):
        load_dataset(h5_path)


def test_load_dataset_returns_stats(tmp_path):
    h5_path = make_hdf5(tmp_path)
    ds = load_dataset(h5_path)
    assert ds.mean.shape == (C,)
    assert ds.std.shape  == (C,)


def test_load_dataset_stats_accuracy(tmp_path):
    h5_path = make_hdf5(tmp_path)
    ds = load_dataset(h5_path)
    train = make_array(seed=SEED).astype(np.float64)
    expected_mean = train.mean(axis=(0, 1, 3, 4))
    expected_std  = train.std(axis=(0, 1, 3, 4))
    np.testing.assert_allclose(ds.mean, expected_mean, rtol=1e-5)
    np.testing.assert_allclose(ds.std,  expected_std,  rtol=1e-5)


def test_load_dataset_valid_metadata_passes(tmp_path):
    h5_path = make_hdf5(tmp_path, with_metadata=True)
    ds = load_dataset(h5_path)
    assert ds.channels == C
    assert ds.img_size == (H, W)


# ---------------------------------------------------------------------------
# 2. TrainingDataset
# ---------------------------------------------------------------------------


def test_training_dataset_len(tmp_path):
    h5_path = make_hdf5(tmp_path)
    ds = TrainingDataset(h5_path, cond_time=COND_TIME)
    assert len(ds) == N * (T - COND_TIME)


def test_training_dataset_item_shapes(tmp_path):
    h5_path = make_hdf5(tmp_path)
    ds = TrainingDataset(h5_path, cond_time=COND_TIME)
    x, y = ds[0]
    assert x.shape == (COND_TIME, C, H, W)
    assert y.shape == (C, H, W)


def test_training_dataset_item_dtype(tmp_path):
    h5_path = make_hdf5(tmp_path)
    ds = TrainingDataset(h5_path, cond_time=COND_TIME)
    x, y = ds[0]
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32


def test_training_dataset_no_normalization(tmp_path):
    h5_path = make_hdf5(tmp_path)
    ds = TrainingDataset(h5_path, cond_time=COND_TIME)
    n, t = ds.indices[0]
    x, y = ds[0]
    data = make_array()
    assert torch.allclose(x, torch.tensor(data[n, t : t + COND_TIME], dtype=torch.float32))
    assert torch.allclose(y, torch.tensor(data[n, t + COND_TIME], dtype=torch.float32))


def test_training_dataset_normalization(tmp_path):
    h5_path = make_hdf5(tmp_path)
    data = make_array()
    mean = torch.full((C, 1, 1), float(data.mean()))
    std  = torch.full((C, 1, 1), float(data.std()))
    ds = TrainingDataset(h5_path, cond_time=COND_TIME, mean=mean, std=std)

    n, t = ds.indices[0]
    raw_x = torch.tensor(data[n, t : t + COND_TIME], dtype=torch.float32)
    raw_y = torch.tensor(data[n, t + COND_TIME], dtype=torch.float32)
    x, y = ds[0]
    assert torch.allclose(x, (raw_x - mean) / std, atol=1e-6)
    assert torch.allclose(y, (raw_y - mean) / std, atol=1e-6)


def test_training_dataset_lazy_open(tmp_path):
    h5_path = make_hdf5(tmp_path)
    ds = TrainingDataset(h5_path, cond_time=COND_TIME)
    assert ds._h5file is None
    _ = ds[0]
    assert ds._h5file is not None


def test_training_dataset_dataloader(tmp_path):
    h5_path = make_hdf5(tmp_path)
    ds = TrainingDataset(h5_path, cond_time=COND_TIME)
    loader = DataLoader(ds, batch_size=2)
    x_batch, y_batch = next(iter(loader))
    assert x_batch.shape == (2, COND_TIME, C, H, W)
    assert y_batch.shape == (2, C, H, W)
    assert x_batch.dtype == torch.float32
    assert y_batch.dtype == torch.float32


def test_training_dataset_worker_safety(tmp_path):
    h5_path = make_hdf5(tmp_path)
    ds = TrainingDataset(h5_path, cond_time=COND_TIME)
    loader = DataLoader(ds, batch_size=2, num_workers=2)
    batches = list(loader)
    assert len(batches) > 0


# ---------------------------------------------------------------------------
# 2b. SnapshotDataset
# ---------------------------------------------------------------------------


SNAPSHOT_LEN = 5


def test_snapshot_dataset_len(tmp_path):
    h5_path = make_hdf5(tmp_path)
    ds = SnapshotDataset(h5_path, snapshot_length=SNAPSHOT_LEN)
    assert len(ds) == N * (T - SNAPSHOT_LEN + 1)


def test_snapshot_dataset_item_shape(tmp_path):
    h5_path = make_hdf5(tmp_path)
    ds = SnapshotDataset(h5_path, snapshot_length=SNAPSHOT_LEN)
    snap = ds[0]
    assert snap.shape == (SNAPSHOT_LEN, C, H, W)


def test_snapshot_dataset_item_dtype(tmp_path):
    h5_path = make_hdf5(tmp_path)
    ds = SnapshotDataset(h5_path, snapshot_length=SNAPSHOT_LEN)
    assert ds[0].dtype == torch.float32


def test_snapshot_dataset_window_lies_in_trajectory(tmp_path):
    h5_path = make_hdf5(tmp_path)
    ds = SnapshotDataset(h5_path, snapshot_length=SNAPSHOT_LEN)
    idx = T - SNAPSHOT_LEN + 1
    snap = ds[idx]
    data = make_array(seed=SEED)
    ref = torch.tensor(data[1, :SNAPSHOT_LEN], dtype=torch.float32)
    assert torch.allclose(snap, ref)


def test_snapshot_dataset_normalization(tmp_path):
    h5_path = make_hdf5(tmp_path)
    data = make_array()
    mean = torch.full((C, 1, 1), float(data.mean()))
    std  = torch.full((C, 1, 1), float(data.std()))

    ds_raw  = SnapshotDataset(h5_path, snapshot_length=SNAPSHOT_LEN)
    ds_norm = SnapshotDataset(h5_path, snapshot_length=SNAPSHOT_LEN, mean=mean, std=std)

    raw  = ds_raw[0]
    norm = ds_norm[0]
    assert torch.allclose(norm, (raw - mean) / std, atol=1e-6)


def test_snapshot_dataset_too_long_raises(tmp_path):
    h5_path = make_hdf5(tmp_path)
    with pytest.raises(ValueError, match="snapshot_length"):
        SnapshotDataset(h5_path, snapshot_length=T + 1)


def test_snapshot_dataset_dataloader(tmp_path):
    h5_path = make_hdf5(tmp_path)
    ds = SnapshotDataset(h5_path, snapshot_length=SNAPSHOT_LEN)
    loader = DataLoader(ds, batch_size=2)
    batch = next(iter(loader))
    assert batch.shape == (2, SNAPSHOT_LEN, C, H, W)
    assert batch.dtype == torch.float32


# ---------------------------------------------------------------------------
# 3. RolloutDataset
# ---------------------------------------------------------------------------


def test_rollout_dataset_len(tmp_path):
    h5_path = make_hdf5(tmp_path)
    ds = RolloutDataset(h5_path, split="test")
    assert len(ds) == N


def test_rollout_dataset_item_shape(tmp_path):
    h5_path = make_hdf5(tmp_path)
    ds = RolloutDataset(h5_path, split="test")
    assert ds[0].shape == (T, C, H, W)


def test_rollout_dataset_item_dtype(tmp_path):
    h5_path = make_hdf5(tmp_path)
    ds = RolloutDataset(h5_path, split="test")
    assert ds[0].dtype == torch.float32


def test_rollout_dataset_no_normalization(tmp_path):
    h5_path = make_hdf5(tmp_path)
    ds = RolloutDataset(h5_path, split="test")
    data = make_array(seed=SEED + 2)  # test split uses seed+2
    assert torch.allclose(ds[0], torch.tensor(data[0], dtype=torch.float32))


def test_rollout_dataset_normalization(tmp_path):
    h5_path = make_hdf5(tmp_path)
    data = make_array(seed=SEED + 2)
    mean = torch.full((C, 1, 1), float(data.mean()))
    std  = torch.full((C, 1, 1), float(data.std()))
    ds = RolloutDataset(h5_path, split="test", mean=mean, std=std)

    raw = torch.tensor(data[0], dtype=torch.float32)
    assert torch.allclose(ds[0], (raw - mean) / std, atol=1e-6)


def test_rollout_dataset_dataloader(tmp_path):
    h5_path = make_hdf5(tmp_path)
    ds = RolloutDataset(h5_path, split="test")
    loader = DataLoader(ds, batch_size=2)
    batch = next(iter(loader))
    assert batch.shape == (2, T, C, H, W)
    assert batch.dtype == torch.float32


# ---------------------------------------------------------------------------
# 4. get_grid
# ---------------------------------------------------------------------------


def test_get_grid_shape():
    grid = get_grid(H, W)
    assert grid.shape == (2, H, W)


def test_get_grid_range():
    grid = get_grid(H, W)
    assert grid.min() >= 0.0
    assert grid.max() <= 1.0


def test_get_grid_dtype():
    grid = get_grid(H, W)
    assert grid.dtype == torch.float32


# ---------------------------------------------------------------------------
# 5. TrainingDataset → StFT interface
# ---------------------------------------------------------------------------

_COND_TIME     = 2
_NUM_IN_STATES = 1
_NUM_VARS      = _NUM_IN_STATES + 2
_IN_CHANNELS   = _NUM_VARS * _COND_TIME
_OUT_CHANNELS  = 1
_IMG_H = _IMG_W = 16
_PATCH_SIZES   = ((8, 8), (4, 4))
_OVERLAPS      = ((1, 1), (1, 1))
_MODES         = ((3, 3), (3, 3))
_VIT_DEPTH     = (1, 1)
_LIFT_CHANNEL  = 8
_DIM           = 16
_NUM_HEADS     = 2
_MLP_DIM       = 16


def test_training_dataset_stft_interface(tmp_path):
    """A DataLoader batch from TrainingDataset feeds into StFT without shape errors."""
    h5_path = make_hdf5(tmp_path, n=4, t=10, c=_NUM_IN_STATES,
                        h=_IMG_H, w=_IMG_W)
    ds = TrainingDataset(h5_path, cond_time=_COND_TIME)
    loader = DataLoader(ds, batch_size=2)
    x, _ = next(iter(loader))
    grid = get_grid(_IMG_H, _IMG_W)

    torch.manual_seed(SEED)
    model = StFT(
        cond_time=_COND_TIME,
        num_vars=_NUM_VARS,
        patch_sizes=_PATCH_SIZES,
        overlaps=_OVERLAPS,
        in_channels=_IN_CHANNELS,
        out_channels=_OUT_CHANNELS,
        modes=_MODES,
        img_size=(_IMG_H, _IMG_W),
        lift_channel=_LIFT_CHANNEL,
        dim=_DIM,
        vit_depth=_VIT_DEPTH,
        num_heads=_NUM_HEADS,
        mlp_dim=_MLP_DIM,
    ).eval()

    with torch.no_grad():
        outputs = model(x, grid)

    assert len(outputs) == len(_PATCH_SIZES)
    B = x.shape[0]
    for d, out in enumerate(outputs):
        assert out.shape == (B, _OUT_CHANNELS, _IMG_H, _IMG_W), (
            f"depth={d}: expected ({B}, {_OUT_CHANNELS}, {_IMG_H}, {_IMG_W}), got {out.shape}"
        )
