"""
Unit tests for stft/data.py: load_dataset, TemporalDataset, get_grid,
and TemporalDataset → StFT interface.
"""

import json

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from stft import StFT
from stft.data import (
    Dataset5D,
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


def make_dataset_dir(tmp_path, with_metadata=False, channels=C, img_size=(H, W)):
    """Write train/val/test .npy files (and optionally metadata.json) to tmp_path."""
    for i, name in enumerate(("train", "val", "test")):
        np.save(tmp_path / f"{name}.npy", make_array(seed=SEED + i))
    if with_metadata:
        (tmp_path / "metadata.json").write_text(
            json.dumps({"channels": channels, "img_size": list(img_size)})
        )
    return tmp_path


# ---------------------------------------------------------------------------
# 1. load_dataset
# ---------------------------------------------------------------------------


def test_load_dataset_returns_correct_shapes(tmp_path):
    make_dataset_dir(tmp_path)
    ds = load_dataset(tmp_path)

    assert isinstance(ds, Dataset5D)
    assert ds.channels == C
    assert ds.img_size == (H, W)
    assert ds.train.shape == (N, T, C, H, W)
    assert ds.val.shape == (N, T, C, H, W)
    assert ds.test.shape == (N, T, C, H, W)


def test_load_dataset_missing_split_raises(tmp_path):
    make_dataset_dir(tmp_path)
    (tmp_path / "val.npy").unlink()
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path)


def test_load_dataset_wrong_ndim_raises(tmp_path):
    make_dataset_dir(tmp_path)
    # Overwrite train.npy with a 4D array
    np.save(tmp_path / "train.npy", np.zeros((N, T, C, H), dtype=np.float32))
    with pytest.raises(ValueError, match="5"):
        load_dataset(tmp_path)


def test_load_dataset_channel_mismatch_raises(tmp_path):
    make_dataset_dir(tmp_path)
    # Overwrite val.npy with different channel count
    np.save(tmp_path / "val.npy", make_array(c=C + 1))
    with pytest.raises(ValueError, match="Channel mismatch"):
        load_dataset(tmp_path)


def test_load_dataset_spatial_mismatch_raises(tmp_path):
    make_dataset_dir(tmp_path)
    np.save(tmp_path / "val.npy", make_array(h=H + 4))
    with pytest.raises(ValueError, match="Spatial size mismatch"):
        load_dataset(tmp_path)


def test_load_dataset_valid_metadata_passes(tmp_path):
    make_dataset_dir(tmp_path, with_metadata=True)
    ds = load_dataset(tmp_path)
    assert ds.channels == C
    assert ds.img_size == (H, W)


def test_load_dataset_metadata_channel_mismatch_raises(tmp_path):
    make_dataset_dir(tmp_path, with_metadata=True, channels=C + 1)
    with pytest.raises(ValueError, match="metadata.json channels"):
        load_dataset(tmp_path)


def test_load_dataset_metadata_img_size_mismatch_raises(tmp_path):
    make_dataset_dir(tmp_path, with_metadata=True, img_size=(H + 4, W))
    with pytest.raises(ValueError, match="metadata.json img_size"):
        load_dataset(tmp_path)


# ---------------------------------------------------------------------------
# 2. TrainingDataset
# ---------------------------------------------------------------------------


def test_training_dataset_len():
    data = make_array()
    ds = TrainingDataset(data, cond_time=COND_TIME)
    assert len(ds) == N * (T - COND_TIME)


def test_training_dataset_item_shapes():
    data = make_array()
    ds = TrainingDataset(data, cond_time=COND_TIME)
    x, y = ds[0]
    assert x.shape == (COND_TIME, C, H, W)
    assert y.shape == (C, H, W)


def test_training_dataset_item_dtype():
    data = make_array()
    ds = TrainingDataset(data, cond_time=COND_TIME)
    x, y = ds[0]
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32


def test_training_dataset_no_normalization():
    data = make_array()
    ds = TrainingDataset(data, cond_time=COND_TIME)
    n, t = ds.indices[0]
    x, y = ds[0]
    assert torch.allclose(x, torch.tensor(data[n, t : t + COND_TIME], dtype=torch.float32))
    assert torch.allclose(y, torch.tensor(data[n, t + COND_TIME], dtype=torch.float32))


def test_training_dataset_normalization():
    data = make_array()
    mean = torch.full((C, 1, 1), float(data.mean()))
    std = torch.full((C, 1, 1), float(data.std()))
    ds = TrainingDataset(data, cond_time=COND_TIME, mean=mean, std=std)

    n, t = ds.indices[0]
    raw_x = torch.tensor(data[n, t : t + COND_TIME], dtype=torch.float32)
    raw_y = torch.tensor(data[n, t + COND_TIME], dtype=torch.float32)
    x, y = ds[0]
    assert torch.allclose(x, (raw_x - mean) / std, atol=1e-6)
    assert torch.allclose(y, (raw_y - mean) / std, atol=1e-6)


def test_training_dataset_dataloader():
    data = make_array()
    ds = TrainingDataset(data, cond_time=COND_TIME)
    loader = DataLoader(ds, batch_size=2)
    x_batch, y_batch = next(iter(loader))
    assert x_batch.shape == (2, COND_TIME, C, H, W)
    assert y_batch.shape == (2, C, H, W)
    assert x_batch.dtype == torch.float32
    assert y_batch.dtype == torch.float32


# ---------------------------------------------------------------------------
# 2b. SnapshotDataset
# ---------------------------------------------------------------------------


SNAPSHOT_LEN = 5


def test_snapshot_dataset_len():
    data = make_array()
    ds = SnapshotDataset(data, snapshot_length=SNAPSHOT_LEN)
    assert len(ds) == N


def test_snapshot_dataset_item_shape():
    data = make_array()
    ds = SnapshotDataset(data, snapshot_length=SNAPSHOT_LEN)
    snap = ds[0]
    assert snap.shape == (SNAPSHOT_LEN, C, H, W)


def test_snapshot_dataset_item_dtype():
    data = make_array()
    ds = SnapshotDataset(data, snapshot_length=SNAPSHOT_LEN)
    assert ds[0].dtype == torch.float32


def test_snapshot_dataset_window_lies_in_trajectory():
    import random as _random

    data = make_array()
    ds = SnapshotDataset(data, snapshot_length=SNAPSHOT_LEN)
    _random.seed(SEED)
    snap = ds[1]
    # The returned window must equal some contiguous slice of trajectory 1.
    matched = False
    for start in range(T - SNAPSHOT_LEN + 1):
        ref = torch.tensor(data[1, start : start + SNAPSHOT_LEN], dtype=torch.float32)
        if torch.allclose(snap, ref):
            matched = True
            break
    assert matched, "SnapshotDataset returned a window not present in trajectory"


def test_snapshot_dataset_normalization():
    import random as _random

    data = make_array()
    mean = torch.full((C, 1, 1), float(data.mean()))
    std = torch.full((C, 1, 1), float(data.std()))

    ds_raw = SnapshotDataset(data, snapshot_length=SNAPSHOT_LEN)
    ds_norm = SnapshotDataset(data, snapshot_length=SNAPSHOT_LEN, mean=mean, std=std)

    _random.seed(SEED)
    raw = ds_raw[0]
    _random.seed(SEED)
    norm = ds_norm[0]
    assert torch.allclose(norm, (raw - mean) / std, atol=1e-6)


def test_snapshot_dataset_too_long_raises():
    data = make_array()
    with pytest.raises(ValueError, match="snapshot_length"):
        SnapshotDataset(data, snapshot_length=T + 1)


def test_snapshot_dataset_dataloader():
    data = make_array()
    ds = SnapshotDataset(data, snapshot_length=SNAPSHOT_LEN)
    loader = DataLoader(ds, batch_size=2)
    batch = next(iter(loader))
    assert batch.shape == (2, SNAPSHOT_LEN, C, H, W)
    assert batch.dtype == torch.float32


# ---------------------------------------------------------------------------
# 3. RolloutDataset
# ---------------------------------------------------------------------------


def test_rollout_dataset_len():
    data = make_array()
    ds = RolloutDataset(data)
    assert len(ds) == N


def test_rollout_dataset_item_shape():
    data = make_array()
    ds = RolloutDataset(data)
    assert ds[0].shape == (T, C, H, W)


def test_rollout_dataset_item_dtype():
    data = make_array()
    ds = RolloutDataset(data)
    assert ds[0].dtype == torch.float32


def test_rollout_dataset_no_normalization():
    data = make_array()
    ds = RolloutDataset(data)
    assert torch.allclose(ds[0], torch.tensor(data[0], dtype=torch.float32))


def test_rollout_dataset_normalization():
    data = make_array()
    mean = torch.full((C, 1, 1), float(data.mean()))
    std = torch.full((C, 1, 1), float(data.std()))
    ds = RolloutDataset(data, mean=mean, std=std)

    raw = torch.tensor(data[0], dtype=torch.float32)
    assert torch.allclose(ds[0], (raw - mean) / std, atol=1e-6)


def test_rollout_dataset_dataloader():
    data = make_array()
    ds = RolloutDataset(data)
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

# Match the minimal config from test_stft.py, but use H=W=16 to satisfy patch sizes.
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


def test_training_dataset_stft_interface():
    """A DataLoader batch from TrainingDataset feeds into StFT without shape errors."""
    data = make_array(n=4, t=10, c=_NUM_IN_STATES, h=_IMG_H, w=_IMG_W)
    ds = TrainingDataset(data, cond_time=_COND_TIME)
    loader = DataLoader(ds, batch_size=2)
    x, _ = next(iter(loader))  # x: (B, cond_time, NUM_IN_STATES, H, W)
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
