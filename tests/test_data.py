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
from stft.data import Dataset5D, TemporalDataset, get_grid, load_dataset

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

N, T, C, H, W = 3, 10, 1, 8, 8
SNAPSHOT_LENGTH = 4
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
# 2. TemporalDataset
# ---------------------------------------------------------------------------


def test_temporal_dataset_len():
    data = make_array()
    ds = TemporalDataset(data, snapshot_length=SNAPSHOT_LENGTH)
    assert len(ds) == N * (T - SNAPSHOT_LENGTH + 1)


def test_temporal_dataset_item_shape():
    data = make_array()
    ds = TemporalDataset(data, snapshot_length=SNAPSHOT_LENGTH)
    assert ds[0].shape == (SNAPSHOT_LENGTH, C, H, W)


def test_temporal_dataset_item_dtype():
    data = make_array()
    ds = TemporalDataset(data, snapshot_length=SNAPSHOT_LENGTH)
    assert ds[0].dtype == torch.float32


def test_temporal_dataset_no_normalization():
    data = make_array()
    ds = TemporalDataset(data, snapshot_length=SNAPSHOT_LENGTH)
    n, t = ds.indices[0]
    expected = torch.tensor(
        data[n, t : t + SNAPSHOT_LENGTH], dtype=torch.float32
    )
    assert torch.allclose(ds[0], expected)


def test_temporal_dataset_normalization():
    data = make_array()
    mean = torch.tensor(data.mean(), dtype=torch.float32)
    std = torch.tensor(data.std(), dtype=torch.float32)
    ds = TemporalDataset(data, snapshot_length=SNAPSHOT_LENGTH, mean=mean, std=std)

    n, t = ds.indices[0]
    raw = torch.tensor(data[n, t : t + SNAPSHOT_LENGTH], dtype=torch.float32)
    expected = (raw - mean) / std
    assert torch.allclose(ds[0], expected, atol=1e-6)


def test_temporal_dataset_snapshot_equals_T():
    data = make_array()
    ds = TemporalDataset(data, snapshot_length=T)
    assert len(ds) == N
    assert ds[0].shape == (T, C, H, W)


def test_temporal_dataset_dataloader():
    data = make_array()
    ds = TemporalDataset(data, snapshot_length=SNAPSHOT_LENGTH)
    loader = DataLoader(ds, batch_size=2)
    batch = next(iter(loader))
    assert batch.shape == (2, SNAPSHOT_LENGTH, C, H, W)
    assert batch.dtype == torch.float32


# ---------------------------------------------------------------------------
# 3. get_grid
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
# 4. TemporalDataset → StFT interface
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


def test_temporal_dataset_stft_interface():
    """A DataLoader batch from TemporalDataset feeds into StFT without shape errors."""
    data = make_array(n=4, t=10, c=_NUM_IN_STATES, h=_IMG_H, w=_IMG_W)
    ds = TemporalDataset(data, snapshot_length=_COND_TIME + 2)
    loader = DataLoader(ds, batch_size=2)
    batch = next(iter(loader))  # (B, snapshot_length, C, H, W)

    x = batch[:, :_COND_TIME]  # (B, cond_time, NUM_IN_STATES, H, W)
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
