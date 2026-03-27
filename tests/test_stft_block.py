"""
Unit tests for StFTBlock to guard against behavioral regressions during refactoring.

Run before refactoring to generate reference outputs:
    pytest tests/test_stft_block.py --gen-ref

Then after refactoring, run normally to verify behavior is unchanged:
    pytest tests/test_stft_block.py
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from StFT_3D import StFTBlock  # noqa: E402

REF_DIR = os.path.join(os.path.dirname(__file__), "reference_outputs")

# ---------------------------------------------------------------------------
# Shared config helpers
# ---------------------------------------------------------------------------

COND_TIME = 2
FREQ_IN_CHANNELS = 4
PH, PW = 8, 8
OUT_CHANNEL = 2
MODES = (3, 3)
N, L = 2, 4          # batch size, num patches
GRID_SIZE = (2, 2)   # grid_size[0]*grid_size[1] == L
LIFT_CHANNEL = 8
DIM = 16             # must be divisible by NUM_HEADS
NUM_HEADS = 2
DEPTH = 1
MLP_DIM = 16
SEED = 42


def _in_channels(layer_indx: int) -> int:
    return COND_TIME * FREQ_IN_CHANNELS + layer_indx * (FREQ_IN_CHANNELS - 2)


def _in_dim(layer_indx: int) -> int:
    return _in_channels(layer_indx) * PH * PW


def _out_dim() -> int:
    return OUT_CHANNEL * PH * PW


def make_block(layer_indx: int, seed: int = SEED) -> StFTBlock:
    torch.manual_seed(seed)
    return StFTBlock(
        cond_time=COND_TIME,
        freq_in_channels=FREQ_IN_CHANNELS,
        in_dim=_in_dim(layer_indx),
        out_dim=_out_dim(),
        out_channel=OUT_CHANNEL,
        modes=MODES,
        lift_channel=LIFT_CHANNEL,
        dim=DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        mlp_dim=MLP_DIM,
        grid_size=GRID_SIZE,
        layer_indx=layer_indx,
    ).eval()


def make_input(layer_indx: int, seed: int = SEED) -> torch.Tensor:
    torch.manual_seed(seed + 1) # use different seed from make_block seed
    return torch.randn(N, L, _in_channels(layer_indx), PH, PW)


# ---------------------------------------------------------------------------
# 1. Output shape tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("layer_indx", [0, 1])
def test_output_shape(layer_indx):
    block = make_block(layer_indx)
    x = make_input(layer_indx)
    with torch.no_grad():
        out = block(x)
    expected_shape = (N, L, OUT_CHANNEL, PH, PW)
    assert out.shape == expected_shape, (
        f"layer_indx={layer_indx}: expected shape {expected_shape}, got {out.shape}"
    )


# ---------------------------------------------------------------------------
# 2. Forward-twice determinism tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("layer_indx", [0, 1])
def test_forward_determinism(layer_indx):
    block = make_block(layer_indx)
    x = make_input(layer_indx)
    with torch.no_grad():
        out1 = block(x)
        out2 = block(x)
    assert torch.equal(out1, out2), (
        f"layer_indx={layer_indx}: two forward passes with the same input produced "
        "different outputs — forward is not deterministic."
    )


# ---------------------------------------------------------------------------
# 3. Numerical regression tests
# ---------------------------------------------------------------------------

def _ref_path(layer_indx: int) -> str:
    return os.path.join(REF_DIR, f"stft_block_layer_indx{layer_indx}.pt")


@pytest.mark.parametrize("layer_indx", [0, 1])
def test_numerical_regression(layer_indx, gen_ref):
    block = make_block(layer_indx)
    x = make_input(layer_indx)

    with torch.no_grad():
        out = block(x)

    ref_path = _ref_path(layer_indx)

    if gen_ref:
        os.makedirs(REF_DIR, exist_ok=True)
        torch.save(out, ref_path)
        pytest.skip(f"Reference output saved to {ref_path}")

    if not os.path.exists(ref_path):
        pytest.skip(
            f"Reference output not found at {ref_path}. "
            "Run with --gen-ref to generate it before refactoring."
        )

    ref = torch.load(ref_path, weights_only=True)
    assert torch.allclose(out, ref, atol=1e-6), (
        f"layer_indx={layer_indx}: output differs from reference. "
        f"Max abs diff: {(out - ref).abs().max().item():.2e}"
    )


# ---------------------------------------------------------------------------
# 4. Gradient flow tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("layer_indx", [0, 1])
def test_gradient_flow(layer_indx):
    block = make_block(layer_indx).train()
    x = make_input(layer_indx).requires_grad_(True)

    out = block(x)
    loss = out.sum()
    loss.backward()

    # Input gradient should exist and be finite
    assert x.grad is not None, f"layer_indx={layer_indx}: no gradient on input"
    assert torch.isfinite(x.grad).all(), (
        f"layer_indx={layer_indx}: non-finite gradient on input"
    )

    # All trainable parameter gradients should exist and be finite
    for name, param in block.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, (
                f"layer_indx={layer_indx}: no gradient for param '{name}'"
            )
            assert torch.isfinite(param.grad).all(), (
                f"layer_indx={layer_indx}: non-finite gradient for param '{name}'"
            )


# ---------------------------------------------------------------------------
# 5. Batch-size independence tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("layer_indx", [0, 1])
def test_batch_independence(layer_indx):
    """Per-sample outputs must not depend on other samples in the batch."""
    block = make_block(layer_indx)
    x = make_input(layer_indx)  # shape (N, L, C, PH, PW), N=2

    with torch.no_grad():
        out_batch = block(x)

        # Run each sample individually (unsqueeze to keep batch dim = 1)
        out_0 = block(x[:1])
        out_1 = block(x[1:])

    assert torch.allclose(out_batch[:1], out_0, atol=1e-6), (
        f"layer_indx={layer_indx}: sample 0 output differs when run alone vs in a batch"
    )
    assert torch.allclose(out_batch[1:], out_1, atol=1e-6), (
        f"layer_indx={layer_indx}: sample 1 output differs when run alone vs in a batch"
    )
