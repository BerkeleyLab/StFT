"""
Unit tests for StFT to guard against behavioral regressions during refactoring.

Run before refactoring to generate reference outputs:
    pytest tests/test_stft.py --gen-ref

Then after refactoring, run normally to verify behavior is unchanged:
    pytest tests/test_stft.py
"""

import os

import pytest
import torch

from stft import StFT

REF_DIR = os.path.join(os.path.dirname(__file__), "reference_outputs")

# ---------------------------------------------------------------------------
# Shared config helpers
# ---------------------------------------------------------------------------

COND_TIME     = 2
NUM_IN_STATES = 1                          # data channels fed to StFT (excluding grid)
NUM_VARS      = NUM_IN_STATES + 2          # freq_in_channels in StFTBlock (includes 2 grid coords)
IN_CHANNELS   = NUM_VARS * COND_TIME       # total channels after grid concat + time-fold
OUT_CHANNELS  = 1
IMG_H, IMG_W  = 16, 16
PATCH_SIZES   = ((8, 8), (4, 4))
OVERLAPS      = ((1, 1), (1, 1))
MODES         = ((3, 3), (3, 3))
VIT_DEPTH     = (1, 1)
LIFT_CHANNEL  = 8
DIM           = 16                         # must be divisible by NUM_HEADS
NUM_HEADS     = 2
MLP_DIM       = 16
B             = 2                          # batch size
SEED          = 42


def make_model(seed: int = SEED) -> StFT:
    torch.manual_seed(seed)
    return StFT(
        cond_time=COND_TIME,
        num_vars=NUM_VARS,
        patch_sizes=PATCH_SIZES,
        overlaps=OVERLAPS,
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        modes=MODES,
        img_size=(IMG_H, IMG_W),
        lift_channel=LIFT_CHANNEL,
        dim=DIM,
        vit_depth=VIT_DEPTH,
        num_heads=NUM_HEADS,
        mlp_dim=MLP_DIM,
    ).eval()


def make_inputs(seed: int = SEED) -> tuple:
    """Returns (x, grid) with shapes (B, COND_TIME, NUM_IN_STATES, H, W) and (2, H, W)."""
    torch.manual_seed(seed + 1)  # use different seed from make_model
    x = torch.randn(B, COND_TIME, NUM_IN_STATES, IMG_H, IMG_W)
    xs = torch.linspace(0, 1, IMG_W).unsqueeze(0).expand(IMG_H, -1)
    ys = torch.linspace(0, 1, IMG_H).unsqueeze(-1).expand(-1, IMG_W)
    grid = torch.stack([xs, ys], dim=0)  # (2, H, W)
    return x, grid


# ---------------------------------------------------------------------------
# 1. Output structure tests
# ---------------------------------------------------------------------------

def test_output_structure():
    model = make_model()
    x, grid = make_inputs()
    with torch.no_grad():
        outputs = model(x, grid)

    assert len(outputs) == len(PATCH_SIZES), (
        f"Expected {len(PATCH_SIZES)} output tensors, got {len(outputs)}"
    )
    expected_shape = (B, OUT_CHANNELS, IMG_H, IMG_W)
    for d, out in enumerate(outputs):
        assert out.shape == expected_shape, (
            f"depth={d}: expected shape {expected_shape}, got {out.shape}"
        )


# ---------------------------------------------------------------------------
# 2. Forward determinism tests
# ---------------------------------------------------------------------------

def test_forward_determinism():
    model = make_model()
    x, grid = make_inputs()
    with torch.no_grad():
        out1 = model(x, grid)
        out2 = model(x, grid)

    for d in range(len(PATCH_SIZES)):
        assert torch.equal(out1[d], out2[d]), (
            f"depth={d}: two forward passes with the same input produced "
            "different outputs — forward is not deterministic."
        )


# ---------------------------------------------------------------------------
# 3. Numerical regression tests
# ---------------------------------------------------------------------------

def _ref_path(depth: int) -> str:
    return os.path.join(REF_DIR, f"stft_depth{depth}.pt")


def test_numerical_regression(gen_ref):
    model = make_model()
    x, grid = make_inputs()

    with torch.no_grad():
        outputs = model(x, grid)

    if gen_ref:
        os.makedirs(REF_DIR, exist_ok=True)
        for d, out in enumerate(outputs):
            torch.save(out, _ref_path(d))
        pytest.skip(f"Reference outputs saved to {REF_DIR}")

    missing = [d for d in range(len(PATCH_SIZES)) if not os.path.exists(_ref_path(d))]
    if missing:
        pytest.skip(
            f"Reference outputs missing for depths {missing}. "
            "Run with --gen-ref to generate them before refactoring."
        )

    for d, out in enumerate(outputs):
        ref = torch.load(_ref_path(d), weights_only=True)
        assert torch.allclose(out, ref, atol=1e-6), (
            f"depth={d}: output differs from reference. "
            f"Max abs diff: {(out - ref).abs().max().item():.2e}"
        )


# ---------------------------------------------------------------------------
# 4. Gradient flow tests
# ---------------------------------------------------------------------------

def test_gradient_flow():
    model = make_model().train()
    x, grid = make_inputs()
    x = x.requires_grad_(True)

    outputs = model(x, grid)
    loss = sum(o.sum() for o in outputs)
    loss.backward()

    # Input gradient should exist and be finite
    assert x.grad is not None, "no gradient on input x"
    assert torch.isfinite(x.grad).all(), "non-finite gradient on input x"

    # All trainable parameter gradients should exist and be finite
    # Note: inter-block concat uses .detach(), so gradients only flow through
    # the currently active block — this is expected behavior, not a bug.
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"no gradient for param '{name}'"
            assert torch.isfinite(param.grad).all(), (
                f"non-finite gradient for param '{name}'"
            )


# ---------------------------------------------------------------------------
# 5. Batch independence tests
# ---------------------------------------------------------------------------

def test_batch_independence():
    """Per-sample outputs must not depend on other samples in the batch."""
    model = make_model()
    x, grid = make_inputs()  # shape (B, COND_TIME, NUM_IN_STATES, H, W), B=2

    with torch.no_grad():
        out_batch = model(x, grid)
        out_0 = model(x[:1], grid)
        out_1 = model(x[1:], grid)

    for d in range(len(PATCH_SIZES)):
        assert torch.allclose(out_batch[d][:1], out_0[d], atol=1e-6), (
            f"depth={d}: sample 0 output differs when run alone vs in a batch"
        )
        assert torch.allclose(out_batch[d][1:], out_1[d], atol=1e-6), (
            f"depth={d}: sample 1 output differs when run alone vs in a batch"
        )


# ---------------------------------------------------------------------------
# 6. Non-square patches and asymmetric overlap variants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("patch_sizes,overlaps,modes", [
    pytest.param(
        ((8, 4), (4, 2)), ((1, 1), (1, 1)), ((3, 2), (2, 1)),
        id="non-square-patches",
    ),
    pytest.param(
        ((8, 8), (4, 4)), ((2, 1), (1, 2)), ((3, 3), (3, 3)),
        id="asymmetric-overlaps",
    ),
    pytest.param(
        ((8, 4), (4, 2)), ((2, 1), (1, 1)), ((3, 2), (2, 1)),
        id="non-square-and-asymmetric",
    ),
])
def test_output_shape_variants(patch_sizes, overlaps, modes):
    torch.manual_seed(SEED)
    model = StFT(
        cond_time=COND_TIME,
        num_vars=NUM_VARS,
        patch_sizes=patch_sizes,
        overlaps=overlaps,
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        modes=modes,
        img_size=(IMG_H, IMG_W),
        lift_channel=LIFT_CHANNEL,
        dim=DIM,
        vit_depth=VIT_DEPTH,
        num_heads=NUM_HEADS,
        mlp_dim=MLP_DIM,
    ).eval()
    x, grid = make_inputs()
    with torch.no_grad():
        outputs = model(x, grid)

    assert len(outputs) == len(patch_sizes)
    expected_shape = (B, OUT_CHANNELS, IMG_H, IMG_W)
    for d, out in enumerate(outputs):
        assert out.shape == expected_shape, (
            f"depth={d}: expected shape {expected_shape}, got {out.shape}"
        )
        assert torch.isfinite(out).all(), (
            f"depth={d}: output contains non-finite values"
        )
