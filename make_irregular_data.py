"""Convert the regular-grid plasma dataset to irregular-domain variants.

Strategy: "logical grid, physical irregularity".
We keep a random irregular subset of grid lines along ONE axis and pack the
kept lines into a dense tensor (so patch unfolding in StFT still works),
while storing the true physical coordinates of every kept line. The
coordinates are what carry the irregularity to the model (grid channels and,
later, the graph-based channel's adjacency).

Axis convention (matches stft.data_utils.get_grid):
  x-axis = H dimension (227 rows),  y-axis = W dimension (101 columns).

Variants produced (random irregular, endpoints always kept, nested masks so
the 25% mask is a subset of the 50% mask):
  plasma_irr_y50.pkl  (1, T, 6, 227, ~50)
  plasma_irr_y25.pkl  (1, T, 6, 227, ~25)
  plasma_irr_x50.pkl  (1, T, 6, ~114, 101)
  plasma_irr_x25.pkl  (1, T, 6, ~57, 101)

Each output pkl keeps the original keys (train/val/test/img_size/channels,
data cast to float32) and adds:
  coords_h, coords_w : physical coordinates in [0,1] of each kept line
  kept_indices       : indices kept from the original axis
  subsampled_axis    : 'x' or 'y'
  keep_ratio, seed   : provenance
"""

import pickle

import numpy as np

SEED = 0
RATIOS = (0.5, 0.25)
SRC = "plasma_(1).pkl"


def nested_masks(n, ratios, rng):
    """Sorted index masks, one per ratio, nested (smaller ⊂ larger).

    Endpoints 0 and n-1 are always included so the domain extent is
    preserved; interior indices come from one shared random permutation.
    """
    perm = rng.permutation(np.arange(1, n - 1))
    masks = {}
    for r in ratios:
        n_keep = max(2, int(round(n * r)))
        interior = perm[: n_keep - 2]
        masks[r] = np.sort(np.concatenate(([0], interior, [n - 1])))
    return masks


def main():
    with open(SRC, "rb") as f:
        d = pickle.load(f)
    H, W = d["img_size"]
    coords_h_full = np.linspace(0, 1, H)
    coords_w_full = np.linspace(0, 1, W)

    rng = np.random.default_rng(SEED)
    masks = {"x": nested_masks(H, RATIOS, rng), "y": nested_masks(W, RATIOS, rng)}

    for axis_name in ("y", "x"):
        for r in RATIOS:
            idx = masks[axis_name][r]
            ax = 3 if axis_name == "x" else 4  # (N, T, C, H, W)
            out = {
                k: np.take(d[k], idx, axis=ax).astype(np.float32)
                for k in ("train", "val", "test")
            }
            if axis_name == "x":
                out["img_size"] = (len(idx), W)
                out["coords_h"] = coords_h_full[idx]
                out["coords_w"] = coords_w_full
            else:
                out["img_size"] = (H, len(idx))
                out["coords_h"] = coords_h_full
                out["coords_w"] = coords_w_full[idx]
            out["channels"] = d["channels"]
            out["kept_indices"] = idx
            out["subsampled_axis"] = axis_name
            out["keep_ratio"] = r
            out["seed"] = SEED

            name = f"plasma_irr_{axis_name}{int(r * 100)}.pkl"
            with open(name, "wb") as f:
                pickle.dump(out, f)
            gaps = np.diff(out["coords_h" if axis_name == "x" else "coords_w"])
            print(
                f"{name}: train={out['train'].shape} img_size={out['img_size']} "
                f"kept={len(idx)}/{H if axis_name == 'x' else W} "
                f"gap min/med/max={gaps.min():.4f}/{np.median(gaps):.4f}/{gaps.max():.4f}"
            )


if __name__ == "__main__":
    main()
