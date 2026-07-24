"""Random-grid-masking variants of the plasma dataset (for Colab runs).

Unlike make_irregular_data.py (which drops whole rows/columns and packs the
survivors into a smaller dense grid), this drops a random scatter of individual
grid points across the full 2D plane. Scattered holes can't form a dense
rectangle, so the grid stays the FULL 227x101 and dropped points become missing
values (set to 0). One fixed spatial mask per seed, shared across every frame
and every split (a fixed "sensor layout"). No model change is needed: the model
trains on the regular 227x101 grid exactly as for `reg`.

Produces (keep_frac of points kept at random, seeds 0 and 1):
  plasma_mask_s0.pkl , plasma_mask_s1.pkl   (1, T, 6, 227, 101)

Each keeps the original keys (train/val/test float32, img_size, channels) plus:
  mask       : (H, W) bool, True = observed/kept
  keep_frac  : fraction of grid points kept
  seed       : RNG seed
No coords_* keys are stored, so train_graph.py uses the regular get_grid(H, W).
"""

import pickle

import numpy as np

KEEP_FRAC = 0.5
SEEDS = (0, 1)
SRC = "plasma_(1).pkl"


def main():
    with open(SRC, "rb") as f:
        d = pickle.load(f)
    H, W = d["img_size"]

    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        mask = rng.random((H, W)) < KEEP_FRAC  # True = kept/observed
        m = mask.astype(np.float32)  # broadcast multiplier over (N,T,C,H,W)

        out = {
            k: (np.asarray(d[k], dtype=np.float32) * m).astype(np.float32)
            for k in ("train", "val", "test")
        }
        out["img_size"] = (H, W)
        out["channels"] = d["channels"]
        out["mask"] = mask
        out["keep_frac"] = KEEP_FRAC
        out["seed"] = seed

        name = f"plasma_mask_s{seed}.pkl"
        with open(name, "wb") as f:
            pickle.dump(out, f)
        print(
            f"{name}: train={out['train'].shape} kept={mask.sum()}/{H * W} "
            f"({mask.mean() * 100:.1f}%) seed={seed}"
        )


if __name__ == "__main__":
    main()
