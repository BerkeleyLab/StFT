"""Convert a directory of train/val/test .npy files to a single HDF5 file.

Usage:
    python scripts/convert_npy_to_hdf5.py --input /path/to/data --output /path/to/dataset.h5

Stats (mean, std per channel) are computed from the training split.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def convert(input_dir: Path, output_path: Path, compression: str | None,
            compression_level: int):
    splits_arr = {}
    for name in ("train", "val", "test"):
        path = input_dir / f"{name}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        arr = np.load(path)
        if arr.ndim != 5:
            raise ValueError(f"{name}.npy has {arr.ndim} dims, expected 5 (N,T,C,H,W)")
        splits_arr[name] = arr

    channels = splits_arr["train"].shape[2]
    img_size = (splits_arr["train"].shape[3], splits_arr["train"].shape[4])
    for name, arr in splits_arr.items():
        if arr.shape[2] != channels:
            raise ValueError(f"Channel mismatch: train={channels}, {name}={arr.shape[2]}")
        if (arr.shape[3], arr.shape[4]) != img_size:
            raise ValueError(f"Spatial mismatch: train={img_size}, {name}={arr.shape[3:5]}")

    metadata = {}
    metadata_path = input_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

    compress_kwargs = {}
    if compression and compression != "none":
        compress_kwargs["compression"] = compression
        if compression == "gzip":
            compress_kwargs["compression_opts"] = compression_level

    train = splits_arr["train"].astype(np.float64)
    # mean/std per channel over (N, T, H, W)
    mean = train.mean(axis=(0, 1, 3, 4))
    std = train.std(axis=(0, 1, 3, 4))
    print(f"  mean={mean}, std={std}")

    print(f"Writing {output_path} ...")
    with h5py.File(output_path, "w") as hf:
        for name, arr in splits_arr.items():
            N, T, C, H, W = arr.shape
            hf.create_dataset(name, data=arr.astype(np.float32),
                              chunks=(1, T, C, H, W), **compress_kwargs)
            print(f"  {name}: {N} samples written")

        stats = hf.create_group("stats")
        stats.create_dataset("mean", data=mean, dtype=np.float64)
        stats.create_dataset("std",  data=std,  dtype=np.float64)

        hf.attrs["channels"] = channels
        hf.attrs["img_size"] = list(img_size)
        for k, v in metadata.items():
            if k not in ("channels", "img_size"):
                hf.attrs[k] = json.dumps(v) if isinstance(v, (dict, list)) else v

    print(f"Done. Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert .npy dataset to HDF5")
    parser.add_argument("--input", required=True, type=Path,
                        help="Directory containing train.npy, val.npy, test.npy")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output .h5 path (default: <input>/dataset.h5)")
    parser.add_argument("--compression", default="gzip",
                        choices=["gzip", "lzf", "none"],
                        help="HDF5 compression (default: gzip)")
    parser.add_argument("--compression-level", type=int, default=4,
                        help="gzip compression level 0-9 (default: 4)")
    args = parser.parse_args()

    input_dir = args.input.resolve()
    output_path = args.output.resolve() if args.output else input_dir / "dataset.h5"
    compression = None if args.compression == "none" else args.compression

    convert(input_dir, output_path, compression, args.compression_level)


if __name__ == "__main__":
    main()
