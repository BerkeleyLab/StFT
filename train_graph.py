"""Train StFT with the graph-based channel (StFTG) on regular or irregular data.

Usage (from the StFT dir, inside the pinned venv, e.g. `uv run --no-sync`):
    python train_graph.py y50            # graph channel ON  (default)
    python train_graph.py y50 --no-graph # baseline: same patching, no graph
    python train_graph.py reg            # original regular grid + graph channel

Variants: reg, y50, y25, x50, x25 (see make_irregular_data.py). Patch sizes,
Fourier modes and graph sub-patch sizes are adapted per variant so that patch
dims stay compatible with the shrunken axis and every patch yields an 8x8 = 64
node graph. Resume files and Ray run names are tagged per (variant, graph) so
installment-style chunked training works exactly like train_resume.py.
"""

import argparse
import os
import pickle
import tempfile

import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from stft.StFT_3D_graph import StFTG
from stft.data_utils import LpLoss, get_grid, get_grid_from_coords, TemporalDataset
from ray.train import Checkpoint
from ray import train, tune
from ray.train import RunConfig, CheckpointConfig

# Per-variant geometry: (dataset file, patch_sizes, modes, graph_patch_sizes).
# modes[level][1] must be <= patch_w // 2 + 1 (rfft width), hence y25's (8, 4).
VARIANTS = {
    "reg": ("plasma_(1).pkl", ((128, 128), (64, 64)), ((8, 8), (8, 8)), ((16, 16), (8, 8))),
    "y50": ("plasma_irr_y50.pkl", ((128, 32), (64, 16)), ((8, 8), (8, 8)), ((16, 4), (8, 2))),
    "y25": ("plasma_irr_y25.pkl", ((128, 16), (64, 8)), ((8, 8), (8, 4)), ((16, 2), (8, 1))),
    "x50": ("plasma_irr_x50.pkl", ((64, 64), (32, 32)), ((8, 8), (8, 8)), ((8, 8), (4, 4))),
    "x25": ("plasma_irr_x25.pkl", ((32, 64), (16, 32)), ((8, 8), (8, 8)), ((4, 8), (2, 4))),
    # Random grid masking (full 227x101 regular grid, scattered points dropped):
    # same geometry as `reg`, only the dataset file differs.
    "mask_s0": ("plasma_mask_s0.pkl", ((128, 128), (64, 64)), ((8, 8), (8, 8)), ((16, 16), (8, 8))),
    "mask_s1": ("plasma_mask_s1.pkl", ((128, 128), (64, 64)), ((8, 8), (8, 8)), ((16, 16), (8, 8))),
}


def save_resume(path, model, optimizer, ep, best_val, best_test, best_test_under_val):
    """Atomically persist the LATEST full training state so a later installment
    can pick up exactly where this one left off. Atomic (tmp + os.replace) so a
    walltime kill mid-write can never corrupt the resume file."""
    tmp = path + ".tmp"
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": int(ep),
            "best_val": float(best_val),
            "best_test": float(best_test),
            "best_test_under_val": float(best_test_under_val),
        },
        tmp,
    )
    os.replace(tmp, path)


def train_model(config):
    def unnorm_data(data, mean, std, B, C, H, W):
        data_copy = data.detach().clone()
        return (data_copy.reshape(B, C, H, W)[:, None, :, :, :]) * std + mean

    many_params = config["many_params"]
    dataset = config["dataset"]
    dim = config["dim"]
    patch_sizes = many_params[0]
    overlaps = many_params[1]
    vit_depth = many_params[2]
    modes = many_params[3]
    mlp_dim = dim
    num_heads = config["num_heads"]
    snapshots = config["snapshots"]
    lr = config["lr"]
    max_epochs = config["max_epochs"]
    batchsize = config["batchsize"]
    cond_time = config["cond_time"]
    lift_channel = config["lift_channel"]
    act = config["act"]
    # graph-channel knobs
    graph_patch_sizes = config["graph_patch_sizes"]
    graph_depth = config["graph_depth"]  # 0 => baseline (no graph branch)
    knn_k = config["knn_k"]
    # Installment knobs: absolute resume-file path (computed in __main__, before
    # Ray chdirs into a per-trial dir) and how often to persist latest state.
    resume_path = config["resume_path"]
    save_every = config.get("save_every", 100)
    myloss = LpLoss(size_average=False)
    num_levels = len(patch_sizes)
    with open(dataset, "rb") as file:
        dataset = pickle.load(file)
    num_in_states = dataset["channels"]
    img_size = dataset["img_size"]
    train_data = torch.tensor(dataset["train"], dtype=torch.float32, device="cuda")
    test = torch.tensor(dataset["test"], dtype=torch.float32, device="cuda")
    val = torch.tensor(dataset["val"], dtype=torch.float32, device="cuda")
    train_mean = train_data.mean(dim=(0, 1, 3, 4), keepdim=True)
    train_std = train_data.std(dim=(0, 1, 3, 4), keepdim=True)
    train_data = (train_data - train_mean) / train_std
    test = (test - train_mean) / train_std
    val = (val - train_mean) / train_std

    train_loader = DataLoader(
        TemporalDataset(train_data, snapshot_length=snapshots),
        batch_size=batchsize,
        shuffle=True,
    )
    in_channels = (2 + num_in_states) * cond_time
    # True physical coordinates for irregular variants; uniform grid otherwise.
    if "coords_h" in dataset:
        grid = get_grid_from_coords(dataset["coords_h"], dataset["coords_w"]).cuda()
    else:
        grid = get_grid(img_size[0], img_size[1]).cuda()
    out_channesl = num_in_states
    model = StFTG(
        cond_time,
        num_in_states + 2,
        patch_sizes,
        overlaps,
        in_channels,
        out_channesl,
        modes,
        img_size=img_size,
        lift_channel=lift_channel,
        dim=dim,
        vit_depth=vit_depth,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        act=act,
        graph_patch_sizes=graph_patch_sizes,
        graph_depth=graph_depth,
        knn_k=knn_k,
    ).to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    best_val = torch.tensor(1e10, dtype=torch.float32, device="cuda")
    best_test = torch.tensor(1e10, dtype=torch.float32, device="cuda")
    best_test_under_val = torch.tensor(1e10, dtype=torch.float32, device="cuda")

    # --- installment resume: load latest state from the previous run, if any ---
    start_ep = 0
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location="cuda")
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_ep = int(ckpt.get("epoch", -1)) + 1
        best_val = torch.tensor(
            ckpt.get("best_val", 1e10), dtype=torch.float32, device="cuda"
        )
        best_test = torch.tensor(
            ckpt.get("best_test", 1e10), dtype=torch.float32, device="cuda"
        )
        best_test_under_val = torch.tensor(
            ckpt.get("best_test_under_val", 1e10), dtype=torch.float32, device="cuda"
        )
        print(
            f"[resume] loaded {resume_path}: continuing at epoch {start_ep}, "
            f"best_val={best_val.item():.6f}",
            flush=True,
        )
    else:
        print(f"[resume] no resume file at {resume_path}; starting fresh", flush=True)

    for ep in range(start_ep, max_epochs):
        model.train()
        train_l2_levels = torch.zeros(num_levels, dtype=torch.float32, device="cuda")
        train_l2 = 0
        train_num_examples = 0
        for _, example in enumerate(train_loader):
            B, L, C, H, W = example.shape
            for i in range(L - cond_time):
                train_num_examples += B * C
                x = example[:, i : (i + cond_time)].cuda()
                y = example[:, i + cond_time].cuda()
                preds = model(x, grid)
                sum_residues = torch.zeros_like(
                    preds[0].reshape(B * num_in_states, -1),
                    device="cuda",
                    dtype=torch.float32,
                )
                for level in range(num_levels):
                    cur_preds = preds[level]
                    sum_residues += cur_preds.reshape(B * num_in_states, -1)
                    train_l2_levels[level] += myloss(
                        cur_preds.reshape(B * num_in_states, -1).reshape(
                            B * num_in_states, -1
                        ),
                        y.reshape(B * num_in_states, -1),
                    )
                loss = myloss(
                    sum_residues.reshape(B * num_in_states, -1),
                    y.reshape(B * num_in_states, -1),
                )
                optimizer.zero_grad()
                loss.backward(retain_graph=False)
                clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
                train_l2 += myloss(
                    sum_residues.reshape(B * num_in_states, -1),
                    y.reshape(B * num_in_states, -1),
                )
        train_l2_levels = train_l2_levels / train_num_examples
        train_l2 = train_l2 / train_num_examples
        model.eval()
        if ep % 10 == 0:
            num_examples = 0
            l2_val = 0.0
            with torch.no_grad():
                B, L, C, H, W = val.shape
                x_old = None
                preds_or = val[:, :cond_time]
                for i in range(L - cond_time):
                    num_examples += B * num_in_states
                    if i == 0:
                        x = preds_or
                    else:
                        x = torch.cat(
                            (x_old[:, 1:, :, :, :], preds_or[:, None, :, :, :]), axis=1
                        )
                    x_old = x.detach().clone()
                    y = val[:, i + cond_time].cuda()
                    preds = model(x, grid)
                    sum_residues = torch.zeros_like(
                        preds[0].reshape(B * num_in_states, -1),
                        device="cuda",
                        dtype=torch.float32,
                    )
                    for level in range(num_levels):
                        cur_preds = preds[level]
                        sum_residues += (
                            cur_preds.reshape(B * num_in_states, -1).detach().clone()
                        )
                    l2_val += myloss(
                        unnorm_data(
                            sum_residues, train_mean, train_std, B, C, H, W
                        ).reshape(B * num_in_states, -1),
                        unnorm_data(y, train_mean, train_std, B, C, H, W).reshape(
                            B * num_in_states, -1
                        ),
                    )
                    preds_or = sum_residues.reshape(B, C, H, W)
            error_val = l2_val / num_examples
            num_examples = 0
            l2_test = 0.0
            with torch.no_grad():
                B, L, C, H, W = test.shape
                x_old = None
                preds_or = test[:, :cond_time]
                for i in range(L - cond_time):
                    num_examples += B * num_in_states
                    if i == 0:
                        x = preds_or
                    else:
                        x = torch.cat(
                            (x_old[:, 1:, :, :, :], preds_or[:, None, :, :, :]), axis=1
                        )
                    x_old = x.detach().clone()
                    y = test[:, i + cond_time].cuda()
                    preds = model(x, grid)
                    sum_residues = torch.zeros_like(
                        preds[0].reshape(B * num_in_states, -1),
                        device="cuda",
                        dtype=torch.float32,
                    )
                    for level in range(num_levels):
                        cur_preds = preds[level]
                        sum_residues += (
                            cur_preds.reshape(B * num_in_states, -1).detach().clone()
                        )
                    l2_test += myloss(
                        unnorm_data(
                            sum_residues, train_mean, train_std, B, C, H, W
                        ).reshape(B * num_in_states, -1),
                        unnorm_data(y, train_mean, train_std, B, C, H, W).reshape(
                            B * num_in_states, -1
                        ),
                    )
                    preds_or = sum_residues.reshape(B, C, H, W)
            error_test = (l2_test / num_examples).clone()
            if error_test < best_test:
                best_test = error_test
            if error_val < best_val:
                best_val = error_val
                best_test_under_val = error_test
                with tempfile.TemporaryDirectory() as tempdir:
                    torch.save(
                        {"model_state": model.state_dict()},
                        os.path.join(tempdir, "checkpoint_harrm.pt"),
                    )
                    metrics = {
                        "epoch": ep,
                        "train_l2": train_l2.item(),
                        "best_val": best_val.item(),
                        "best_test_under_val": best_test_under_val.item(),
                        "best_test": best_test.item(),
                        "test_error": error_test.item(),
                        "val_error": error_val.item(),
                    }
                    for _ in range(num_levels):
                        metrics["level_" + str(_) + "_loss"] = train_l2_levels[_].item()
                    train.report(
                        metrics=metrics, checkpoint=Checkpoint.from_directory(tempdir)
                    )
            else:
                metrics = {
                    "epoch": ep,
                    "train_l2": train_l2.item(),
                    "best_val": best_val.item(),
                    "best_test_under_val": best_test_under_val.item(),
                    "best_test": best_test.item(),
                    "test_error": error_test.item(),
                    "val_error": error_val.item(),
                }
                for _ in range(num_levels):
                    metrics["level_" + str(_) + "_loss"] = train_l2_levels[_].item()
                train.report(metrics=metrics)

        # --- installment checkpoint: persist LATEST full state for resume ---
        if ep % save_every == 0:
            save_resume(
                resume_path, model, optimizer, ep, best_val, best_test, best_test_under_val
            )

    # final save so a clean finish also leaves a resume point
    save_resume(
        resume_path, model, optimizer, max_epochs - 1, best_val, best_test, best_test_under_val
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=sorted(VARIANTS), help="dataset variant")
    parser.add_argument(
        "--no-graph", action="store_true", help="disable the graph channel (baseline)"
    )
    parser.add_argument("--max-epochs", type=int, default=100000)
    parser.add_argument("--graph-depth", type=int, default=2)
    parser.add_argument("--knn-k", type=int, default=8)
    # Ray reserves this many CPUs for the trial. On NERSC we used 16; on Colab
    # (often only 2 vCPUs) a request larger than the machine has leaves the trial
    # stuck PENDING forever, so pass e.g. --cpus 2 there.
    parser.add_argument("--cpus", type=int, default=2)
    # Where run outputs (ray_results/ + resume_<tag>.pt) go. On Colab, point this
    # at local disk (e.g. --out-dir /content/stft_out) so the many/large checkpoint
    # writes DON'T hit the Google Drive quota (a full model checkpoint is hundreds
    # of MB). Default "." = current dir (fine on NERSC scratch).
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    dataset_file, patch_sizes, modes, graph_patch_sizes = VARIANTS[args.variant]
    tag = args.variant + ("_nograph" if args.no_graph else "_graph")
    config = {
        "dataset": os.path.abspath(dataset_file),
        "many_params": (
            patch_sizes,
            ((1, 1), (1, 1)),
            (6, 6),
            modes,
        ),
        "dim": 128,
        "num_heads": 1,
        "snapshots": 20,
        "lr": 1e-4,
        "max_epochs": args.max_epochs,
        "batchsize": 20,
        "cond_time": 5,
        "lift_channel": 64,
        "act": "gelu",
        "graph_patch_sizes": graph_patch_sizes,
        "graph_depth": 0 if args.no_graph else args.graph_depth,
        "knn_k": args.knn_k,
        # Resume file under out_dir, tagged per run. Absolute path computed here
        # (before Ray chdirs into its per-trial dir).
        "resume_path": os.path.join(out_dir, f"resume_{tag}.pt"),
        "save_every": 100,
    }
    save_path = os.path.join(out_dir, "ray_results")
    trainable_with_cpu_gpu = tune.with_resources(train_model, {"cpu": args.cpus, "gpu": 1})
    tuner = tune.Tuner(
        trainable_with_cpu_gpu,
        param_space=config,
        run_config=RunConfig(
            name=f"train_{tag}",
            storage_path=save_path,
            # Keep only the latest (best-val) checkpoint instead of accumulating
            # one per improvement — bounds disk use to a single checkpoint.
            checkpoint_config=CheckpointConfig(num_to_keep=1),
        ),
    )
    tuner.fit()
