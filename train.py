import os
import torch
import wandb
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from stft import StFT, LpLoss, get_grid, TemporalDataset
import pickle


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
    save_path = config["save_path"]
    save_every_n = config["save_every_n"]
    os.makedirs(save_path, exist_ok=True)
    wandb.init(project="stft", config=config)
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
    grid = get_grid(img_size[0], img_size[1]).cuda()
    out_channesl = num_in_states
    model = StFT(
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
    ).to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    best_val = torch.tensor(1e10, dtype=torch.float32, device="cuda")
    best_test = torch.tensor(1e10, dtype=torch.float32, device="cuda")
    best_test_under_val = torch.tensor(1e10, dtype=torch.float32, device="cuda")

    for ep in range(max_epochs):
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
            improved_val = error_val < best_val
            if improved_val:
                best_val = error_val
                best_test_under_val = error_test

            metrics = {
                "epoch": ep,
                "train_l2": train_l2.item(),
                "best_val": best_val.item(),
                "best_test_under_val": best_test_under_val.item(),
                "best_test": best_test.item(),
                "test_error": error_test.item(),
                "val_error": error_val.item(),
            }
            for level in range(num_levels):
                metrics[f"level_{level}_loss"] = train_l2_levels[level].item()
            wandb.log(metrics)

            if improved_val:
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        **metrics,
                    },
                    os.path.join(save_path, "best.pt"),
                )

        if ep % save_every_n == 0:
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": ep,
                },
                os.path.join(save_path, f"checkpoint_ep{ep:06d}.pt"),
            )

    wandb.finish()


if __name__ == "__main__":
    config = {
        "dataset": "/path/to/my/data/plasma.pkl",
        "many_params": (
            ((128, 128), (64, 64)),
            ((1, 1), (1, 1)),
            (6, 6),
            ((8, 8), (8, 8)),
        ),
        "dim": 128,
        "num_heads": 1,
        "snapshots": 20,
        "lr": 1e-4,
        "max_epochs": 100000,
        "batchsize": 20,
        "cond_time": 5,
        "lift_channel": 64,
        "act": "gelu",
        "save_path": "/path/to/my/results",
        "save_every_n": 100,
    }
    train_model(config)
