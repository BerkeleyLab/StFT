from pathlib import Path
import signal
import sys
import time
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from stft import StFT, get_grid, TrainingDataset, SnapshotDataset, RolloutDataset, load_dataset

class LpLoss(object):
    def __init__(self, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        assert p > 0

        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(
            x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1
        )
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

class Trainer:
    def __init__(self, config):
        self.config = config
        self.patch_sizes = config["patch_sizes"]
        self.overlaps = config["overlaps"]
        self.vit_depth = config["vit_depth"]
        self.modes = config["modes"]
        self.num_levels = len(self.patch_sizes)
        if not (self.num_levels == len(self.overlaps) == len(self.vit_depth) == len(self.modes)):
            raise ValueError(
                f"patch_sizes, overlaps, vit_depth, and modes must all have the same number of levels, "
                f"got lengths: patch_sizes={self.num_levels}, overlaps={len(self.overlaps)}, "
                f"vit_depth={len(self.vit_depth)}, modes={len(self.modes)}"
            )
        self.dataset_path = config["dataset"]
        self.dim = config["dim"]
        self.num_heads = config["num_heads"]
        self.lr = config["lr"]
        self.max_epochs = config["max_epochs"]
        self.batchsize = config["batchsize"]
        self.cond_time = config["cond_time"]
        self.lift_channel = config["lift_channel"]
        self.act = config["act"]
        self.save_path = Path(config["save_path"])
        self.save_every_n = config["save_every_n"]
        self.condition = config["condition_blocks"]
        self.use_snapshots = config["use_snapshots"]
        self.snapshot_length = config["snapshot_length"]
        if self.use_snapshots and (self.snapshot_length <= self.cond_time):
            raise ValueError(
                f"snapshot_length ({self.snapshot_length}) must be greater than "
                f"cond_time ({self.cond_time}) when use_snapshots=True"
            )
        self.epoch = 0
        self.start_epoch = 0
        self.train_time = 0.0
        self._stopped = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setup(self):
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.load_data()
        self.build_model()
        signal.signal(signal.SIGTERM, self._handle_sigterm)
        latest = self.save_path / "latest.pt"
        if latest.exists():
            self.load_checkpoint(latest)

    def _handle_sigterm(self, signum, frame):
        self._stopped = True

    def _get_wandb_run_id(self):
        run_id_file = self.save_path / "wandb_run_id.txt"
        if run_id_file.exists():
            return run_id_file.read_text().strip()
        run_id = wandb.util.generate_id()
        run_id_file.write_text(run_id)
        return run_id
        
    def run(self):
        self.setup()
        run_id = self._get_wandb_run_id()
        wandb.init(project="stft", config=self.config, id=run_id, resume="allow")
        for epoch in range(self.start_epoch, self.max_epochs):
            self.epoch = epoch
            self.model.train()
            model_metrics, comp_metrics = self.train_epoch()
            if self._stopped:
                self.save_checkpoint()
                wandb.finish()
                sys.exit(0)
            wandb.log({"epoch": epoch, **comp_metrics})
            self.model.eval()
            if epoch % 10 == 0:
                self.evaluate_and_log(model_metrics)
            if epoch % self.save_every_n == 0:
                self.save_checkpoint()
        wandb.finish()

    def load_data(self):
        dataset = load_dataset(self.dataset_path)
        self.num_in_states = dataset.channels
        self.img_size = dataset.img_size
        train_mean = np.mean(dataset.train, axis=(0, 1, 3, 4), keepdims=True)
        train_std = np.std(dataset.train, axis=(0, 1, 3, 4), keepdims=True)
        self.train_mean = torch.tensor(train_mean, dtype=torch.float32, device=self.device)
        self.train_std = torch.tensor(train_std, dtype=torch.float32, device=self.device)
        norm_mean = torch.tensor(train_mean, dtype=torch.float32).squeeze(0).squeeze(0)
        norm_std = torch.tensor(train_std, dtype=torch.float32).squeeze(0).squeeze(0)
        if self.use_snapshots:
            train_dataset = SnapshotDataset(
                dataset.train,
                snapshot_length=self.snapshot_length,
                mean=norm_mean,
                std=norm_std,
            )
        else:
            train_dataset = TrainingDataset(
                dataset.train,
                cond_time=self.cond_time,
                mean=norm_mean,
                std=norm_std,
            )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batchsize,
            shuffle=True,
        )
        self.test_loader = DataLoader(
            RolloutDataset(dataset.test, mean=norm_mean, std=norm_std),
            batch_size=self.batchsize,
        )
        self.val_loader = DataLoader(
            RolloutDataset(dataset.val, mean=norm_mean, std=norm_std),
            batch_size=self.batchsize,
        )

    def build_model(self):
        in_channels = (2 + self.num_in_states) * self.cond_time
        self.grid = get_grid(self.img_size[0], self.img_size[1]).to(self.device)
        self.myloss = LpLoss(size_average=False)
        self.model = StFT(
            self.cond_time,
            self.num_in_states + 2,
            self.patch_sizes,
            self.overlaps,
            in_channels,
            self.num_in_states,
            self.modes,
            img_size=self.img_size,
            lift_channel=self.lift_channel,
            dim=self.dim,
            vit_depth=self.vit_depth,
            num_heads=self.num_heads,
            mlp_dim=self.dim,
            act=self.act,
            condition_blocks=self.condition
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.best_val = torch.tensor(1e10, dtype=torch.float32, device=self.device)
        self.best_test = torch.tensor(1e10, dtype=torch.float32, device=self.device)
        self.best_test_under_val = torch.tensor(1e10, dtype=torch.float32, device=self.device)

    def train_epoch(self):
        if self.device != "cpu":
            torch.cuda.reset_peak_memory_stats(self.device)
        t0 = time.time()
        self._epoch_l2_levels = torch.zeros(self.num_levels, dtype=torch.float32, device=self.device)
        self._epoch_l2 = torch.zeros((), dtype=torch.float32, device=self.device)
        self._epoch_num_examples = 0
        for batch in self.train_loader:
            if self.use_snapshots:
                self.train_snapshot_batch(batch)
            else:
                x, y = batch
                self.train_batch(x, y)
            if self._stopped:
                break
        elapsed = time.time() - t0
        self.train_time += elapsed / (60 * 60)
        n = self._epoch_num_examples or 1
        model_metrics = {
            "train_l2": self._epoch_l2 / n,
            "level_losses": self._epoch_l2_levels / n,
        }
        comp_metrics = (
            {"throughput_samples_per_sec": self._epoch_num_examples / elapsed}
            if self._epoch_num_examples > 0
            else {}
        )
        if self.device != "cpu":
            comp_metrics["peak_gpu_memory_gb"] = torch.cuda.max_memory_allocated(self.device) / 1024**3
            comp_metrics["reserved_gpu_memory_gb"] = torch.cuda.max_memory_reserved(self.device) / 1024**3
        return model_metrics, comp_metrics

    def train_batch(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)
        B = x.shape[0]
        preds = self.model(x, self.grid)
        sum_residues = torch.zeros_like(
            preds[0].reshape(B * self.num_in_states, -1),
            dtype=torch.float32,
        )
        for level in range(self.num_levels):
            cur_preds = preds[level]
            sum_residues += cur_preds.reshape(B * self.num_in_states, -1)
            self._epoch_l2_levels[level] += self.myloss(
                cur_preds.reshape(B * self.num_in_states, -1),
                y.reshape(B * self.num_in_states, -1),
            ).detach()
        loss = self.myloss(
            sum_residues,
            y.reshape(B * self.num_in_states, -1),
        )
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()
        self._epoch_l2 += loss.detach()
        self._epoch_num_examples += B * self.num_in_states

    def train_snapshot_batch(self, snapshot):
        snapshot = snapshot.to(self.device)
        L = snapshot.shape[1]
        for i in range(L - self.cond_time):
            x = snapshot[:, i : i + self.cond_time]
            y = snapshot[:, i + self.cond_time]
            self.train_batch(x, y)
            if self._stopped:
                break

    def evaluate(self, loader):
        num_examples = 0
        l2 = 0.0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                B, L, C, H, W = batch.shape
                x_old = None
                preds_or = batch[:, : self.cond_time]
                for i in range(L - self.cond_time):
                    num_examples += B * self.num_in_states
                    if i == 0:
                        x = preds_or
                    else:
                        x = torch.cat(
                            (x_old[:, 1:, :, :, :], preds_or[:, None, :, :, :]), axis=1
                        )
                    x_old = x.detach().clone()
                    y = batch[:, i + self.cond_time]
                    preds = self.model(x, self.grid)
                    sum_residues = torch.zeros_like(
                        preds[0].reshape(B * self.num_in_states, -1),
                        device=self.device,
                        dtype=torch.float32,
                    )
                    for level in range(self.num_levels):
                        sum_residues += preds[level].reshape(B * self.num_in_states, -1).detach().clone()
                    l2 += self.myloss(
                        self.unnorm_data(sum_residues, B, C, H, W).reshape(B * self.num_in_states, -1),
                        self.unnorm_data(y, B, C, H, W).reshape(B * self.num_in_states, -1),
                    )
                    preds_or = sum_residues.reshape(B, C, H, W)
        return l2 / num_examples

    def evaluate_and_log(self, model_metrics):
        error_val = self.evaluate(self.val_loader)
        error_test = self.evaluate(self.test_loader)
        if error_test < self.best_test:
            self.best_test = error_test
        improved_val = error_val < self.best_val
        if improved_val:
            self.best_val = error_val
            self.best_test_under_val = error_test
        metrics = {
            "epoch": self.epoch,
            "train_l2": model_metrics["train_l2"].item(),
            "best_val": self.best_val.item(),
            "best_test_under_val": self.best_test_under_val.item(),
            "best_test": self.best_test.item(),
            "test_error": error_test.item(),
            "val_error": error_val.item(),
        }
        for level in range(self.num_levels):
            metrics[f"level_{level}_loss"] = model_metrics["level_losses"][level].item()
        wandb.log(metrics)
        if improved_val:
            self.save_checkpoint(is_best=True)

    def save_checkpoint(self, is_best=False):
        checkpoint = {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "epoch": self.epoch,
                "train_time": self.train_time,
        }
        checkpoint_path = self.save_path / "latest.pt"
        if is_best:
            checkpoint_path = self.save_path / "best.pt"
        tmp_path = checkpoint_path.with_suffix(".pt.tmp")
        print(f"Saving checkpoint to {tmp_path}...")
        torch.save(checkpoint, tmp_path)
        tmp_path.replace(checkpoint_path)
        print(f"Checkpoint successfully saved to {checkpoint_path}")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.train_time = checkpoint["train_time"]
        self.epoch = checkpoint["epoch"]
        self.start_epoch = self.epoch + 1

    def unnorm_data(self, data, B, C, H, W):
        return data.detach().clone().reshape(B, C, H, W).unsqueeze(1) * self.train_std + self.train_mean
