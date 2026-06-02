from pathlib import Path
import signal
import time
import numpy as np
import torch
import wandb
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, DistributedSampler

from stft import StFT, get_grid, TrainingDataset, SnapshotDataset, RolloutDataset, load_dataset
from stft.distributed import (
    barrier,
    cleanup_distributed,
    distributed_is_enabled,
    log_distributed_preflight,
    reduce_max,
    reduce_sum,
    setup_distributed,
    unwrap_model,
)

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
        self._wandb_run = None

    def setup(self):
        self.device, self.local_rank, self.rank, self.world_size = setup_distributed()
        self.distributed = distributed_is_enabled()
        self.is_main = self.rank == 0
        signal.signal(signal.SIGTERM, self._handle_stop_signal)
        
        if self.is_main:
            self.save_path.mkdir(parents=True, exist_ok=True)
        barrier(self.distributed)
        log_distributed_preflight(self.device, self.local_rank, self.rank, self.world_size)
        self.load_data()
        self.build_model()
        latest = self.save_path / "latest.pt"
        if latest.exists():
            self.load_checkpoint(latest)

    def _handle_stop_signal(self, signum, frame):
        self._stopped = True

    def _get_wandb_run_id(self):
        run_id_file = self.save_path / "wandb_run_id.txt"
        if run_id_file.exists():
            return run_id_file.read_text().strip()
        run_id = wandb.util.generate_id()
        run_id_file.write_text(run_id)
        return run_id
        
    def run(self):
        try:
            self.setup()
            if self.is_main:
                run_id = self._get_wandb_run_id()
                self._wandb_run = wandb.init(
                    project="stft",
                    config=self.config,
                    id=run_id,
                    resume="allow",
                )
                print(
                    " ".join(
                        [
                            f"per_rank_batch={self.batchsize}",
                            f"world_size={self.world_size}",
                            "gradient_accumulation_steps=1",
                            f"effective_global_batch={self.batchsize * self.world_size}",
                        ]
                    ),
                    flush=True,
                )
            for epoch in range(self.start_epoch, self.max_epochs):
                self.epoch = epoch
                self.model.train()
                model_metrics, comp_metrics = self.train_epoch()
                print(
                    f"epoch {epoch} | "
                    f"peak allocated: {comp_metrics["peak_gpu_memory_gb"]} GB | "
                    f"peak reserved: {comp_metrics["reserved_gpu_memory_gb"]} GB",
                    flush=True
                )
                if self._sync_stop_requested():
                    self.save_checkpoint()
                    break
                if self.is_main:
                    wandb.log({"epoch": epoch, **comp_metrics})
                self.model.eval()
                if epoch % 10 == 0:
                    if self.is_main:
                        self.evaluate_and_log(model_metrics)
                    barrier(self.distributed)
                if epoch % self.save_every_n == 0:
                    self.save_checkpoint()
        finally:
            if self._wandb_run is not None:
                wandb.finish()
            cleanup_distributed()

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
        self.train_sampler = (
            DistributedSampler(train_dataset, shuffle=True)
            if self.distributed
            else None
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batchsize,
            sampler=self.train_sampler,
            shuffle=self.train_sampler is None,
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
        raw_model = StFT(
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
        self.optimizer = torch.optim.AdamW(raw_model.parameters(), lr=self.lr)
        if self.distributed:
            if self.device.type == "cuda":
                self.model = DistributedDataParallel(
                    raw_model,
                    device_ids=[self.device.index],
                    output_device=self.device.index,
                )
            else:
                self.model = DistributedDataParallel(raw_model)
        else:
            self.model = raw_model
        self.best_val = torch.tensor(1e10, dtype=torch.float32, device=self.device)
        self.best_test = torch.tensor(1e10, dtype=torch.float32, device=self.device)
        self.best_test_under_val = torch.tensor(1e10, dtype=torch.float32, device=self.device)

    def train_epoch(self):
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(self.epoch)
        if self.device.type == "cuda":
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
            if self._sync_stop_requested():
                break
        elapsed = time.time() - t0
        self.train_time += elapsed / (60 * 60)
        local_n = torch.tensor(self._epoch_num_examples, dtype=torch.float64, device=self.device)
        global_n = reduce_sum(local_n, self.distributed).clamp_min(1)
        global_l2 = reduce_sum(self._epoch_l2.to(torch.float64), self.distributed)
        global_l2_levels = reduce_sum(self._epoch_l2_levels.to(torch.float64), self.distributed)
        model_metrics = {
            "train_l2": global_l2 / global_n,
            "level_losses": global_l2_levels / global_n,
        }
        global_examples = reduce_sum(local_n, self.distributed)
        max_elapsed = reduce_max(
            torch.tensor(elapsed, dtype=torch.float64, device=self.device),
            self.distributed,
        )
        comp_metrics = (
            {"throughput_samples_per_sec": global_examples.item() / max_elapsed.item()}
            if global_examples.item() > 0 and max_elapsed.item() > 0
            else {}
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize() # REMOVE AFTER TESTING
            peak_memory = torch.tensor(
                torch.cuda.max_memory_allocated(self.device) / 1024**3,
                dtype=torch.float64,
                device=self.device,
            )
            reserved_memory = torch.tensor(
                torch.cuda.max_memory_reserved(self.device) / 1024**3,
                dtype=torch.float64,
                device=self.device,
            )
            comp_metrics["peak_gpu_memory_gb"] = reduce_max(peak_memory, self.distributed).item()
            comp_metrics["reserved_gpu_memory_gb"] = reduce_max(reserved_memory, self.distributed).item()
            
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
        clip_grad_norm_(unwrap_model(self.model).parameters(), max_norm=10.0)
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
            if self._sync_stop_requested():
                break

    def evaluate(self, loader):
        num_examples = 0
        l2 = torch.zeros((), dtype=torch.float32, device=self.device)
        eval_model = unwrap_model(self.model)
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
                    preds = eval_model(x, self.grid)
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
        return l2 / max(num_examples, 1)

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
            self.save_checkpoint(is_best=True, sync=False)

    def save_checkpoint(self, is_best=False, sync=True):
        if not self.is_main:
            if sync:
                barrier(self.distributed)
            return
        checkpoint = {
                "model_state": unwrap_model(self.model).state_dict(),
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
        if sync:
            barrier(self.distributed)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        unwrap_model(self.model).load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.train_time = checkpoint["train_time"]
        self.epoch = checkpoint["epoch"]
        self.start_epoch = self.epoch + 1

    def _sync_stop_requested(self):
        stop = torch.tensor(int(self._stopped), dtype=torch.int32, device=self.device)
        self._stopped = bool(reduce_sum(stop, self.distributed).item())
        return self._stopped

    def unnorm_data(self, data, B, C, H, W):
        return data.detach().clone().reshape(B, C, H, W).unsqueeze(1) * self.train_std + self.train_mean
    
    def run_test(self, n_epochs, warmup_steps, measure_steps, B, C, H, W):
        self.warmup_steps = warmup_steps
        self.measure_steps = measure_steps
        self.B = B
        self.C = C
        self.H = H 
        self.W = W
        self.test_setup()
        for epoch in range(n_epochs):
            self.epoch = epoch
            self.model.train()
            comp_metrics = self.test_train_epoch()
            print(
                f"epoch {epoch} | "
                f"peak allocated: {comp_metrics[0]} GB | "
                f"peak reserved: {comp_metrics[1]} GB",
                flush=True
            )

    def test_setup(self):
        signal.signal(signal.SIGTERM, self._handle_stop_signal)
        self.device = torch.device("cuda:0")
        self.distributed = False
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.is_main = True
        self.num_in_states = self.C
        self.img_size = (self.H, self.W)
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
    
    def test_train_epoch(self):
        self._epoch_l2_levels = torch.zeros(self.num_levels, dtype=torch.float32, device=self.device)
        self._epoch_l2 = torch.zeros((), dtype=torch.float32, device=self.device)
        self._epoch_num_examples = 0
        for i in range(self.warmup_steps):
            self.test_train_snapshot_batch()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(self.device)
        for i in range(self.measure_steps):
            self.test_train_snapshot_batch()
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated(self.device) / 1024**3
        reserved_memory = torch.cuda.max_memory_reserved(self.device) / 1024**3
        comp_metrics = (peak_memory, reserved_memory)
        return comp_metrics
    
    def test_train_snapshot_batch(self):
        for i in range(self.snapshot_length - self.cond_time):
            x = torch.randn(self.B, self.cond_time, self.C, self.H, self.W, device=self.device)
            y = torch.randn(self.B, self.C, self.H, self.W, device=self.device)
            self.train_batch(x, y)
            if self._sync_stop_requested():
                break
        return
