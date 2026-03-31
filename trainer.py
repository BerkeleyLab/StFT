import pickle
from pathlib import Path
import time
import torch
import wandb
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from stft import StFT, get_grid, TemporalDataset

'''
TODO
- add checks for consistency between metadata in dictionaries and properties of data tensors,
  e.g., num_channels is the size of the data tensor in the channel dimension
'''

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
        many_params = config["many_params"]
        self.patch_sizes = many_params[0]
        self.overlaps = many_params[1]
        self.vit_depth = many_params[2]
        self.modes = many_params[3]
        self.num_levels = len(self.patch_sizes)
        self.dataset_path = config["dataset"]
        self.dim = config["dim"]
        self.num_heads = config["num_heads"]
        self.snapshots = config["snapshots"]
        self.lr = config["lr"]
        self.max_epochs = config["max_epochs"]
        self.batchsize = config["batchsize"]
        self.cond_time = config["cond_time"]
        self.lift_channel = config["lift_channel"]
        self.act = config["act"]
        self.save_path = Path(config["save_path"])
        self.save_every_n = config["save_every_n"]
        self.epoch = 0
        self.start_epoch = 0
        self.train_time = 0.0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setup(self):
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.load_data()
        self.build_model()
        checkpoints = sorted(self.save_path.glob("checkpoint_ep*.pt"))
        if checkpoints:
            self.load_checkpoint(checkpoints[-1])
        
    def run(self):
        self.setup()
        wandb.init(project="stft", config=self.config)
        for epoch in range(self.start_epoch, self.max_epochs):
            self.epoch = epoch
            self.model.train()
            train_metrics = self.train_epoch()
            self.model.eval()
            if epoch % 10 == 0:
                self.evaluate_and_log(train_metrics)
            if epoch % self.save_every_n == 0:
                self.save_checkpoint()
        wandb.finish()

    def load_data(self):
        with open(self.dataset_path, "rb") as file:
            dataset = pickle.load(file)
        self.num_in_states = dataset["channels"]
        self.img_size = dataset["img_size"]
        train_data = torch.tensor(dataset["train"], dtype=torch.float32, device=self.device)
        test = torch.tensor(dataset["test"], dtype=torch.float32, device=self.device)
        val = torch.tensor(dataset["val"], dtype=torch.float32, device=self.device)
        self.train_mean = train_data.mean(dim=(0, 1, 3, 4), keepdim=True)
        self.train_std = train_data.std(dim=(0, 1, 3, 4), keepdim=True)
        train_data = (train_data - self.train_mean) / self.train_std
        self.test = (test - self.train_mean) / self.train_std
        self.val = (val - self.train_mean) / self.train_std
        self.train_loader = DataLoader(
            TemporalDataset(train_data, snapshot_length=self.snapshots),
            batch_size=self.batchsize,
            shuffle=True,
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
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.best_val = torch.tensor(1e10, dtype=torch.float32, device=self.device)
        self.best_test = torch.tensor(1e10, dtype=torch.float32, device=self.device)
        self.best_test_under_val = torch.tensor(1e10, dtype=torch.float32, device=self.device)

    def train_epoch(self):
        t0 = time.time()
        train_l2_levels = torch.zeros(self.num_levels, dtype=torch.float32, device=self.device)
        train_l2 = 0
        train_num_examples = 0
        for _, example in enumerate(self.train_loader):
            B, L, C, H, W = example.shape
            for i in range(L - self.cond_time):
                train_num_examples += B * C
                x = example[:, i : (i + self.cond_time)].to(self.device)
                y = example[:, i + self.cond_time].to(self.device)
                preds = self.model(x, self.grid)
                sum_residues = torch.zeros_like(
                    preds[0].reshape(B * self.num_in_states, -1),
                    device=self.device,
                    dtype=torch.float32,
                )
                for level in range(self.num_levels):
                    cur_preds = preds[level]
                    sum_residues += cur_preds.reshape(B * self.num_in_states, -1)
                    train_l2_levels[level] += self.myloss(
                        cur_preds.reshape(B * self.num_in_states, -1),
                        y.reshape(B * self.num_in_states, -1),
                    )
                loss = self.myloss(
                    sum_residues.reshape(B * self.num_in_states, -1),
                    y.reshape(B * self.num_in_states, -1),
                )
                self.optimizer.zero_grad()
                loss.backward(retain_graph=False)
                clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.optimizer.step()
                train_l2 += loss.detach()
        self.train_time += time.time() - t0
        return {
            "train_l2": train_l2 / train_num_examples,
            "level_losses": train_l2_levels / train_num_examples,
        }

    def evaluate(self, data):
        B, L, C, H, W = data.shape
        num_examples = 0
        l2 = 0.0
        x_old = None
        preds_or = data[:, : self.cond_time]
        with torch.no_grad():
            for i in range(L - self.cond_time):
                num_examples += B * self.num_in_states
                if i == 0:
                    x = preds_or
                else:
                    x = torch.cat(
                        (x_old[:, 1:, :, :, :], preds_or[:, None, :, :, :]), axis=1
                    )
                x_old = x.detach().clone()
                y = data[:, i + self.cond_time].to(self.device)
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

    def evaluate_and_log(self, train_metrics):
        error_val = self.evaluate(self.val)
        error_test = self.evaluate(self.test)
        if error_test < self.best_test:
            self.best_test = error_test
        improved_val = error_val < self.best_val
        if improved_val:
            self.best_val = error_val
            self.best_test_under_val = error_test
        metrics = {
            "epoch": self.epoch,
            "train_l2": train_metrics["train_l2"].item(),
            "best_val": self.best_val.item(),
            "best_test_under_val": self.best_test_under_val.item(),
            "best_test": self.best_test.item(),
            "test_error": error_test.item(),
            "val_error": error_val.item(),
        }
        for level in range(self.num_levels):
            metrics[f"level_{level}_loss"] = train_metrics["level_losses"][level].item()
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
        checkpoint_path = self.save_path / f"checkpoint_ep{self.epoch:06d}.pt"
        if is_best:
            checkpoint_path = self.save_path / "best.pt"
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.train_time = checkpoint["train_time"]
        self.epoch = checkpoint["epoch"]
        self.start_epoch = self.epoch + 1

    def unnorm_data(self, data, B, C, H, W):
        return data.detach().clone().reshape(B, C, H, W).unsqueeze(1) * self.train_std + self.train_mean
