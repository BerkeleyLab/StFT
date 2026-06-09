from pathlib import Path

import numpy as np
import torch

from stft.distributed import cleanup_distributed, unwrap_model
from trainer import Trainer


def rollout(trainer, batch):
    batch = batch.to(trainer.device)
    bsz, length, channels, height, width = batch.shape
    preds = []
    targets = []
    x = batch[:, : trainer.cond_time]
    model = unwrap_model(trainer.model)

    with torch.no_grad():
        for step in range(length - trainer.cond_time):
            level_preds = model(x, trainer.grid)
            summed = torch.zeros_like(
                level_preds[0].reshape(bsz * trainer.num_in_states, -1),
                device=trainer.device,
                dtype=torch.float32,
            )
            for level in range(trainer.num_levels):
                summed += level_preds[level].reshape(bsz * trainer.num_in_states, -1)
            pred = summed.reshape(bsz, channels, height, width)
            target = batch[:, step + trainer.cond_time]
            preds.append(trainer.unnorm_data(pred, bsz, channels, height, width).squeeze(1).cpu())
            targets.append(trainer.unnorm_data(target, bsz, channels, height, width).squeeze(1).cpu())
            x = torch.cat((x[:, 1:], pred[:, None]), dim=1)

    return torch.stack(preds, dim=1), torch.stack(targets, dim=1)


if __name__ == "__main__":
    config = {
        "dataset": "/pscratch/sd/a/atrupe/StFT/data/shallow-water",
        "patch_sizes": ((128, 128), (64, 64), (32, 32)),
        "overlaps": ((1, 1), (1, 1), (1, 1)),
        "vit_depth": (6, 6, 6),
        "modes": ((8, 8), (8, 8), (8, 8)),
        "dim": 512,
        "num_heads": 1,
        "lr": 1e-4,
        "max_epochs": 10_000,
        "batchsize": 16,
        "cond_time": 5,
        "lift_channel": 64,
        "act": "gelu",
        "save_path": "/pscratch/sd/a/atrupe/StFT/experiments/run_1",
        "save_every_n": 5,
        "validate_every_n": 10,
        "condition_blocks": True,
        "use_snapshots": True,
        "snapshot_length": 20,
    }

    trainer = Trainer(config)
    try:
        trainer.setup()
        checkpoint_path = Path(config["save_path"]) / "latest.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

        trainer.model.eval()
        batch = next(iter(trainer.val_loader))
        preds, targets = rollout(trainer, batch)
        context = (
            batch[:, : trainer.cond_time].to(trainer.device) * trainer.train_std + trainer.train_mean
        ).cpu()
        rel_l2 = torch.linalg.vector_norm(preds - targets) / torch.linalg.vector_norm(targets)

        output_path = Path(config["save_path"]) / "val_rollout_latest.npz"
        np.savez(
            output_path,
            context=context.numpy(),
            preds=preds.numpy(),
            targets=targets.numpy(),
            rel_l2=rel_l2.item(),
        )
        print(f"saved validation rollout to {output_path}")
        print(f"preds shape: {tuple(preds.shape)}")
        print(f"targets shape: {tuple(targets.shape)}")
        print(f"relative L2: {rel_l2.item():.6f}")
    finally:
        cleanup_distributed()
