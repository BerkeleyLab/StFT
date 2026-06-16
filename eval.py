import argparse
from pathlib import Path

import numpy as np
import torch

from stft.config import load_run_config, resolve_run_dir
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


def parse_args():
    parser = argparse.ArgumentParser(description="Run validation rollout for a saved StFT run.")
    parser.add_argument("--run", default="2", help="Run number or run_<number> directory name.")
    parser.add_argument("--run-dir", default=None, help="Explicit experiment run directory.")
    parser.add_argument(
        "--experiments-dir",
        default="experiments",
        help="Parent directory used with --run.",
    )
    parser.add_argument(
        "--checkpoint",
        choices=("latest", "best"),
        default="latest",
        help="Checkpoint basename to load.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_dir = resolve_run_dir(
        run=args.run,
        run_dir=args.run_dir,
        experiments_dir=args.experiments_dir,
    )
    config = load_run_config(run_dir=run_dir)

    trainer = Trainer(config, persist_config=False)
    try:
        trainer.setup()
        checkpoint_path = Path(config["save_path"]) / f"{args.checkpoint}.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
        trainer.load_checkpoint(checkpoint_path)

        trainer.model.eval()
        batch = next(iter(trainer.val_loader))
        preds, targets = rollout(trainer, batch)
        context = (
            batch[:, : trainer.cond_time].to(trainer.device) * trainer.train_std + trainer.train_mean
        ).cpu()
        rel_l2 = torch.linalg.vector_norm(preds - targets) / torch.linalg.vector_norm(targets)

        output_path = Path(config["save_path"]) / f"val_rollout_{args.checkpoint}.npz"
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
