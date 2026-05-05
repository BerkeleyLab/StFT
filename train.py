from trainer import Trainer

if __name__ == "__main__":
    config = {
        "dataset": "/pscratch/sd/a/atrupe/StFT/data/shallow-water",
        "patch_sizes": ((128, 128), (64, 64), (32, 32)),
        "overlaps": ((1, 1), (1, 1), (1, 1)),
        "vit_depth": (6, 6, 6),
        "modes": ((8, 8), (8, 8), (8, 8)),
        "dim": 32,
        "num_heads": 1,
        "snapshots": 20,
        "lr": 1e-4,
        "max_epochs": 1,
        "batchsize": 2,
        "cond_time": 5,
        "lift_channel": 64,
        "act": "gelu",
        "save_path": "/pscratch/sd/a/atrupe/StFT/experiments/test",
        "save_every_n": 100,
        "condition_blocks": True
    }
    trainer = Trainer(config)
    trainer.run()