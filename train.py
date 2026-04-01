from trainer import Trainer


if __name__ == "__main__":
    config = {
        "dataset": "/path/to/my/data/plasma/",
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
        "condition_blocks": True
    }
    trainer = Trainer(config)
    trainer.run()
