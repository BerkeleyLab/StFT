import sys, hydra

from stft.config import to_plain_config
from trainer import Trainer

def _require_config_name():
    args = sys.argv[1:]
    has_config_name = any(
        arg == "--config-name"
        or arg.startswith("--config-name=")
        or arg == "-cn"
        or arg.startswith("-cn=")
        for arg in args
    )
    if not has_config_name:
        raise SystemExit(
            "Missing required Hydra config. Launch with: "
            "python train.py --config-name run_2"
        )

@hydra.main(version_base=None, config_path="configs", config_name=None)
def main(cfg):
    _require_config_name()
    config = to_plain_config(cfg)
    trainer = Trainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
