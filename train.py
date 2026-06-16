import hydra

from stft.config import to_plain_config
from trainer import Trainer


@hydra.main(version_base=None, config_path="configs", config_name="run_2")
def main(cfg):
    config = to_plain_config(cfg)
    trainer = Trainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
