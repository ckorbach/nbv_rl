import hydra
from omegaconf import DictConfig
from classification.trainer import Trainer


@hydra.main(config_path="../configs/config.yaml")
def train(cfg: DictConfig) -> None:
    print(cfg.classificator.pretty())
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    train()
