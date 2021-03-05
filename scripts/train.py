from next_best_view.nextbestview_task import NextBestViewTask
import pybullet as p
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../configs/config.yaml")
def train(cfg: DictConfig) -> None:
    print(cfg.train.pretty())
    task = NextBestViewTask(cfg, mode="train")
    task.train()
    p.disconnect()


if __name__ == "__main__":
    train()
