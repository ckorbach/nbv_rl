import os
from next_best_view.nextbestview_task import NextBestViewTask
import pybullet as p
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../configs/config.yaml")
def evaluate(cfg: DictConfig) -> None:
    print(cfg.evaluate.pretty())
    task = NextBestViewTask(cfg, mode="eval")
    task.evaluate()
    p.disconnect()


if __name__ == "__main__":
    evaluate()
