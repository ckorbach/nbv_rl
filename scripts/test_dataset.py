from io import open
import os
import json
from classification.predictor import Predictor
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../configs/config.yaml")
def test(cfg: DictConfig) -> None:
    predictor = Predictor(cfg, test=True)
    predictor.evaluate()


if __name__ == "__main__":
    test()

