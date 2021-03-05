#!/usr/bin/env python

from definitions import ROOT_DIR
import pybullet as p
from datetime import datetime
import numpy as np
import yaml
import json
import os
import imageio
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../configs/config.yaml")
def get_hydra_config(cfg: DictConfig) -> None:
    return cfg


def get_random_position(lower_bounds=None, upper_bounds=None):
    """
    Create a random position with boundaries
    :return position vector
    """
    if lower_bounds is None:
        lower_bounds = [0, 0, 0]
    if upper_bounds is None:
        upper_bounds = [1, 1, 1]
    x = np.random.uniform(lower_bounds[0], upper_bounds[0])
    y = np.random.uniform(lower_bounds[1], upper_bounds[1])
    z = np.random.uniform(lower_bounds[2], upper_bounds[2])
    vector = [x, y, z]
    return vector


def get_random_orientation(lower_bounds=None, upper_bounds=None):
    """
    Create a random orientation with boundaries
    :return orientation quaternion
    """
    if lower_bounds is None:
        lower_bounds = [0, 0, 0]
    if upper_bounds is None:
        upper_bounds = [1, 1, 1]
    x = np.random.uniform(lower_bounds[0], upper_bounds[0])
    y = np.random.uniform(lower_bounds[1], upper_bounds[1])
    z = np.random.uniform(lower_bounds[2], upper_bounds[2])
    w = np.random.uniform(lower_bounds[3], upper_bounds[3])
    quaternion = [x, y, z, w]
    return quaternion


def get_random_pose(pos_lower_bounds=None, pos_upper_bounds=None, orn_lower_bounds=None, orn_upper_bounds=None):
    """
    Create a random pose with boundaries
    :return pose [*position_vector, *orientation_quaternion]
    """
    position = get_random_position(pos_lower_bounds, pos_upper_bounds)
    orientation = get_random_orientation(orn_lower_bounds, orn_upper_bounds)
    pose = [*position, *orientation]
    return pose


def save_image(image, name="default_name", path=None):
    if path is None:
        path = os.path.abspath(os.getcwd())
    img_path = os.path.join(path, name + ".png")
    imageio.imwrite(img_path, image)


def save_json(data, filename):
    with open(filename, 'w') as outfile:
        json.dump(data, outfile)


def make_timestamp_id():
    now = datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M%S")
    return date_time


def load_config(root_dir, config_name):
    config = None
    conf_dir = os.path.join(root_dir, "configs")
    conf_path = os.path.join(conf_dir, config_name)
    print(" Load config: ", conf_path)
    with open(conf_path, "r") as config_file:
        try:
            config = yaml.safe_load(config_file)
        except yaml.YAMLError as error:
            print("[Error (config load)", error)
    return config
