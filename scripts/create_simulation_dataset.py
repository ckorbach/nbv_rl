import math
import time
from datetime import datetime
import argparse
import pybullet as p
import os
import hydra
from omegaconf import DictConfig


from next_best_view.pybullet_simulator import PyBulletSimulator
from next_best_view import utils


class PyBulletSimulatorDatasetCreator(PyBulletSimulator):
    def __init__(self, cfg):
        super(PyBulletSimulatorDatasetCreator, self).__init__(cfg)
        self.cfg = cfg
        pos_camera = self.camera.eye_position
        plb = self.cfg.gym.pos_lower_bounds
        pub = self.cfg.gym.pos_upper_bounds
        self.pos_lower_bounds = [pos_camera[0] + plb[0], pos_camera[1] + plb[1], pos_camera[2] + plb[2]]
        self.pos_upper_bounds = [pos_camera[0] + pub[0], pos_camera[1] + pub[1], pos_camera[2] + pub[2]]
        self.orn_lower_bounds = self.cfg.gym.orn_lower_bounds
        self.orn_upper_bounds = self.cfg.gym.orn_upper_bounds
        self.pose_array = []

        # to define
        self.only_object = False
        self.num_poses = 2000
        self.mode = 0  # 0 = by random poses, 1 = orn rotated

        # path
        # TODO fix path, bugs with hydra
        self.dataset_name = "yab_robot_complete"
        current_path = os.path.abspath(os.getcwd() + "../../../../")
        # current_path = os.path.dirname(os.path.realpath(__file__) + "../../")
        print(current_path)
        self.dataset_path = os.path.join(current_path, self.dataset_name)
        print(self.dataset_path)
        if os.path.exists(self.dataset_path):
            print(f"Directory {self.dataset_path} already exists, please use another name.")
            return
        os.mkdir(self.dataset_path)

    def create_dataset(self):
        print(f"Create dataset {self.dataset_name} in directory: {self.dataset_path}")
        self.create_pose_array()
        for key in self.class_map:
            self.process_single_object(key)
        print(f"Dataset {self.dataset_name} created")

    def process_single_object(self, obj_id):
        self.load_object(obj_id)
        obj_name = self.object.object_name

        object_path = os.path.join(self.dataset_path, obj_name)
        os.mkdir(object_path)
        print(f"Create images for object {obj_name} with id {obj_id} in directory: {object_path}")
        for i, pose in enumerate(self.pose_array):
            self.move_to_position(pose)
            name = obj_name + "_" + str(i)
            self.save_image(name=name, path=object_path)
        # for i in range(self.num_poses):
        #     valid = False
        #     while not valid:
        #         pose = self.get_random_pose()
        #         self.move_to_position(pose)
        #         valid_position = self.camera.object_in_view(self.object)
        #         valid = valid_position.all()
        #     name = obj_name + "_" + str(i)
        #     self.save_image(name=name, path=object_path)
        print(f"--- finished!")

    def create_pose_array(self, mode=0):
        # by pose numbers
        if self.mode == 0:
            for i in range(self.num_poses):
                pose = self.get_random_pose()
                self.pose_array.append(pose)

        # by pose increments
        inc = 0.1
        if self.mode == 1:
            plb = self.pos_lower_bounds
            pub = self.pos_upper_bounds
            pos_mid = [(plb[0] + pub[0]) / 2.0, (plb[1] + pub[1]) / 2.0, (plb[2] + pub[2]) / 2.0]
            pos = pos_mid

            for i in range(3):
                orn = self.orn_lower_bounds
                while orn[i] <= self.orn_upper_bounds - inc:
                    orn[i] += inc
                    pose = [*pos, *orn]
                    self.pose_array.append(pose)

        print(f"Created {len(self.pose_array)} poses with mode {self.mode}")

    # --- UTILS ---

    def save_image(self, name, path):
        rgb_img = self.get_rgb_image()
        utils.save_image(rgb_img, name, path)

    def move_to_position(self, pose):
        pos = pose[:3]
        orn = pose[-4:]
        if self.only_object:
            p.resetBasePositionAndOrientation(self.object.id, pos, orn)
        else:
            self.robot.move_joints(pos=pos, orn=orn)
            self.attach_object_to_end_effector()

    def get_random_pose(self):
        pose = utils.get_random_pose(self.pos_lower_bounds, self.pos_upper_bounds,
                                     self.orn_lower_bounds, self.orn_upper_bounds)
        return pose

    def get_random_pos(self):
        pos = utils.get_random_position(self.pos_lower_bounds, self.pos_upper_bounds)
        return pos


@hydra.main(config_path="../configs/config.yaml")
def create(cfg: DictConfig) -> None:
    creator = PyBulletSimulatorDatasetCreator(cfg)
    creator.create_dataset()
    p.disconnect()


if __name__ == "__main__":
    current_path = os.path.abspath(os.getcwd())
    print(current_path)
    create()
