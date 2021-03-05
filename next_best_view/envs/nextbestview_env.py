#!/usr/bin/env python

import os
import numpy as np
import warnings
import random
from pathlib import Path
from datetime import datetime

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import pybullet as p

from next_best_view.pybullet_simulator import PyBulletSimulator
from next_best_view import utils

CONFIG_SIM = "config_j2s7s300.yaml"


class NextBestViewEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self, cfg=None, data_path=None):
        print("[NextBestViewEnv] initializing ...")
        print(cfg.gym.pretty())
        self.cfg = cfg
        self.id_stamp = utils.make_timestamp_id()
        self.root_dir = str(Path(__file__).parent.parent.parent)
        self.object_id = 0  # TODO REWORK
        self.object_orientation = self.cfg.gym.orientation
        self.data_log_dir = os.path.join(self.root_dir, "data", "data_log")
        if data_path is None:
            self.data_dir = os.path.join(self.data_log_dir, self.id_stamp + "_" + str(self.object_id))
        else:
            self.data_dir = data_path
        self.simulator = PyBulletSimulator(cfg, obj_id=self.object_id, obj_orn=self.object_orientation)
        self.physicsClient = self.simulator.client_id
        self.seed_id = self.cfg.gym.seed

        self.reward = 0
        self.accuracy = 0
        self.accuracy_diff = 0

        # set custom robot parameters
        pos_camera = self.simulator.camera.eye_position
        plb = self.cfg.gym.pos_lower_bounds
        pub = self.cfg.gym.pos_upper_bounds
        self.pos_lower_bounds = [pos_camera[0] + plb[0], pos_camera[1] + plb[1], pos_camera[2] + plb[2]]
        self.pos_upper_bounds = [pos_camera[0] + pub[0], pos_camera[1] + pub[1], pos_camera[2] + pub[2]]
        self.orn_lower_bounds = self.cfg.gym.orn_lower_bounds
        self.orn_upper_bounds = self.cfg.gym.orn_upper_bounds

        # cfg
        self.start_pose = self.cfg.gym.start_pose
        self.random_start_pose = self.cfg.gym.random_start_pose
        self.ds_obj_mode = self.cfg.train.ds_obj_mode
        self.ds_sort_mode = self.cfg.train.ds_sort_mode
        self.max_steps = self.cfg.train.steps_per_episode
        self.episodes_per_object = self.cfg.train.episodes_per_object
        self.consecutive_episodes = 0
        self.cur_object_episode = 1
        self.objects, self.learning_starts = self.get_objects()
        self.object_arr_id = 0
        self.arr_tmp_trained_objs = []
        self.finished_obj_list = False

        self.save_best_images = self.cfg.gym.save_best_images
        self.save_all_images = self.cfg.gym.save_all_images
        self.save_init_images = self.cfg.gym.save_init_images
        self.print_epochs = self.cfg.gym.print_epochs
        self.init_reset = True  # to work around first reset

        self.y_true, self.y_pred, self.output_data = [], [], []

        self.seq_acc_diff = None
        self.seq_y_pred = None
        self.seq_y_true = None
        self.seq_output_data = None
        self.seq_acc_diff_best = -np.inf

        # set specific env settings
        if self.simulator.shift_arm:
            if not self.simulator.ignore_robot:
                self.pos_lower_bounds[1] += 2
                self.pos_upper_bounds[1] += 2
        if self.start_pose is None:
            if self.random_start_pose:
                self.start_pose = self.get_random_pose()
            else:
                self.start_pose = self.simulator.robot.get_pose()
        self.pose = self.start_pose  # [*pos, *orn]

        # set action and observation space
        self.action_space = spaces.Box(np.array([*self.pos_lower_bounds, *self.orn_lower_bounds]),
                                       np.array([*self.pos_upper_bounds, *self.orn_upper_bounds]),
                                       dtype=np.float64)

        self.observation = []
        self.observation_space = spaces.Box(np.array([*self.pos_lower_bounds, *self.orn_lower_bounds,
                                                      -1, 0]),
                                            np.array([*self.pos_upper_bounds, *self.orn_upper_bounds,
                                                      1, self.simulator.num_classes]),
                                            dtype=np.float64)

        # init parameters
        self.data = []  # data storage: [env_episode][episode_dict, env_step_counter]
        # self.data_last = dict.fromkeys(["accuracy", "accuracy_diff", "accuracy_arr",
        #                                 "is_true", "index", "name", "observation"])
        # self.data_curr = dict.fromkeys(["accuracy", "accuracy_diff", "accuracy_arr",
        #                                 "is_true", "index", "name", "observation"])
        # self.data_last = self.data_curr
        #
        # self.data_curr["accuracy"] = accuracy
        # self.data_curr["accuracy_diff"] = accuracy_diff
        # self.data_curr["accuracy_arr"] = accuracy_arr
        # self.data_curr["is_true"] = is_true
        # self.data_curr["index"] = index
        # self.data_curr["name"] = name
        # self.data_curr["observation"] = observation
        #self.data = dict.fromkeys("")
        self.start_prediction_array = []
        self.rgb_image = None
        self.env_step_counter = 0
        self.env_episode = 0
        self.env_step_total = 0
        self.valid_position = None
        self.invalid_position_count = 0
        self.compute_done_string = ""
        self.log_name = None  # TODO maybe define here instead of task
        self.start_prediction_array = []
        self.t_epoch_arr = []
        self.validating = False   # TODO check if modes

        # data tracking - all time best
        self.accuracy_best = -np.inf
        self.accuracy_diff_best = -np.inf
        self.reward_best = -np.inf
        self.reward_sum = -np.inf
        self.reward_mean = -np.inf

        # for easy tensorboard access
        self.pos_change = None
        self.orn_change = None
        self.pose_change = None

        if not os.path.exists(self.data_log_dir):
            os.mkdir(self.data_log_dir)
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)
        # TODO rework saving images
        #if self.save_best_images or self.save_all_images:
        #    self.check_image_folders()
        #self.check_image_folders()
        self.start_time = datetime.now().replace(microsecond=0)
        self.tmp_time = self.start_time
        self.seed(self.seed_id)
        print("[NextBestViewEnv] initialized!")

    def step(self, action):

        """
        return a tuple with:
        - an observation (observation_array, e.g. [0, 0, 0, 0, 0, 0, 0]
        - a reward: scalar
        - a boolean flag indicates in whether an episode has ended (True triggers _reset)
        - an info dictionary (optional), can be blank dict
        """

        if self.env_step_counter == 0:
            self.valid_position = self.simulator.camera.object_in_view(self.simulator.object)
            self.rgb_image = self.simulator.get_rgb_image()
            accuracy_arr = self.simulator.get_prediction(self.rgb_image)
            index, name, accuracy, is_true = self.simulator.get_object_prediction_info(accuracy_arr, self.object_id)
            accuracy_diff = self.get_accuracy_diff(accuracy_arr, accuracy)
            episode_start_prediction = [accuracy, accuracy_diff]
            self.start_prediction_array.append(episode_start_prediction)

        self.env_step_counter += 1

        # main steps
        self.assign_action_axis(action)
        p.stepSimulation()
        self.observation = self.compute_observation(reset=False)
        reward = self.compute_reward()
        if not self.validating:
            self.update_best_values(reset=False)
        done = self.compute_done()
        if done and not self.validating:
            self.update_episode_data(do_print=True)

        return np.array(self.observation), reward, done, {}

    def assign_action_axis(self, action):
        old_pose = self.pose
        if action is not None:
            self.pose = action

        pos = self.pose[:3]
        orn = self.pose[-4:]
        if self.simulator.ignore_robot:
            p.resetBasePositionAndOrientation(self.simulator.object.id, pos, orn)
        else:
            self.simulator.robot.move_joints(pos, orn)
            self.simulator.attach_object_to_end_effector()

        self.pos_change = sum(abs(np.array(old_pose[:3]) - np.array(pos)))
        self.orn_change = sum(abs(np.array(old_pose[-4:]) - np.array(orn)))
        self.pose_change = self.pos_change + self.orn_change

    def compute_observation(self, reset=False):
        # get observation information
        self.valid_position = self.simulator.camera.object_in_view(self.simulator.object)
        self.rgb_image = self.simulator.get_rgb_image()
        accuracy_arr = self.simulator.get_prediction(self.rgb_image)
        index, name, accuracy, is_true = self.simulator.get_object_prediction_info(accuracy_arr, self.object_id)
        accuracy_diff = self.get_accuracy_diff(accuracy_arr, accuracy)
        if not self.valid_position.any():
            self.invalid_position_count += 1
        else:
            self.invalid_position_count = 0

        self.accuracy = accuracy
        self.accuracy_diff = accuracy_diff

        # tmp for cm
        if accuracy_diff is not None and accuracy_diff > self.seq_acc_diff_best:
            self.seq_acc_diff_best = accuracy_diff
            self.seq_y_pred = index
            self.seq_y_true = self.object_id
            self.seq_output_data = accuracy_arr
        self.seq_acc_diff = accuracy_diff


        # store values in data
        self.pose = self.simulator.robot.get_pose()
        observation = [*self.pose, accuracy_diff, self.object_id]
        if not reset or self.validating:
            self.data[self.env_episode-1][1].append({})  # prepare for data
            self.set_sdata(accuracy, "accuracy")
            self.set_sdata(accuracy_diff, "accuracy_diff")
            self.set_sdata(accuracy_arr, "accuracy_arr")
            self.set_sdata(is_true, "is_true")
            self.set_sdata(index, "index")
            self.set_sdata(name, "name")
            self.set_sdata([], "observation")

        # TODO rename "_best"
            if self.save_all_images:
                obj_dir = os.path.join(self.data_dir, str(self.object_id) + "_" + self.simulator.object.object_name)
                if not os.path.exists(obj_dir):
                    os.makedirs(obj_dir)
                self.check_image_folders(obj_dir)

                img_name = "accuracy_" + str(self.env_episode) + "_" + str(self.env_step_counter) + \
                           "_" + str(round(accuracy, 4))
                utils.save_image(self.rgb_image, img_name, os.path.join(obj_dir, "accuracy"))
                img_name = "accuracy_diff_" + str(self.env_episode) + "_" + str(self.env_step_counter) + \
                           "_" + str(round(accuracy_diff, 4))
                utils.save_image(self.rgb_image, img_name, os.path.join(obj_dir, "accuracy_diff"))

            elif self.save_init_images and not self.save_all_images:
                obj_dir = os.path.join(self.data_dir, str(self.object_id) + "_" + self.simulator.object.object_name)
                if not os.path.exists(obj_dir):
                    os.makedirs(obj_dir)
                self.check_image_folders(obj_dir)

                img_name = "accuracy_" + str(self.env_episode) + "_init_" + str(round(accuracy, 4))
                utils.save_image(self.rgb_image, img_name, os.path.join(obj_dir, "accuracy"))
                img_name = "accuracy_diff_" + str(self.env_episode) + "_init_" + str(round(accuracy_diff, 4))
                utils.save_image(self.rgb_image, img_name, os.path.join(obj_dir, "accuracy_diff"))

        return observation

    def compute_reward(self):
        # get values
        reward = 0
        if self.env_step_counter < 2:
            self.set_sdata(reward, "reward")
            return reward
        acc_diff = self.get_sdata("accuracy_diff")

        if self.valid_position.all():
            reward += acc_diff
        else:
            reward -= 1
        reward *= 10
        self.reward = reward
        self.set_sdata(reward, "reward")
        return reward

    def compute_done(self):
        self.compute_done_string = ""
        is_done = False
        if self.is_in_step_range():
            pass
            # if self.is_out_of_bounds():
            #     self.compute_done_string = "pos_out_of_bounds"
            #     is_done = True
            # elif self.object_moving_out_of_view(self.max_steps/2.0):
            #     self.compute_done_string = "object_moving_out_of_view"
            #     is_done = True
        else:
            self.compute_done_string = "is_not_in_step_size"
            is_done = True

        return is_done

    def reset(self, val=False, val_obj_id=None):
        if not val and not self.validating:
            self.reset_episode()

        skip = self.init_reset or val

        # p.resetSimulation()
        p.setTimeStep(0.01)

        if val:
            rand_pose = self.get_random_pose()
            pos = rand_pose[:3]
            orn = rand_pose[-4:]
            self.simulator.load_object(obj_id=val_obj_id, pos=pos, orn=orn)
            self.object_id = val_obj_id
        else:
            # reset pybullet simulator
            if self.random_start_pose:
                self.start_pose = self.get_random_pose()
            pos = self.start_pose[:3]
            orn = self.start_pose[-4:]
            if self.simulator.ignore_robot:
                p.resetBasePositionAndOrientation(self.simulator.object.id, pos, orn)
            else:
                if self.random_start_pose:
                    self.simulator.reset(self.start_pose)
                else:
                    self.simulator.reset()
            if not skip:
                self.consecutive_episodes += 1  # "which cons_eps would come now"
            if self.ds_sort_mode in [0, 1, 3]:
                if not skip and self.object_id < self.simulator.num_classes:
                    if self.consecutive_episodes >= self.episodes_per_object:
                        if self.ds_sort_mode in [0, 1]:
                            if self.object_arr_id == self.simulator.num_classes - 1:
                                self.object_arr_id = 0
                            else:
                                self.object_arr_id += 1
                        elif self.ds_sort_mode in [3]:
                            self.object_arr_id = np.random.randint(0, self.simulator.num_classes)
                        self.object_id = self.objects[self.object_arr_id]
                        self.consecutive_episodes = 0
            elif self.ds_sort_mode in [2]:
                # TODO
                self.object_id = random.randint(0, self.simulator.num_classes - 1)
            self.simulator.load_object(obj_id=self.object_id, pos=pos, orn=orn)
            self.tmp_save_trained_objs(self.object_id)
        self.init_reset = False
        # you *have* to compute and return the observation from reset()
        if self.validating:
            self.observation = self.compute_observation(reset=False)
        else:
            self.observation = self.compute_observation(reset=True)
        return np.array(self.observation)

    def render(self, mode="human", close=False):
        # PyBullet is doing this, can be blank
        pass

    def seed(self, seed=None):
        print(f"[NextBestViewEnv] Seed: {seed}")
        np.random.seed(seed)
        random.seed(seed)
        self.np_random, sd = seeding.np_random(seed)
        print(f"[NextBestViewEnv] Numpy seed: {self.np_random}, {sd}")
        return [sd]

    # -------------

    # --- UTILS ---

    # -------------

    def tmp_save_trained_objs(self, obj_id):
        if obj_id not in self.arr_tmp_trained_objs:
            self.arr_tmp_trained_objs.append(obj_id)

        # processing and deleting in tensorboard_callback

    def get_objects(self):
        objs = []
        if self.ds_obj_mode in [0, 1]:
            objs = [x for x in range(0, 18)]
        elif self.ds_obj_mode in [2]:
            objs = [x for x in range(0, 36)]
        objs = objs

        if self.ds_sort_mode == 0:
            pass
        elif self.ds_sort_mode == 1:
            objs.reverse()
        elif self.ds_sort_mode == 2:
            objs = self.cfg.train.custom_objs
        elif self.ds_sort_mode == 3:
            pass

        if self.cfg.algorithm.learning_starts is not None:
            ls = self.cfg.algorithm.learning_starts
        else:
            ls = self.simulator.num_classes * 1000

        print(f"[SAC] LEARNING STARTS WITH {ls}")

        return objs, ls

    def get_accuracy_diff(self, arr, acc):
        indexed = enumerate(arr)
        decorated = ((value, index) for index, value in indexed)
        sorted_pairs = sorted(decorated)
        if sorted_pairs[-1][1] == self.simulator.obj_id:
            val = sorted_pairs[-2][0]
        else:
            val = sorted_pairs[-1][0]

        diff = acc - val
        return diff

    def reset_episode(self):
        # cm

        if self.seq_y_pred is not None:
            self.y_true.append(self.seq_y_true)
            self.y_pred.append(self.seq_y_pred)
            self.output_data.append(self.seq_output_data)
        self.seq_y_pred = None
        self.seq_y_true = None
        self.seq_output_data = None
        self.seq_acc_diff_best = -np.inf

        # general
        self.env_step_total += self.env_step_counter  # update total steps
        self.env_episode += 1  # update episode number
        self.env_step_counter = 0  # update step number of current episode
        self.data.append([{}, []])

        self.invalid_position_count = 0
        self.compute_done_string = ""

        # data tracking - all time best
        self.accuracy_best = -np.inf
        self.accuracy_diff_best = -np.inf
        self.reward_best = -np.inf
        self.reward_sum = -np.inf
        self.reward_mean = -np.inf


    def reset_env(self):
        self.data = []  # data storage: [env_episode][episode_dict, env_step_counter]
        self.rgb_image = None
        self.invalid_position_count = 0
        self.env_step_counter = 0
        self.env_episode = 0
        self.env_step_total = 0
        self.valid_position = None
        self.compute_done_string = ""
        self.start_prediction_array = []

        # data tracking - all time best
        self.accuracy_best = -np.inf
        self.accuracy_diff_best = -np.inf
        self.reward_best = -np.inf
        self.reward_sum = -np.inf
        self.reward_mean = -np.inf

        # for easy tensorboard access
        self.pos_change = None
        self.orn_change = None
        self.pose_change = None
        self.start_time = datetime.now().replace(microsecond=0)
        self.tmp_time = self.start_time

    def update_best_values(self, reset=False):
        if reset:
            return
        accuracy = self.get_sdata("accuracy")
        if accuracy > self.accuracy_best:
            self.accuracy_best = accuracy
            if self.save_best_images and not self.save_all_images:
                img_name = "accuracy_best" + str(self.env_episode) + "_" + str(self.env_step_counter) + \
                           "_" + str(round(accuracy, 4))
                utils.save_image(self.rgb_image, img_name, os.path.join(self.data_dir, "accuracy_best"))

        accuracy_diff = self.get_sdata("accuracy_diff")
        if accuracy_diff > self.accuracy_diff_best:
            self.accuracy_diff_best = accuracy_diff
            if self.save_best_images and not self.save_all_images:
                img_name = "accuracy_diff_best" + str(self.env_episode) + "_" + str(self.env_step_counter) + \
                           "_" + str(round(accuracy_diff, 4))
                utils.save_image(self.rgb_image, img_name, os.path.join(self.data_dir, "accuracy_diff_best"))

        reward = self.get_sdata("reward")
        if reward > self.reward_best:
            self.reward_best = reward
            if self.save_best_images and not self.save_all_images:
                img_name = "reward_best" + str(self.env_episode) + "_" + str(self.env_step_counter) + \
                           "_" + str(round(reward, 4))
                utils.save_image(self.rgb_image, img_name, os.path.join(self.data_dir, "reward_best"))

    def update_episode_data(self, do_print=False):
        self.set_edata(self.env_step_counter, "num_steps")
        self.set_edata(self.compute_done_string, "compute_done")

        self.set_edata(self.accuracy_best, "accuracy_best")
        self.set_edata(self.accuracy_diff_best, "accuracy_diff_best")
        self.set_edata(self.reward_best, "reward_best")

        reward_sum = 1 # TODO buggy np.sum(self.get_sdata_array("reward"))
        reward_mean = reward_sum / self.env_step_counter

        self.set_edata(reward_sum, "reward_sum")
        self.set_edata(reward_mean, "reward_mean")

        t_train = datetime.now().replace(microsecond=0) - self.start_time
        t_epoch = datetime.now().replace(microsecond=0) - self.tmp_time
        self.t_epoch_arr.append(t_epoch)
        left_epoch = self.cfg.train.episodes - self.env_episode

        tmp_t_epoch_arr = self.t_epoch_arr
        k = self.env_episode / 10 * self.env_step_counter
        k = int(k)
        if self.env_episode <= k:
            k = self.env_episode
        else:
            for x in range(0, int(k / 4)):
                tmp_t_epoch_arr = self.t_epoch_arr[-k:]
                tmp_t_epoch_arr.remove(np.max(tmp_t_epoch_arr))
                tmp_t_epoch_arr.remove(np.min(tmp_t_epoch_arr))

        t_epoch_avg = np.mean(self.t_epoch_arr)
        t_est = left_epoch * t_epoch_avg
        t_est /= 60.0

        if reward_mean >= 0.0:
            str_reward_mean = f"reward_mean =  {reward_mean:.2f}"
        else:
            str_reward_mean = f"reward_mean = {reward_mean:.2f}"
        if self.accuracy_diff_best >= 0.0:
            str_acc_diff_best = f"acc_diff_best =  {self.accuracy_diff_best:.2f} | "
        else:
            str_acc_diff_best = f"acc_diff_best = {self.accuracy_diff_best:.2f} | "

        if self.print_epochs and do_print and (self.env_episode % 100 == 0):
            print(f"[{self.env_episode}] "
                  f"(obj: {self.object_id}) "
                  f"{str_reward_mean} | "
                  f"{str_acc_diff_best} | "
                  f"t_epoch: {t_epoch} | "
                  f"t_train: {t_train} | "
                  f"est: {t_est} min")

        self.tmp_time = datetime.now().replace(microsecond=0)
        self.env_step_counter = 0

    # --- IS_DONE UTILS ---

    def is_out_of_bounds(self):
        """
        :return: True, if object is out of bounds
        """
        pos = self.get_sdata("observation")[:3]
        for i, val in enumerate(pos):
            if val < self.pos_lower_bounds[i] or val > self.pos_upper_bounds[i]:
                return True
        return False

    def is_in_step_range(self):
        """
        :return: True, if valid step value
        """
        return self.env_step_counter < self.max_steps

    def object_moving_out_of_view(self, t):
        """
        :return if object is t consecutive steps out of view / image
        """
        if self.invalid_position_count > t:
            return True
        return False

    def image_is_empty(self):
        """
        :return if empty is blank
        """
        all_same = np.all(self.rgb_image == 255) or np.all(self.rgb_image == 0)
        return all_same

    # --- POSE UTILS ---

    def get_random_position(self):
        """
        Create a random position with boundaries
        :return position vector
        """
        vector = utils.get_random_position(self.pos_lower_bounds, self.pos_upper_bounds)
        return vector

    def get_random_orientation(self):
        """
        Create a random orientation with boundaries
        :return orientation quaternion
        """
        quaternion = utils.get_random_orientation(self.orn_lower_bounds, self.orn_upper_bounds)
        return quaternion

    def get_random_pose(self):
        """
        Create a random pose with boundaries
        :return pose [*position_vector, *orientation_quaternion]
        """
        position = self.get_random_position()
        orientation = self.get_random_orientation()
        pose = [*position, *orientation]
        return pose

    # --- DATA UTILS ---

    # TODO overwrite_acc_diff hack
    def compute_evaluation_json(self, n_eval_episodes, round_to=4, overwrite_acc_diff=False):
        d = {"accuracy": [],
             "accuracy_diff": [],
             "acc_improvements_first_pose_mean": [],
             "acc_improvements_best_pose_mean": [],
             "acc_diff_improvements_first_pose_mean": [],
             "acc_diff_improvements_best_pose_mean": []
             }

        y_true, y_pred = [], []
        for i in range(1, n_eval_episodes + 1):
            e = {}
            acc_arr = self.get_sdata_array(key_1="accuracy", episode=i)
            start_pose = round(self.start_prediction_array[i-1][0], round_to)
            e["start_pose"] = start_pose
            for j in range(0, self.cfg.evaluate.steps_per_episode):
                e["pose_" + str(j+1)] = round(acc_arr[j], round_to)
            m = max(acc_arr)
            e["best_pose"] = round(m, round_to)
            e["steps"] = acc_arr.index(m) + 1
            e["improvement_to_best_pose"] = round(m - start_pose, round_to)
            e["improvement_to_first_pose"] = round(e["pose_1"] - start_pose, round_to)
            d["accuracy"].append(e)

        if overwrite_acc_diff:
            d["accuracy_diff"].append(e)
        else:
            e = {}
            acc_diff_arr = self.get_sdata_array(key_1="accuracy_diff", episode=i)
            start_pose = round(self.start_prediction_array[i-1][1], round_to)
            e["start_pose"] = start_pose
            for j in range(0, self.cfg.evaluate.steps_per_episode):
                e["pose_" + str(j+1)] = round(acc_diff_arr[j], round_to)
            m = max(acc_diff_arr)
            e["best_pose"] = round(m, round_to)
            e["steps"] = acc_diff_arr.index(m) + 1
            e["improvement_to_best_pose"] = round(m - start_pose, round_to)
            e["improvement_to_first_pose"] = round(e["pose_1"] - start_pose, round_to)
            d["accuracy_diff"].append(e)


        d = self.set_dict_mean(d, "accuracy", "improvement_to_first_pose", "acc_improvements_first_pose_mean")
        d = self.set_dict_mean(d, "accuracy", "improvement_to_best_pose", "acc_improvements_best_pose_mean")
        d = self.set_dict_mean(d, "accuracy_diff", "improvement_to_first_pose", "acc_diff_improvements_first_pose_mean")
        d = self.set_dict_mean(d, "accuracy_diff", "improvement_to_best_pose", "acc_diff_improvements_best_pose_mean")

        m = self.get_means(d)

        return d, m

    def get_means(self, d):
        means = {"acc_fp": self.get_mean(d, "accuracy", "pose_1"),
                 "acc_diff_fp": self.get_mean(d, "accuracy_diff", "pose_1"),
                 "acc_bp": self.get_mean(d, "accuracy", "best_pose"),
                 "acc_diff_bp": self.get_mean(d, "accuracy_diff", "best_pose"),
                 "acc_impr_fp": self.get_mean(d, "accuracy", "improvement_to_first_pose"),
                 "acc_diff_impr_fp": self.get_mean(d, "accuracy_diff", "improvement_to_first_pose"),
                 "acc_impr_bp": self.get_mean(d, "accuracy", "improvement_to_best_pose"),
                 "acc_diff_impr_bp": self.get_mean(d, "accuracy_diff", "improvement_to_best_pose"),
                 "acc_diff_bp_steps": self.get_mean(d, "accuracy_diff", "steps")
                 }
        return means

    def get_mean(self, d, main_key, sub_key):
        sum = 0.0
        count = 0
        data = d[main_key]
        for val in data:
            count += 1
            sum += val[sub_key]
        mean = sum / count
        return mean

    def set_dict_mean(self, d, main_key, improvement_key, target_key, round_to=4):
        plain_list = [[] for k in range(self.max_steps)]
        for num, dic in enumerate(d[main_key]):
            step = dic["steps"]
            plain_list[step-1].append(dic[improvement_key])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            new_list = list(map(lambda x: round(np.mean(x), round_to), plain_list))
        nums = list(map(lambda x: len(x), plain_list))
        for j in range(0, self.max_steps):
            g = {"num": nums[j], "mean": new_list[j]}
            d[target_key].append(g)
        return d

    def get_sdata_array(self, key_1, key_2=None, episode=None):
        """
        Get an array of all values of the input step key(s)
        """
        if not episode:
            episode = self.env_episode
        if key_2:
            arr = [self.get_sdata(key_1, key_2, episode, x) for x in range(1, len(self.data[episode-1][1]) + 1)]
        else:
            arr = [self.get_sdata(key_1, None, episode, x) for x in range(1, len(self.data[episode-1][1]) + 1)]
        return arr

    def get_arr_data_episode_key(self, key_1, key_2=None):
        """
        Get an array of all values of the input episode key(s)
        """
        if key_2:
            arr = [self.get_edata(key_1, key_2, x) for x in range(0, len(self.data))]
        else:
            arr = [self.get_edata(key_1, None, x) for x in range(0, len(self.data))]
        return arr

    def get_sdata(self, key_1=None, key_2=None, episode=None, step=None):
        """
        Get an dict entry in data[episode][1][step] for the input key(s)
        """
        value = None
        if episode is None:
            episode = self.env_episode
        if step is None:
            step = self.env_step_counter

        try:
            if key_1:
                if key_2:
                    value = self.data[episode-1][1][step-1][key_1][key_2]
                else:
                    value = self.data[episode-1][1][step-1][key_1]
            else:
                value = self.data[episode-1][1][step-1]
        except (KeyError, IndexError) as e:
            print(f"val: {value}")
            print(f"eps: {episode-1}, step: {step-1} key_1: {key_1}, key_2: {key_2}")
            # print(f"data: {self.data[episode-1][1]}")
            raise KeyError("invalid dict keys", e)
        return value

    def get_edata(self, key_1=None, key_2=None, episode=None):
        """
        Get an dict entry in data[episode][0[ for the input key(s)
        """
        value = None
        if episode is None:
            episode = self.env_episode
        try:
            if key_1:
                if key_2:
                    value = self.data[episode-1][0][key_1][key_2]
                else:
                    value = self.data[episode-1][0][key_1]
            else:
                value = self.data[episode-1][0]
        except (KeyError, IndexError) as e:
            print(f"val: {value}")
            print(f"eps: {episode-1}, key_1: {key_1}, key_2: {key_2}")
            # print(f"data: {self.data[episode-1][0]}")
            raise KeyError("invalid dict keys", e)
        return value

    def set_sdata(self, value, key_1, key_2=None, episode=None, step=None):
        """
        Sets an dict entry in data[episode][1][step] for the input key(s)
        """
        if episode is None:
            episode = self.env_episode
        if step is None:
            step = self.env_step_counter
        try:
            if key_2:
                self.data[episode-1][1][step-1][key_1][key_2] = value
            else:
                self.data[episode-1][1][step-1][key_1] = value
        except (KeyError, IndexError) as e:
            print(f"val: {value}")
            print(f"eps: {episode-1}, key_1: {key_1}, key_2: {key_2}")
            # print(f"data: {self.data[episode-1][1]}")
            raise KeyError("invalid dict keys", e)

    def set_edata(self, value, key_1, key_2=None, episode=None):
        """
        Sets an dict entry in data[episode][0] for the input key(s)
        """
        if episode is None:
            episode = self.env_episode
        try:
            if key_2:
                self.data[episode-1][0][key_1][key_2] = value
            else:
                self.data[episode-1][0][key_1] = value
        except (KeyError, IndexError) as e:
            print(f"val: {value}")
            print(f"eps: {episode}, key_1: {key_1}, key_2: {key_2}")
            # print(f"data: {self.data[episode-1][0]}")
            raise KeyError("invalid dict keys", e)

    # --- PRINTING ---

    def print_step(self, num_print_mod=50):
        acc_entry = round(self.get_sdata("accuracy"), 2)
        is_true_entry = self.get_sdata("is_true")
        reward_entry = self.get_sdata("reward")
        print(f"({self.env_episode} | {self.env_step_counter}): Reward = {reward_entry} | "
              f"Accuracy = {acc_entry:.2f} | Prediction = {is_true_entry}")
        print(f"({self.env_episode} | {self.env_step_counter}): Observation: {self.observation}")

    def check_image_folders(self, folder=None, subfolder=None):
        dir = folder
        if dir is None:
            dir = self.data_dir
        if subfolder:
            dir = os.path.join(dir, subfolder)
        dir_acc_best = os.path.join(dir, "accuracy_best")
        dir_acc_diff_best = os.path.join(dir, "accuracy_diff_best")
        dir_acc= os.path.join(dir, "accuracy")
        dir_acc_diff= os.path.join(dir, "accuracy_diff")
        if not os.path.exists(dir_acc_best):
            os.mkdir(dir_acc_best)
        if not os.path.exists(dir_acc_diff_best):
            os.mkdir(dir_acc_diff_best)
        if not os.path.exists(dir_acc):
            os.mkdir(dir_acc)
        if not os.path.exists(dir_acc_diff):
            os.mkdir(dir_acc_diff)
