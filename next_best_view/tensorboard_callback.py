#!/usr/bin/env python

import numpy as np
import tensorflow as tf
import os

from stable_baselines.common.callbacks import BaseCallback


class CustomTensorboardCallback(BaseCallback):
    def __init__(self, env, log_path=".log", model=None, do_add_summary=True, verbose=0):
        self.env = env
        self.model = model
        self.is_tb_set = False
        self.best_reward_mean = - np.inf
        self.best_ds_acc_diff_mean = - np.inf
        self.log_path = os.path.join(log_path, "tensorboard")
        self.do_add_summary = do_add_summary
        self.validated_eps_arr = []
        self.writer = tf.summary.FileWriter(self.log_path)
        super(CustomTensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        if self.do_add_summary:
            self.add_summary("train/accuracy", self.env.accuracy)
            self.add_summary("train/accuracy_diff", self.env.accuracy_diff)
            self.add_summary("train/step_reward", self.env.reward)
            self.add_summary("movement/pos_change", self.env.pos_change)
            self.add_summary("movement/orn_change", self.env.orn_change)
            self.add_summary("movement/pose_change", self.env.pose_change)

            name = f"train:dataset/{self.env.object_id}"
            self.add_summary(name, self.env.accuracy_diff)

            # if self.env.env_step_counter >= self.env.max_steps - 1 and self.model is not None:
                # print(self.env.env_episode)
                # print(self.env.cfg.train.val_num)
                # print(self.env.env_episode - 1 % self.env.cfg.train.val_num) == 0

        # print(self.env.cfg.train.val_num is None)
        if self.env.cfg.train.val_num is None:
            print(self.env.cfg.train.val_num)
            if self.env.simulator.num_classes == len(self.env.arr_tmp_trained_objs) and self.env.object_id != self.env.arr_tmp_trained_objs[-1]:
                self.validate()
        elif self.env.env_episode >= self.env.cfg.train.val_num and \
                ((self.env.env_episode - 1) % self.env.cfg.train.val_num) == 0\
                and self.env.env_episode not in self.validated_eps_arr:
            self.validate()

    # def _on_training_start(self):
    #     self.validate()

    def validate(self):
        if self.env.env_step_counter >= self.env.max_steps - 1:
            reward_sum = np.sum(self.env.get_sdata_array("reward"))
            reward_mean = reward_sum / self.env.env_step_counter
            if reward_mean > self.best_reward_mean:
                self.model.save(os.path.join(self.log_path, f"{self.env.log_name}_{self.env.env_episode}"))
                print(f"Saving new best model with reward_mean: {reward_mean:.4f} to {self.log_path}")
                self.best_reward_mean = reward_mean
                self.add_summary("train/reward_sum", reward_sum)
                self.add_summary("train/reward_mean", reward_mean)

        self.validated_eps_arr.append(self.env.env_episode)
        self.env.validating = True
        tmp_obj_id = self.env.object_id
        obj_obs_arr_mean = []
        obj_obs_arr_first = []
        ds_obs_arr_mean = []
        ds_obs_arr_first = []

        val_eps_num = 5
        val_steps = 10
        print(f"[[validation] validating accuracy difference with {val_steps} steps and {val_eps_num} episodes per object ...")
        for obj_id in range(0, self.env.simulator.num_classes):
            for eps_num in range(0, val_eps_num):
                obs = self.env.reset(val=True, val_obj_id=obj_id)
                # pos = self.env.start_pose[:3]
                # orn = self.env.start_pose[-4:]
                # self.env.simulator.move_object(pos=pos, orn=orn)
                for i in range(0, val_steps):
                    action, _states = self.model.predict(obs)
                    obs, rewards, done, info = self.env.step(action)
                    obj_obs_arr_mean.append(obs[-2])
                    if i == 0:
                        obj_obs_first = obs[-2]
                        obj_obs_arr_first.append(obj_obs_first)
                    # print(f"obj_id: {obj_id}, i: {i}/{val_steps}")

            obj_acc_diff_mean = np.mean(obj_obs_arr_mean)
            obj_acc_diff_first = np.mean(obj_obs_arr_first)
            ds_obs_arr_mean.append(obj_acc_diff_mean)
            ds_obs_arr_first.append(obj_acc_diff_first)

            name = f"validation:dataset:mean/{self.env.object_id}"
            self.add_summary(name, obj_acc_diff_mean)
            name = f"validation:dataset:first_pose/{self.env.object_id}"
            self.add_summary(name, obj_acc_diff_first)

            obj_obs_arr_mean = []
            obj_obs_arr_first = []

            #print(f"[val](obj {self.env.object_id}) first_pose: {obj_acc_diff_first:.2f} | mean: {obj_acc_diff_mean:.2f}")

        ds_acc_diff_mean = np.mean(ds_obs_arr_mean)
        ds_acc_diff_first = np.mean(ds_obs_arr_first)
        print(f"[validation](acc_diff) first_pose: {ds_acc_diff_first:.2f}  |  mean: {ds_acc_diff_mean:.2f}")
        self.add_summary("validation/mean", ds_acc_diff_mean)
        self.add_summary("validation/first_pose", ds_acc_diff_first)

        if ds_acc_diff_mean > self.best_ds_acc_diff_mean:
            self.model.save(os.path.join(self.log_path, f"{self.env.log_name}_{self.env.env_episode}"))
            print(f"[validation] saving new best model to {self.log_path}")
            self.best_ds_acc_diff_mean = ds_acc_diff_mean

        # set back
        self.env.arr_tmp_trained_objs = []
        self.env.object_id = tmp_obj_id
        self.env.simulator.load_object(self.env.object_id)
        self.env.validating = False

    def _on_rollout_end(self) -> None:
        pass

    def add_summary(self, tag, simple_value):
        summary = tf.Summary(value=[tf.Summary.Value(
            tag=tag, simple_value=simple_value)])
        self.writer.add_summary(summary, self.num_timesteps)

    # def _on_training_end(self) -> None:
