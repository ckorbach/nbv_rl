#!/usr/bin/env python

import os
import gym
import tensorflow as tf
from datetime import datetime

import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from random import randint

from stable_baselines import SAC
from stable_baselines.sac.policies import MlpPolicy, LnMlpPolicy, CnnPolicy, LnCnnPolicy
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import VecNormalize
# from stable_baselines.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from next_best_view.tensorboard_callback import CustomTensorboardCallback
from next_best_view import utils
import next_best_view
from sklearn.manifold import TSNE
import umap.umap_ as umap
import umap.plot

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class NextBestViewTask:

    def __init__(self, cfg, mode):
        self.cfg = cfg
        self.mode = mode
        self.seed = self.cfg.gym.seed
        self.model = None
        self.id_stamp = utils.make_timestamp_id()
        self.train_path = None
        self.evaluate_path = None
        self.log_path = None
        self.data_path = self.check_folders()
        self.env = self.load_env()
        print(cfg.algorithm.pretty())
        print(f"[Task] Gym nextbestview-v0 created!")

    def load_env(self):
        env = gym.make(self.cfg.gym.name, cfg=self.cfg, data_path=self.data_path)
        # env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
        set_global_seeds(self.seed)
        return env

    def load(self, model_path, env=None):
        if env is None:
            env = self.env
        # load the agent
        self.model = SAC.load(model_path, env)

    def init_model(self, log_path):
        if self.cfg.algorithm.policy == "Mlp":
            policy = MlpPolicy
        elif self.cfg.algorithm.policy == "LnMlp":
            policy = LnMlpPolicy
        elif self.cfg.algorithm.policy == "Cnn":
            policy = CnnPolicy
        elif self.cfg.algorithm.policy == "LnCnn":
            policy = LnCnnPolicy
        else:
            raise ValueError("No valid policy.")
        print("SEED SAC ALGO:", self.seed)
        if self.seed is None:
            self.seed = randint(0, self.cfg.algorithm.max_seed)
        model = SAC(policy,
                    self.env,
                    learning_rate=self.cfg.algorithm.learning_rate,
                    buffer_size=self.cfg.algorithm.buffer_size,
                    batch_size=self.cfg.algorithm.batch_size,
                    target_update_interval=self.cfg.algorithm.target_update_interval,
                    ent_coef=self.cfg.algorithm.ent_coef,
                    verbose=self.cfg.algorithm.verbose,
                    seed=self.seed,
                    learning_starts=self.env.learning_starts,
                    tensorboard_log=log_path,
                    n_cpu_tf_sess=self.cfg.algorithm.n_cpu_tf_sess)
        # model.get_parameters()
        self.model = model

    def check_folders(self):
        root_path = os.path.dirname(os.path.realpath(__file__ + "/../"))
        data_path = os.path.join(root_path, "data", "data_log")
        train_path = os.path.join(data_path, "train")
        evaluate_path = os.path.join(data_path, "evaluate")
        if not os.path.exists(train_path):
            os.makedirs(train_path)
        if not os.path.exists(evaluate_path):
            os.makedirs(evaluate_path)
        self.train_path = train_path
        self.evaluate_path = evaluate_path

        log_name = self.cfg.evaluate.model_path.split(".")[-2].split("/")[-1] + "_" + str(self.id_stamp)
        self.evaluate_path = os.path.join(self.evaluate_path, log_name)
        #if not os.path.exists(log_path):
        #   os.makedirs(log_path)

        if self.mode == "train":
            return self.train_path
        elif self.mode == "eval":
            return self.evaluate_path

    def train(self):
        timesteps = self.cfg.train.steps_per_episode * self.cfg.train.episodes
        log_name = f"{self.cfg.algorithm.name}_arm={not self.cfg.model.ignore_robot}" \
                   f"_mode-{self.cfg.train.ds_obj_mode}-{self.cfg.train.ds_sort_mode}" \
                   f"_{self.cfg.train.steps_per_episode}steps"
        self.env.log_name = log_name
        # TODO same prob as in 186
        # if self.cfg.train.model_path:
        #     log_name += "_continued"
        self.log_path = os.path.join(self.train_path, log_name + "_" + str(self.id_stamp))
        model_path = os.path.join(self.log_path, log_name)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        # TODO data folder handling
        # TODO txt log file

        self.env.reset_env()

        # load objects
        object_idx = self.env.objects
        if object_idx is not None:
            self.env.simulator.load_object(object_idx[0])
            self.env.object_id = object_idx[0]
        obj_name = self.env.simulator.object.object_name

        # load SAC
        print(self.cfg.train.model_path)
        if False:  # TODO
        # if self.cfg.train.model_path is not None:
            # TODO continue training doesn't really work with callback
            callback = CustomTensorboardCallback(self.env, log_path, self.model)
            self.load(model_path=self.cfg.train.model_path, env=self.env)
            print(f"[Task] Learning agent created! Continue ... !")
            self.model.learn(
                total_timesteps=timesteps,
                callback=callback,
                tb_log_name=model_path,
                log_interval=10)
        else:

            # todo add eval_env
            callback = CustomTensorboardCallback(self.env, self.log_path)
            self.init_model(self.log_path)
            print(f"[Task] Learning agent created! Starting ... !")
            self.model.learn(
                total_timesteps=timesteps,
                callback=callback,
                tb_log_name=model_path,
                log_interval=10)

        if self.mode == 0:
            print(f"[Task] Finished {self.cfg.algorithm.name} learning with object {obj_name} in {(datetime.now() - self.env.start_time)}")
        elif self.mode == 1:
            print(f"[Task] Finished {self.cfg.algorithm.name} learning (sorted) with objects {self.env.objects} in {(datetime.now() - self.env.start_time)}")
        elif self.mode == 2:
            print(f"[Task] Finished {self.cfg.algorithm.name} learning (random) with objects {self.env.objects} in {(datetime.now() - self.env.start_time)}")

        self.model.save(model_path)
        print("[Task] Saving final model to %s" % model_path)
        print("[Task] Start evaluation ...")
        # self.evaluate(train_eval=True)
        self.env.close()

    # --- EVALUATION ---

    def evaluate_policy(self):
        self.data = []
        n_runs = self.cfg.evaluate.n_eval_episodes
        mean_reward, std_reward = evaluate_policy(self.model, self.env, n_eval_episodes=n_runs)
        print(f"[Evaluate] mean_reward: {mean_reward}")
        print(f"[Evaluate] std_reward: {std_reward}")
        return mean_reward, std_reward

    def evaluate(self, train_eval=False):
        # Set eval values
        self.env.max_steps = self.cfg.evaluate.steps_per_episode
        self.env.ds_obj_mode = self.cfg.evaluate.ds_obj_mode
        self.env.ds_sort_mode = self.cfg.evaluate.ds_sort_mode
        means = []
        obj_names = []
        self.env.episodes_per_object = self.cfg.evaluate.n_eval_episodes

        if train_eval:
            self.env.print_epochs = False
            for obj_id in range(0, self.env.simulator.num_classes):
                self.env.reset(val=True, val_obj_id=obj_id)
               # self.env.reset_env()
                self.env.object_id = obj_id
                self.env.simulator.load_object(obj_id)
                _, _ = self.evaluate_policy()
                _, m = self.env.compute_evaluation_json(self.cfg.evaluate.n_eval_episodes)
                means.append(m)
                obj_names.append(self.env.simulator.object.object_name)
        else:
            #self.env.save_all_images = True
            self.load(model_path=self.cfg.evaluate.model_path)
            for obj_id in self.cfg.evaluate.objects:
                self.env.object_id = obj_id
                self.env.simulator.load_object(obj_id)
                _, _ = self.evaluate_policy()
                d, m = self.env.compute_evaluation_json(self.cfg.evaluate.n_eval_episodes, overwrite_acc_diff=True)
                if self.cfg.evaluate.process_data:
                    self.process_eval_data(d, obj_id)
                    means.append(m)
                    obj_names.append(self.env.simulator.object.object_name)
                self.env.reset_env()

        self.get_confusion_matrix()
        self.print_means(means, obj_names)

    def get_confusion_matrix(self):
        data = {"y_Actual": self.env.y_true,
                "y_Predicted": self.env.y_pred
                }
        tmp_labels = []
        for i in range(1, 19):
            tmp_labels.append(str(i))

        df = pd.DataFrame(data, columns=["y_Actual", "y_Predicted"])
        #df = pd.DataFrame(data, index=tmp_labels, columns=tmp_labels)
        confusion_matrix = pd.crosstab(df["y_Actual"], df["y_Predicted"],
                                       rownames=["Actual"], colnames=["Predicted"],
                                       margins=False, dropna=False)
        if self.cfg.classificator.normalize_cm:
            confusion_matrix = confusion_matrix.astype("float") / confusion_matrix.sum(axis=1)[:, np.newaxis]
            print("confusion matrix normalized:")
        confusion_matrix.fillna(0, inplace=True)
        confusion_matrix = confusion_matrix.reindex(range(0, self.env.simulator.num_classes), axis=0, fill_value=0)
        confusion_matrix = confusion_matrix.reindex(range(0, self.env.simulator.num_classes), axis=1, fill_value=0)
        confusion_matrix.sort_values(by="Actual", axis=0, ascending=True, inplace=True)
        confusion_matrix.sort_values(by="Predicted", axis=1, ascending=True, inplace=True)
        plt.figure(figsize=(40, 30))
        sn.set(font_scale=4.0)  # for label size
        sn.heatmap(confusion_matrix, annot=True, fmt=".2f", annot_kws={"size": 40}, vmin=0, vmax=1, xticklabels=tmp_labels, yticklabels=tmp_labels)
        test_path = self.data_path
        plt.savefig(os.path.join(test_path, "agent_confusion_matrix_" + str(self.cfg.evaluate.steps_per_episode) + self.cfg.classificator.file_type), bbox_inches="tight")
        print(f"Saved confusion_matrix to: {test_path}")
        plt.close()

        # metric
        num_samples = self.cfg.evaluate.n_eval_episodes * self.env.simulator.num_classes
        for mode in self.cfg.classificator.modes:
            if mode == "tsne":
                x_reduced = TSNE(n_components=2).fit_transform(self.env.output_data)
            elif mode == "umap":
                x_reduced = umap.UMAP(random_state=42).fit_transform(self.env.output_data)
            else:
                print(f"Mode {self.cfg.classificator.mode} is not defined. Cancel ...")
                return
            y_true = np.array(self.env.y_true)
            y = y_true[:num_samples].flatten()

            plt.figure(figsize=(20, 15))
            plt.rcParams['legend.fontsize'] = 'xx-small'
            colors = plt.cm.get_cmap(self.cfg.classificator.color_map).colors
            for i in range(0, self.env.simulator.num_classes):
                c = colors[i % 18]
                #label = classes[i]
                label = i + 1
                c = np.array(c).reshape(1, -1)
                markerx = "x" if i < 18 else "o"  # for distractors vs objects
                print(label, len(x_reduced[y == i]))
                plt.scatter(x_reduced[y == i, 0][:], x_reduced[y == i, 1][:],
                            label=label, c=c, s=self.cfg.classificator.size, marker=markerx, alpha=0.8)
                # plt.scatter(x_reduced[y == i, 0], x_reduced[y == i, 1], label=label, s=5, marker=markerx, alpha=0.5)
            save_path = os.path.join(test_path,"agent_" + mode + "_")
            if self.cfg.classificator.legend:
                plt.legend(loc='center right', bbox_to_anchor=[1.3, 0.5], fontsize="x-small", markerscale=3.0)
                save_path += "legend"
            plt.savefig(save_path + self.cfg.classificator.file_type, bbox_inches="tight")
            print(f"Saved plot_embedding to: {test_path}")

    # TODO save in log
    def print_means(self, means, obj_names):
        print("[Task] Evaluation results:")
        mean_acc_diff_fp_avg = []
        mean_acc_diff_bp_avg = []
        mean_acc_diff_bp_steps_avg = []
        mean_acc_diff_impr_fp_avg = []
        mean_acc_diff_impr_bp_avg = []
        #log_output_file = os.path.join(self.log_path, "log_output")
        #f = open(log_output_file, "w")
        for index, obj_id in enumerate(self.env.objects):
            mean_acc_diff_first_pose = means[index]["acc_diff_fp"]
            mean_acc_diff_best_pose = means[index]["acc_diff_bp"]
            mean_acc_diff_best_pose_steps = means[index]["acc_diff_bp_steps"]
            mean_acc_diff_impr_first_pose = means[index]["acc_diff_impr_fp"]
            mean_acc_diff_impr_best_pose = means[index]["acc_diff_impr_bp"]
            log_output = f"[Eval] {obj_names[index]}: "\
                         f"first: {mean_acc_diff_first_pose:.2f} ({mean_acc_diff_impr_first_pose:.2f}) | "\
                         f"best: {mean_acc_diff_best_pose:.2f} | ({mean_acc_diff_impr_best_pose:.2f}) | "\
                         f"steps: {mean_acc_diff_best_pose_steps:.2f}"
            print(log_output)
            # f.write(log_output)

            mean_acc_diff_fp_avg.append(mean_acc_diff_first_pose)
            mean_acc_diff_bp_avg.append(mean_acc_diff_best_pose)
            mean_acc_diff_bp_steps_avg.append(mean_acc_diff_best_pose_steps)
            mean_acc_diff_impr_fp_avg.append(mean_acc_diff_impr_first_pose)
            mean_acc_diff_impr_bp_avg.append(mean_acc_diff_impr_best_pose)

        mean_avg_fp = np.mean(mean_acc_diff_fp_avg)
        mean_avg_bp = np.mean(mean_acc_diff_bp_avg)
        mean_avg_impr_fp = np.mean(mean_acc_diff_impr_fp_avg)
        mean_avg_impr_bp = np.mean(mean_acc_diff_impr_bp_avg)
        mean_avg_bp_steps = np.mean(mean_acc_diff_bp_steps_avg)
        log_output_avg =f"[Eval] Average: "\
                        f"first: {mean_avg_fp:.2f} ({mean_avg_impr_fp:.2f}) | "\
                        f"best: {mean_avg_bp:.2f} | ({mean_avg_impr_bp:.2f}) | "\
                        f"steps: {mean_avg_bp_steps:.2f}"
        print(log_output_avg)
        # f.write(log_output)
        # f.close()

    def process_eval_data(self, eval_data, obj_id):
        # TODO save name global file
        log_name = f"eval_{self.cfg.algorithm.name}_mode{self.cfg.evaluate.ds_obj_mode}-{self.cfg.evaluate.ds_sort_mode}_id{obj_id}"
        log_name + "_" + str(self.id_stamp)
        log_path = os.path.join(self.evaluate_path, log_name)
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        print("--------------------")
        print(f"save_name: {log_path}")
        print(f"model_path: {self.cfg.evaluate.model_path}")

        # get data
        df_acc = pd.DataFrame(eval_data["accuracy"])
        df_acc_diff = pd.DataFrame(eval_data["accuracy_diff"])

        ### BEST POSE ###
        # IMPROVEMENT #

        #print("----- ACCURACY -----")
        df_acc.plot.scatter(x="start_pose", y="improvement_to_best_pose", c="steps", s=50, colormap="gist_rainbow", ylim=(-1, 1),
                            xlim=(0, 1))
        plt.axis([None, None, None, None])
        plt.axhline(0, color="k", alpha=0.5)
        save_path = os.path.join(log_path, log_name) + "_best_pose_acc_improvement"
        plt.savefig(save_path, bbox_inches="tight")
        #print(f"Saved plot to: {save_path}")
        plt.close()

        #print("----- ACCURACY DIFF -----")
        df_acc_diff.plot.scatter(x="start_pose", y="improvement_to_best_pose", c="steps", s=50, colormap="gist_rainbow",
                                 ylim=(-2, 2), xlim=(-1, 1))
        plt.axhline(0, color="k", alpha=0.5)
        plt.axvline(0, color="k", alpha=0.5)
        plt.axis([None, None, None, None])
        save_path = os.path.join(log_path, log_name) + "_acc_diff_improvement"
        save_path = log_path + "_best_pose_acc_diff_improvement"
        plt.savefig(save_path, bbox_inches="tight")
        #print(f"Saved plot to: {save_path}")
        plt.close()

        # POSE #

        #print("----- ACCURACY -----")
        df_acc.plot.scatter(x="start_pose", y="best_pose", c="steps", s=50, colormap="gist_rainbow", ylim=(0, 1),
                            xlim=(0, 1))
        plt.axis([None, None, None, None])
        # x1, y1 = [1, 0], [0, 1]
        # plt.plot(x1, y1, marker='o')
        save_path = os.path.join(log_path, log_name) + "_best_pose_acc"
        plt.savefig(save_path, bbox_inches="tight")
        #print(f"Saved plot to: {save_path}")
        plt.close()

        #print("----- ACCURACY DIFF -----")
        df_acc_diff.plot.scatter(x="start_pose", y="best_pose", c="steps", s=50, colormap="gist_rainbow", ylim=(-1, 1),
                                 xlim=(-1, 1))
        plt.axhline(0, color="k", alpha=0.5)
        plt.axvline(0, color="k", alpha=0.5)

        plt.axis([None, None, None, None])
        save_path = log_path + "_best_pose_acc_diff"
        plt.savefig(save_path, bbox_inches="tight")
        plt.savefig(save_path, bbox_inches="tight")
        #print(f"Saved plot to: {save_path}")
        plt.close()


        ### FIRST POSE ###
        # IMPROVEMENT #

        #print("----- ACCURACY -----")
        df_acc.plot.scatter(x="start_pose", y="improvement_to_first_pose", s=50, ylim=(-1, 1),
                            xlim=(0, 1))
        plt.axis([None, None, None, None])
        plt.axhline(0, color="k", alpha=0.5)
        save_path = os.path.join(log_path, log_name) + "_first_pose_acc_improvement"
        plt.savefig(save_path, bbox_inches="tight")
        #print(f"Saved plot to: {save_path}")
        plt.close()

        #print("----- ACCURACY DIFF -----")
        df_acc_diff.plot.scatter(x="start_pose", y="improvement_to_first_pose", s=50, ylim=(-2, 2), xlim=(-1, 1))
        plt.axhline(0, color="k", alpha=0.5)
        plt.axis([None, None, None, None])
        save_path = os.path.join(log_path, log_name) + "_first_pose_acc_diff_improvement"
        save_path = log_path + "_acc_diff_improvement"
        plt.savefig(save_path, bbox_inches="tight")
        #print(f"Saved plot to: {save_path}")
        plt.close()

        # POSE #

        #print("----- ACCURACY -----")
        df_acc.plot.scatter(x="start_pose", y="pose_1", s=50,  ylim=(0, 1), xlim=(0, 1))
        plt.axis([None, None, None, None])
        # x1, y1 = [1, 0], [0, 1]
        # plt.plot(x1, y1, marker='o')
        save_path = os.path.join(log_path, log_name) + "_first_pose_acc"
        plt.savefig(save_path, bbox_inches="tight")
        #print(f"Saved plot to: {save_path}")
        plt.close()

        #print("----- ACCURACY DIFF -----")
        df_acc_diff.plot.scatter(x="start_pose", y="pose_1", s=50, ylim=(-1, 1), xlim=(-1, 1))
        plt.axhline(0, color="k", alpha=0.5)
        plt.axvline(0, color="k", alpha=0.5)

        plt.axis([None, None, None, None])
        save_path = log_path + "_first_pose_acc_diff"
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

        # --- LATEX --- #
        with open(save_path + ".tex", 'w') as tf:
            tf.write(df_acc.to_latex())
        # with open(save_path + "_acc_average.tex", 'w') as tf:
        #     df_acc_average = pd.DataFrame(eval_data["accuracy_improvement_mean"])
        #     df_acc_average.index = np.arange(1, len(df_acc_average) + 1)
        #     tf.write(df_acc_average.to_latex())

        with open(save_path + "_diff.tex", 'w') as tf:
            tf.write(df_acc_diff.to_latex())
        # with open(save_path + "_diff_average.tex", 'w') as tf:
        #     df_acc_diff_average = pd.DataFrame(eval_data["accuracy_diff_improvement_mean"])
        #     tf.write(df_acc_diff_average.to_latex())

        print(f"Saved plots to: {save_path}")
    def save_plt_img(self, save_path, bbox_inches="tight", plt_close=True):
        plt.savefig(save_path, bbox_inches=bbox_inches)
        #print(f"Saved plot to: {save_path}")
        if plt_close:
            plt.close()
