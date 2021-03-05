"""
input: json
[[time_stamp, step, value], [...]
"""

import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import numpy as np
from scipy.signal import savgol_filter
import re
import ntpath


### natural sorting

def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


plt.close("all")


class TensorboardPlotter:
    def __init__(self, key=None, root_path=None, subdir_path=None, smooth_setting=[101, 1, "nearest"]):
        self.smooth_setting = smooth_setting
        self.key = key
        self.root_path = root_path
        self.subdir_path = subdir_path
        self.colors = ["b", "g", "r", "c", "m", "y", "k", "limegreen", "orange", "dodgerblue"]
        self.color_index = 0
        plt.figure(figsize=(10, 10))

    # def plot2(self, legend=None, title=None, save_name=None, neg_val=False, loc="lower right"):
    #     # style
    #     plt.style.use('seaborn-darkgrid')
    #
    #     # create a color palette
    #     palette = plt.get_cmap('Set1')
    #
    #     # multiple line plot
    #     num = 0
    #     for column in self.df.drop('x', axis=1):
    #         num += 1
    #     plt.plot(self.df['x'], self.df[column], marker='', color=palette(num), linewidth=1, alpha=0.9, label=column)
    #
    #     # Add legend
    #     plt.legend(loc=2, ncol=2)
    #
    #     # Add titles
    #     plt.title("A (bad) Spaghetti plot", loc='left', fontsize=12, fontweight=0, color='orange')
    #     plt.xlabel("Time")
    #     plt.ylabel("Score")

    def plot(self, legend=None, title=None, save_name=None, neg_val=False, loc="lower right"):

        if neg_val:
            plt.ylim(bottom=-1, top=1)
            plt.axhline(0, color="k", alpha=0.3)
            plt.yticks(np.arange(-1.0, 1.1, step=0.1))
            x_line_start, x_line_num = -1, 20
        else:
            plt.ylim(bottom=0, top=1)
            plt.yticks(np.arange(1.1, step=0.1))
            x_line_start, x_line_num = 0, 10
        plt.xticks(np.arange(0, 1000000, step=100000))
        plt.xticks(fontsize="larger")
        plt.yticks(fontsize="x-large")

        for i in range(0, x_line_num):
            x_line_start += 0.1
            plt.axhline(x_line_start, color="k", alpha=0.1)

        y_line_start, y_line_num = -100000, 10
        for i in range(0, y_line_num):
            y_line_start += 100000
            plt.axvline(y_line_start, color="k", alpha=0.1)

        ax = plt.axes()
        ax.set_xlabel("steps ", labelpad=10, fontsize=20)
        ax.set_ylabel("validation sequence mean (confidence difference in %)", labelpad=10, fontsize=20)
        # ax.set_facecolor("lightgrey")

        if title:
            if legend is None:
                legend = plt.legend(title=title, loc=loc, shadow=False, fontsize="xx-large")
            else:
                legend = plt.legend(legend, title=title, loc=loc, shadow=False, fontsize="xx-large")

            # legend.get_frame().set_facecolor("C0")
        if save_name:
            save_path = os.path.dirname(os.path.realpath(__file__))
            save_path = os.path.join(save_path, save_name)
            plt.savefig(save_path, bbox_inches="tight")
            print(f"Saved plot to: {save_path}")
        plt.show()

    def get_subfolders(self, folder=None, name=True):
        if folder is None:
            folder = self.root_path
        if name:
            subfolders = [f.name for f in os.scandir(folder) if f.is_dir()]
        else:
            subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]
        return subfolders

    def plot_benchmark(self, smooth=False):
        subfolders = self.get_subfolders(name=True)
        subfolders = sorted(subfolders, key=natural_key)
        print(subfolders)
        root_tmp = self.root_path
        for sub in subfolders:
            print(sub)
            path = os.path.join(self.root_path, sub)
            self.root_path = path
            print(path)
            print(f"Add group {sub}")
            self.add_group(folder_path=path, smooth=smooth)
            self.root_path = root_tmp


    def compute_fill(self, json_arr):
        """
        :require: steps need to be the same
        """
        # init values
        values_arr = []
        steps = None
        for e in json_arr:
            tmp_steps = self.get_steps(e)
            if steps is None or len(tmp_steps) < len(steps):
                steps = tmp_steps
        for json_data in json_arr:
            values_arr.append(self.get_values(json_data))

        # calculate min, max, median
        min_arr = []
        max_arr = []
        mean_arr = []
        for i in range(len(steps)):
            current = list(map(lambda x: x[i], values_arr))
            min_arr.append(np.min(current))
            max_arr.append(np.max(current))
            mean_arr.append(np.mean(current))
        return min_arr, max_arr, mean_arr, steps

    def add_group(self, file_names=None, folder_path=None, color=None, name="", smooth=False):
        if not folder_path:
            folder_path = self.root_path
        if not name:
            name = folder_path
        #json_data_arr = self.load_data(file_names=file_names, folder_path=folder_path, smooth=smooth)
        json_data_arr = self.add_singles(folder_name=folder_path, smooth=smooth, bypass_pd=True)

        min_arr, max_arr, mean_arr, steps = self.compute_fill(json_arr=json_data_arr)
        col = self.get_color(color)
        plt.plot(steps, mean_arr, col, label=name)
        plt.fill_between(steps, min_arr, max_arr,  color=col, alpha=0.07)


    def add_singles(self, folder_name=None, smooth=False, bypass_pd=False):
        json_data_arr = []

        subfolders = self.get_subfolders(folder=self.root_path)
        for subfolder in subfolders:
            subfolder = os.path.join(self.root_path, subfolder)
            sub_path = os.path.join(subfolder, "json")
            file_names = os.listdir(sub_path)
            file_names = sorted(file_names, key=natural_key)
            for file_name in file_names:
                file_name = os.path.join(sub_path, file_name)
                json_data = self.add_single(file_name=file_name, smooth=smooth, bypass_pd=bypass_pd)
                if json_data is not None:
                    json_data_arr.append(json_data)
        return json_data_arr

    def add_single(self, file_name, color=None, smooth=False, bypass_pd=False):
        print(f"[Plotter] add {file_name}")
        pd_data, json_data = self.get_pd(file_name=file_name, smooth=smooth)
        self.df = pd_data
        if pd_data is not None and not bypass_pd:
            plt.plot(pd_data.index, pd_data, self.get_color(color))
        return json_data

    def get_pd(self, json_data=None, file_name=None, smooth=False):
        pd_data = None
        if file_name:
            json_data = self.load_json(file_name, smooth=smooth)
            if json_data is not None:
                pd_data = pd.Series(self.get_values(json_data), index=self.get_steps(json_data))
        return pd_data, json_data

    def get_color(self, color=None):
        if color is None:
            if self.color_index >= len(self.colors):
                self.color_index = 0
            color = self.colors[self.color_index]
            self.color_index += 1
        return color

    def load_data(self, file_names=None, folder_path=None, smooth=False):
        print(file_names)
        print("!")
        json_data_arr = []
        if file_names is None:
            if folder_path:
                file_names = os.listdir(os.path.join(self.root_path, folder_path))
            else:
                file_names = os.listdir(self.root_path)

        for file_name in file_names:
            if folder_path:
                json_data = self.load_json(file_name, folder_path=folder_path, smooth=smooth)
            else:
                json_data = self.load_json(file_name, smooth=smooth)
            if json_data is not None:
                json_data_arr.append(json_data)
        return json_data_arr


    def check_key(self, file_name):
        file_name_base = os.path.basename(file_name)
        key = self.key
        k = key.split("_")
        k_len = len(k)
        #k.append("json")

        fn = file_name_base.split(".")
        # f = re.findall(r"[\w']+", fn[0])
        f = re.split(", |_|-|!", fn[0])
        #f = fn[0].split("_")

        return f[-k_len:] == k


    # def load_data(self, file_names=None, folder_path=None, smooth=False):
    #     json_data_arr = []
    #     if file_names is None:
    #         if folder_path:
    #             file_names = os.listdir(os.path.join(self.root_path, folder_path))
    #         else:
    #             file_names = os.listdir(self.root_path)
    #     file_names = sorted(file_names, key=natural_key)
    #     for file_name in file_names:
    #         if folder_path:
    #             json_data = self.load_json(file_name, folder_path=folder_path, smooth=smooth)
    #         else:
    #             json_data = self.load_json(file_name, smooth=smooth)
    #         json_data_arr.append(json_data)
    #     return json_data_arr

    def load_json(self, file_name, folder_path=None, smooth=False):
        """
        return: json data with [x] = [time_stamp, step, value]
        """
        json_data = None
        if file_name and self.check_key(file_name):
            with open(file_name, "r") as f:
                json_data = json.load(f)
                if smooth:
                    json_data = self.smooth_data(json_data)
        else:
            if not self.check_key(file_name):
                print("File skipped")
            else:
                print("JSON not found")
        return json_data

    def smooth_data(self, json_data):

        data_smoothed = savgol_filter(list(map(lambda x: x[2], json_data)), window_length=self.smooth_setting[0],
                                      polyorder=self.smooth_setting[1], mode=self.smooth_setting[2])
        for i in range(len(json_data)):
            json_data[i][2] = data_smoothed[i]
        return json_data

    @staticmethod
    def get_steps(json_data):
        steps = list(map(lambda x: x[1], json_data))
        return steps

    @staticmethod
    def get_values(json_data):
        values = list(map(lambda x: x[2], json_data))
        return values

    @staticmethod
    def get_timestamps(json_data):
        timestamps = list(map(lambda x: x[0], json_data))
        return timestamps

