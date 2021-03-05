import os
import numpy as np

import torch
from torchvision.transforms import transforms
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import torch.nn as nn

from sklearn.manifold import TSNE
import umap.umap_ as umap
import umap.plot
import json

from classification.netloader import NetLoader


class Predictor:

    def __init__(self, cfg, test=False):
        self.cfg = cfg
        #torch.manuel_seed(self.cfg.classificator.seed)
        #torch.backends.cudnn.deterministic = True
        #torch.backends.cudnn.benchmark = False
        self.print_debug = False
        self.test = test
        root_dir = self.cfg.system.project_path
        if self.test:
            # self.model_dir_path = root_dir + self.cfg.classificator.test_model_path
            self.model_dir_path = self.cfg.classificator.test_model_path
            self.model_path = os.path.join(self.model_dir_path, self.cfg.classificator.test_model)
            self.class_map_path = os.path.join(self.model_dir_path, self.cfg.classificator.class_map)
            self.net = self.cfg.classificator.net
            self.resize_size = self.cfg.classificator.resize_size
            self.activation = self.cfg.classificator.activation
        else:
            self.model_dir_path = root_dir + self.cfg.model.path
            self.model_path = os.path.join(self.model_dir_path, self.cfg.model.model)
            self.class_map_path = os.path.join(self.model_dir_path, self.cfg.model.class_map)
            self.net = self.cfg.model.net
            self.resize_size = self.cfg.model.resize_size
            self.activation = self.cfg.model.activation
        self.model_name = self.model_path.split("/")[-1].split(".")[0]
        print("[Predictor] Model: %s: " % self.model_path)

        self.class_map = json.load(open(self.class_map_path))
        self.classes = len(self.class_map.keys())

        netloader = NetLoader(model_name=self.net,
                              num_classes=self.classes,
                              resize_size=self.resize_size)
        self.model, self.resize = netloader.get_model()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

        self.cuda_avail = torch.cuda.is_available()
        if self.cuda_avail:
            self.model = self.model.cuda()

        checkpoint = torch.load(self.model_path)
        self.model.load_state_dict(checkpoint)
        self.model.eval()

    def predict_image(self, image, resize=224, use_activation=True):
        if isinstance(image, str):
            image = Image.open(image)
            image = Image.fromarray(np.uint8(image)).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(np.uint8(image)).convert("RGB")

        # Define transformations for the image, should
        transformation = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Pre-process the image
        image_tensor = transformation(image).float()

        # Add an extra batch dimension since pytorch treats all images as batches
        image_tensor = image_tensor.unsqueeze_(0)
        if self.cuda_avail:
            image_tensor.cuda()
            dtype = torch.cuda.FloatTensor
            input = torch.autograd.Variable(image_tensor.type(dtype))
        else:
            # Turn the input into a Variable
            input = Variable(image_tensor)

        # Predict the class of the image
        output = self.model(input)
        if use_activation:
            if self.activation == "softmax":
                output = self.softmax(output)
            elif self.activation == "sigmoid":
                output = self.sigmoid(output)
            elif self.activation is None:
                pass
            else:
                print("[Error] self.activation wrong defined, using None")

        # print(output)
        output_data = output.data.cpu().numpy()

        if self.print_debug:
            print(f"{output}: {output}")
        return output_data[0]

    def get_predicted_object(self, output_data):
        index = output_data.argmax()
        accuracy = output_data[index]
        acc_rounded = float(str(round(accuracy)))
        name = self.get_name_from_index(index)
        if self.print_debug:
            print(f"Predicted Class: {name}, {acc_rounded}")
        return index, accuracy

    def get_name_from_index(self, index):
        return self.class_map[str(index)]

    def evaluate(self):
        classes_ids = list(self.class_map.keys())
        classes = list(self.class_map.values())

        test_path = self.cfg.classificator.test_path
        print(f"Creating confusion_matrix of dataset {test_path}")
        print(f"Predict images ...")
        y_true, y_pred = [], []
        predictions = []
        num_samples = 0
        for i in range(self.classes):
            print(f"... class {i}/{self.classes - 1}")
            object_name = self.class_map[str(i)]
            object_folder = os.path.join(test_path, object_name)
            for j, filename in enumerate(os.listdir(object_folder)):
                if filename.endswith(self.cfg.classificator.file_type):
                    image_path = os.path.join(object_folder, filename)
                    output_data = self.predict_image(image_path)
                    predicted_index, _ = self.get_predicted_object(output_data)
                    i_tmp = i+1
                    predicted_index_tmp = predicted_index + 1

                    y_true.append(i_tmp)
                    y_pred.append(predicted_index_tmp)
                    predictions.append(output_data)
                    num_samples += 1
        print(f"... finished predicting images!")

        data = {"y_Actual": y_true,
                "y_Predicted": y_pred
                }

        df = pd.DataFrame(data, columns=["y_Actual", "y_Predicted"])
        confusion_matrix = pd.crosstab(df["y_Actual"], df["y_Predicted"],
                                       rownames=["Actual"], colnames=["Predicted"],
                                       margins=False, dropna=False)

        if self.cfg.classificator.normalize_cm:
            confusion_matrix = confusion_matrix.astype("float") / confusion_matrix.sum(axis=1)[:, np.newaxis]
            print("confusion matrix normalized:")
        else:
            print("confusion matrix:")
        print(confusion_matrix)
        confusion_matrix.sort_values(by="Actual", axis=0, ascending=True, inplace=True)
        confusion_matrix.sort_values(by="Predicted", axis=1, ascending=True, inplace=True)
        plt.figure(figsize=(40, 30))
        sn.set(font_scale=4.0)  # for label size
        sn.heatmap(confusion_matrix, annot=True, fmt=".2f", annot_kws={"size": 40}, vmin=0, vmax=1)
        plt.savefig(os.path.join(test_path,self.model_name + "_confusion_matrix" + self.cfg.classificator.file_type), bbox_inches="tight")
        print(f"Saved confusion_matrix to: {test_path}")
        plt.close()

        print(num_samples)

        # metric
        for mode in self.cfg.classificator.modes:
            if mode == "tsne":
                x_reduced = TSNE(n_components=2).fit_transform(predictions)
            elif mode == "umap":
                x_reduced = umap.UMAP(random_state=42).fit_transform(predictions)
            else:
                print(f"Mode {self.cfg.classificator.mode} is not defined. Cancel ...")
                return
            y_true = np.array(y_true)
            y = y_true[:num_samples].flatten()

            plt.figure(figsize=(20, 15))
            plt.rcParams['legend.fontsize'] = 'xx-small'
            colors = plt.cm.get_cmap(self.cfg.classificator.color_map).colors
            for i in range(0, len(classes_ids)):
                c = colors[i % 18]
                #label = classes[i]
                label = i + 1
                c = np.array(c).reshape(1, -1)
                markerx = "x" if i < 18 else "o"  # for distractors vs objects
                print(label, len(x_reduced[y == i]))
                plt.scatter(x_reduced[y == i, 0][:], x_reduced[y == i, 1][:],
                            label=label, c=c, s=self.cfg.classificator.size, marker=markerx, alpha=0.8)
                # plt.scatter(x_reduced[y == i, 0], x_reduced[y == i, 1], label=label, s=5, marker=markerx, alpha=0.5)
            save_path = os.path.join(test_path, self.model_name + "_" + mode + "_")
            if self.cfg.classificator.legend:
                plt.legend(loc='center right', bbox_to_anchor=[1.3, 0.5], fontsize="x-small", markerscale=3.0)
                save_path += "legend"
            plt.savefig(save_path + self.cfg.classificator.file_type, bbox_inches="tight")
            print(f"Saved plot_embedding to: {test_path}")
