#!/usr/bin/env python

import os
import numpy as np

import torch.onnx
import onnxruntime

import onnx2keras
from onnx2keras import onnx_to_keras
import keras
import onnx
from classification.basicnet import BasicNet
from tensorflow.python.tools.import_pb_to_tensorboard import import_to_tensorboard
from onnx_tf.backend import prepare

CLASSES = 18
NAME = "yad_objects_random_basicnet_112"
DEFAULT_PATH = "/home/ckorbach/nbv/next_best_view_rl/data/models/classifier/"
TORCH_PATH = os.path.join(DEFAULT_PATH, NAME + ".model")
ONNX_PATH = os.path.join(DEFAULT_PATH, NAME + ".onnx")
KERAS_PATH = os.path.join(DEFAULT_PATH, NAME +  ".h5")
TF_PATH = os.path.join(DEFAULT_PATH, NAME +  ".pb")

print("Load Model")
model = torch.load(TORCH_PATH)

device = torch.device('cuda')
model = BasicNet(CLASSES)
model.load_state_dict(torch.load(TORCH_PATH, map_location=device))

print("Export model to onnx")
batch_size = 1
x = torch.randn(batch_size, 3, 224, 224, requires_grad=True)
torch_out = model
torch.onnx.export(torch_out, x, ONNX_PATH, export_params=True)

print("Check exported model")
onnx_model = onnx.load(ONNX_PATH)
# onnx.checker.check_model(onnx_model)
# ort_session = onnxruntime.InferenceSession(ONNX_PATH)

tf_rep = prepare(onnx_model)
tf_rep.export_graph(TF_PATH)
import_to_tensorboard(TF_PATH.pb, "tb_log")

# def to_numpy(tensor):
#     return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
#
# # compute ONNX Runtime output prediction
# ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
# ort_outs = ort_session.run(None, ort_inputs)
# # compare ONNX Runtime and PyTorch results
# np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
# print("Exported model has been tested with ONNXRuntime, and the result looks good!")
#
# print("Transfer onnx model to keras")
# k_model = onnx_to_keras(onnx_model, ['input_1'])
# keras.models.save_model(k_model, KERAS_PATH,overwrite=True,include_optimizer=True)
# print("Successful")