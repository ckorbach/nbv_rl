import pybullet as p
import pybullet_data
import time
import json
import os
from pathlib import Path

from next_best_view import utils
from classification.predictor import Predictor
from next_best_view.pybullet_robot import PyBulletRobot
from next_best_view.pybullet_object import PyBulletObject
from next_best_view.pybullet_camera import PyBulletCamera


class PyBulletSimulator(object):
    def __init__(self, cfg, obj_id=None, obj_orn=None):
        print("[PyBulletSimulator] initializing ...")
        self.cfg = cfg
        self.root_dir = str(Path(__file__).parent.parent)
        self.client_id = p.connect(p.SHARED_MEMORY)
        self.use_gui = self.cfg.simulation.gui_client
        if self.client_id < 0:
            if self.use_gui:
                self.client_id = p.connect(p.GUI,
                                           options="--background_color_red=1.0 --background_color_green=1.0 --background_color_blue=1.0")
            else:
                self.client_id = p.connect(p.DIRECT)

        self.use_real_time_simulation = self.cfg.simulation.use_real_time_simulation
        p.setRealTimeSimulation(self.use_real_time_simulation)

        self.gravity = self.cfg.simulation.gravity
        p.setGravity(0, 0, self.gravity)

        self.robot = PyBulletRobot(self.cfg)
        self.ignore_robot = self.cfg.model.ignore_robot
        self.y_shift = 2
        self.shift_arm = self.cfg.model.shift_arm
        if self.ignore_robot:
            self.shift_arm = True
        else:
            self.shift_arm = self.cfg.model.shift_arm
        if self.shift_arm:
            shifted_pos = self.robot.origin_pos
            shifted_pos[1] += self.y_shift
            self.robot.reset_base_position(shifted_pos)
        time.sleep(1)
        self.camera = PyBulletCamera(self.cfg)
        self.reset_debug_camera(1.2, 90.0, -10.0, [0, self.camera.eye_position[1], 0.6])

        self.obj_id = obj_id
        self.obj_orn = obj_orn
        if self.obj_id is None:
            self.obj_id = self.cfg.simulate.obj_id
        if self.obj_orn is None:
            self.obj_orn = self.cfg.simulate.obj_orn
        self.num_classes = None
        self.class_map = None
        self.object = None
        self.load_class_map()
        print(self.class_map)
        self.predictor = None
        if self.cfg.simulation.predict:
            self.load_model()
        self.rgb_img = None
        self.called_obj_list = [None] * self.num_classes
        self.obj_store_pos = [-1, -1, -1]

        self.load_object(obj_id=self.obj_id, pos=None, orn=None)
        self.reset()
        print("[PyBulletSimulator] initialized!")
        # self.robot.close_gripper()

    def reset(self, pose=None):
        self.rgb_img = None
        if pose:
            # self.robot.reset_base_position(pose[:3], p.getQuaternionFromEuler(pose[3:6]))
            self.robot.move_joints(pose[:3], pose[3:6])
        else:
            self.robot.reset_joint_states()
        if self.object:
            self.attach_object_to_end_effector()

    # --- OTHERS ---

    def move_object(self, pb_obj_id=None, pos=None, orn=None):
        if pos is None:
            pos = [0, 0, 0]
        if orn is None:
            orn = [0, 0, 0, 1]
        if pb_obj_id is None:
            pb_obj_id = self.object.id

        p.resetBasePositionAndOrientation(pb_obj_id, pos, orn)
        return pos, orn

    def attach_object_to_end_effector(self, pb_obj_id=None, pos=None, orn=None):
        # TODO HACK solved validation init pose  # FUNCTION NAME LIES!
        if pos is None:
            pos = [0, 0, 0]
        if orn is None:
            orn = [0, 0, 0, 1]

        if not self.ignore_robot:
            ls = self.robot.get_link_state()
            pos = list(ls[0])
            if self.shift_arm:
                pos[1] -= self.y_shift
            orn = list(ls[1])
        self.move_object(pb_obj_id=pb_obj_id, pos=pos, orn=orn)

    def get_prediction(self, rgb_img=None):
        # get prediction
        if rgb_img is None:
            rgb_img = self.get_rgb_image()
            if rgb_img is None:
                raise AttributeError("No valid rgb_image")
        prediction_arr = self.predictor.predict_image(rgb_img)
        return prediction_arr

    def get_object_prediction_info(self, prediction_arr, class_index):
        # get accuracy of specific class_index
        assert 0 <= class_index <= len(prediction_arr) and isinstance(class_index, int)
        accuracy = prediction_arr[class_index]
        name = self.predictor.get_name_from_index(class_index)
        # TODO check bug
        class_index, _, _, is_true = self.get_predicted_object(prediction_arr)
        return class_index, name, accuracy, is_true

    def get_predicted_object(self, prediction_arr):
        class_index, accuracy = self.predictor.get_predicted_object(prediction_arr)
        name = self.predictor.get_name_from_index(class_index)
        is_true = True if self.object.object_name == name else False
        return class_index, name, accuracy, is_true

    def get_images(self):
        width, height, rgb_img, depth_img, seg_img = \
            p.getCameraImage(width=self.camera.w, height=self.camera.h, viewMatrix=self.camera.view_matrix,
                             projectionMatrix=self.camera.projection_matrix)
        return rgb_img, depth_img, seg_img

    def get_rgb_image(self):
        rgb_img, _, _ = self.get_images()
        return rgb_img

    # --- INIT & LOADING ---

    @staticmethod
    def reset_debug_camera(distance=1.2, yaw=90.0, pitch=-10.0, target_position=None):
        if target_position is None:
            target_position = [0, 0, 0.6]
        p.resetDebugVisualizerCamera(cameraDistance=distance, cameraYaw=yaw,
                                     cameraPitch=pitch, cameraTargetPosition=target_position)

    def load_class_map(self):
        print("[PyBulletSimulator] Load class map ...")
        model_dir = self.cfg.model.path
        class_path = self.root_dir + model_dir
        class_path = os.path.join(class_path, self.cfg.model.class_map)
        print(class_path)
        self.class_map = json.load(open(class_path))
        self.num_classes = len(self.class_map)
        print("[PyBulletSimulator] Class map loaded")

    def load_model(self):
        print("[PyBulletSimulator] Load model and predictor ...")
        print(self.cfg.model.pretty())
        self.predictor = Predictor(cfg=self.cfg)
        print("[PyBulletSimulator] Predictor loaded")

    def load_object(self, obj_id=None, pos=None, orn=None):
        old_obj_id = self.obj_id
        self.obj_id = obj_id
        if self.object:
            old_pb_obj_id = self.object.id
        else:
            old_pb_obj_id = None

        if obj_id is None:
            return

        # TODO can be removed ?!
        # get end_effector position for object spawning
        # ls = p.getLinkState(self.robot.id, self.robot.end_effector_index)
        # spawn_pos = list(ls[0])
        # self.obj_id = obj_id
        # if obj_orn:
        #     self.obj_orn = obj_orn
        # if self.obj_orn:
        #     spawn_orn = self.obj_orn
        # else:
        #     spawn_orn = list(ls[1])
        if self.called_obj_list[obj_id] is None:
            if self.object:
                self.move_object(pb_obj_id=old_pb_obj_id, pos=self.obj_store_pos, orn=self.obj_orn)
            object_name = self.class_map[str(self.obj_id)]
            # file_name = object_name + ".obj" # TODO
            file_name = object_name + ".STL"
            self.object = PyBulletObject(file_name, self.cfg)
            self.called_obj_list[self.obj_id] = self.object
            self.attach_object_to_end_effector(pos=pos, orn=orn)

        else:
            if self.obj_id == old_obj_id:
                self.attach_object_to_end_effector(pos=pos, orn=orn)
            else:
                if self.object:
                    self.move_object(pb_obj_id=old_pb_obj_id, pos=self.obj_store_pos, orn=self.obj_orn)
                self.object = self.called_obj_list[self.obj_id]
                self.attach_object_to_end_effector(pos=pos, orn=orn)
