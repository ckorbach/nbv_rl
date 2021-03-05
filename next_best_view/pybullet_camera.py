import pybullet as p
import numpy as np
from pathlib import Path


class PyBulletCamera:
    def __init__(self, cfg):
        print("[PyBulletCamera] init ...")
        print(cfg.camera.pretty())
        model = cfg.camera.model
        root = str(Path(__file__).parent.parent)
        self.model = root + model
        self.eye_position = cfg.camera.cameraEyePosition
        self.target_position = cfg.camera.cameraTargetPosition
        self.up_vector = cfg.camera.cameraUpVector

        self.fov = cfg.camera.fov
        self.aspect = cfg.camera.aspect
        self.near_val = cfg.camera.nearVal
        self.far_val = cfg.camera.farVal

        self.w = cfg.camera.w
        self.h = cfg.camera.h
        self.F = cfg.camera.focal

        self.id = p.loadURDF(self.model, self.eye_position)

        self.view_matrix = p.computeViewMatrix(cameraEyePosition=self.eye_position,
                                               cameraTargetPosition=self.target_position,
                                               cameraUpVector=self.up_vector)

        self.projection_matrix = p.computeProjectionMatrixFOV(fov=self.fov,
                                                              aspect=self.aspect,
                                                              nearVal=self.near_val,
                                                              farVal=self.far_val)
        print("[PyBulletCamera] initialized!")

    def object_in_view(self, obj):
        """
        :cam PybulletObject
        :return if object is full viewable
        """

        aabb_in_view = [True, True]
        boundaries = p.getAABB(obj.id)
        min_p = np.array(boundaries[0])
        max_p = np.array(boundaries[1])
        x1, y1 = self.point_to_2d(min_p)
        x2, y2 = self.point_to_2d(max_p)
        if not np.array([0 <= x < self.w for x in [x1, x2]]).all():
            aabb_in_view[1] = False
        if not np.array([0 <= y < self.h for y in [y1, y2]]).all():
            aabb_in_view[0] = False
        return np.array(aabb_in_view)

    def point_to_2d(self, p):
        p_x = p[0] - self.eye_position[0]
        p_y = p[1] - self.eye_position[1]
        p_z = p[2] - self.eye_position[2]
        x = self.F * p_y / p_x
        y = self.F * p_z / p_x
        x += self.w / 2
        y += self.h / 2
        return x, y

