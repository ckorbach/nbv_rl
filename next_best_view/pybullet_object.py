import pybullet as p
from pathlib import Path


class PyBulletObject:
    def __init__(self, file_name, cfg, spawn_pos=None, spawn_orn=None):
        if cfg.gym.print_level >= 1:
            print("[PyBulletObject] load %s ..." % file_name)
        # print(cfg.objects.pretty())
        if spawn_orn is None:
            spawn_orn = [0, 0, 0, 0]
        if spawn_pos is None:
            spawn_pos = [0, 0, 0]
        self.file_name = file_name
        self.object_name = file_name.split(".")[0]
        root = str(Path(__file__).parent.parent)
        self.file_path = root + cfg.objects.data_path + "/" + self.object_name + "/" + file_name
        self.visual_shape_id = p.createVisualShape(shapeType=p.GEOM_MESH,
                                                   fileName=self.file_path,
                                                   rgbaColor=None,
                                                   meshScale=cfg.objects.mesh_scale)

        self.collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                                         fileName=self.file_path,
                                                         meshScale=cfg.objects.mesh_scale)

        self.id = p.createMultiBody(baseMass=0.5,
                                    # baseCollisionShapeIndex=self.collision_shape_id,
                                    baseVisualShapeIndex=self.visual_shape_id,
                                    basePosition=spawn_pos,
                                    baseOrientation=spawn_orn)

        # create texture
        if cfg.objects.use_texture:
            self.texture_id = p.loadTexture(root + cfg.objects.texture_path)
            p.changeVisualShape(self.id, -1, textureUniqueId=self.texture_id)
        if cfg.gym.print_level >= 1:
            print("[PyBulletObject] initialized!")