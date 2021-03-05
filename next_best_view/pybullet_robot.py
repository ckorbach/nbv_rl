import pybullet as p
import numpy as np
from pathlib import Path
from next_best_view import utils


class PyBulletRobot:
    def __init__(self, cfg):
        print("[PyBulletRobot] initializing ...")
        print(cfg.robot.pretty())
        self.cfg = cfg
        self.name = self.cfg.robot.name
        root = str(Path(__file__).parent.parent)
        self.model = root + self.cfg.robot.urdf_path
        self.origin_pos = self.cfg.robot.origin_pos
        self.origin_orientation = self.cfg.robot.origin_orientation
        self.end_effector_index = self.cfg.robot.end_effector_index
        self.finger_index = self.cfg.robot.finger_index
        self.finger_tip_index = self.cfg.robot.finger_tip_index
        self.use_real_time_simulation = self.cfg.simulation.use_real_time_simulation

        self.jd = self.cfg.robot.jd  # joint damping coefficents
        self.ll = self.cfg.robot.ll  # lower limits
        self.ul = self.cfg.robot.ul  # upper limits
        self.jr = self.cfg.robot.jr  # joint ranges
        self.rp = self.cfg.robot.rp  # rest poses
        self.hp = self.cfg.robot.hp  # home pose

        self.id = p.loadURDF(self.model, self.origin_pos)
        self.reset_base_position(self.origin_pos, self.origin_orientation)

        self.num_fixed_joints = 0
        self.num_movable_joints = 0
        self.num_joints = p.getNumJoints(self.id)
        self.reset_joint_states(print_debug=True)

        print("[PyBulletRobot] initialized!")

    def reset_base_position(self, pos=None, orn=None):
        pass
        # if pos:
        #     self.origin_pos = pos
        # if orn:
        #     self.origin_orientation = orn
        # p.resetBasePositionAndOrientation(self.id, self.origin_pos, self.origin_orientation)

    def reset_joint_states(self, print_debug=False):
        if print_debug:
            print("[PyBulletRobot] Reset Joint States")
        for i in range(self.num_joints):
            p.resetJointState(self.id, i, self.hp[i])
            info = p.getJointInfo(self.id, i)
            if info[3] == -1:
                self.num_fixed_joints += 1
            if print_debug:
                print("Joint number: ", i)
                print(" -- Info: ", info)
                print(" -- State: ", p.getJointState(self.id, i))
        self.num_movable_joints = self.num_joints - self.num_fixed_joints
        if print_debug:
            print("Number of joints: ", self.num_joints)
            print("Number of fixed joints: ", self.num_fixed_joints)
            print("Number of movable joints: ", self.num_movable_joints)

    def move_joints(self, pos, orn=False, target_velocity=0, force=500,
                    position_gain=0.01, velocity_gain=1, use_accurate=False):
        if use_accurate:
            joint_poses = self.accurateIK(pos, orn=orn)
        else:
            if not np.any(orn):
                joint_poses = p.calculateInverseKinematics(self.id, self.end_effector_index, pos)
            else:
                joint_poses = p.calculateInverseKinematics(self.id, self.end_effector_index, pos, orn)

        for i in range(self.num_joints):
            q_index = self.get_joint_info(i)[3]
            if q_index > -1:
                if self.use_real_time_simulation:
                    self.set_motor(joint_index=i,
                                   target_position=joint_poses[q_index - 7],
                                   target_velocity=target_velocity,
                                   force=force,
                                   position_gain=position_gain,
                                   velocity_gain=velocity_gain)
                else:
                    p.resetJointState(self.id, i, joint_poses[q_index - 7])

    def set_motor(self, joint_index, target_position,
                  target_velocity=0, force=500, position_gain=0.01, velocity_gain=1):
        p.setJointMotorControl2(bodyIndex=self.id,
                                jointIndex=joint_index,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=target_position,
                                targetVelocity=target_velocity,
                                force=force,
                                positionGain=position_gain,
                                velocityGain=velocity_gain)

    def open_gripper(self, target_velocity=0, force=500, position_gain=0.01, velocity_gain=1):
        for i in range(0, len(self.finger_index)):
            self.set_motor(joint_index=self.finger_index[i],
                           target_position=self.ll[self.finger_index[0]],
                           target_velocity=target_velocity,
                           force=force,
                           position_gain=position_gain,
                           velocity_gain=velocity_gain)
        for i in range(0, len(self.finger_tip_index)):
            self.set_motor(joint_index=self.finger_tip_index[i],
                           target_position=self.ll[self.finger_tip_index[0]],
                           target_velocity=target_velocity,
                           force=force,
                           position_gain=position_gain,
                           velocity_gain=velocity_gain)

    def close_gripper(self, target_velocity=0, force=500, position_gain=0.01, velocity_gain=1):
        for i in range(0, len(self.finger_index)):
            self.set_motor(joint_index=self.finger_index[i],
                           target_position=self.ul[self.finger_index[0]],
                           target_velocity=target_velocity,
                           force=force,
                           position_gain=position_gain,
                           velocity_gain=velocity_gain)
        for i in range(0, len(self.finger_tip_index)):
            self.set_motor(joint_index=self.finger_tip_index[i],
                           target_position=self.ll[self.finger_tip_index[0]],
                           target_velocity=target_velocity,
                           force=force,
                           position_gain=position_gain,
                           velocity_gain=velocity_gain)

    def accurateIK(self, target_position, orn=False, use_null_space=True, max_iter=10, threshold=1e-4):
        """
        https://github.com/erwincoumans/pybullet_robots/blob/master/baxter_ik_demo.py
        """

        joint_poses = None
        close_enough = False
        iter = 0
        dist2 = 1e30

        while not close_enough and iter < max_iter:
            if use_null_space:
                if not orn:
                    joint_poses = p.calculateInverseKinematics(self.id, self.end_effector_index,
                                                               target_position,
                                                               lowerLimits=self.ll, upperLimits=self.ul,
                                                               jointRanges=self.jr, restPoses=self.rp)
                else:
                    joint_poses = p.calculateInverseKinematics(self.id, self.end_effector_index,
                                                               target_position, orn,
                                                               lowerLimits=self.ll, upperLimits=self.ul,
                                                               jointRanges=self.jr, restPoses=self.rp)
            else:
                if not orn:
                    joint_poses = p.calculateInverseKinematics(self.id, self.end_effector_index,
                                                               target_position)
                else:
                    joint_poses = p.calculateInverseKinematics(self.id, self.end_effector_index,
                                                               target_position, orn)

            for i in range(self.num_joints):
                joint_info = self.get_joint_info(i)
                q_index = joint_info[3]
                if q_index > -1:
                    p.resetJointState(self.id, i, joint_poses[q_index - 7])
            ls = self.get_link_state()
            new_pos = ls[4]
            diff = [target_position[0] - new_pos[0],
                    target_position[1] - new_pos[1],
                    target_position[2] - new_pos[2]]
            dist2 = np.sqrt((diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]))
            # print("dist2=", dist2)
            close_enough = (dist2 < threshold)
            iter += 1
        # print("iter=", iter)
        return joint_poses

    def get_link_state(self):
        """
        ls[0] = position
        ls[1] = orientation
        """
        return p.getLinkState(self.id, self.end_effector_index)

    def get_pose(self):
        ls = self.get_link_state()
        pose = [*ls[0], *ls[1]]
        return pose

    def get_joint_info(self, i):
        return p.getJointInfo(self.id, i)
