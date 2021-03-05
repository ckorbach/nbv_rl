import math
import time
from datetime import datetime
import argparse
import pybullet as p
import hydra
from omegaconf import DictConfig
import omegaconf

from next_best_view.pybullet_simulator import PyBulletSimulator
from next_best_view import utils


class Colors:
    BLACK = [0, 0, 0]
    RED = [1, 0, 0]
    GREEN = [0, 0.5, 0]
    BLUE = [0, 0, 1]
    WHITE = [1, 1, 1]


class PyBulletSimulatorTest(PyBulletSimulator):
    def simulate_test(self):
        if self.use_gui:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)

            # init debug parameter
            self.use_accurate_id = p.addUserDebugParameter("useAccurate", 0, 1, 0)
            self.use_debug_id = p.addUserDebugParameter("useDebug", 0, 1, 0)
            self.use_manual_id = p.addUserDebugParameter("useManual", 0, 1, 1)
            self.use_gripper_id = p.addUserDebugParameter("useGripper", 0, 1, 0)
            self.target_pos_x_id = p.addUserDebugParameter("targetPosX", -1, 1, 0.3)
            self.target_pos_y_id = p.addUserDebugParameter("targetPosY", -1, 1, 0.2)
            self.target_pos_z_id = p.addUserDebugParameter("targetPosZ", -1, 1, 0.5)
            self.target_orn_x_id = p.addUserDebugParameter("targetOrnX", -6.28, 6.28, 1.0)
            self.target_orn_y_id = p.addUserDebugParameter("targetOrnY", -6.28, 6.28, 1.5)
            self.target_orn_z_id = p.addUserDebugParameter("targetOrnZ", -6.28, 6.28, 4.5)
            self.dt_0_id = self.dt_1_id = self.dt_2_id = self.dt_3_id = self.dt_4_id = None

        # use 0 for no-removal
        trail_duration = 100
        pos = None
        prev_pose = [0, 0, 0]
        prev_pose1 = [0, 0, 0]
        has_prev_pose = 0

        print("[PyBulletSimulator] Start Simulation ... ")
        time.sleep(1)
        t = 0.
        step = 0
        while 1:
            step += 1
            # p.stepSimulation()
            if self.use_real_time_simulation:
                dt = datetime.now()
                t = (dt.second / 60.) * 2. * math.pi
            else:
                t += 0.01
                time.sleep(0.01)

            # handle debug
            use_debug = False
            use_accurate = False
            if self.use_gui:
                accurate = p.readUserDebugParameter(self.use_accurate_id)
                use_accurate = accurate > 0.5
                debug = p.readUserDebugParameter(self.use_debug_id)
                use_debug = debug > 0.5
            if use_debug:
                self.process_debug_parameter(use_accurate)
            else:
                for i in range(1):
                    pos = [0.6, 0.25 + 0.2 * math.cos(t), 0.5 + 0.2 * math.sin(t)]
                    if self.shift_arm:
                        pos[1] += self.y_shift
                    orn = p.getQuaternionFromEuler([0, -math.pi, math.sin(t)])

                    self.robot.move_joints(pos, orn=orn, use_accurate=use_accurate)

                ls = self.robot.get_link_state()
                current_pos = list(ls[0])
                current_orn = list(ls[1])

                # update and visualize path
                # if has_prev_pose and pos:
                #     self.print_debug_line(prev_pose, pos, [0, 0, 0.3], trail_duration)
                #     self.print_debug_line(prev_pose1, current_pos, [1, 0, 0], trail_duration)
                prev_pose = pos
                prev_pose1 = current_pos
                has_prev_pose = 1

            # update object position
            current_pos, current_orn = self.attach_object_to_end_effector()

            # predict object
            self.rgb_img = self.get_rgb_image()
            if self.cfg.simulation.predict:
                prediction_arr = self.get_prediction(self.rgb_img)
                _, prediction_name, prediction_accuracy, prediction_is_true = self.get_predicted_object(prediction_arr)

            # print Debug
            # if self.use_gui:
            #     text_step = f"Step: {step}"
            #     text_object = f"Object: {self.object.object_name}"
            #     text_prediction = f"Prediction: {prediction_name}, {float(str(round(prediction_accuracy, 2)))}"
            #     text_current_pos = f"Position: ({round(current_pos[0], 2)}, {round(current_pos[1], 2)}, " \
            #                        f"{round(current_pos[2], 2)})"
            #     text_current_orn = f"Orientation: ({round(current_orn[0], 2)}, {round(current_orn[1], 2)}," \
            #                        f"{round(current_orn[2], 2)}, {round(current_orn[3], 2)})"
            #     text_arr = [text_step, text_object, text_prediction, text_current_pos, text_current_orn]
            #     self.update_debug_text(text_arr, prediction_is_true)

        p.disconnect()

    @staticmethod
    def print_debug_line(prev_pose, current_pos, color=None, trail_duration=0):
        if color is None:
            color = [0, 0, 0]
        p.addUserDebugLine(prev_pose, current_pos, color, 1, trail_duration)

    def process_debug_parameter(self, use_accurate=False):
        manual = p.readUserDebugParameter(self.use_manual_id)
        use_manual = manual > 0.5

        if not use_manual:
            gripper = p.readUserDebugParameter(self.use_gripper_id)
            target_pos_x = p.readUserDebugParameter(self.target_pos_x_id)
            target_pos_y = p.readUserDebugParameter(self.target_pos_y_id)
            target_pos_z = p.readUserDebugParameter(self.target_pos_z_id)
            target_orn_x = p.readUserDebugParameter(self.target_orn_x_id)
            target_orn_y = p.readUserDebugParameter(self.target_orn_y_id)
            target_orn_z = p.readUserDebugParameter(self.target_orn_z_id)
            target_position = [target_pos_x, target_pos_y, target_pos_z]
            target_orientation = p.getQuaternionFromEuler([target_orn_x, target_orn_y, target_orn_z])

            self.robot.move_joints(target_position, orn=target_orientation,
                                   use_accurate=use_accurate)

            open_gripper = gripper < 0.5
            if open_gripper:
                self.robot.open_gripper()
            else:
                self.robot.close_gripper()

    def update_debug_text(self, text_arr, prediction_is_true):
        if self.dt_0_id:
            p.removeUserDebugItem(self.dt_0_id)
            p.removeUserDebugItem(self.dt_1_id)
            p.removeUserDebugItem(self.dt_2_id)
            p.removeUserDebugItem(self.dt_3_id)
            p.removeUserDebugItem(self.dt_4_id)

        self.dt_0_id = p.addUserDebugText(text_arr[0], [0, 0.4, 1], Colors.BLACK)
        self.dt_1_id = p.addUserDebugText(text_arr[1], [0, 0.4, 0.95], Colors.BLACK)
        if prediction_is_true:
            self.dt_2_id = p.addUserDebugText(text_arr[2], [0, 0.4, 0.9], Colors.GREEN)
        else:
            self.dt_2_id = p.addUserDebugText(text_arr[2], [0, 0.4, 0.9], Colors.RED)
        self.dt_3_id = p.addUserDebugText(text_arr[3], [0, 0.4, 0.80], Colors.BLACK)
        self.dt_4_id = p.addUserDebugText(text_arr[4], [0, 0.4, 0.75], Colors.BLACK)


@hydra.main(config_path="../configs/config.yaml")
def simulate(cfg: DictConfig) -> None:
    hydra.utils.get_original_cwd()
    print(cfg.simulate.pretty())
    simulator = PyBulletSimulatorTest(cfg)
    simulator.simulate_test()
    p.disconnect()


if __name__ == "__main__":
    simulate()
