import pybullet as p
import pybullet_data
from datetime import datetime
import time
import math
import yaml
import argparse
import json
import os

from enum import Enum
import utils
from predictor import Predictor


class State(Enum):
    IDLE = 0
    INIT = 1
    DEBUG = 2
    RUN = 3


class Colors:
    BLACK = [0, 0, 0]
    RED = [255, 0, 0]
    GREEN = [0, 170, 0]
    BLUE = [0, 0, 255]
    WHITE = [0, 0, 0]


# --- TEST PARAMS ---
STATE = State.INIT
TEST = True

# --- LOAD ARGS ---

parser = argparse.ArgumentParser()
parser.add_argument('-config', default="../configs/config_j2s7s300.yaml", help="Config file")
args = parser.parse_args()

with open(args.config, "r") as config_file:
    try:
        config = yaml.safe_load(config_file)
    except yaml.YAMLError as error:
        print("error (config load)", error)


# --- LOAD ENV ---

print("Load environment ...")
clId = p.connect(p.SHARED_MEMORY)
if clId < 0:
    if config["simulation"]["gui_client"]:
        clId = p.connect(p.GUI, options='--background_color_red=1.0 --background_color_green=1.0 --background_color_blue=1.0')
    else:
        clId = p.connect(p.DIRECT)
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)

gravity = config["simulation"]["gravity"]
useRealTimeSimulation = config["simulation"]["useRealTimeSimulation"]
p.setGravity(0, 0, gravity)
p.setRealTimeSimulation(useRealTimeSimulation)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
# planeId = p.loadURDF(config["environment"]["model"])

useAccurateId = p.addUserDebugParameter("useAccurate", 0, 1, 0)
useDebugId = p.addUserDebugParameter("useDebug", 0, 1, 0)
useManualId = p.addUserDebugParameter("useManual", 0, 1, 1)
useGripperId = p.addUserDebugParameter("useGripper", 0, 1, 0)
targetPosXId = p.addUserDebugParameter("targetPosX", -1, 1, 0.3)
targetPosYId = p.addUserDebugParameter("targetPosY", -1, 1, 0.2)
targetPosZId = p.addUserDebugParameter("targetPosZ", -1, 1, 0.5)
targetOrnXId = p.addUserDebugParameter("targetOrnX", -6.28, 6.28, 1.0)
targetOrnYId = p.addUserDebugParameter("targetOrnY", -6.28, 6.28, 1.5)
targetOrnZId = p.addUserDebugParameter("targetOrnZ", -6.28, 6.28, 4.5)
dt0Id = dt1Id = dt2Id = dt3Id = dt4Id = None


# --- LOAD CAMERA AND VIEWPOINT---

print("Load camera ...")
camera_model = config["camera"]["model"]
camera_eye_position = config["camera"]["cameraEyePosition"]
camera_target_position = config["camera"]["cameraTargetPosition"]
camera_up_vector = config["camera"]["cameraUpVector"]

camera_fov = config["camera"]["fov"]
camera_aspect = config["camera"]["aspect"]
camera_nearVal = config["camera"]["nearVal"]
camera_farVal = config["camera"]["farVal"]

cameraId = p.loadURDF(camera_model, camera_eye_position)

viewMatrix = p.computeViewMatrix(cameraEyePosition=camera_eye_position,
                                 cameraTargetPosition=camera_target_position,
                                 cameraUpVector=camera_up_vector)

projectionMatrix = p.computeProjectionMatrixFOV(fov=camera_fov,
                                                aspect=camera_aspect,
                                                nearVal=camera_nearVal,
                                                farVal=camera_farVal)

p.resetDebugVisualizerCamera(cameraDistance=1.2,
                             cameraYaw=90.0,
                             cameraPitch=-10.0,
                             cameraTargetPosition=[0, camera_eye_position[1], 0.6])

# --- LOAD ROBOT ---

print("Load robot ...")
robot_name = config["robot"]["name"]
robot_model = config["robot"]["model"]
robot_origin_pos = config["robot"]["origin_pos"]
robot_origin_orientation = config["robot"]["origin_orientation"]
robotEndeffectorIndex = config["robot"]["endeffector_index"]
robotId = p.loadURDF(robot_model, robot_origin_pos)
p.resetBasePositionAndOrientation(robotId, robot_origin_pos, robot_origin_orientation)
numJoints = p.getNumJoints(robotId)
robotFingerIndex = config["robot"]["finger_index"]
robotFingerTipIndex = config["robot"]["finger_tip_index"]

jd = config["jd"]  # joint damping coefficents
ll = config["ll"]  # lower limits
ul = config["ul"]  # upper limits
jr = config["jr"]  # joint ranges
rp = config["rp"]  # rest poses
hp = config["hp"]  # home pose

print("Reset Joint States")
numFixedJoints = 0
for i in range(numJoints):
    p.resetJointState(robotId, i, hp[i])
    info = p.getJointInfo(robotId, i)
    if info[3] == -1:
        numFixedJoints += 1
    print("Joint number: ", i)
    print(" -- Info: ", info)
    print(" -- State: ", p.getJointState(robotId, i))
numMoveableJoints = numJoints - numFixedJoints
print("Number of joints: ", numJoints)
print("Number of fixed joints: ", numFixedJoints)
print("Number of moveable joints: ", numMoveableJoints)
time.sleep(1)

# --- LOAD OBJECT ---

fileName = "Object_5.obj"
objectName = fileName.split(".")[0]
filePath = config["objects"]["path"] + "/" + fileName
print("Load object %s ..." % fileName)
visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH,
                                    fileName=filePath,
                                    rgbaColor=None,
                                    meshScale=config["objects"]["meshScale"])

collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                          fileName=filePath,
                                          meshScale=config["objects"]["meshScale"])

# get robotEndeffectorPos for object spawning
ls = p.getLinkState(robotId, robotEndeffectorIndex)
current_pos = list(ls[0])
current_orn = list(ls[1])
utils.rotate_quaternion(current_orn, 2, 0.0)

objectId = p.createMultiBody(baseMass=0.5,
                             # baseCollisionShapeIndex=collisionShapeId,
                             baseVisualShapeIndex=visualShapeId,
                             basePosition=current_pos,
                             baseOrientation=current_orn)

# create texture
textureId = p.loadTexture('../data/objects/texture.jpg')
p.changeVisualShape(objectId, -1, textureUniqueId=textureId)
time.sleep(1)


# --- LOAD CLASSIFICATION MODEL ---
model_dir = "/home/chris/git/next_best_view_rl/data/models"
class_map = json.load(open(os.path.join(model_dir, "class_map.json")))
predictor = Predictor(5, os.path.join(model_dir, "complex_white_1_67.model"), class_map)
prediction_text_position = camera_eye_position
prediction_text_position[2] += 0.05


# --- SIMULATION ---

# use 0 for no-removal
trailDuration = 100
prevPose = [0, 0, 0]
prevPose1 = [0, 0, 0]
hasPrevPose = 0

print("Start Simulation ... ")
STATE = State.RUN
show_debug_slider = False
active_debug_slider = False
time.sleep(1)
t = 0.
step = 0
while 1:
    step += 1
    # p.stepSimulation()
    if useRealTimeSimulation:
        dt = datetime.now()
        t = (dt.second / 60.) * 2. * math.pi
    else:
        t += 0.01
        time.sleep(0.01)

    # get image
    width, height, rgbImg, depthImg, segImg = p.getCameraImage(width=800,
                                                               height=800,
                                                               viewMatrix=viewMatrix,
                                                               projectionMatrix=projectionMatrix)
    index = predictor.predict_image(rgbImg)
    prediction = predictor.get_class_from_index(index)

    # handle manual debug
    accurate = p.readUserDebugParameter(useAccurateId)
    useAccurate = accurate > 0.5
    debug = p.readUserDebugParameter(useDebugId)
    useDebug = debug > 0.5
    if useDebug:
        manual = p.readUserDebugParameter(useManualId)
        useManual = manual > 0.5

        if not useManual:
            gripper = p.readUserDebugParameter(useGripperId)
            targetPosX = p.readUserDebugParameter(targetPosXId)
            targetPosY = p.readUserDebugParameter(targetPosYId)
            targetPosZ = p.readUserDebugParameter(targetPosZId)
            targetOrnX = p.readUserDebugParameter(targetOrnXId)
            targetOrnY = p.readUserDebugParameter(targetOrnYId)
            targetOrnZ = p.readUserDebugParameter(targetOrnZId)
            targetPosition = [targetPosX, targetPosY, targetPosZ]
            targetOrientation = p.getQuaternionFromEuler([targetOrnX, targetOrnY, targetOrnZ])

            utils.move_joints(robotId, robotEndeffectorIndex, targetPosition, ll, ul, jr, hp, orn=targetOrientation,
                              targetVelocity=0, force=500, positionGain=0.01, velocityGain=1, use_accurate=useAccurate)

            openGripper = gripper < 0.5
            if openGripper:
                utils.open_gripper(robotId, robotFingerIndex, robotFingerTipIndex, ll, ul)
            else:
                utils.close_gripper(robotId, robotFingerIndex, robotFingerTipIndex, ll, ul)

    if not useDebug:
        if TEST:
            for i in range(1):
                pos = [0.4, 0.25 + 0.2 * math.cos(t), 0.5 + 0.2 * math.sin(t)]
                orn = p.getQuaternionFromEuler([0, -math.pi, math.sin(t)])

                utils.move_joints(robotId, robotEndeffectorIndex, pos, ll, ul, jr, hp, orn=False,
                                  targetVelocity=0, force=500, positionGain=0.01, velocityGain=1, use_accurate=useAccurate)

            ls = p.getLinkState(robotId, robotEndeffectorIndex)
            current_pos = list(ls[0])
            current_orn = list(ls[1])

            # update and visualize path
            if hasPrevPose:
                p.addUserDebugLine(prevPose, pos, [0, 0, 0.3], 1, trailDuration)
                p.addUserDebugLine(prevPose1, current_pos, [1, 0, 0], 1, trailDuration)
            prevPose = pos
            prevPose1 = current_pos
            hasPrevPose = 1

    # update object position
    ls = p.getLinkState(robotId, robotEndeffectorIndex)
    current_pos = list(ls[0])
    current_orn = list(ls[1])
    current_orn = utils.rotate_quaternion(current_orn, 1, 0.0)
    p.resetBasePositionAndOrientation(objectId, current_pos, current_orn)

    # print Debug
    if dt0Id:
        p.removeUserDebugItem(dt0Id)
        p.removeUserDebugItem(dt1Id)
        p.removeUserDebugItem(dt2Id)
        p.removeUserDebugItem(dt3Id)
        p.removeUserDebugItem(dt4Id)

    text_step = f"Step: {step}"
    text_object = f"Object: {objectName}"
    text_prediction = f"Prediction: {prediction}"
    text_current_pos = f"Position: ({round(current_pos[0], 2)}, {round(current_pos[1], 2)}, {round(current_pos[2], 2)})"
    text_current_orn = f"Orientation: ({round(current_orn[0], 2)}, {round(current_orn[1], 2)}, {round(current_orn[2], 2)})"
    dt0Id = p.addUserDebugText(text_step, [0, 0.5, 1], Colors.BLACK)
    dt1Id = p.addUserDebugText(text_object, [0, 0.5, 0.95], Colors.BLACK)
    if objectName == prediction:
        dt2Id = p.addUserDebugText(text_prediction, [0, 0.5, 0.9], Colors.GREEN)
    else:
        dt2Id = p.addUserDebugText(text_prediction, [0, 0.5, 0.9], Colors.RED)
    dt3Id = p.addUserDebugText(text_current_pos, [0, 0.5, 0.80], Colors.BLACK)
    dt4Id = p.addUserDebugText(text_current_orn, [0, 0.5, 0.75], Colors.BLACK)

p.disconnect()


# --- DEPRECATED ---

# cid = p.createConstraint(robotId, 9, objectId, -1, p.JOINT_POINT2POINT, [0, 1, 0], [0, 0, 0], [0, 0, 1])

# --- GRIP OBJECT ---

# pos = [0.6, 0, 0.1]
# orn_down = p.getQuaternionFromEuler([0, -math.pi, 0])

# utils.move_joints(robotId, robotEndeffectorIndex, pos, ll, ul, jr, hp, orn=orn_down,
# target_velocity=0, force=500, positionGain=0.01, velocityGain=1, use_accurate=True)

# time.sleep(5)
# utils.close_gripper(robotId, robotFingerIndex, robotFingerTipIndex, ll, ul)
# time.sleep(1)
