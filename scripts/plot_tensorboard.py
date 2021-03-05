from next_best_view.plotter import TensorboardPlotter
import os
TITLE = None
LEGEND = None
mode = None
NEG_VAL = True
ROOT = "/home/ckorbach/nbv/data/iros/benchmarks/"
ROOT_DIR = "/home/ckorbach/nbv/data/iros/ds_objects/p1/"
SAVE_NAME = "test"

#TODO
# define axis title here (atm in plotter lol)
######################
mode = 5
if mode is not None:
    if mode == 1:
        NEG_VAL = True
        TITLE = "100 steps per episode"
        LEGEND = ["Class. model A, robot-arm mode",
                  "Class. model C, robot-arm mode",
                  "Class. model A, object-only mode"]
    elif mode == 2:  # no arm
        NEG_VAL = False
        TITLE = "Object-only mode with class. model A"
        n = ["ascending order, 100 steps/episode",
             "ascending order, 10 steps/episode",
             "descending order, 10 steps/episode",
             "random order, 10 steps/episode"]
        LEGEND = [n[0], n[1], n[2], n[3]]
    elif mode == 3:  #
        NEG_VAL = True
        TITLE = "10 steps per episode"
        LEGEND = ["Class. model A, object-only mode",
                  "Class. model A, robot-arm mode, pretrained",
                  "Class. model C, robot-arm mode, pretrained"]
    elif mode == 4:
        NEG_VAL = True
        TITLE = " "
        LEGEND = ["Class. model A, robot-arm mode",
                  "Class. model C, robot-arm mode",
                  "Class. model A, object-only mode",
                  "Class. model A, robot-arm mode, pre-trained",
                  "Class. model C, robot-arm mode, pre-trained"]
    elif mode == 5:
        NEG_VAL = True
        TITLE = " "
        LEGEND = ["$Agent_1$",
                  "$Agent_2$",
                  "$Agent_3$",
                  "$Agent_4$",
                  "$Agent_5$"]
    elif mode == 6:
        NEG_VAL = True
        TITLE = " "
        LEGEND = ["Object 11",
                  "Object 12",
                  "Object 13",
                  "Object 14",
                  "Object 15",
                  "Object 16",
                  "Object 17"]

    SAVE_NAME = str(mode) + "_"
    ROOT_DIR = os.path.join(ROOT, str(mode))
#################

smooth_setting = [11, 3, "nearest"]
KEY = "validation_mean"
SMOOTH = False
# ROOT_DIR = "/home/ckorbach/nbv/data/iros/group_id_a0" + KEY
plotter = TensorboardPlotter(key=KEY, root_path=ROOT_DIR, smooth_setting=smooth_setting)


plotter.plot_benchmark(smooth=SMOOTH)
#plotter.add_group(smooth=SMOOTH)
#plotter.add_singles(smooth=SMOOTH)

SAVE_ROOT = "/home/ckorbach/nbv/data/plots/iros"
if SMOOTH:
    SAVE_NAME += "smoothed_"
SAVE_NAME += KEY + ".png"
SAVE_PATH = os.path.join(SAVE_ROOT, SAVE_NAME)



# robot, no premodel
a = ["Class. model A, 100 steps/episode",
     "Class. model C, 100 steps/episode"]

LOC = "lower right"

plotter.plot(legend=LEGEND, title=TITLE, save_name=SAVE_PATH, neg_val=NEG_VAL, loc=LOC)

