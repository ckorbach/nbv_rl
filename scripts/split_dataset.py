import os
import shutil
import random
import numpy as np

PATH = "/home/ckorbach/nbv/data/yab_robot_complete"
# PATH_IMAGES = PATH
PATH_IMAGES = os.path.join(PATH, "images")

val_ratio = test_ration = 0.1

root_dir = os.path.join(PATH, "splitted")
train_dir = os.path.join(root_dir, "train")
val_dir = os.path.join(root_dir, "val")
test_dir = os.path.join(root_dir, "test")

if not os.path.exists(root_dir):
    os.makedirs(root_dir)

for cls in os.listdir(PATH_IMAGES):
    train_cls_dir = os.path.join(train_dir, cls)
    val_cls_dir = os.path.join(val_dir, cls)
    test_cls_dir = os.path.join(test_dir, cls)
    if not os.path.exists(train_cls_dir):
        os.makedirs(train_cls_dir)
    if not os.path.exists(val_cls_dir):
        os.makedirs(val_cls_dir)
    if not os.path.exists(test_cls_dir):
        os.makedirs(test_cls_dir)

    # Creating partitions of the data after shuffeling
    src = os.path.join(PATH_IMAGES, cls)

    all_file_names = os.listdir(src)
    np.random.shuffle(all_file_names)
    val_file_names, train_file_names, test_file_names = np.split(np.array(all_file_names),
        [int(len(all_file_names) * val_ratio),
         int(len(all_file_names) * (1 - val_ratio))])

    train_file_names = [src + "/" + name for name in train_file_names.tolist()]
    val_file_names = [src + "/" + name for name in val_file_names.tolist()]
    test_file_names = [src + "/" + name for name in test_file_names.tolist()]

    print("-" * 30)
    print(f"Total images: {len(all_file_names)}")
    print(f"Training: {len(train_file_names)}")
    print(f"Validation: {len(val_file_names)}")
    print(f"Testing: {len(test_file_names)}")

    # Copy-pasting images
    for name in train_file_names:
        shutil.copy(name, train_cls_dir)

    for name in val_file_names:
        shutil.copy(name, val_cls_dir)

    for name in test_file_names:
        shutil.copy(name, test_cls_dir)


