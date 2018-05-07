import pickle
import os
from os.path import join
import numpy as np
from process_data import BB_ts_to_curve as bbts
from util import read_json, root, config

data = 0
scenes = []
sets = []
folder = config["scene_folder"]
for file in sorted(os.listdir(folder)):
    if file.endswith("_scene.pkl"):
        with open(join(folder, file), "rb") as f:
            scene = pickle.load(f)
            scene.P_of_c = np.ones_like(scene.P_of_c)/float(len(scene.P_of_c))
            scenes.append(scene)

    elif file.endswith("_set.pkl"):
        with open(join(folder, file),'r') as f:
            sets.append(pickle.load(f))

names = [x.name for x in scenes]

scene = scenes[0]
set = sets[0]


