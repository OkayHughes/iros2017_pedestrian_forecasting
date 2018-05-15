"""Data management module

This module imports and handles generated scenes after they have been learned.


Attributes:
    names (string list): module level variable that contains the names of scenes

    scenes (Scene list): module level variable containing Scene objects that are
        read from the folder denoted in the config file (in the field "scene_folder")

    sets (list list): module level variable containing the data sets
        corresponding to the Scenes in `scenes`

"""

import pickle
import os
from os.path import join
import numpy as np
from util import config

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
        with open(join(folder, file), 'rb') as f:
            sets.append(pickle.load(f))

names = [x.name for x in scenes]

ex_scene = scenes[0]
ex_set = sets[0]
