"""Scene making module

Module contains the function for training scenes from data using default parameters.

"""
from sys import argv
from os.path import join
import pickle
from sklearn.model_selection import train_test_split

from scene import Scene
from util import read_json, config
import process_data


scene_dir = config['scene_folder']

def make_scene(folder):
    """
    Reads necessary data and annotations from folder parameter and
    trains a scene with the default training parameters.

    takes:
    folder: string, path to folder formatted like subfolders in the "annotation directory"
    """
    prefix = read_json(join(folder, "params.json"))["prefix"]
    print("Initializing a scene from " + folder)
    BB_ts_list, width, height = process_data.get_BB_ts_list(folder)
    train_set, test_set = train_test_split(BB_ts_list, random_state=0)
    test_scene = Scene(prefix, train_set, width, height)

    print("P(k) = {}".format(test_scene.P_of_c))

    print("sum(P(k)) = {}".format(test_scene.P_of_c.sum()))
    print("Pickling scene and test set.")
    with open(join(scene_dir, "{}_scene.pkl").format(prefix), "wb") as f:
        pickle.dump(test_scene, f)
        print("Pickled scene")

    with open(join(scene_dir, "{}_set.pkl").format(prefix), "wb") as f:
        pickle.dump(test_set, f)
        print("Pickled the test set with {} agents".format(len(test_set)))

if __name__ == "__main__":
    folder = argv[1]
    make_scene(folder)
