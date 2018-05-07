from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold
import pickle
import process_data
from scene import Scene
from sys import argv
from util import read_json, root, config
from os.path import join

scene_dir = config['scene_folder']

def make_scene(folder):

    prefix = read_json(join(folder, "params.json"))["prefix"]
    print("Initializing a scene from " + folder)
    BB_ts_list, width, height = process_data.get_BB_ts_list(folder)
    kf = KFold(n_splits = 10)
    train_set, test_set = train_test_split( BB_ts_list, random_state = 0 )
    test_scene = Scene(prefix, train_set, width, height )

    print("P(k) = {}".format(test_scene.P_of_c))

    print("sum(P(k)) = {}".format( test_scene.P_of_c.sum()))
    print("Pickling scene and test set.")
    with open(join(scene_dir, "{}_scene.pkl").format(prefix), "w") as f:
        pickle.dump( test_scene, f)
        print("Pickled scene")

    with open(join(scene_dir, "{}_set.pkl").format(prefix), "w") as f:
        pickle.dump( test_set, f)
        print("Pickled the test set with {} agents".format(len(test_set)))

if __name__ == "__main__":
    folder = argv[1]
    make_scene(folder)
