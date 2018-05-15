"""Example training script

This script trains a model on the data given in examples/video_data,
and plots an example prediction.

"""

import sys
sys.path.insert(0, '.')

import matplotlib.pyplot as plt

from make_scene import make_scene
from data import scenes, sets, names
from process_data import BB_ts_to_curve
from visualization import singular_distribution_to_image
from visualization import convolved_distribution_to_image
from visualization import visualize_generator
from distributions import full_model_generator

folder = 'examples/video_data'

# the folder should contain an "annotations.txt" that corresponds
# to the format of the stanford drone dataset, and a "config.json"
# which denotes the width and height of the scene in pixels,
# the label from which training data should be drawn,
# and the prefix which you will use to access this scene.
make_scene(folder)

# `scenes` is a list of scene objects, names is a list
# of scene names which has the same order as `scenes`.
# all of this data is read when you import the `data` module.
scene = scenes[names.index("train_example")]
# `sets` is a list of testing data (each entry is a list of bounding-box sequences)
# which has the same order as `scenes` and `names`
ex_set = sets[names.index("train_example")]

# we select the fourth trajectory from our testing data
# note: it is currently a time series of bounding boxes
test_BB_ts = ex_set[3]

# the BB_ts_to_curve function converts a series of bounding boxes to a
# series of (x,y) points of size(2, num_time_points)
curve = BB_ts_to_curve(test_BB_ts)
offset = 30
# choose a position measurement. In the paper we denote position measurements by \hat{x}.
x_hat = curve[:, offset]
# use finite differences to calculate a symmetric velocity at the time point corresponding
# to \hat{x}. We denote measured velocity by \hat{v} in the paper.
v_hat = (curve[:, offset+2] - curve[:, offset-2])/4
print("x_hat = " + str(x_hat))
print("v_hat = " + str(v_hat))

# number of steps to discretize the time interval [0, t_final] into.
N_steps = 100
# number of seconds in the future you want to predict.
t_final = 100

# the size of the domain over which we will do predictions.
# all domains are assumed to be rectangles centered at (0, 0)
domain = [scene.width, scene.height]

# this is where the magic happens. The result of this function 
# is a generator which, upon each call of `next` returns the nested tuple
# ((x_nl_arr, w_nl_arr), (x_lin_arr, w_lin_arr)) 
# where `x_nl_arr` has shape (2, N_grid_points_1) and denotes the positions of the 
#           dirac deltas which should be convolved with a gaussian distribution.
#       `w_nl_arr` has shape (N_grid_points_1) and denotes the weights of the dirac deltas
#           with positions given in `x_nl_arr`
#       `x_lin_arr` has shape (2, N_grid_points_2) and denotes the positions of the dirac
#           deltas which SHOULD NOT be convolved with a gaussian distribution
#           because these points come from the linear class.
#       `w_lin_arr` has shape (N_grid_points_2) and denotes the weights of the dirac deltas
#           with positions given in `x_lin_arr`
gen = full_model_generator("train_example", x_hat, v_hat, t_final, N_steps)

plt.ion()
# width denotes the width (in scene coordinates, not pixels) of bounding boxes
# over which the probability distribution will be integrated, in order to plot the results
width = 0.02
results = list(enumerate(gen))
for frame_n, (_, (x_lin_arr, w_lin_arr)) in results:
    # use the visualization function `singular_distribution_to_image` to integrate
    # the linear distribution (WITHOUT CONVOLUTION) to an image.
    # X is a grid of the x positions of the bounding boxes over which the 
    # probability distribution is integrated
    # Y is the y positions
    # Z is a grid of the integral values over the associated bounding boxes.
    X, Y, Z = singular_distribution_to_image(x_lin_arr,
                                             w_lin_arr,
                                             width,
                                             domain)
    plt.pcolormesh(X, Y, Z, cmap='viridis')
    plt.plot(curve[0], curve[1])
    plt.scatter(curve[0, offset], curve[1, offset])
    plt.show()
    plt.pause(0.004)
    plt.clf()

for frame_n, ((x_nl_arr, w_nl_arr), _) in results:
    # NOTE: the naive convolution that happens in `convolved_distribution_to_image`
    # is very slow. In a real application, this would be done in a more clever way.
    # for now, we merely omit 4/5 of the frames to make it tolerable.
    if frame_n%5 == 0:
        # use the visualization function `singular_distribution_to_image` to integrate
        # the nonlinear distribution to an image.
        # X, Y, and Z are the same as in the preceding for loop.
        X, Y, Z = convolved_distribution_to_image(x_nl_arr,
                                                  w_nl_arr,
                                                  width,
                                                  domain,
                                                  scene.kappa * frame_n * t_final / N_steps)
        plt.pcolormesh(X, Y, Z, cmap='viridis')
        plt.plot(curve[0], curve[1])
        plt.scatter(curve[0, offset], curve[1, offset])
        plt.show()
        plt.pause(0.02)
        plt.clf()

# this function plots $\int\mathrm{d}(\rho_{lin} + \rho{non-linear})$, i.e. it 
# visualizes the full distribution.
visualize_generator(full_model_generator("train_example",
                                         x_hat,
                                         v_hat,
                                         t_final,
                                         N_steps),
                    scene, t_final, N_steps, width)
