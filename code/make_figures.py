from distributions import full_model_generator
from helper_routines import convolve_and_score, stupid_sum
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from data import names, scenes
from varidis import varidis
from os.path import join
from util import config

name = "test"

scene = scenes[names.index(name)]

ref_img = "../annotations/coupa/video2/reference.jpg"

x_hat = np.array([0,0])
v_hat = np.array([0,-.001])
t_final = 300
n_steps = 50

gen = full_model_generator(name, x_hat, v_hat, t_final, n_steps)

bounds = [scene.width, scene.height]
width = scene.bbox_width/8
ctx = int(np.ceil(bounds[0]/width))
cty = int(np.ceil(bounds[1]/width))

bboxes = []
for i in range(ctx):
    for j in range(cty):
        bboxes.append([-1 * bounds[0]/2.0 + width * i, -1 * bounds[0]/2.0 + width * (i+1),
                       -1 * bounds[1]/2.0 + width * j, -1 * bounds[1]/2.0 + width * (j+1)])

for ind, ((nl_pos, nl_weights), (lin_pos, lin_weights)) in enumerate(gen):
    msh = convolve_and_score(nl_pos, nl_weights, scene.kappa * (float(ind)/n_steps) * t_final, bboxes)
    msh += stupid_sum(lin_pos, lin_weights, bboxes)
    msh = msh.reshape((ctx, cty))

    img = img.imread(ref_img)

    extent = [-scene.width/2, scene.width/2, -scene.height/2, scene.height/2]
    plt.imshow(img, extent=extent)
    plt.imshow(msh, a=0.5, cmap=varidis, extent = extent)
    plt.savefig(join(config["output_folder"], "frame_{}.png".format(ind)))
    




