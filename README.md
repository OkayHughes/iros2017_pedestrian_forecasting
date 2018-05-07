## Synopsis

This project seeks to predict pedestrian position seconds in the future with a millisecond time-budget.  The motivation for this project is for inclusion in autonomous vehicle software, which requires real-time forecasting of pedestrian actions. The paper can be read [here](https://google.com).

It should be fairly self evident from this document how this is implemented. If you find errors, unclear sections in this document, or problems with installation, please contact me at [owhughes@umich.edu](mailto:owhughes@umich.edu).

## Installation gotchas ##
Ensure that `scipy` is the most recent version available on your OS.


## Usage ##

The `train_example.py` file in `code/example` will train a sample model, and show how to plot the relevant quantities that are calculated by our model. 

### Making sure your data is formatted correctly ###

We used the Stanford Drone Dataset as our test data, so we use their format which can be observed in the ```annotations``` folder. It describes observations in terms of a bounding box around the pedestrian. The column labels are:

```id, x1, y1,x2, y2, frame, lost, occluded, generated, label```

We don't pay attention to `lost, occluded,` or `generated`. 

`id` is a unique identifier corresponding to a pedestrian. `x1, y1` is the lower-left corner of the bounding box. `x2, y2` is the upper right. `frame` corresponds to the frame number of the observation, and `label` corresponds to the type of agent observed. Our model defaultly reads `Pedestrian` labels, but you can set this yourself (see below). The spatial coordinates are measured in pixels, so express it in integers.

Put your data in a file called `annotations.txt`.

### Learn a scene ###

In the same folder as `annotations`, make a file called `params.json`. The file should look something like 

```
{
   "width": 100,
   "height": 100,
   "label": "Pedestrian",
   "prefix": "test"
}
```

`width` is the width of the scene in pixels, `height` is the height in pixels, `label` is as above, `prefix` is what the pickle files should be prefixed with when the learned scene is saved.

Then run `python make_scene.py path/to/folder`, and pass the path to the folder containing `annotations.txt` and `params.json`. If your data is well formatted, it should build a scene and put it in the `scene_folder` specified in `config.json`.

### Generating results ###

The scene that's generated has dimensions `[1.0, height/width]`, centered at `[0,0]`. This determines the units that you pass into the model. Our model implementation preserves the units you pass into it.

The inputs to the model are the name of the scene you want to use, the position measurement (a 1x2 numpy array) that's measured in the scene dimensions listed above, the velocity measured in units/frame (1x2 numpy array), the time horizon you're simulating over (float), and the number of steps to simulate (int).

To generate results:
```
from distributions import full_model_generator
import numpy as np


prefix = "test" 
x_hat = np.array([0, 0])
v_hat = np.array([0, -.001])
t_final = 300
n_steps = 300


gen = full_model_generator(prefix, x_hat, v_hat, t_final, N_steps)
for ((nl_pts_n, nl_wts_n), (linear_pts_n, linear_wts_n)) in gen:
     pass
```

The output of the model is a tuple `((non_linear_points, non_linear_weights), (linear_points, linear_weights))`. Each points array is of size `(2, N)`, which means the array looks like `[[x1, x2, x3, x4, ...], [y1, y2, y3, y4, ...]]`. Each of the weights arrays are of size `(N)`, each of which correspond to one of the points in the point arrays.

** _The non-linear output must be convolved with the gaussian distribution described here in order to be correct. _ ** This is why the different points are split into two outputs. The linear points are treated as dirac deltas, so the probability of a pedestrian being in a set A is just the sum of the dirac deltas in that set. 

The non-linear weights are treated as dirac deltas convolved with gaussians of standard deviation `scene.kappa * (n / n_steps) * t_final`, where n corresponds to the number of frames that have come from the generator. The `convolve_and_score` method from `helper_routines` takes a set of points, weights, the standard deviation, and a set of bounding boxes to integrate over, and returns the convolved integral over those boxes. If you know the form, you can likely extend and improve what we've put together.



## Reproduce

The `reproduce` folder contains a number of files that were used to do the analysis for the paper. Due to the tight deadline, very few of the files that run the actual analysis are well documented. It is preserved here for posterity, but most of the helper functions in the folder should just be rewritten. 

To actually use the good version of our code, look in `code`.


## Contributors

[Owen Hughes](mailto:owhughes@umich.edu)

Henry O. Jacobs

## License


