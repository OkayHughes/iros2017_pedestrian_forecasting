"""Visualization module
Contains methods for visualizing distributions and clusters.
"""
import numpy as np
import matplotlib.pyplot as plt
import posteriors
from evaluation import classifier, classifier_no_convolve

def visualize_cluster(scene, k):
    """ Displays a plot of all the clusters and there potential functions

    args:
        scene:
        k: (int)

    returns:
        plot of requested cluster, as well as a potential function and a vector
        field.
    """
    assert k < scene.num_nl_classes
    cluster = scene.clusters[k]
    _, ax_arr = plt.subplots(1, 2, figsize=(15, 5))
    width = scene.width
    height = scene.height
    X_grid, Y_grid = np.meshgrid(np.linspace(-width/2, width/2, 20),
                                 np.linspace(-height/2, height/2, 20))
    x = np.vstack([X_grid.flatten(), Y_grid.flatten()])
    p_of_x = posteriors.x_given_k(x, k).reshape(X_grid.shape)
    #V = legval2d( 2*X_grid / width, 2*Y_grid / height, alpha )
    #V -= V.min()
    #p_of_x = np.exp( - V)

    ax_arr[0].contourf(X_grid, Y_grid, p_of_x, 40, cmap='viridis', interpolation='cubic')
    for xy in cluster:
        ax_arr[0].plot(xy[0], xy[1], 'w-')
    ax_arr[0].axis('equal')
    UV = scene.director_field_vectorized(k, np.vstack([X_grid.flatten(), Y_grid.flatten()]))
    U_grid = UV[0].reshape(X_grid.shape)
    V_grid = UV[1].reshape(X_grid.shape)
    ax_arr[1].quiver(X_grid, Y_grid, U_grid, V_grid, scale=30)
    ax_arr[1].axis('equal')
    plt.show()
    return -1

def singular_distribution_to_image(pts, weights, width, domain):
    """ Converts a singular distribution into an image for plotting.

    args:
        pts: np.array.shape = (2,N)
        weights: np.array.shape = (N,)
        width: float, width of the individual boxes to sum over
        domain: (width, height)

    returns:
        np.array.shape = (domain[0]//width, domain[1]//width), x positions of bounding box centers
        np.array.shape = (domain[0]//width, domain[1]//width), y positions of bounding box centers
        np.array.shape = (domain[0]//width, domain[1]//width), probability associated with bbox
    """
    sums, bboxes = classifier_no_convolve(domain, width, (pts, weights))
    ctx, cty = int(np.ceil(domain[0]/width)), int(np.ceil(domain[1]/width))
    xs = (bboxes[:, 0] + bboxes[:, 1])/2
    ys = (bboxes[:, 2] + bboxes[:, 3])/2
    sums = sums.reshape(ctx, cty)
    xs = xs.reshape(ctx, cty)
    ys = ys.reshape(ctx, cty)
    return xs, ys, sums

def convolved_distribution_to_image(pts_nl, weights_nl, width, domain, sigma_x):
    """ Converts a singular distribution into an image for plotting.

    args:
        pts_nl: np.array.shape = (2,N)
        weights_nl: np.array.shape = (N,)
        pts_lin: np.array.shape = (2,N)
        weights_lin: np.array.shape = (N,)
        width: float, width of the individual bounding boxes
        domain: (width, height)

    returns:
        np.array.shape = (domain[0]//width * domain[1]//width)
        np.array.shape = (domain[0]//width, domain[1]//width)
    """
    sums, bboxes = classifier(domain, width, (pts_nl, weights_nl), sigma_x)
    ctx, cty = int(np.ceil(domain[0]/width)), int(np.ceil(domain[1]/width))
    xs = (bboxes[:, 0] + bboxes[:, 1])/2
    ys = (bboxes[:, 2] + bboxes[:, 3])/2
    sums = sums.reshape(ctx, cty)
    xs = xs.reshape(ctx, cty)
    ys = ys.reshape(ctx, cty)
    return xs, ys, sums

def visualize_generator(generator, scene, t_final, N_max, width):
    """Visualizes a distribution generator
    args:
        generator: generator returned from full_model_generator
            Note: consumes generator
        t_final: float, time of final prediction
        N_max: int, max number of frames returned from generator
    """
    domain = (scene.width, scene.height)
    kappa = scene.kappa
    plt.ion()
    for n, ((x_nl_arr, w_nl_arr), (x_lin_arr, w_lin_arr)) in enumerate(generator):
        conv = convolved_distribution_to_image(x_nl_arr,
                                               w_nl_arr,
                                               width,
                                               domain,
                                               kappa * n * t_final / N_max)
        lin = singular_distribution_to_image(x_lin_arr,
                                             w_lin_arr,
                                             width,
                                             domain)

        X, Y, Z = conv[0], conv[1], conv[2] + lin[2]
        plt.pcolormesh(X, Y, Z, cmap='viridis')
        plt.show()
        plt.pause(0.02)
        plt.clf()


if __name__ == '__main__':
    from data import scene
    visualize_cluster(scene, 0)

    print("Testing routine for visualizing singular distributions")
    weights = np.ones(100000)
    pts = np.random.randn(2, weights.size)
    pts[1] *= 0.5
    pts[0] += 0.5
    domain = (4, 4)
    res = (30, 20)
    X, Y, Z = singular_distribution_to_image(pts, weights, 0.02, domain)
    plt.contourf(X, Y, Z, 30, cmap='viridis')
    plt.title('Should be a Gaussian with $\\mu = (0.5,0.0), \\sigma_x=1.0, \\sigma_y=0.5$')
    plt.show()
