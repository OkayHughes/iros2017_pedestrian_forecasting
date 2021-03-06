#!/usr/bin/env python
"""Helper Function Module

This module contains an assortment of functions which
are referenced by an assortment of other modules.
"""

import numpy as np
from scipy.special import erf # pylint: disable=no-name-in-module




def ct(scene, width):
    """
    computes the number of bboxes that are returned by bounding box generating functions.
    takes:
    scene: Scene
    width: float
    returns:
    ctx: int, number of boxes in the x axis
    cty: int, number of boxes in the y axis
    """
    ctx = int(np.ceil(scene.width/width))
    cty = int(np.ceil(scene.height/width))
    return ctx, cty

def stupid_sum(pts, weights, bbox_ls):
    """
    Counts the number of points in each bounding box box naively
    takes:
    pts: np.array of shape (2, N_pts)
    weights: np.array of shape (N_pts)
    bbox_ls: np.array of shape (N_pts, 4)
    """
    #Garbage one-liner
    filt = lambda x, z: np.logical_and((z[0] > x[0]), (z[0] < x[1]), (z[1] > x[2]), (z[1] < x[3]))
    return [sum(weights[np.where(filt(bbox, pts))[0]]) for bbox in bbox_ls]



def convolve_and_score(pts, weights, sigma, bbox_ls):
    """ Smooths a singular distribution, and integrates it over a list of bboxes

    args:
        pts: numpy array of shape (2, N_pts)
        weights: numpy array of shape (N_pts)
        sigma: float (the standard deviation of the Gaussian)
        bbox_arr: numpy array of shape (N_boxes, 4)

    returns:
        score: numpy_arr of shape (N_boxes
        )
    """

    num_bboxes = len(bbox_ls)
    score = np.zeros(num_bboxes)
    #basex = np.tile(pts[0], (num_bboxes, 1))
    #basey = np.tile(pts[1], (num_bboxes,1))
    #xmins = bbox_ls[:, 0]
    #xmaxes = bbox_ls[:, 1]
    #ymins = bbox_ls[:, 2]
    #ymaxes = bbox_ls[:, 3]
    #maxes_x = (xmaxes - basex.transpose()).transpose()/(np.sqrt(2)*sigma)
    #mins_x = (xmins - basex.transpose()).transpose()/(np.sqrt(2)*sigma)

    #out_x = (erf(maxes_x) - erf(mins_x)) / 2

    #maxes_y = (ymaxes - basey.transpose()).transpose()/(np.sqrt(2)*sigma)
    #mins_y = (ymins - basey.transpose()).transpose()/(np.sqrt(2)*sigma)
    #out_y = (erf(maxes_y) - erf(mins_y)) / 2

    #out = (out_x * out_y) * weights

    #return np.sum(out, axis=1)

    for k, bbox in enumerate(bbox_ls):
        score[k] = np.dot(cdf_of_2d_normal(pts, sigma, bbox),
                          weights)
    return score

def cdf_of_normal(mu, sigma, x_min, x_max):
    """ Computes the integral of a normal distribution over a bounding box.

    args:
        mu: numpy array of size 2
        sigma: float, standard deviation
        x_min: float
        x_max: float

    returns:
        out: float
    """
    out = erf((x_max-mu) / (np.sqrt(2) * sigma))
    out -= erf((x_min-mu) / (np.sqrt(2) * sigma))
    out *= 0.5
    return out

def cdf_of_2d_normal(mu, sigma, bbox):
    """ Computes the integral of a normal distribution over a bounding box.

    args:
        mu: numpy array of size 2, mean of normal distribution
        sigma: float, standard deviation
        bbox: 4-tuple (x_min, x_max, y_min, y_max)

    returns:
        out: float

    NOTE: This is vectorized in the parameter mu as long as mu is of shape
    (2,N).
    """
    x_min, x_max, y_min, y_max = bbox
    out_x = cdf_of_normal(mu[0], sigma, x_min, x_max)
    out_y = cdf_of_normal(mu[1], sigma, y_min, y_max)
    out = out_x * out_y
    return out

def precision_series(pred_scores, true_scores, tols):
    """ Computes the precision for a range of tolerances

    args:
        pred_scores: numpy array of shape (N,)
        true_scores: numpy array of shape (N,)
        tols: numpy array of shape (N_tols,)

    returns:
        out: numpy array of shape (N_tols,)
    """
    N_tols = len(tols)
    N_scores = len(pred_scores)
    pred = np.outer(pred_scores, np.ones(N_tols)) > np.outer(np.ones(N_scores), tols)
    true = np.outer(true_scores, np.ones(N_tols)) > np.outer(np.ones(N_scores), tols)
    PP = pred.sum(axis=0).astype(np.int32)
    TP = true.sum(axis=0).astype(np.int32)

    #where PP is 0 we should output a precision of 1
    TP = (PP == 0) + TP * (PP != 0)
    PP = (PP == 0) + PP * (PP != 0)
    return TP / PP.astype(np.float64)

if __name__ == "__main__":
    print("Testing routine cdf_of_normal")
    sigma = 1.1
    mu = 0.1
    x_min = -0.2
    x_max = 1
    result = cdf_of_normal(mu, sigma, x_min, x_max)
    from scipy.integrate import quad
    def integrand(x):
        """
        Test integrand f(x) = \\frac{\\exp(\frac{-x^2}{2 * \\sigma^2}}{\\sqrt{2 * \\pi * \\sigma^2}}
        """
        return np.exp(-(x-mu)**2 / (2 * sigma**2)) / (np.sqrt(2 * np.pi * sigma**2))
    expected, error = quad(integrand, x_min, x_max)
    print("result   = {}".format(result))
    print("expected = {} +/- {}".format(expected, error))

    print("Testing integration_of_Gaussian_over_bbox")
    sigma = 1.1
    mu = np.random.randn(2)
    bbox = (-1.0, 5.1, -1, 0)
    result = cdf_of_2d_normal(mu, sigma, bbox)
    from scipy.integrate import dblquad
    def integrand_2(y, x):
        """
        Test integrand f(x) = \\exp(\frac{-(x_1^2 + x_2^2)}{2 * \\sigma^2}
        """
        out = np.exp(-((x - mu[0])**2 + (y - mu[1])**2) / (2 * sigma**2))
        out /= 2 * np.pi * sigma**2
        return out
    expected, error = dblquad(integrand_2,
                              bbox[0],
                              bbox[1],
                              lambda x: bbox[2],
                              lambda x: bbox[3])
    print("result   = {}".format(result))
    print("expected = {} +/- {}".format(expected, error))

    print("Testint cdf_of_normal is vectorized")
    mu = np.random.randn(10)
    result = cdf_of_normal(mu, sigma, x_min, x_max)
    print(result)

    print("Testint cdf_of_2d_normal is vectorized")
    mu = np.random.randn(2, 10)
    result = cdf_of_2d_normal(mu, sigma, bbox)
    print(result)

    print("Testing convolve_and_score")
    sigma = 0.1
    N_pts = 1
    pts = np.zeros((2, N_pts))
    weights = np.array([1.0])
    bbox_ls = [(-1, 1, -1, 1)]
    score = convolve_and_score(pts, weights, sigma, bbox_ls)
    expected = 1.0
    computed = score[0]
    print("expected = {}".format(expected))
    print("computed = {}".format(computed))

    print("Testing if convolve_and_score is vectorized.")
    N_pts = 100
    pts = np.random.randn(2, N_pts)
    weights = np.random.randn(N_pts)**2
    X_grid, Y_grid = np.meshgrid(np.linspace(-2, 2, 30),
                                 np.linspace(-2, 2, 30))
    x_min_ls = list(X_grid.flatten() - 0.1)
    x_max_ls = list(X_grid.flatten() + 0.1)
    y_min_ls = list(Y_grid.flatten() - 0.1)
    y_max_ls = list(Y_grid.flatten() + 0.1)
    bbox_ls = list(zip(x_min_ls, x_max_ls, y_min_ls, y_max_ls))
    from time import time
    t0 = time()
    score = convolve_and_score(pts, weights, sigma, bbox_ls)
    t1 = time()
    dt = t1 - t0
    print("Done. Total compute time = {} seconds".format(t1-t0))
    print("Estimated time to compute 300 frames = {} seconds".format(300*dt))
    print("max score = {}".format(score.max()))
    print("min score = {}".format(score.min()))

    tols = np.linspace(score.min(), score.max(), 100)
    precision_arr = precision_series(score, score, tols)
