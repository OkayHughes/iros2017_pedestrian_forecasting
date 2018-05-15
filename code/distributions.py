"""Distributions module

This module contains methods for generating the final probability distributions.
Note that set_scene must be called if you aren't using the default scene.

"""
import numpy as np


import posteriors
import derived_posteriors
from derived_posteriors import joint_lin_x_t_x_hat_v_hat
from derived_posteriors import joint_k_x_x_hat_v_hat

from data import scene, scenes, names


Vk = scene.alpha_arr
scene_scale = np.array([scene.width, scene.height])
#temporary
dist_width = np.ones([2]) * scene.bbox_width
vel_width = np.ones([2]) * scene.bbox_velocity_width
s_max = scene.s_max
sigma_x = scene.sigma_x
sigma_v = scene.sigma_v
sigma_L = scene.sigma_L
kappa = scene.kappa

def set_scene(scn):
    """
    This function is a hack that's used to set all of the global scene variables.
    """
    global scene, Vk, scene_scale, dist_width, vel_width, s_max, sigma_x, sigma_v, sigma_L, kappa
    scene = scn
    Vk = scn.alpha_arr
    scene_scale = np.array([scene.width, scene.height])
    #temporary
    dist_width = np.ones([2]) * scene.bbox_width
    vel_width = np.ones([2]) * scene.bbox_velocity_width
    s_max = scene.s_max
    sigma_x = scene.sigma_x
    sigma_v = scene.sigma_v
    sigma_L = scene.sigma_L
    kappa = scene.kappa
    derived_posteriors.set_scene(0, custom_scene=scene)


def integrate_class(k, x0, T, N_steps):
    """
    integrates a director field over a range [-T,T]
    args:
    x0: np.array.shape == (2,N)
    T: float
    N_steps: int

    Returns:
    np.array.shape == (2*N_steps+1, 2, N)
    """
    N = x0.shape[1] #number of particles
    from scipy.integrate import odeint
    def f(x, t):
        """
        Generates vector field at gridpoints `x` at time `t` for odeint.
        """
        x = x.reshape(2, len(x) // 2) + 0 * t
        return s_max * scene.director_field_vectorized(k, x).flatten()

    # Integrate the ODE backwards in time
    f_backwards = lambda x, t: -1 * f(x, t)
    t_arr = np.linspace(0, T, N_steps+1)
    x_backward = odeint(f_backwards, x0.flatten(), t_arr)[::-1]
    x_forward = odeint(f, x0.flatten(), t_arr)
    x_arr = np.concatenate([x_forward, x_backward[1:]])
    return x_arr.reshape((2*N_steps+1, 2, N))

def full_model_generator(scene_name, x_hat, v_hat, t_final, N_steps, convolve=False):
    """
    a generator which gives particles and weights
    Takes:
    x_hat: np.array(2): position measurement
    v_hat: np.array(2): velocity measurement
    t_final: float
    dt: float

    Example:
    for x,w in particle_generator( scene, x_hat, v_hat, 3.0):
        plt.scatter(x[0], x[1]) #NOTE: This plots the points
        print w.sum() #This p(rints the total mass
    """
    set_scene(scenes[names.index(scene_name)])
    num_nl_classes = len(scene.P_of_c)-1

    #Initializes particles for nonlinear classes
    x_span = np.linspace(-3 * sigma_x, 3 * sigma_x, 5)
    dvol_nl = (x_span[1]-x_span[0])**2
    X, Y = np.meshgrid(x_span + x_hat[0], x_span + x_hat[1])
    x0 = np.vstack([X.flatten(), Y.flatten()])

    #Initializes a regular grid for evaluation of the linear class
    x_span = np.linspace(-scene.width/2, scene.width/2, 250)
    dx = x_span[1] - x_span[0]
    y_span = np.linspace(-scene.height/2, scene.height/2, 250)
    dy = y_span[1] - y_span[0]
    X, Y = np.meshgrid(x_span, y_span)
    x_lin = np.vstack([X.flatten(), Y.flatten()])

    N_ptcl = x0.shape[1]
    x_arr = np.zeros((num_nl_classes, 2*N_steps+1, 2, N_ptcl))
    for k in range(num_nl_classes):
        x_arr[k] = integrate_class(k, x0, t_final, N_steps)
    #At time t=0, rho(x,t=0) is nothing but P(x0 | x0_hat),
    # which equals P(x0_hat | x0).
    w_arr = posteriors.x_hat_given_x(x0, x_hat)*dvol_nl
    w_out = w_arr.flatten()
    x_out = x0
    yield (x_out, w_out), (x_lin, np.zeros_like(x_lin[0]))
    #For later times, the class of the agent matters.
    w_arr_base = np.zeros((num_nl_classes, 2*N_steps+1, N_ptcl))
    for k in range(num_nl_classes):
        for m in range(-N_steps, N_steps+1):
            w_arr_base[k, m] = joint_k_x_x_hat_v_hat(
                k, x0, x_hat)

    veloc = [scene.director_field_vectorized(k, x0) for k in range(num_nl_classes)]
    for n in range(1, N_steps):
        #The following computations handle the nonlinear classes
        t = n * t_final / float(N_steps)
        ds = s_max / n
        w_arr = np.zeros((num_nl_classes, 2*n+1, N_ptcl))

        for k in range(num_nl_classes):
            for m in range(-n, n+1):
                w_arr[k, m] = w_arr_base[k, m]
                s = s_max * m / n
                v = s * veloc[k]
                r2 = (v[0]-v_hat[0])**2 + (v[1]-v_hat[1])**2
                w_arr[k, m] *= np.exp(-r2/(2*sigma_v**2)) / (2*np.pi*sigma_v**2)
                w_arr[k, m] *= 1.0/(2*s_max) * (s <= s_max) * (s >= -s_max)
                w_arr[k, m] *= ds * dvol_nl
        x_out = np.zeros((num_nl_classes, 2 * n+1, 2, N_ptcl))
        x_out[:, -n:, :, :] = x_arr[:, -n:, :, :]
        x_out[:, :n+1, :, :] = x_arr[:, :n+1, :, :]
        w_out = w_arr.flatten()
        x_out = np.vstack([x_out[:, :, 0, :].flatten(), x_out[:, :, 1, :].flatten()])
        if convolve:
        	#BEGIN GAUSSIAN CONVOLVE
            from scipy.stats import multivariate_normal
            N_conv = 15
            length = len(w_out) * N_conv
            gauss = np.vstack((np.random.normal(0, kappa * t_final/float(N_steps) * n, length),
                               np.random.normal(0, kappa * t_final/float(N_steps) * n, length)))
            positions = np.repeat(x_out, N_conv, axis=1) + gauss
            weights = (multivariate_normal.pdf(gauss.transpose(),
                                               mean=np.array([0, 0]),
                                               cov=(kappa * t_final/float(N_steps) * n)**2)
                       * np.repeat(w_out, N_conv) / N_conv)
            x_out = positions
            w_out = weights
            #END GAUSSIAN CONVOLVE
        #The following computations handle the linear predictor class
        w_lin = joint_lin_x_t_x_hat_v_hat(t, x_lin, x_hat, v_hat) * dy*dx
        #x_out = np.concatenate( [x_out, x_lin], axis=1)
        #w_out = np.concatenate( [w_out, w_lin])
        prob_of_mu = w_out.sum() + w_lin.sum()
        yield (x_out, w_out / prob_of_mu), (x_lin, w_lin / prob_of_mu)

def linear_generator(name, x_hat, v_hat, t_final, N_steps):
    """
    Generates probability distribution for the linear motion class.
    """
    set_scene(scenes[names.index(name)])
    x_span = np.linspace(-scene.width/2, scene.width/2, 250)
    dx = x_span[1] - x_span[0]
    y_span = np.linspace(-scene.height/2, scene.height/2, 250)
    dy = y_span[1] - y_span[0]
    X, Y = np.meshgrid(x_span, y_span)
    x_lin = np.vstack([X.flatten(), Y.flatten()])
    yield x_lin, np.zeros_like(x_lin[0])

    for n in range(1, N_steps):
        t = n * t_final / float(N_steps)
        w_lin = joint_lin_x_t_x_hat_v_hat(t, x_lin, x_hat, v_hat) * dy*dx
        yield x_lin, w_lin






if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from data import ex_set as test_set
    test_BB_ts = test_set[3]

    from process_data import BB_ts_to_curve
    curve = BB_ts_to_curve(test_BB_ts)
    offset = 30
    x_hat = curve[:, offset]
    v_hat = (curve[:, offset+2] - curve[:, offset-2])/4
    print("x_hat = " + str(x_hat))
    print("v_hat = " + str(v_hat))
    speed = np.sqrt(np.sum(v_hat**2))
    print("Measured speed / sigma_L = {:f}".format(speed / scene.sigma_L))
    print("sigma_L = {:f}".format(scene.sigma_L))
    k = 0
    N_steps = 100
    t_final = 100
    #Domain is actually larger than the domain we care about
    domain = [-scene.width, scene.width, -scene.height, scene.height]

    gen = full_model_generator("test", x_hat, v_hat, t_final, N_steps)
    n = 0
    from visualization import singular_distribution_to_image
    plt.ion()
    res = (50, 60)
    for (x_nl_arr, w_nl_arr), (x_lin_arr, w_lin_arr) in gen:
        if n%5 == 0:
            X, Y, Z = singular_distribution_to_image(x_nl_arr,
                                                     w_nl_arr,
                                                     domain,
                                                     res=res)
            plt.contourf(X, Y, Z, 20, cmap='viridis')
            plt.plot(curve[0], curve[1])
            plt.scatter(curve[0, offset], curve[1, offset])
            plt.show()
            plt.pause(0.02)
            plt.clf()
        n += 1
