import numpy as np
import matplotlib.pyplot as plt

from numba import jit

from read_data import read_h5, read_h5_yx
from seeding import seed_uniform, seed_random, seed_vorticity
from integration import forward_euler, runge_kutta_4

def f(p: np.ndarray, ux: np.ndarray, uy: np.ndarray):
    # If out of bounds, return 0s
    pr = np.array([round(p[1]), round(p[0])])
    if pr[0] < 0 or pr[0] >= ux.shape[0] or pr[1] < 0 or pr[1] >= ux.shape[1]:
        return np.array([0,0])
    return np.array([ux[pr[0], pr[1]], uy[pr[0], pr[1]]])

def field_line(ux, uy, p0, l, dx=0.2, n_max=10000, integrator="fe", backwards=True):
    """
    ## Parameters:
        - l_max: length of field line in either direction
    """
    #if integrator == "fe":
    #    integrator = forward_euler
    #if integrator == "rk4":
        #    integrator = runge_kutta_4

    line_f = [p0]
    line_b = [p0]
    length = 0
    f_stall = False
    b_stall = not backwards
    while length < l and len(line_f) + len(line_b) < n_max:
        if not f_stall:
            #p_next = integrator(line_f[-1], ux, uy, dx)
            if integrator == "fe":
                p_next = forward_euler(line_f[-1], ux, uy, dx)
            elif integrator == "rk4":
                p_next = runge_kutta_4(line_f[-1], ux, uy, dx)
            else:
                p_next = None

            if np.all(p_next - line_f[-1] == 0):
                f_stall = True
            length += np.linalg.norm(p_next - line_f[-1])
            line_f.append(p_next)
        if backwards and not b_stall:
            #pb_next = integrator(line_b[-1], -ux, -uy, dx)
            if integrator == "fe":
                pb_next = forward_euler(line_b[-1], -ux, -uy, dx)
            elif integrator == "rk4":
                pb_next = runge_kutta_4(line_b[-1], -ux, -uy, dx)
            else:
                pb_next = None

            if np.all(pb_next - line_b[-1] == 0):
                b_stall = True
            length += np.linalg.norm(pb_next - line_b[-1])
            line_b.append(pb_next)
        if f_stall and b_stall:
            break

    return np.array(line_b[-1:0:-1] + line_f)


def generate_field_lines(ux, uy, length, n_seeds, dx=0.2, seeding="uniform", integrator="fe", backwards=True):
    if seeding == "uniform":
        p0 = seed_uniform(0, ux.shape[0] - 1, 0, uy.shape[1] - 1, n_seeds)
    elif seeding == "random":
        p0 = seed_random(0, ux.shape[0] - 1, 0, uy.shape[1] - 1, n_seeds)
    elif seeding == "vorticity":
        p0 = seed_vorticity(ux, uy, n_seeds)
    else:
        raise ValueError(f"Seeding must be of valid type, not {seeding}")
    #p0 = seedingf(0, ux.shape[0] - 1, 0, uy.shape[1] - 1, n_seeds)
    fl = [field_line(ux, uy, i, length * ux.shape[0], dx=dx, integrator=integrator, backwards=backwards) for i in p0]
    return fl

def plot_field_lines(ax, fl, metsim=False):
    for l in fl:
        if metsim:
            ax.plot(l[:,0].T, l[:,1].T, color="black")
        else:
            ax.plot(l[:,1], l[:,0], color="black")
