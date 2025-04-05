import numpy as np
import matplotlib.pyplot as plt

from read_data import read_h5, read_h5_yx
from seeding import seed_uniform, seed_random
from integration import forward_euler, runge_kutta_4

def f(p, ux, uy, interpolate=True):
    # If out of bounds, return 0s
    pr = np.array([round(p[1]), round(p[0])])
    if pr[0] < 0 or pr[0] >= ux.shape[0] or pr[1] < 0 or pr[1] >= ux.shape[1]:
        return np.array([0,0])
    return np.array([ux[*pr], uy[*pr]])

def field_line(ux, uy, p0, nstep, dx=0.1, integrator=forward_euler, backwards=False):
    """
    ## Parameters:
        - l_max: length of field line in either direction
    """
    line_f = [p0]
    line_b = [p0]
    length = 0
    #while length < l:
    for i in range(nstep):
        p_next = integrator(line_f[-1], f, ux, uy, dx, backwards=False)
        length += np.linalg.norm(p_next - line_f[-1])
        line_f.append(p_next)
        if backwards:
            pb_next = integrator(line_b[-1], f, ux, uy, dx, backwards=True)
            length += np.linalg.norm(pb_next - line_b[-1])
            line_b.append(pb_next)

    return np.array(line_b[-1:0:-1] + line_f)

def main():
    ux, uy = read_h5_yx("data/isabel_2d.h5")
    #ux, uy = read_h5("data/metsim1_2d.h5")
    x = np.arange(ux.shape[0])
    y = np.arange(ux.shape[1])
    x, y = np.meshgrid(x, y)

    fig, ax = plt.subplots(1, 1)
    #n = 10
    #plt.quiver(x[::n,::n], y[::n,::n], ux[::n,::n], uy[::n,::n], scale=1000)
    p0 = seed_uniform(0, 499, 0, 499, 20)
    #p0 = seed_random(0, 499, 0, 499, 400)
    fl = [field_line(ux, uy, i, 1000, integrator=forward_euler, backwards=True) for i in p0]
    #fl = [field_line(ux, uy, i, 1000, integrator=runge_kutta_4, backwards=True) for i in p0]
    for i in fl:
        ax.plot(i[:,0], i[:,1])
    plt.show()

if __name__ == "__main__":
    main()
