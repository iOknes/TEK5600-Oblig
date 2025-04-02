import numpy as np
import matplotlib.pyplot as plt

from read_data import read_h5, read_h5_yx
#from integration import forward_euler, runge_kutta_4

def f(p, ux, uy):
    p = np.array([round(p[1]), round(p[0])])
    if p[0] < 0 or p[0] >= ux.shape[0] or p[1] < 0 or p[1] >= ux.shape[1]:
        return np.array([0,0])
    return np.array([ux[*p], uy[*p]])

def field_line(ux, uy, p0, nstep, dx=1, integrator=None):
    """
    ## Parameters:
        - l_max: length of field line in either direction
    """
    line = [p0]
    length = 0
    #while length < l:
    for i in range(nstep):
        dl = dx * f(line[-1], ux, uy)
        length += np.linalg.norm(dl)
        line.append(line[-1] + dl)
    return np.array(line)

def main():
    ux, uy = read_h5_yx("data/isabel_2d.h5")
    #ux, uy = read_h5("data/metsim1_2d.h5")
    x = np.arange(ux.shape[0])
    y = np.arange(ux.shape[1])
    x, y = np.meshgrid(x, y)
    f1 = field_line(ux, uy, [340, 280], 100)
    print(f"f1 len: {len(f1)}")

    fig, ax = plt.subplots(1, 1)
    n = 10
    plt.quiver(x[::n,::n], y[::n,::n], ux[::n,::n], uy[::n,::n], scale=1000)
    #ax.plot(f1[:,0], f1[:,1])
    #p0 = [[499, i*50] for i in range(10)]
    p0 = [[i*50, 100] for i in range(10)]
    fl = [field_line(ux, uy, i, 100) for i in p0]
    for i in fl:
        ax.plot(i[:,0], i[:,1])
    #ax.invert_yaxis()
    #ax.streamplot(x, y, ux, uy)
    plt.show()

if __name__ == "__main__":
    main()
