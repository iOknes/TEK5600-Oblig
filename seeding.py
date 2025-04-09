import numpy as np

from numba import jit
from meshgrid import meshgrid

def seed_uniform(xmin, xmax, ymin, ymax, n):
    """
    Create seeding points uniformly in the specified range.
    A total n^2 points are created.
    """
    x = np.linspace(ymin, ymax, n)
    y = np.linspace(ymin, ymax, n)
    #x, y = np.meshgrid(x, y)
    x, y = meshgrid(x, y)
    #return np.stack([x.flat, y.flat], axis=1)
    return np.array([x.flat,y.flat]).T

def seed_random(xmin, xmax, ymin, ymax, n):
    """
    Create seeding points randomly in the specified range.
    A total of n points are created.
    """
    x = np.random.uniform(xmin, xmax, n)
    y = np.random.uniform(ymin, ymax, n)
    #return np.stack([x,y], axis=1)
    return np.array([x,y]).T

def seed_vorticity(ux, uy, n):
    """
    Create seeding points randomly, but with higher density in the regions of high vorticity.
    A total of n points are created.
    """
    dudx, dudy = np.gradient(ux)
    dvdx, dvdy = np.gradient(uy)
    curl = dvdx - dudy

    p = []

    x = np.arange(ux.shape[0])
    y = np.arange(uy.shape[1])

    for i in range(n):
        for j in range(n):
            x_ = np.random.choice(x, p=np.abs(np.mean(curl, axis=1))/np.sum(np.abs(np.mean(curl, axis=1))))
            y_ = np.random.choice(y, p=np.abs(curl[x_,:]) / np.sum(np.abs(curl[x_,:])))
            p.append([y_, x_])

    return np.array(p)
