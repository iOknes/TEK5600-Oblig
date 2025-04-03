import numpy as np

def seed_uniform(xmin, xmax, ymin, ymax, n):
    """
    Create seeding points uniformly in the specified range.
    A total n^2 points are created.
    """
    stepx = (xmax - xmin)/n
    stepy = (ymax - ymin)/n
    x = np.arange(xmin, xmax, stepx)
    y = np.arange(ymin, ymax, stepy)
    x, y = np.meshgrid(x, y)
    return np.stack([x.flat, y.flat], axis=1)

def seed_random(xmin, xmax, ymin, ymax, n):
    """
    Create seeding points randomly in the specified range.
    A total of n^2 points are created.
    """
    x = np.random.uniform(xmin, xmax, n)
    y = np.random.uniform(ymin, ymax, n)
    x, y = np.meshgrid(x, y)
    return np.stack([x.flat, y.flat], axis=1)
