import numpy as np

def seed_uniform(xmin, xmax, ymin, ymax, n):
    """
    Create seeding points uniformly in the specified range.
    A total n^2 points are created.
    """
    x = np.linspace(ymin, ymax, n)
    y = np.linspace(ymin, ymax, n)
    x, y = np.meshgrid(x, y)
    return np.stack([x.flat, y.flat], axis=1)

def seed_random(xmin, xmax, ymin, ymax, n):
    """
    Create seeding points randomly in the specified range.
    A total of n points are created.
    """
    x = np.random.uniform(xmin, xmax, n)
    y = np.random.uniform(ymin, ymax, n)
    return np.stack([x,y], axis=1)
