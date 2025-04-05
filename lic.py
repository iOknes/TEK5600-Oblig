import numpy as np

def noise_texture(nx, ny):
    return (255 * np.random.rand(nx, ny)).astype(int)

def lic(filter: np.matrix):
    pass
