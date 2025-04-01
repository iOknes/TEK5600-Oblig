import numpy as np
import h5py

def read_h5(fpath: str):
    data = h5py.File(fpath)
    vel = data['Velocity']
    return np.array(vel["X-comp"]), np.array(vel["Y-comp"])

def read_h5_yx(fpath: str):
    x, y = read_h5(fpath)
    return x.T, y.T
