import numpy as np

def normalised(x):
    if np.linalg.norm(x) == 0:
        return x
    return x / np.linalg.norm(x)

def f(p: np.ndarray, ux: np.ndarray, uy: np.ndarray):
    # If out of bounds, return 0s
    pr = np.array([round(p[1]), round(p[0])])
    if pr[0] < 0 or pr[0] >= ux.shape[0] or pr[1] < 0 or pr[1] >= ux.shape[1]:
        return np.array([0,0])
    return np.array([ux[pr[0], pr[1]], uy[pr[0], pr[1]]])

def forward_euler(p, ux, uy, dx, backwards=False, normalise=False):
    if normalise and np.all(f(p, ux, uy) != 0):
        return p + dx * f(p, ux, uy) / np.linalg.norm(f(p, ux, uy))
    return p + dx * f(p, ux, uy)

def runge_kutta_4(p, ux, uy, dx, backwards=False, normalise=False):
    if normalise and np.all(f(p, ux, uy) != 0):
        K1 = dx * normalised(f(p, ux, uy))
        K2 = dx * normalised(f(p + 0.5*K1, ux, uy))
        K3 = dx * normalised(f(p + 0.5*K2, ux, uy))
        K4 = dx * normalised(f(p + K3, ux, uy))
    else:
        K1 = dx*f(p, ux, uy)
        K2 = dx*f(p + 0.5*K1, ux, uy)
        K3 = dx*f(p + 0.5*K2, ux, uy)
        K4 = dx*f(p + K3, ux, uy)
    return p + (1/6.0)*(K1 + 2*K2 + 2*K3 + K4)
