import numpy as np
from numba import jit


def forward_euler(p, f, ux, uy, dx, backwards=False):
    if backwards:
        return p - dx * f(p, ux, uy)
    return p + dx * f(p, ux, uy)


def runge_kutta_4(p, f, ux, uy, dx, backwards=False):
    if backwards:
        K1 = dx*f(p, ux, uy)
        K2 = dx*f(p - 0.5*K1, ux, uy)
        K3 = dx*f(p - 0.5*K2, ux, uy)
        K4 = dx*f(p - K3, ux, uy)
        return p - (1/6.0)*(K1 + 2*K2 + 2*K3 + K4)
    K1 = dx*f(p, ux, uy)
    K2 = dx*f(p + 0.5*K1, ux, uy)
    K3 = dx*f(p + 0.5*K2, ux, uy)
    K4 = dx*f(p + K3, ux, uy)
    return p + (1/6.0)*(K1 + 2*K2 + 2*K3 + K4)
