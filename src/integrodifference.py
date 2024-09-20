import numpy as np
from scipy import ndimage
from typing import Callable

def simulate_integrodifference(
    growth_func: Callable,
    kernel_func: Callable,
    initial_cond: Callable,
    n_steps: int,
    xmin: float,
    xmax: float,
    dx: float,
    kernel_radius:
    float,
    normalize_kernel=True
    ):

    assert (xmin / dx).is_integer()
    assert (xmax / dx).is_integer()
    assert (kernel_radius / dx).is_integer()
    
    X = np.arange(xmin, xmax + dx, dx)
    soln = np.zeros((n_steps + 1,) + X.shape)
    soln[0] = initial_cond(X)

    kX = np.arange(-kernel_radius, kernel_radius + dx, dx)
    K = kernel_func(kX)
    K *= dx
    if normalize_kernel:
        K = K / np.sum(K)

    for t in range(n_steps):
        soln[t + 1] = ndimage.convolve(growth_func(soln[t]), K, mode="nearest")
        soln[t + 1][0] = growth_func(soln[t][0])
        soln[t + 1][-1] = growth_func(soln[t][-1])

    return soln

# named kernels
def LaplaceKernel(a: float=1):
    return lambda x: a/2 * np.exp(-a * np.abs(x))

# named growth functions
def RickerMap(r: float):
    return lambda u: u * np.exp(r * (1 - u))

def VortcampMap(r: float, a: float):
    return lambda u: u * np.exp(r * (1 - u) * (u/a - 1))

def VortcampMap_v2(r: float, a: float):
    return lambda u: u * np.exp(r * (1 - u) * (u - a))