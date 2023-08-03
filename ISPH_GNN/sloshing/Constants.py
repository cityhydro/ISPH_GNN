import numba as nb
import numpy as np
import torch

BASE_RADIUS = 1.0e-2  # the closest distance between each particles
ND_RAIUS = 2.2 * BASE_RADIUS
GRAD_RADIUS = 2.2 * BASE_RADIUS  # the control distance used to calculate gradient
LAP_RADIUS = 3.1 * BASE_RADIUS   # the control distance used to calculate Laplacian operator
COL_RADIUS = 0.92 * BASE_RADIUS   # the control radius of particles' collision
COL_COEF = 0.2
NU = 1.0e-6   # viscority coefficient
RHO = 1.0e3    # density of water
G = -9.81
DIMS = 2
DT = 0.001


@nb.njit(nb.float64(nb.float64, nb.float64))
def MPS_KERNEL(r, re):
    if r > re or r < 1e-8:
        return 0.0
    else:
        return (re / r) - 1.0
