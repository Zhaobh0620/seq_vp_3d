"""T01 component-major injection/restriction and exact six-face boundary."""
import numpy as np

from ..validation import InputValidationError, as_float64_array
from .validation import zero_boundary


def inject_free(theta, grid):
    theta = as_float64_array(theta, "theta", (3 * grid.n_free_scalar,))
    out = np.zeros(grid.shape + (3,))
    out[1:-1, 1:-1, 1:-1] = np.moveaxis(theta.reshape((3,) + grid.interior_shape), 0, -1)
    return out


def restrict_free(field, grid):
    field = as_float64_array(field, "field", grid.shape + (3,))
    return np.moveaxis(field[1:-1, 1:-1, 1:-1], -1, 0).copy().ravel(order="C")


def pack_initial_displacement(u, grid):
    u = as_float64_array(u, "u", grid.shape + (3,))
    if not zero_boundary(u):
        raise InputValidationError("u: all three components on all six faces must be exactly zero")
    return restrict_free(u, grid)
