"""Component-major S / S.T; only initial-displacement packing checks boundary."""

import numpy as np

from .types import FloatArray, Grid2D
from .validation import as_float64_array, require_zero_boundary


def inject_free(z: FloatArray, grid: Grid2D) -> FloatArray:
    """Lift x-interior then y-interior values into a fresh zero-boundary field."""
    z = as_float64_array(z, "z", (2 * grid.n_free_scalar,))
    u = np.zeros((grid.ny, grid.nx, 2), dtype=np.float64)
    u[1:-1, 1:-1, :] = z.reshape(2, grid.ny - 2, grid.nx - 2).transpose(1, 2, 0)
    return u


def restrict_free(v_full: FloatArray, grid: Grid2D) -> FloatArray:
    """Euclidean S.T, not Q-weighted; nonzero full-field boundaries are allowed."""
    v_full = as_float64_array(v_full, "v_full", (grid.ny, grid.nx, 2))
    return np.concatenate((v_full[1:-1, 1:-1, 0].ravel(order="C"),
                           v_full[1:-1, 1:-1, 1].ravel(order="C")))


def pack_initial_displacement(u0: FloatArray, grid: Grid2D) -> FloatArray:
    u0 = as_float64_array(u0, "u0", (grid.ny, grid.nx, 2))
    require_zero_boundary(u0, "u0")
    return restrict_free(u0, grid)
