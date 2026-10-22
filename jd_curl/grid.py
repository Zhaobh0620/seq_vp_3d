"""Physical rectangular grids and full-node trapezoidal quadrature (F06)."""

import numpy as np

from .types import Grid2D
from .validation import InputValidationError, finite_real, grid_size, readonly_copy


def make_grid_2d(
    *, nx: int, ny: int, hx: float, hy: float,
    x0: float = 0.0, y0: float = 0.0,
) -> Grid2D:
    nx, ny = grid_size(nx, "nx"), grid_size(ny, "ny")
    hx = finite_real(hx, "hx", positive=True)
    hy = finite_real(hy, "hy", positive=True)
    x0, y0 = finite_real(x0, "x0"), finite_real(y0, "y0")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        x = x0 + hx * np.arange(nx, dtype=np.float64)
        y = y0 + hy * np.arange(ny, dtype=np.float64)
        omega = hx * hy
        tx, ty = np.ones(nx), np.ones(ny)
        tx[[0, -1]], ty[[0, -1]] = 0.5, 0.5
        q = omega * ty[:, None] * tx[None, :]
    for name, axis in (("x", x), ("y", y)):
        if not np.all(np.isfinite(axis)) or not np.all(np.diff(axis) > 0):
            raise InputValidationError(f"{name}: coordinates must be finite and strictly increasing")
    if not np.all(np.isfinite(q)) or not np.all(q > 0):
        raise InputValidationError("q: spacings must produce positive finite float64 weights")
    X, Y = np.meshgrid(x, y, indexing="xy")
    return Grid2D(
        nx=nx, ny=ny, hx=hx, hy=hy,
        x=readonly_copy(x), y=readonly_copy(y),
        X=readonly_copy(X), Y=readonly_copy(Y),
        q=readonly_copy(q), sqrt_q=readonly_copy(np.sqrt(q)),
        omega=omega, n_free_scalar=(nx - 2) * (ny - 2),
    )
