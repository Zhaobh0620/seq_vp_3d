"""F02 full-node geometry, reusing the F06 differences without boundary repair."""

import numpy as np

from .differences import DifferenceOperators2D
from .types import FloatArray, GeometryState, Grid2D
from .validation import as_float64_array, readonly_copy


def bilinear_cell_jacobians(u: FloatArray, grid: Grid2D) -> FloatArray:
    """Four corner determinants of the bilinear interpolant of id+u per cell.

    A bilinear map's determinant is affine within each cell, so its extrema
    occur at corners. This diagnostic is not the objective's nodal difference
    JD, an acceptance filter, or a proof of global injectivity.
    Corner order: (x,y) = (0,0), (1,0), (0,1), (1,1).
    """
    u = as_float64_array(u, "cell_geometry.u", (grid.ny, grid.nx, 2))
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        dx = (u[:, 1:]-u[:, :-1])/grid.hx
        dy = (u[1:, :]-u[:-1, :])/grid.hy
        dx[..., 0] += 1.
        dy[..., 1] += 1.
        pairs = ((dx[:-1], dy[:, :-1]), (dx[:-1], dy[:, 1:]),
                 (dx[1:], dy[:, :-1]), (dx[1:], dy[:, 1:]))
        J = np.stack([a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0] for a, b in pairs], axis=-1)
    if not np.all(np.isfinite(J)):
        raise FloatingPointError("cell geometry: nonfinite result")
    return readonly_copy(J)


def compute_geometry(u: FloatArray, diff: DifferenceOperators2D) -> GeometryState:
    """Allow affine test fields; zero boundary is imposed by injection, not here.

    Invalid inputs raise InputValidationError. Numerical overflow raises
    FloatingPointError for evaluate_state to classify separately.
    Nonpositive finite J is valid diagnostic output, never clipped or rejected.
    """
    u = as_float64_array(u, "geometry.u", diff.shape + (2,))
    with np.errstate(over="raise", invalid="raise"):
        F = np.empty(diff.shape + (2, 2), dtype=np.float64)
        F[..., 0, 0] = 1.0 + diff.dx(u[..., 0])
        F[..., 0, 1] = diff.dy(u[..., 0])
        F[..., 1, 0] = diff.dx(u[..., 1])
        F[..., 1, 1] = 1.0 + diff.dy(u[..., 1])
        A = np.empty_like(F)
        A[..., 0, 0] = F[..., 1, 1]
        A[..., 0, 1] = -F[..., 1, 0]
        A[..., 1, 0] = -F[..., 0, 1]
        A[..., 1, 1] = F[..., 0, 0]
        J = F[..., 0, 0] * F[..., 1, 1] - F[..., 0, 1] * F[..., 1, 0]
        curl = F[..., 1, 0] - F[..., 0, 1]
    if not all(np.all(np.isfinite(a)) for a in (F, A, J, curl)):
        raise FloatingPointError("geometry: nonfinite result from finite displacement")
    return GeometryState(*(readonly_copy(a) for a in (F, A, J, curl)))
