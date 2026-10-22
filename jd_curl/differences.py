"""F06 full-node differences using 1D sparse matrices and exact transposes."""

import numpy as np
from scipy.sparse import csr_matrix, diags

from .types import FloatArray, Grid2D
from .validation import as_float64_array


def _derivative_1d(n: int, h: float) -> csr_matrix:
    matrix = diags((-np.ones(n - 1), np.ones(n - 1)), (-1, 1), shape=(n, n), format="lil")
    matrix[0, :3] = [-3.0, 4.0, -1.0]
    matrix[-1, -3:] = [1.0, -4.0, 3.0]
    return (matrix.tocsr() * (0.5 / h)).tocsr()


class DifferenceOperators2D:
    """One cached pair of 1D operators; no production 2D Kronecker matrices."""

    def __init__(self, grid: Grid2D):
        self._shape = (grid.ny, grid.nx)
        self._Dx = _derivative_1d(grid.nx, grid.hx)
        self._Dy = _derivative_1d(grid.ny, grid.hy)
        self._Dx_T = self._Dx.T.tocsr()
        self._Dy_T = self._Dy.T.tocsr()

    @property
    def shape(self) -> tuple[int, int]:
        """Scalar-field shape, used by geometry without exposing sparse matrices."""
        return self._shape

    def dx(self, s: FloatArray) -> FloatArray:
        s = as_float64_array(s, "dx.s", self._shape)
        return (self._Dx @ s.T).T

    def dy(self, s: FloatArray) -> FloatArray:
        s = as_float64_array(s, "dy.s", self._shape)
        return self._Dy @ s

    def dx_T(self, w: FloatArray) -> FloatArray:
        w = as_float64_array(w, "dx_T.w", self._shape)
        return (self._Dx_T @ w.T).T

    def dy_T(self, w: FloatArray) -> FloatArray:
        w = as_float64_array(w, "dy_T.w", self._shape)
        return self._Dy_T @ w


def make_difference_operators(grid: Grid2D) -> DifferenceOperators2D:
    return DifferenceOperators2D(grid)
