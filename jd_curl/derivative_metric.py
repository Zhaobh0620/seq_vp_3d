"""Separable SPD metric matched to the full-node weighted first derivatives.

This is an optional gradient metric, not a replacement energy or the five-point
F11 Poisson operator.  With zero-boundary injection S and q=omega*t_y*t_x,
its scalar block is omega*a*I + b*S.T*(Dx.T*Q*Dx+Dy.T*Q*Dy)*S.
"""

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from scipy.linalg import eigh

from .differences import _derivative_1d
from .types import FloatArray, Grid2D
from .validation import InputValidationError, as_float64_array, finite_real, readonly_copy, require_finite_result


@lru_cache(maxsize=32)
def _weighted_gram(n: int, h: float) -> tuple[FloatArray, FloatArray, FloatArray]:
    """One-dimensional S.T*D.T*T*D*S and its orthonormal eigensystem."""
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        derivative = _derivative_1d(n, h).toarray()[:, 1:-1]
        weights = np.ones(n)
        weights[[0, -1]] = .5
        gram = derivative.T @ (weights[:, None]*derivative)
    require_finite_result(gram, "derivative metric Gram")
    values, vectors = eigh(gram, check_finite=False)
    require_finite_result(values, "derivative metric spectrum")
    require_finite_result(vectors, "derivative metric eigenvectors")
    if np.any(values <= 0):
        raise FloatingPointError("derivative metric spectrum is not positive float64")
    return readonly_copy(gram), readonly_copy(values), readonly_copy(vectors)


@dataclass(frozen=True)
class DerivativeMetricInverse:
    """Two identical component-major blocks with exact separable inversion."""

    grid: Grid2D
    a: float
    b: float
    _gram_x: FloatArray = field(repr=False)
    _gram_y: FloatArray = field(repr=False)
    _vectors_x: FloatArray = field(repr=False)
    _vectors_y: FloatArray = field(repr=False)
    _denominator: FloatArray = field(repr=False)

    @property
    def shape(self) -> tuple[int, int]:
        n = 2*self.grid.n_free_scalar
        return n, n

    def apply(self, rhs: FloatArray) -> FloatArray:
        rhs = as_float64_array(rhs, "rhs", (self.shape[0],))
        fields = rhs.reshape(2, self.grid.ny-2, self.grid.nx-2)
        x, y = self._vectors_x, self._vectors_y
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            out = np.stack([y @ ((y.T @ s @ x)/self._denominator) @ x.T for s in fields])
        require_finite_result(out, "derivative metric inverse")
        return out.ravel(order="C")

    def matvec(self, v: FloatArray) -> FloatArray:
        v = as_float64_array(v, "v", (self.shape[0],))
        fields = v.reshape(2, self.grid.ny-2, self.grid.nx-2)
        with np.errstate(over="raise", invalid="raise"):
            out = np.stack([self.grid.omega*(self.a*s+self.b*(self._gram_y @ s+s @ self._gram_x))
                            for s in fields])
        require_finite_result(out, "derivative metric action")
        return out.ravel(order="C")


def build_derivative_metric_inverse(grid: Grid2D, *, a: float, b: float) -> DerivativeMetricInverse:
    a, b = finite_real(a, "a"), finite_real(b, "b", positive=True)
    if a < 0:
        raise InputValidationError("a: must be nonnegative")
    try:
        gx, ex, vx = _weighted_gram(grid.nx, grid.hx)
        gy, ey, vy = _weighted_gram(grid.ny, grid.hy)
        with np.errstate(over="raise", invalid="raise"):
            denominator = grid.omega*(a+b*(ey[:, None]+ex[None, :]))
        require_finite_result(denominator, "derivative metric denominator")
        if np.any(denominator <= 0):
            raise FloatingPointError("nonpositive derivative metric denominator")
    except FloatingPointError as exc:
        raise InputValidationError("derivative metric: positive finite spectrum required") from exc
    return DerivativeMetricInverse(grid, a, b, gx, gy, vx, vy, readonly_copy(denominator))
