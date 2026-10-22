"""F11 five-point Dirichlet negative Laplacian and mass-scaled DST-I inverse."""

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from scipy.fft import dstn, idstn

from .types import FloatArray, Grid2D
from .validation import InputValidationError, as_float64_array, finite_real, readonly_copy, require_finite_result


def apply_dirichlet_negative_laplacian(s_interior: FloatArray, grid: Grid2D) -> FloatArray:
    s = as_float64_array(s_interior, "s_interior", (grid.ny-2, grid.nx-2))
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        wx, wy = np.square(1/np.float64(grid.hx)), np.square(1/np.float64(grid.hy))
        out = (2*wx+2*wy)*s
        out[:, 1:] -= wx*s[:, :-1]
        out[:, :-1] -= wx*s[:, 1:]
        out[1:, :] -= wy*s[:-1, :]
        out[:-1, :] -= wy*s[1:, :]
    require_finite_result(out, "L action")
    return out


@lru_cache(maxsize=16)
def _eigenvalues(nx: int, ny: int, hx: float, hy: float) -> FloatArray:
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        x = np.square(2*np.sin(np.pi*np.arange(1, nx-1)/(2*(nx-1)))/hx)
        y = np.square(2*np.sin(np.pi*np.arange(1, ny-1)/(2*(ny-1)))/hy)
        spectrum = y[:, None]+x[None, :]
    require_finite_result(spectrum, "Poisson spectrum")
    if np.any(spectrum <= 0):
        raise FloatingPointError("Poisson spectrum underflowed to zero")
    return readonly_copy(spectrum)


def dirichlet_laplacian_eigenvalues(grid: Grid2D) -> FloatArray:
    try:
        return _eigenvalues(grid.nx, grid.ny, grid.hx, grid.hy)
    except FloatingPointError as exc:
        raise InputValidationError("Poisson grid: spectrum is not representable in float64") from exc


@dataclass(frozen=True)
class SPDInverse:
    """Factory-built M=omega*blockdiag(aI+bL,aI+bL); apply is its inverse."""

    grid: Grid2D
    a: float
    b: float
    _denominator: FloatArray = field(repr=False)

    @property
    def shape(self) -> tuple[int, int]:
        n = 2*self.grid.n_free_scalar
        return n, n

    def apply(self, rhs: FloatArray) -> FloatArray:
        rhs = as_float64_array(rhs, "rhs", (self.shape[0],))
        fields = rhs.reshape(2, self.grid.ny-2, self.grid.nx-2)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            transformed = dstn(fields, type=1, norm="ortho", axes=(-2, -1))
            result = idstn(transformed/self._denominator, type=1, norm="ortho", axes=(-2, -1))
        require_finite_result(result, "Poisson inverse")
        return result.ravel(order="C")

    def matvec(self, v: FloatArray) -> FloatArray:
        v = as_float64_array(v, "v", (self.shape[0],))
        fields = v.reshape(2, self.grid.ny-2, self.grid.nx-2)
        with np.errstate(over="raise", invalid="raise"):
            out = np.stack([self.grid.omega*(self.a*s+self.b*apply_dirichlet_negative_laplacian(s, self.grid))
                            for s in fields])
        require_finite_result(out, "M action")
        return out.ravel(order="C")


def build_screened_poisson_inverse(grid: Grid2D, *, a: float, b: float) -> SPDInverse:
    a, b = finite_real(a, "a"), finite_real(b, "b", positive=True)
    if a < 0:
        raise InputValidationError("a: must be nonnegative")
    spectrum = dirichlet_laplacian_eigenvalues(grid)
    try:
        with np.errstate(over="raise", invalid="raise"):
            denominator = grid.omega*(a+b*spectrum)
        require_finite_result(denominator, "Poisson denominator")
        if np.any(denominator <= 0):
            raise FloatingPointError("nonpositive denominator")
    except FloatingPointError as exc:
        raise InputValidationError("Poisson denominator: must be positive finite float64") from exc
    return SPDInverse(grid, a, b, readonly_copy(denominator))
