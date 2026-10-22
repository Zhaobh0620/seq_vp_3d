"""Mass-scaled Q1 finite-element stiffness metric on a rectangular grid.

For either displacement component, M=omega*a*I+b*K_Q1, where K_Q1 is
the consistently integrated bilinear-element stiffness with zero boundary
values.  Its DST-I eigenbasis is shared with, but its eigenvalues and stencil
are different from, the five-point operator.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .poisson import SPDInverse
from .types import FloatArray, Grid2D
from .validation import InputValidationError, as_float64_array, finite_real, readonly_copy, require_finite_result


def apply_q1_stiffness(s_interior: FloatArray, grid: Grid2D) -> FloatArray:
    """Apply K_Q1, including physical cell area, without assembling a matrix."""
    s = as_float64_array(s_interior, "s_interior", (grid.ny-2, grid.nx-2))
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        wx, wy = np.square(1/np.float64(grid.hx)), np.square(1/np.float64(grid.hy))
        center = 4*(wx+wy)/3
        horizontal, vertical, diagonal = (-2*wx+wy)/3, (wx-2*wy)/3, -(wx+wy)/6
        out = center*s
        out[:, 1:] += horizontal*s[:, :-1]
        out[:, :-1] += horizontal*s[:, 1:]
        out[1:, :] += vertical*s[:-1, :]
        out[:-1, :] += vertical*s[1:, :]
        out[1:, 1:] += diagonal*s[:-1, :-1]
        out[1:, :-1] += diagonal*s[:-1, 1:]
        out[:-1, 1:] += diagonal*s[1:, :-1]
        out[:-1, :-1] += diagonal*s[1:, 1:]
        out *= grid.omega
    require_finite_result(out, "Q1 stiffness action")
    return out


@lru_cache(maxsize=16)
def _eigenvalues(nx: int, ny: int, hx: float, hy: float) -> FloatArray:
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        theta_x = np.pi*np.arange(1, nx-1)/(nx-1)
        theta_y = np.pi*np.arange(1, ny-1)/(ny-1)
        ell_x = np.square(2*np.sin(theta_x/2)/hx)
        ell_y = np.square(2*np.sin(theta_y/2)/hy)
        mass_x, mass_y = (2+np.cos(theta_x))/3, (2+np.cos(theta_y))/3
        spectrum = mass_y[:, None]*ell_x[None, :]+ell_y[:, None]*mass_x[None, :]
    require_finite_result(spectrum, "Q1 metric spectrum")
    if np.any(spectrum <= 0):
        raise FloatingPointError("Q1 metric spectrum is not positive float64")
    return readonly_copy(spectrum)


def q1_stiffness_eigenvalues(grid: Grid2D) -> FloatArray:
    """Eigenvalues of K_Q1/omega, not of K_Q1 or the five-point Laplacian."""
    try:
        return _eigenvalues(grid.nx, grid.ny, grid.hx, grid.hy)
    except FloatingPointError as exc:
        raise InputValidationError("Q1 metric grid: positive finite spectrum required") from exc


@dataclass(frozen=True)
class Q1MetricInverse(SPDInverse):
    """M=blockdiag(omega*a*I+b*K_Q1, omega*a*I+b*K_Q1).

    Only the transform inverse/layout is inherited; matvec uses Q1 stiffness.
    """

    def matvec(self, v: FloatArray) -> FloatArray:
        v = as_float64_array(v, "v", (self.shape[0],))
        fields = v.reshape(2, self.grid.ny-2, self.grid.nx-2)
        with np.errstate(over="raise", invalid="raise"):
            out = np.stack([self.grid.omega*self.a*s+self.b*apply_q1_stiffness(s, self.grid)
                            for s in fields])
        require_finite_result(out, "Q1 metric action")
        return out.ravel(order="C")


def build_q1_metric_inverse(grid: Grid2D, *, a: float, b: float) -> Q1MetricInverse:
    a, b = finite_real(a, "a"), finite_real(b, "b", positive=True)
    if a < 0:
        raise InputValidationError("a: must be nonnegative")
    spectrum = q1_stiffness_eigenvalues(grid)
    try:
        with np.errstate(over="raise", invalid="raise"):
            denominator = grid.omega*(a+b*spectrum)
        require_finite_result(denominator, "Q1 metric denominator")
        if np.any(denominator <= 0):
            raise FloatingPointError("nonpositive Q1 metric denominator")
    except FloatingPointError as exc:
        raise InputValidationError("Q1 metric denominator: positive finite float64 required") from exc
    return Q1MetricInverse(grid, a, b, readonly_copy(denominator))
