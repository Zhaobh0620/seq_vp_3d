"""T10 mass-scaled seven-point or consistent Q1 Poisson inverse, via DST-I."""
from dataclasses import dataclass, field

import numpy as np
from scipy.fft import dstn, idstn

from ..validation import InputValidationError, as_float64_array, finite_real, readonly_copy, require_finite_result
from .grid import Grid3D


def _metric_name(metric):
    if metric not in ("seven_point", "q1"):
        raise InputValidationError("metric: expected seven_point or q1")


def poisson_spectrum(grid, *, metric):
    _metric_name(metric)
    ell, mass = [], []
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for n, h in zip(grid.shape[::-1], grid.spacing):
            theta = np.pi * np.arange(1, n - 1) / (n - 1)
            ell.append((2 * np.sin(theta / 2) / h)**2)
            mass.append((2 + np.cos(theta)) / 3)
        ex, ey, ez = ell[0][None, None, :], ell[1][None, :, None], ell[2][:, None, None]
        mx, my, mz = mass[0][None, None, :], mass[1][None, :, None], mass[2][:, None, None]
        result = ex + ey + ez if metric == "seven_point" else ex * my * mz + mx * ey * mz + mx * my * ez
    require_finite_result(result, "Poisson spectrum")
    if np.any(result <= 0):
        raise FloatingPointError("nonpositive Poisson spectrum")
    return readonly_copy(result)


# E89 (P07, explicit, off by default): poisson_spectrum per (grid object, metric), computed once by the
# same function (the spectrum depends on the grid and metric only; the denominator is rebuilt every time).
_SPECTRA = None


def enable_spectrum_cache(enabled=True):
    global _SPECTRA
    _SPECTRA = {} if enabled else None


def _spectrum(grid, metric):
    memo = _SPECTRA
    if memo is not None:
        entry = memo.get((id(grid), metric))
        if entry is not None and entry[0]() is grid:
            return entry[1]
    value = poisson_spectrum(grid, metric=metric)
    if memo is not None:
        import weakref
        if len(memo) > 16:
            for key in [k for k, (ref, _) in memo.items() if ref() is None]:
                del memo[key]
        memo[(id(grid), metric)] = (weakref.ref(grid), value)
    return value


def _tridiagonal(field, axis, diagonal, off_diagonal):
    result = diagonal * field
    left, right = [slice(None)] * field.ndim, [slice(None)] * field.ndim
    left[axis], right[axis] = slice(1, None), slice(None, -1)
    result[tuple(left)] += off_diagonal * field[tuple(right)]
    result[tuple(right)] += off_diagonal * field[tuple(left)]
    return result


def apply_poisson_operator(field, grid, *, metric):
    """Scalar L7 or KQ/omega action, independently of its spectral inverse."""
    _metric_name(metric)
    field = as_float64_array(field, "field", grid.interior_shape)
    out = np.zeros_like(field)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for axis, h in enumerate(grid.spacing[::-1]):
            factor = (1 / np.float64(h))**2
            term = _tridiagonal(field, axis, 2 * factor, -factor)
            if metric == "q1":
                for other in range(3):
                    if other != axis:
                        term = _tridiagonal(term, other, 2 / 3, 1 / 6)
            out += term
    require_finite_result(out, "Poisson operator")
    return out


@dataclass(frozen=True)
class PoissonInverse3D:
    grid: Grid3D
    a: float
    b: float
    metric: str
    denominator: np.ndarray = field(repr=False)

    @property
    def shape(self):
        n = 3 * self.grid.n_free_scalar
        return n, n

    def apply(self, rhs):
        rhs = as_float64_array(rhs, "rhs", (self.shape[0],))
        fields = rhs.reshape((3,) + self.grid.interior_shape)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            transformed = dstn(fields, type=1, norm="ortho", axes=(-3, -2, -1))
            out = idstn(transformed / self.denominator, type=1, norm="ortho", axes=(-3, -2, -1))
        require_finite_result(out, "Poisson inverse")
        return out.ravel(order="C")

    def matvec(self, v):
        v = as_float64_array(v, "v", (self.shape[0],))
        fields = v.reshape((3,) + self.grid.interior_shape)
        with np.errstate(over="raise", invalid="raise"):
            out = np.stack([self.grid.omega * (self.a * s + self.b * apply_poisson_operator(s, self.grid, metric=self.metric))
                            for s in fields])
        require_finite_result(out, "Poisson matrix action")
        return out.ravel(order="C")


def build_poisson_inverse(grid, *, a, b, metric="q1"):
    a, b = finite_real(a, "a"), finite_real(b, "b", positive=True)
    if a < 0:
        raise InputValidationError("a: must be nonnegative")
    _metric_name(metric)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            denominator = grid.omega * (a + b * _spectrum(grid, metric))
        require_finite_result(denominator, "Poisson denominator")
        if np.any(denominator <= 0):
            raise FloatingPointError("nonpositive denominator")
    except FloatingPointError as exc:
        raise InputValidationError("Poisson coefficients/grid not representable in float64") from exc
    return PoissonInverse3D(grid, a, b, metric, readonly_copy(denominator))
