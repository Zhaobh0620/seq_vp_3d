"""T05 float64 intervals with one outward nextafter per elementary operation.

Finite endpoints only; no reassociated reductions, BLAS, FMA or fast-math.
Overflow/zero-containing division raises ArithmeticUncertain for the caller
to classify as uncertain, never as a positivity certificate.
"""
from dataclasses import dataclass

import numpy as np


class ArithmeticUncertain(FloatingPointError):
    pass


def arithmetic_supported():
    """Fail closed if binary64/gradual-underflow prerequisites are not observed.

    This runtime smoke check supplements, not replaces, rational inclusion
    tests of this NumPy CPU implementation. Unsupported backends aren't used.
    """
    f = np.finfo(np.float64)
    with np.errstate(all="ignore"):
        sub = np.nextafter(np.float64(0), np.float64(1))
        return bool(f.nmant == 52 and f.maxexp == 1024 and sub > 0
                    and np.multiply(sub, np.float64(1)) == sub
                    and np.divide(np.float64(f.tiny), np.float64(2)) > 0
                    and np.add(np.float64(1), np.float64(2**-53)) == 1)


@dataclass(frozen=True)
class Interval:
    lo: np.ndarray
    hi: np.ndarray

    def __post_init__(self):
        lo, hi = np.broadcast_arrays(np.asarray(self.lo, dtype=np.float64), np.asarray(self.hi, dtype=np.float64))
        if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)) or np.any(lo > hi):
            raise ArithmeticUncertain("nonfinite or invalid interval endpoints")
        object.__setattr__(self, "lo", lo)
        object.__setattr__(self, "hi", hi)

    @classmethod
    def point(cls, value):
        a = np.asarray(value, dtype=np.float64)
        return cls(a, a)

    def __getitem__(self, key):
        return Interval(self.lo[key], self.hi[key])

    @classmethod
    def outward(cls, lo, hi):
        with np.errstate(all="ignore"):
            return cls(np.nextafter(lo, -np.inf), np.nextafter(hi, np.inf))

    def __add__(self, other):
        other = other if isinstance(other, Interval) else Interval.point(other)
        with np.errstate(all="ignore"):
            return Interval.outward(self.lo + other.lo, self.hi + other.hi)

    def __sub__(self, other):
        other = other if isinstance(other, Interval) else Interval.point(other)
        with np.errstate(all="ignore"):
            return Interval.outward(self.lo - other.hi, self.hi - other.lo)

    def __mul__(self, other):
        other = other if isinstance(other, Interval) else Interval.point(other)
        with np.errstate(all="ignore"):
            endpoints = np.broadcast_arrays(self.lo * other.lo, self.lo * other.hi,
                                             self.hi * other.lo, self.hi * other.hi)
            return Interval.outward(np.minimum.reduce(endpoints), np.maximum.reduce(endpoints))

    def scale_nonnegative(self, factor):
        """Multiply by a finite nonnegative point, with the SAME outward step.

        Monotonicity selects the two endpoints without allocating the four
        broadcast products of general interval multiplication. Zero and
        subnormal products still pass through outward(), never an identity
        shortcut. This is not multiplication by an uncertain interval.
        """
        factor = np.asarray(factor, dtype=np.float64)
        if not np.all(np.isfinite(factor)) or np.any(factor < 0):
            raise ArithmeticUncertain("scale requires a finite nonnegative point")
        with np.errstate(all="ignore"):
            return Interval.outward(self.lo*factor, self.hi*factor)

    def __truediv__(self, other):
        other = other if isinstance(other, Interval) else Interval.point(other)
        if np.any((other.lo <= 0) & (other.hi >= 0)):
            raise ArithmeticUncertain("division interval contains zero")
        with np.errstate(all="ignore"):
            endpoints = np.broadcast_arrays(self.lo / other.lo, self.lo / other.hi,
                                             self.hi / other.lo, self.hi / other.hi)
            return Interval.outward(np.minimum.reduce(endpoints), np.maximum.reduce(endpoints))


def stack(values, axis):
    return Interval(np.stack([v.lo for v in values], axis=axis), np.stack([v.hi for v in values], axis=axis))


def take(value, index, axis):
    return Interval(np.take(value.lo, index, axis=axis), np.take(value.hi, index, axis=axis))


def lagrange_to_bernstein(values):
    """Tensor transform of degree <= (2,2,2), last axes are reference z/y/x."""
    result = values
    if result.lo.shape[-3:] != (3, 3, 3):
        raise ValueError("Bernstein values require three length-3 trailing axes")
    for axis in (-1, -2, -3):
        a, b, c = (take(result, j, axis) for j in range(3))
        result = stack([a, b*2 - (a+c)*.5, c], axis)
    return result


def split_bernstein(coefficients, axis):
    """Midpoint de Casteljau along a reference xyz axis (0/1/2)."""
    if axis not in (0, 1, 2):
        raise ValueError("split axis must be 0,1,2")
    storage_axis = -1-axis
    a, b, c = (take(coefficients, j, storage_axis) for j in range(3))
    ab, bc = (a+b)*.5, (b+c)*.5
    mid = (ab+bc)*.5
    return stack([a, ab, mid], storage_axis), stack([mid, bc, c], storage_axis)
