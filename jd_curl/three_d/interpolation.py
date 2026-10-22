"""T07 one tensor cubic not-a-knot spline for values/physical xyz gradients."""
from dataclasses import dataclass, field
import hashlib

import numpy as np
from scipy.interpolate import make_interp_spline, NdBSpline

from ..types import InvalidTrial
from ..validation import InputValidationError, as_float64_array, readonly_copy, require_finite_result
from .grid import Grid3D
from .validation import integer_at_least
from .work_control import checkpoint


@dataclass(frozen=True)
class SampleResult3D:
    value: np.ndarray
    gradient: np.ndarray | None


@dataclass(frozen=True)
class ImageSampler3D:
    """Factory-built, owned coefficients; caller must not mutate private spline."""
    _spline: NdBSpline = field(repr=False)
    grid: Grid3D
    image_id: str


def build_image_sampler(image, grid):
    image = as_float64_array(image, "image", grid.shape)
    coefficients, knots = image.copy(), []
    for axis, coordinates in enumerate(grid.axes[::-1]):
        checkpoint("cpu_spline_build", axis=axis)
        spline = make_interp_spline(coordinates, coefficients, k=3, axis=axis, bc_type="not-a-knot")
        coefficients = np.moveaxis(spline.c, 0, axis)
        knots.append(readonly_copy(spline.t))
    require_finite_result(coefficients, "spline coefficients")
    spline = NdBSpline(tuple(knots), readonly_copy(coefficients), (3, 3, 3), extrapolate=False)
    digest = hashlib.sha256(grid.grid_id.encode() + image.tobytes(order="C")).hexdigest()
    return ImageSampler3D(spline, grid, digest)


def check_map_domain(u, grid):
    """Check id+u without allocating dense identity; boundary displacement is separate."""
    u = as_float64_array(u, "u", grid.shape + (3,), check_finite=False)
    from . import fast_checks
    if fast_checks.enabled(u.size):
        # E51 exact parallel scan: same per-node rule, same integer counts.
        nonfinite, invalid = fast_checks.domain_counts(u, grid.axes)
        if nonfinite:
            return InvalidTrial("NONFINITE_STATE", "u contains nonfinite coordinates", int(nonfinite))
        if invalid:
            return InvalidTrial("INVALID_DOMAIN", "id+u violates closed boundary / open free-node support", int(invalid))
        return None
    bad = ~np.all(np.isfinite(u), axis=-1)
    if np.any(bad):
        return InvalidTrial("NONFINITE_STATE", "u contains nonfinite coordinates", int(bad.sum()))
    bad = np.zeros(grid.shape, dtype=bool)
    with np.errstate(over="ignore", invalid="ignore"):
        for j, coordinates in enumerate(grid.axes):
            shape = [1, 1, 1]
            shape[2 - j] = len(coordinates)
            value = u[..., j] + coordinates.reshape(shape)
            bad |= ~np.isfinite(value) | (value < coordinates[0]) | (value > coordinates[-1])
            free = value[1:-1, 1:-1, 1:-1]
            bad[1:-1, 1:-1, 1:-1] |= (free <= coordinates[0]) | (free >= coordinates[-1])
    if np.any(bad):
        return InvalidTrial("INVALID_DOMAIN", "id+u violates closed boundary / open free-node support", int(bad.sum()))
    return None


def sample_points(sampler, points_xyz, *, strict=None, with_gradient=True, chunk_size=65536):
    """Arbitrary batches; strict mask marks queries required to be in open support."""
    points = as_float64_array(points_xyz, "points_xyz", check_finite=False)
    if points.ndim < 2 or points.shape[-1] != 3:
        raise InputValidationError("points_xyz: expected (...,3) with a query axis")
    chunk_size = integer_at_least(chunk_size, "chunk_size")
    if not isinstance(with_gradient, bool):
        raise InputValidationError("with_gradient: expected bool")
    if strict is None:
        strict = np.zeros(points.shape[:-1], dtype=bool)
    if not isinstance(strict, np.ndarray) or strict.dtype != bool or strict.shape != points.shape[:-1]:
        raise InputValidationError("strict: boolean mask with query shape required")
    bad = ~np.all(np.isfinite(points), axis=-1)
    if np.any(bad):
        return InvalidTrial("NONFINITE_STATE", "nonfinite sampling coordinates", int(bad.sum()))
    for j, axis in enumerate(sampler.grid.axes):
        x = points[..., j]
        bad |= (x < axis[0]) | (x > axis[-1]) | (strict & ((x <= axis[0]) | (x >= axis[-1])))
    if np.any(bad):
        return InvalidTrial("INVALID_DOMAIN", "sampling support violated before evaluation", int(bad.sum()))
    queries = points.reshape(-1, 3)
    value = np.empty(len(queries))
    gradient = np.empty((len(queries), 3)) if with_gradient else None
    for start in range(0, len(queries), chunk_size):
        checkpoint("cpu_spline_sample", completed=start, total=len(queries))
        stop = min(start + chunk_size, len(queries))
        q = queries[start:stop, ::-1]  # NdBSpline axes zyx, gradient components xyz
        value[start:stop] = sampler._spline(q)
        if with_gradient:
            for j, nu in enumerate(((0, 0, 1), (0, 1, 0), (1, 0, 0))):
                gradient[start:stop, j] = sampler._spline(q, nu=nu)
    if not np.all(np.isfinite(value)) or (with_gradient and not np.all(np.isfinite(gradient))):
        return InvalidTrial("NONFINITE_STATE", "nonfinite spline output", len(queries))
    return SampleResult3D(readonly_copy(value.reshape(points.shape[:-1])),
                          readonly_copy(gradient.reshape(points.shape)) if with_gradient else None)


def sample_value_gradient(sampler, phi, *, chunk_size=65536):
    phi = as_float64_array(phi, "phi", sampler.grid.shape + (3,), check_finite=False)
    strict = np.zeros(sampler.grid.shape, dtype=bool)
    strict[1:-1, 1:-1, 1:-1] = True
    return sample_points(sampler, phi, strict=strict, chunk_size=chunk_size)
