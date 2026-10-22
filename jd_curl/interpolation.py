"""One cubic spline for image values and physical derivatives; no extrapolation."""

from dataclasses import dataclass, field
import hashlib

import numpy as np
from scipy.interpolate import RectBivariateSpline

from .types import FloatArray, Grid2D, InvalidTrial, SampleResult
from .validation import InputValidationError, as_float64_array, readonly_copy


@dataclass(frozen=True)
class ImageSampler2D:
    """Factory-built sampler with private owned coefficients and frozen metadata.

    support is (xmin, xmax, ymin, ymax); identifiers hash float64 image/grid data
    at construction only. Do not hand-build inconsistent fields or mutate _spline.
    """

    _spline: RectBivariateSpline = field(repr=False)
    shape: tuple[int, int]
    support: tuple[float, float, float, float]
    image_id: str
    grid_id: str


def build_image_sampler(moving_image: FloatArray, grid: Grid2D) -> ImageSampler2D:
    shape = (grid.ny, grid.nx)
    image = as_float64_array(moving_image, "moving_image", shape)
    spline = RectBivariateSpline(grid.y, grid.x, image, kx=3, ky=3, s=0)
    if not np.all(np.isfinite(spline.get_coeffs())):
        raise InputValidationError("moving_image: spline coefficients are not finite float64")
    image_id = hashlib.sha256(str(shape).encode() + image.tobytes(order="C")).hexdigest()
    grid_id = hashlib.sha256(str(shape).encode() + grid.x.tobytes() + grid.y.tobytes()).hexdigest()
    support = (float(grid.x[0]), float(grid.x[-1]), float(grid.y[0]), float(grid.y[-1]))
    return ImageSampler2D(spline, shape, support, image_id, grid_id)


def sample_value_gradient(sampler: ImageSampler2D, phi: FloatArray) -> SampleResult | InvalidTrial:
    """Check finite coordinates and support BEFORE evaluating any spline output.

    Boundary *node indices* use the closed support, free nodes its strict interior.
    The caller separately enforces zero boundary displacement using inject_free.
    invalid_count counts nodes, not failing components. No input is altered.
    """
    phi = as_float64_array(phi, "phi", sampler.shape + (2,), check_finite=False)
    bad = ~np.all(np.isfinite(phi), axis=-1)
    if np.any(bad):
        return InvalidTrial("NONFINITE_STATE", "phi contains nonfinite coordinates", int(bad.sum()))
    x, y = phi[..., 0], phi[..., 1]
    xmin, xmax, ymin, ymax = sampler.support
    bad = (x < xmin) | (x > xmax) | (y < ymin) | (y > ymax)
    bad[1:-1, 1:-1] |= ((x <= xmin) | (x >= xmax) | (y <= ymin) | (y >= ymax))[1:-1, 1:-1]
    if np.any(bad):
        return InvalidTrial("INVALID_DOMAIN", "phi violates closed boundary / strict free-node support",
                            int(bad.sum()))
    value = sampler._spline.ev(y, x)
    gradient = np.stack((sampler._spline.ev(y, x, dx=0, dy=1),
                         sampler._spline.ev(y, x, dx=1, dy=0)), axis=-1)
    bad = ~np.isfinite(value) | ~np.all(np.isfinite(gradient), axis=-1)
    if np.any(bad):
        return InvalidTrial("NONFINITE_STATE", "spline value or gradient is nonfinite", int(bad.sum()))
    return SampleResult(readonly_copy(value), readonly_copy(gradient))
