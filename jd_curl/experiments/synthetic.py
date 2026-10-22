"""Task contract section 6: deterministic discrete manufactured registration."""

import numpy as np

from ..differences import make_difference_operators
from ..geometry import compute_geometry
from ..grid import make_grid_2d
from ..interpolation import build_image_sampler, sample_value_gradient
from ..types import Controls2D, InvalidTrial, RegistrationCase2D
from ..validation import InputValidationError, grid_size


def make_synthetic_case(*, nx: int = 33, ny: int = 29, identity: bool = False) -> RegistrationCase2D:
    nx, ny = grid_size(nx, "nx"), grid_size(ny, "ny")
    if not isinstance(identity, bool):
        raise InputValidationError("identity: expected bool")
    grid = make_grid_2d(nx=nx, ny=ny, hx=1/(nx-1), hy=1/(ny-1))
    x, y = grid.X, grid.Y
    moving = (np.exp(-((x-.35)**2+(y-.45)**2)/.03)
              +.7*np.exp(-((x-.72)**2+(y-.62)**2)/.02)
              +.1*np.sin(2*np.pi*x)*np.sin(3*np.pi*y))
    truth = np.zeros((ny, nx, 2))
    if not identity:
        truth[..., 0] = .02*np.sin(np.pi*x)*np.sin(np.pi*y)
        truth[..., 1] = -.015*np.sin(2*np.pi*x)*np.sin(np.pi*y)
        truth[[0, -1], :, :] = 0.
        truth[:, [0, -1], :] = 0.
    geometry = compute_geometry(truth, make_difference_operators(grid))
    sample = sample_value_gradient(build_image_sampler(moving, grid), np.stack((x, y), axis=-1)+truth)
    if isinstance(sample, InvalidTrial):
        raise InputValidationError("synthetic truth: "+sample.reason)
    kind = "identity" if identity else "manufactured"
    controls = Controls2D(np.zeros_like(truth), geometry.J, geometry.curl, f"synthetic-{kind}-{nx}x{ny}")
    metadata = dict(axis_order="yx", component_order="xy", mapping="fixed_to_moving",
        displacement_unit="unit_length", coordinate_unit="unit_length", data_kind="synthetic_discrete",
        preprocessing="Analytic moving samples; no normalization, resize, smoothing or noise.",
        provenance=dict(generator="jd_curl.experiments.synthetic.make_synthetic_case", version=1,
                        contract="03 section 6", variant=kind, seed=None,
                        fixed_image="moving.copy" if identity else "same cubic spline at identity+u_true",
                        controls="F06 discrete JD and raw curl of u_true"),
        truth_note="Manufactured displacement, generally not a minimizer with zero-reference proximity.")
    return RegistrationCase2D(grid, moving if identity else sample.value, moving, np.zeros_like(truth),
                               controls, metadata, truth)
