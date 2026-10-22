"""G's two preregistered cases: native real pair and oracle-control manufactured MRI."""

import numpy as np

from ..differences import make_difference_operators
from ..geometry import compute_geometry
from ..interpolation import build_image_sampler, sample_value_gradient
from ..nifti import NiftiSlice2D, require_same_slice_grid
from ..types import Controls2D, InvalidTrial, RegistrationCase2D
from ..validation import InputValidationError


def make_mri_case(*, moving: NiftiSlice2D, fixed: NiftiSlice2D | None = None) -> RegistrationCase2D:
    """fixed=None selects the declared manufactured case, never an inferred real target."""
    grid = moving.grid
    zero = np.zeros((grid.ny, grid.nx, 2))
    Lx, Ly = grid.x[-1], grid.y[-1]
    if fixed is not None:
        require_same_slice_grid(fixed, moving)
        truth = None
        fixed_image = fixed.image
        f, g = np.ones_like(moving.image), np.zeros_like(moving.image)
        kind = "mri_pair"
        control_source = "Fixed identity-geometry prior f=1,g=0; not measured anatomical JD/curl."
    else:
        truth = np.zeros_like(zero)
        x, y = grid.X/Lx, grid.Y/Ly
        truth[..., 0] = .02*Lx*np.sin(np.pi*x)*np.sin(np.pi*y)
        truth[..., 1] = -.015*Ly*np.sin(2*np.pi*x)*np.sin(np.pi*y)
        truth[[0, -1], :, :] = 0.
        truth[:, [0, -1], :] = 0.
        geometry = compute_geometry(truth, make_difference_operators(grid))
        phi = np.stack((grid.X, grid.Y), axis=-1)+truth
        sample = sample_value_gradient(build_image_sampler(moving.image, grid), phi)
        if isinstance(sample, InvalidTrial):
            raise InputValidationError("MRI manufactured truth: "+sample.reason)
        fixed_image, f, g = sample.value, geometry.J, geometry.curl
        kind = "mri_known_warp"
        control_source = "Oracle f/g: F06 discrete JD/raw curl of u_true; not an image-only accuracy benchmark."
    controls = Controls2D(zero, f, g, f"{kind}-{grid.nx}x{grid.ny}")
    metadata = dict(axis_order="yx", component_order="xy", mapping="fixed_to_moving",
        displacement_unit="mm", coordinate_unit="mm", data_kind=kind,
        preprocessing="Native transpose into (y,x); NIfTI stored scaling only; no reorientation, resize, normalization, clipping or smoothing.",
        provenance=dict(generator="jd_curl.experiments.mri.make_mri_case", version=1, seed=None,
            moving=moving.metadata, fixed=None if fixed is None else fixed.metadata,
            fixed_construction="same cubic moving spline at identity+u_true" if truth is not None else "source NIfTI samples",
            controls=control_source, prior="u0=u_ref=0, controls fixed throughout solve"),
        length_scale=dict(Lx_mm=Lx, Ly_mm=Ly, ell_mm=float(np.sqrt(Lx*Ly)),
                          lambda_reference=.1/(Lx*Ly), mu_reference=1., nu_reference=1.,
                          rationale="lambda=.1/ell^2; explicit mm objective; algorithm defaults not rescaled"),
        truth_note="Oracle manufactured displacement, not necessarily full-objective minimizer." if truth is not None
                   else "Unknown anatomical displacement; no ground-truth displacement accuracy claim.")
    return RegistrationCase2D(grid, fixed_image, moving.image, zero, controls, metadata, truth)
