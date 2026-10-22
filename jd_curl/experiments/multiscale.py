"""F21: physical-grid image pyramid and audited feasible Q1 initialization.

Only zero-reference constant-control cases are supported. No solver or energy
is duplicated here, and the finest images are never filtered or resampled.
"""
from dataclasses import replace
from numbers import Integral

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

from ..boundary import pack_initial_displacement
from ..geometry import bilinear_cell_jacobians
from ..grid import make_grid_2d
from ..jacobian_barrier import feasible_step_bound
from ..linearization import energy_gradient
from ..residuals import evaluate_state
from ..solvers.common import gradient_norm_mass
from ..types import Controls2D, EvalState, RegistrationCase2D
from ..validation import InputValidationError, as_float64_array, require_zero_boundary


def resample_q1(values, source, target):
    """Physical tensor-product linear interpolation; bounds errors are explicit."""
    values = as_float64_array(values, 'transfer.values')
    if values.shape[:2] != (source.ny, source.nx):
        raise InputValidationError('transfer.values: source grid shape mismatch')
    sample = RegularGridInterpolator((source.y, source.x), values, method='linear', bounds_error=True)
    return sample(np.stack((target.Y, target.X), axis=-1))


def pyramid_case(case: RegistrationCase2D, factor: int) -> RegistrationCase2D:
    if isinstance(factor, (bool, np.bool_)) or not isinstance(factor, Integral) or factor < 1:
        raise InputValidationError('factor: require positive integer')
    if (np.any(case.u0) or np.any(case.controls.u_ref) or np.any(case.controls.f != 1.)
            or np.any(case.controls.g) or case.u_true is not None):
        raise InputValidationError('pyramid requires zero u0/u_ref, f=1, g=0 and no oracle displacement')
    if factor == 1:
        return case
    native = case.grid
    nx, ny = ((n-2+factor)//factor+1 for n in (native.nx, native.ny))
    if min(nx, ny) < 5:
        raise InputValidationError('pyramid level requires at least 5 nodes on each axis')
    grid = make_grid_2d(nx=nx, ny=ny, hx=(native.x[-1]-native.x[0])/(nx-1),
                       hy=(native.y[-1]-native.y[0])/(ny-1), x0=native.x[0], y0=native.y[0])
    sigma = tuple(.5*np.sqrt((h/h0)**2-1.) for h, h0 in ((grid.hy, native.hy), (grid.hx, native.hx)))
    images = [resample_q1(gaussian_filter(a, sigma=sigma, mode='reflect', truncate=4.), native, grid)
              for a in (case.fixed_image, case.moving_image)]
    zero = np.zeros((ny, nx, 2))
    controls = Controls2D(zero, np.ones((ny, nx)), np.zeros((ny, nx)), case.controls.controls_id+f':scale{factor}')
    metadata = dict(case.metadata, preprocessing='Coarse image only: native Gaussian reflect/truncate=4 then physical Q1 sampling.',
                    multiscale=dict(factor=int(factor), native_shape=[native.ny, native.nx], sigma_native_yx=list(sigma),
                                    native_preprocessing=case.metadata.get('preprocessing')))
    return RegistrationCase2D(grid, *images, zero, controls, metadata)


def transfer_initialization(u, source, problem):
    """Return feasible physical displacement and diagnostics; never silently reset.

    The repair is allowed only for initialization, not for optimization trials.
    All evaluate_state calls here are extra setup costs, not solver iterations.
    """
    u = as_float64_array(u, 'transfer.u', (source.ny, source.nx, 2))
    require_zero_boundary(u, 'transfer.u')
    grid = problem.grid
    for a, b in ((source.x, grid.x), (source.y, grid.y)):
        if not np.allclose(a[[0, -1]], b[[0, -1]], rtol=0., atol=1e-12):
            raise InputValidationError('transfer: physical domains must match')
    if problem.jacobian_barrier is None:
        raise InputValidationError('transfer requires explicit Q1 Jacobian barrier')
    v = resample_q1(u, source, grid)
    v[[0, -1], :, :] = 0.
    v[:, [0, -1], :] = 0.
    z = pack_initial_displacement(v, grid)
    trial = evaluate_state(problem, z, eval_id=0)
    record = dict(alpha=1., raw_min_corner_jd=float(bilinear_cell_jacobians(v, grid).min()),
                  raw_rejection=None, state_evaluations=1, feasible_bound_evaluations=0)
    if not isinstance(trial, EvalState):
        record['raw_rejection'] = trial.reason
        base = evaluate_state(problem, np.zeros_like(z), eval_id=1)
        if not isinstance(base, EvalState):
            raise RuntimeError('Identity initialization unexpectedly infeasible')
        bound = feasible_step_bound(problem, base, z)
        alpha = min(1., problem.jacobian_barrier.fraction_to_boundary*bound)
        record.update(alpha=alpha, state_evaluations=3, feasible_bound_evaluations=1)
        if not np.isfinite(alpha) or alpha <= 0:
            raise RuntimeError(f'No feasible positive initialization scale: {record}')
        trial = evaluate_state(problem, alpha*z, eval_id=2)
        if not isinstance(trial, EvalState):
            raise RuntimeError(f'Transferred initialization remains invalid: {trial}; {record}')
    record['min_corner_jd'] = float(trial.corner_geometry.J.min())
    return trial.u, record


def anchored_config(problem, config, max_iterations):
    """Fix F14 tolerance to this level's zero-state gradient, not its warm start."""
    base = evaluate_state(problem, np.zeros(2*problem.grid.n_free_scalar), eval_id=0)
    if not isinstance(base, EvalState):
        raise InputValidationError('zero gradient anchor is not feasible')
    G0 = gradient_norm_mass(energy_gradient(problem, base), problem.grid.omega)
    threshold = config.stop.grad_atol+config.stop.grad_rtol*G0
    resolved = replace(config, stop=replace(config.stop, max_iterations=max_iterations,
                                           grad_atol=threshold, grad_rtol=0.))
    return resolved, dict(zero_G=G0, threshold=threshold, state_evaluations=1, gradient_evaluations=1)
