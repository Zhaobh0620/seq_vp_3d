"""F's fixed-base frequency/refinement probes; not nonlinear multiscale registration."""

from dataclasses import asdict
from time import perf_counter

import numpy as np

from .boundary import restrict_free
from .grid import make_grid_2d
from .linearization import linearize
from .poisson import build_screened_poisson_inverse
from .residuals import build_problem_2d, evaluate_state
from .solvers.common import LMConfig, gradient_norm_mass
from .solvers.lm import make_lm_operator, solve_lm_direction
from .types import Controls2D, EvalState, Weights
from .validation import InputValidationError, grid_size


def high_frequency_diagnostics(sizes: tuple[int, ...] = (17, 33, 65, 129)) -> list[dict]:
    config, weights, gamma = LMConfig(), Weights(.1, 1., 1.), .001
    rows = []
    for n in sizes:
        n = grid_size(n, "n")
        grid = make_grid_2d(nx=n, ny=n, hx=1/(n-1), hy=1/(n-1))
        zero = np.zeros((n, n))
        problem = build_problem_2d(grid=grid, weights=weights,
            controls=Controls2D(np.zeros((n, n, 2)), np.ones_like(zero), zero, f"frequency-{n}"),
            fixed_image=zero, moving_image=zero)
        base = evaluate_state(problem, np.zeros(2*grid.n_free_scalar), eval_id=0)
        if not isinstance(base, EvalState):
            raise InputValidationError("diagnostic identity state: "+base.reason)
        lin = linearize(problem, base)
        L_mass = build_screened_poisson_inverse(grid, a=0., b=1.)
        A = make_lm_operator(lin, omega=grid.omega, gamma=gamma)
        envelope = np.sin(np.pi*grid.X)*np.sin(np.pi*grid.Y)
        for mode in ("low", "high"):
            v = np.zeros((n, n, 2))
            v[..., 0] = envelope
            if mode == "high":
                v[..., 0] *= (-1.)**np.indices((n, n)).sum(axis=0)
            v[[0, -1], :, :] = 0.
            v[:, [0, -1], :] = 0.
            vector = restrict_free(v, grid)
            blocks = lin.jvp(vector).reshape(5, n, n)
            geometry_quadratic = float(np.sum(blocks[3:]**2))
            poisson_quadratic = float(vector@L_mass.matvec(vector))
            gradient = -(A@vector)  # p=vector is the exact solution; not an image gradient.
            start = perf_counter()
            direction = solve_lm_direction(lin, gradient, gamma=gamma, config=config)
            elapsed = perf_counter()-start
            error = (gradient_norm_mass(direction.p-vector, 1.)/gradient_norm_mass(vector, 1.)
                     if direction.p is not None and np.all(np.isfinite(direction.p)) else None)
            rows.append(dict(n=n, hx=grid.hx, hy=grid.hy, mode=mode, weights=asdict(weights), gamma=gamma,
                geometry_quadratic=geometry_quadratic, poisson_quadratic=poisson_quadratic,
                ratio=geometry_quadratic/poisson_quadratic, linear_valid=direction.valid,
                linear_reason=direction.reason, cg_info=direction.info, cg_iterations=direction.iterations,
                true_relative_residual=direction.relative_residual, solution_relative_error=error,
                n_hvp=direction.n_hvp, n_preconditioner_applications=direction.n_preconditioner_applications,
                rhs_generation_hvp=1, wall_seconds=elapsed, cg_rtol=config.cg_rtol,
                cg_atol=config.cg_atol, cg_maxiter=config.cg_maxiter))
    return rows
