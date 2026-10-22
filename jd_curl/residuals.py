"""F07: one full-node weighted residual and energy, with isolated trial states."""

from numbers import Integral
from dataclasses import replace
from uuid import uuid4

import numpy as np

from .boundary import inject_free
from .cell_q1 import CellQ1Operators, compute_cell_geometry
from .differences import make_difference_operators
from .geometry import compute_geometry
from .interpolation import build_image_sampler, sample_value_gradient
from .jacobian_barrier import (JacobianBarrierConfig, JacobianFeasibilityConfig, barrier_residual,
                               validate_barrier, validate_feasibility, feasibility_config)
from .types import (
    RESIDUAL_NAMES, Controls2D, EnergyTerms, EvalState, FloatArray, GeometryState,
    Grid2D, InvalidTrial, Problem2D, Q1Controls2D, RegistrationCase2D, ResidualBlocks, Weights,
)
from .validation import (InputValidationError, as_float64_array, readonly_copy, require_finite_result,
                         validate_geometry_discretization)


def build_problem_2d(
    *, grid: Grid2D, weights: Weights, controls: Controls2D | Q1Controls2D,
    fixed_image: FloatArray, moving_image: FloatArray,
    geometry_discretization: str = "nodal",
    jacobian_barrier: JacobianBarrierConfig | None = None,
    jacobian_feasibility: JacobianFeasibilityConfig | None = None,
) -> Problem2D:
    """Construct matched operators and owned inputs; no initial state or optimizer."""
    shape = (grid.ny, grid.nx)
    validate_geometry_discretization(geometry_discretization)
    validate_barrier(jacobian_barrier, geometry_discretization)
    validate_feasibility(jacobian_feasibility, jacobian_barrier, geometry_discretization)
    fixed = as_float64_array(fixed_image, "fixed_image", shape)
    if not isinstance(weights, Weights) or not isinstance(controls, (Controls2D, Q1Controls2D)):
        raise InputValidationError("weights/controls: expected Weights and nodal or Q1 controls")
    as_float64_array(controls.u_ref, "controls.u_ref", shape+(2,))
    if isinstance(controls, Q1Controls2D):
        if geometry_discretization != "q1":
            raise InputValidationError("Gauss controls require q1 geometry")
        owned_controls = Q1Controls2D(controls.u_ref, controls.f_gauss, controls.g_gauss, controls.controls_id)
    else:
        as_float64_array(controls.f, "controls.f", shape)
        owned_controls = Controls2D(controls.u_ref, controls.f, controls.g, controls.controls_id)
    sampler = build_image_sampler(moving_image, grid)
    cells = CellQ1Operators(grid) if geometry_discretization == "q1" else None
    return Problem2D(grid, weights, owned_controls, readonly_copy(fixed), sampler,
                     make_difference_operators(grid), uuid4().hex, geometry_discretization,
                     cells, (owned_controls.f_gauss if isinstance(owned_controls, Q1Controls2D)
                             else readonly_copy(cells.interpolate(owned_controls.f)) if cells else None),
                     (owned_controls.g_gauss if isinstance(owned_controls, Q1Controls2D)
                      else readonly_copy(cells.interpolate(owned_controls.g)) if cells else None),
                     jacobian_barrier, CellQ1Operators(grid, location="corners")
                     if jacobian_barrier or jacobian_feasibility else None, jacobian_feasibility)


def build_problem(case: RegistrationCase2D, weights: Weights, *, geometry_discretization: str = "nodal",
                  jacobian_barrier: JacobianBarrierConfig | None = None,
                  jacobian_feasibility: JacobianFeasibilityConfig | None = None) -> Problem2D:
    """Case adapter; the same C model is used for synthetic and later real data."""
    return build_problem_2d(grid=case.grid, weights=weights, controls=case.controls,
                            fixed_image=case.fixed_image, moving_image=case.moving_image,
                            geometry_discretization=geometry_discretization, jacobian_barrier=jacobian_barrier,
                            jacobian_feasibility=jacobian_feasibility)


def with_q1_self_reference(problem: Problem2D, base: EvalState, *, include_barrier: bool = False) -> Problem2D:
    """F22a: fresh owned references and IDs; reuse original spline/operators.

    Only call after the previous candidate has been accepted. Existing states
    remain tied to the old problem and cannot be used with the returned one.
    """
    from .linearization import _check_state
    _check_state(problem, base)
    if not isinstance(include_barrier, bool):
        raise InputValidationError("include_barrier: expected bool")
    if problem.geometry_discretization != "q1" or (problem.jacobian_barrier is not None and not include_barrier):
        raise InputValidationError("self-reference requires Q1 without absolute barrier")
    controls = Q1Controls2D(base.u, base.cell_geometry.J, base.cell_geometry.curl, uuid4().hex)
    return replace(problem, controls=controls, cell_f=controls.f_gauss,
                   cell_g=controls.g_gauss, problem_id=uuid4().hex)


def _residual_scales(problem: Problem2D) -> tuple[FloatArray, ...]:
    """Same F07 square-root factors for residual, JVP, and VJP; no extra omega."""
    q = problem.grid.sqrt_q
    weights = problem.weights
    proximity = np.sqrt(weights.lam) * q
    geometry_q = np.sqrt(problem.grid.omega/4) if problem.geometry_discretization == "q1" else q
    return q, proximity, proximity, np.sqrt(weights.mu)*geometry_q, np.sqrt(weights.nu)*geometry_q


def _weighted_blocks(problem: Problem2D, *fields: FloatArray, barrier=None) -> ResidualBlocks:
    with np.errstate(over="raise", invalid="raise"):
        weighted = [scale*value for scale, value in zip(_residual_scales(problem), fields)]
    for value in weighted:
        require_finite_result(value, "weighted residual")
    return ResidualBlocks(*weighted, barrier=barrier)


def make_residuals(
    problem: Problem2D, u: FloatArray, geometry: GeometryState, warped_image: FloatArray,
    *, cell_geometry=None, barrier=None,
) -> ResidualBlocks:
    shape = (problem.grid.ny, problem.grid.nx)
    u = as_float64_array(u, "u", shape + (2,))
    warped = as_float64_array(warped_image, "warped_image", shape)
    J = as_float64_array(geometry.J, "geometry.J", shape)
    curl = as_float64_array(geometry.curl, "geometry.curl", shape)
    controls = problem.controls
    if (barrier is None) != (problem.jacobian_barrier is None):
        raise InputValidationError("barrier residual must match problem configuration")
    if problem.geometry_discretization == "q1":
        if cell_geometry is None:
            raise InputValidationError("Q1 residuals require cell geometry")
        J, curl, f, g = cell_geometry.J, cell_geometry.curl, problem.cell_f, problem.cell_g
    else:
        f, g = controls.f, controls.g
    with np.errstate(over="raise", invalid="raise"):
        d = u - controls.u_ref
        return _weighted_blocks(problem, warped-problem.fixed_image,
                                d[..., 0], d[..., 1], J-f, curl-g, barrier=barrier)


def pack_residuals(blocks: ResidualBlocks) -> FloatArray:
    arrays = [getattr(blocks, name).ravel(order="C") for name in RESIDUAL_NAMES]
    if blocks.barrier is not None:
        arrays.append(blocks.barrier.ravel(order="C"))
    return np.concatenate(arrays)


def residual_size(grid: Grid2D, geometry_discretization: str = "nodal", *, with_barrier: bool = False) -> int:
    validate_geometry_discretization(geometry_discretization)
    if with_barrier and geometry_discretization != "q1":
        raise InputValidationError("barrier residual requires Q1")
    n = grid.ny*grid.nx
    return 3*n+(12 if with_barrier else 8)*(grid.ny-1)*(grid.nx-1) if geometry_discretization == "q1" else 5*n


def unpack_residuals(w: FloatArray, grid: Grid2D, geometry_discretization: str = "nodal",
                     *, with_barrier: bool = False) -> ResidualBlocks:
    w = as_float64_array(w, "residual vector", (residual_size(grid, geometry_discretization, with_barrier=with_barrier),))
    if geometry_discretization == "nodal":
        return ResidualBlocks(*w.reshape(5, grid.ny, grid.nx))
    n = grid.ny*grid.nx
    nodes = w[:3*n].reshape(3, grid.ny, grid.nx)
    cells = w[3*n:].reshape(3 if with_barrier else 2, grid.ny-1, grid.nx-1, 4)
    return ResidualBlocks(*nodes, *cells[:2], barrier=cells[2] if with_barrier else None)


def energy_from_residuals(blocks: ResidualBlocks) -> EnergyTerms:
    with np.errstate(over="raise", invalid="raise"):
        terms = [float(.5*np.dot(a, a)) for a in
                 (getattr(blocks, name).ravel() for name in RESIDUAL_NAMES)]
        image, px, py, jd, curl = terms
        proximity = px + py
        total = image + proximity + jd + curl
        barrier = 0. if blocks.barrier is None else float(.5*np.dot(blocks.barrier.ravel(), blocks.barrier.ravel()))
        total += barrier
    require_finite_result(terms + [proximity, barrier, total], "energy")
    return EnergyTerms(image, proximity, jd, curl, total, barrier)


def evaluate_state(problem: Problem2D, z: FloatArray, *, eval_id: int) -> EvalState | InvalidTrial:
    """Evaluate the same model for base and trial; never mutate a previous state.

    invalid_count=0 denotes unlocalized arithmetic overflow, not a valid trial.
    Only numerical FloatingPointError is translated, never programming errors.
    """
    if isinstance(eval_id, (bool, np.bool_)) or not isinstance(eval_id, Integral) or eval_id < 0:
        raise InputValidationError("eval_id: expected a nonnegative integer")
    z = as_float64_array(z, "z", (2*problem.grid.n_free_scalar,), check_finite=False)
    bad = ~np.all(np.isfinite(z.reshape(2, -1)), axis=0)
    if np.any(bad):
        return InvalidTrial("NONFINITE_STATE", "z contains nonfinite coordinates", int(bad.sum()))
    u = inject_free(z, problem.grid)
    try:
        with np.errstate(over="raise", invalid="raise"):
            phi = np.stack((problem.grid.X, problem.grid.Y), axis=-1) + u
            sample = sample_value_gradient(problem.sampler, phi)
        if isinstance(sample, InvalidTrial):
            return sample
        geometry = compute_geometry(u, problem.diff)
        cell_geometry = compute_cell_geometry(u, problem.cell_operators) if problem.cell_operators else None
        corners, barrier, factor = None, None, None
        feasibility = feasibility_config(problem)
        if feasibility is not None:
            corners = compute_cell_geometry(u, problem.corner_operators)
            bad = corners.J <= feasibility.min_jd
            if np.any(bad):
                return InvalidTrial("JACOBIAN_BOUND_VIOLATION", "Q1 corner JD must exceed min_jd", int(bad.sum()))
            if problem.jacobian_barrier is not None:
                barrier, factor = barrier_residual(corners.J, problem.jacobian_barrier, problem.grid.omega)
        blocks = make_residuals(problem, u, geometry, sample.value, cell_geometry=cell_geometry, barrier=barrier)
    except FloatingPointError as exc:
        return InvalidTrial("NONFINITE_STATE", str(exc), 0)
    try:
        energy = energy_from_residuals(blocks)
    except FloatingPointError as exc:
        return InvalidTrial("NONFINITE_ENERGY", str(exc), 0)
    return EvalState(problem.problem_id, problem.controls.controls_id, int(eval_id),
                     readonly_copy(z), readonly_copy(u), readonly_copy(phi), geometry,
                     sample.value, sample.gradient, blocks, energy, cell_geometry, corners,
                     readonly_copy(factor) if factor is not None else None)
