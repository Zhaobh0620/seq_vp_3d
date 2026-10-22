"""F08--F10 Euclidean JVP/VJP, gradient and GN; not the exact Newton Hessian."""

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from scipy.sparse.linalg import LinearOperator

from .boundary import inject_free, restrict_free
from .residuals import _residual_scales, _weighted_blocks, pack_residuals, unpack_residuals, residual_size
from .types import RESIDUAL_NAMES, EvalState, FloatArray, Problem2D, ResidualBlocks
from .validation import InputValidationError, StateMismatchError, require_finite_result


def _check_state(problem: Problem2D, base: EvalState) -> None:
    if not isinstance(base, EvalState):
        raise InputValidationError("base: expected a valid EvalState")
    if base.problem_id != problem.problem_id or base.controls_id != problem.controls.controls_id:
        raise StateMismatchError("base: problem_id or controls_id does not match")


def _geometry_operators(problem: Problem2D, base: EvalState):
    if problem.geometry_discretization == "q1":
        return base.cell_geometry.A, problem.cell_operators
    return base.geometry.A, problem.diff


def residual_jvp(problem: Problem2D, base: EvalState, v: FloatArray) -> FloatArray:
    return pack_residuals(_residual_jvp_blocks(problem, base, v))


def _residual_jvp_blocks(problem: Problem2D, base: EvalState, v: FloatArray) -> ResidualBlocks:
    """Internal block action; avoid packing only to immediately unpack in K.T."""
    _check_state(problem, base)
    full = inject_free(v, problem.grid)
    A, diff = _geometry_operators(problem, base)
    b = base.image_grad
    with np.errstate(over="raise", invalid="raise"):
        vx_x, vx_y = diff.dx(full[..., 0]), diff.dy(full[..., 0])
        vy_x, vy_y = diff.dx(full[..., 1]), diff.dy(full[..., 1])
        image = np.sum(b*full, axis=-1)
        jd = A[..., 0, 0]*vx_x + A[..., 0, 1]*vx_y + A[..., 1, 0]*vy_x + A[..., 1, 1]*vy_y
        barrier = None
        if problem.jacobian_barrier is not None:
            C, corners = base.corner_geometry.A, problem.corner_operators
            dJ = (C[..., 0, 0]*corners.dx(full[..., 0])+C[..., 0, 1]*corners.dy(full[..., 0])
                  +C[..., 1, 0]*corners.dx(full[..., 1])+C[..., 1, 1]*corners.dy(full[..., 1]))
            barrier = base.barrier_jacobian_scale*dJ
            require_finite_result(barrier, "barrier JVP")
        blocks = _weighted_blocks(problem, image, full[..., 0], full[..., 1], jd, vy_x-vx_y, barrier=barrier)
    return blocks


def residual_vjp(problem: Problem2D, base: EvalState, w: FloatArray) -> FloatArray:
    _check_state(problem, base)
    blocks = unpack_residuals(w, problem.grid, problem.geometry_discretization,
                              with_barrier=problem.jacobian_barrier is not None)
    return _residual_vjp_blocks(problem, base, blocks)


def _residual_vjp_blocks(problem: Problem2D, base: EvalState, blocks: ResidualBlocks) -> FloatArray:
    """Only for validated, owned blocks from this problem's internal actions."""
    _check_state(problem, base)
    A, diff = _geometry_operators(problem, base)
    b = base.image_grad
    with np.errstate(over="raise", invalid="raise"):
        ti, tx, ty, tj, tc = [scale*getattr(blocks, name) for scale, name in
                             zip(_residual_scales(problem), RESIDUAL_NAMES)]
        # Check before passing arithmetic results to input-validating differences.
        fluxes = [A[..., 0, 0]*tj, A[..., 0, 1]*tj, A[..., 1, 0]*tj, A[..., 1, 1]*tj, tc]
        for flux in fluxes:
            require_finite_result(flux, "VJP flux")
        x = b[..., 0]*ti + tx + diff.dx_T(fluxes[0]) + diff.dy_T(fluxes[1]) - diff.dy_T(tc)
        y = b[..., 1]*ti + ty + diff.dx_T(fluxes[2]) + diff.dy_T(fluxes[3]) + diff.dx_T(tc)
        if problem.jacobian_barrier is not None:
            C, corners = base.corner_geometry.A, problem.corner_operators
            tb = base.barrier_jacobian_scale*blocks.barrier
            flux = C*tb[..., None, None]
            require_finite_result(flux, "barrier VJP flux")
            x += corners.dx_T(flux[..., 0, 0])+corners.dy_T(flux[..., 0, 1])
            y += corners.dx_T(flux[..., 1, 0])+corners.dy_T(flux[..., 1, 1])
        full = np.stack((x, y), axis=-1)
    require_finite_result(full, "VJP")
    return restrict_free(full, problem.grid)


def energy_gradient(problem: Problem2D, base: EvalState) -> FloatArray:
    _check_state(problem, base)
    return _residual_vjp_blocks(problem, base, base.residuals)


def image_gradient(problem: Problem2D, base: EvalState) -> FloatArray:
    """F22: Euclidean gradient of D alone, using the original image spline."""
    _check_state(problem, base)
    with np.errstate(over="raise", invalid="raise"):
        # Match the image block's sqrt(q) twice, including its rounding path.
        full = (problem.grid.sqrt_q*base.residuals.image)[..., None]*base.image_grad
    require_finite_result(full, "image gradient")
    return restrict_free(full, problem.grid)


def image_barrier_gradient(problem: Problem2D, base: EvalState) -> FloatArray:
    """F23: gradient of D+B only; exclude all three changing reference terms.

    Reuse the exact VJP, not subtraction of two potentially cancelling full
    gradients. The barrier is absolute and is never reset with the reference.
    """
    _check_state(problem, base)
    if problem.jacobian_barrier is None:
        raise InputValidationError("image_barrier_gradient requires an absolute barrier")
    r = base.residuals
    zero = np.zeros_like(r.image)
    geometry_zero = np.zeros_like(r.jd)
    selected = ResidualBlocks(r.image, zero, zero, geometry_zero, geometry_zero, barrier=r.barrier)
    return _residual_vjp_blocks(problem, base, selected)


def gn_hvp(problem: Problem2D, base: EvalState, v: FloatArray) -> FloatArray:
    return _residual_vjp_blocks(problem, base, _residual_jvp_blocks(problem, base, v))


@dataclass(frozen=True)
class Linearization:
    """A fixed problem/base pair; all actions reuse the same coefficients."""

    base: EvalState
    shape: tuple[int, int]
    _problem: Problem2D = field(repr=False)

    @cached_property
    def _barrier_ggn_factor(self) -> FloatArray:
        """F26 coefficient, owned by this frozen base, not by an LM retry."""
        _check_state(self._problem, self.base)
        cfg = self._problem.jacobian_barrier
        if cfg is None or self._problem.geometry_discretization != "q1":
            raise InputValidationError("GGN model requires q1 and an absolute barrier")
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            factor = (np.sqrt(cfg.weight*self._problem.grid.omega/4)
                      /(self.base.corner_geometry.J-cfg.min_jd)
                      /self.base.barrier_jacobian_scale)**2
        require_finite_result(factor, "barrier GGN factor")
        factor.flags.writeable = False
        return factor

    @cached_property
    def _barrier_ggn_diagonal(self) -> FloatArray:
        """F29 exact diagonal of H_C (not the full Newton Hessian)."""
        _check_state(self._problem, self.base)
        p, base = self._problem, self.base
        cfg = p.jacobian_barrier
        if cfg is None or p.geometry_discretization != "q1":
            raise InputValidationError("GGN diagonal requires q1 and an absolute barrier")
        w, omega = p.weights, p.grid.omega
        A, C = base.cell_geometry.A, base.corner_geometry.A
        gauss, corners = p.cell_operators, p.corner_operators
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            diagonal = p.grid.q[..., None]*(base.image_grad**2+w.lam)
            barrier_weight = (cfg.weight*omega/4)/(base.corner_geometry.J-cfg.min_jd)**2
            for component in (0, 1):
                diagonal[..., component] += gauss.squared_derivative_diagonal(
                    A[..., component, 0], A[..., component, 1], w.mu*omega/4)
                diagonal[..., component] += gauss.squared_derivative_diagonal(
                    component, component-1, w.nu*omega/4)
                diagonal[..., component] += corners.squared_derivative_diagonal(
                    C[..., component, 0], C[..., component, 1], barrier_weight)
        require_finite_result(diagonal, "GGN diagonal")
        result = restrict_free(diagonal, p.grid)
        if np.any(result <= 0):
            raise FloatingPointError("GGN diagonal must be strictly positive")
        result.flags.writeable = False
        return result

    def jvp(self, v: FloatArray) -> FloatArray:
        return residual_jvp(self._problem, self.base, v)

    def vjp(self, w: FloatArray) -> FloatArray:
        return residual_vjp(self._problem, self.base, w)

    def gn_hvp(self, v: FloatArray) -> FloatArray:
        return gn_hvp(self._problem, self.base, v)

    def as_linear_operator(self) -> LinearOperator:
        # SciPy accepts (n,) and (n,1); the core array contract stays strictly 1D.
        return LinearOperator(self.shape, dtype=np.float64,
                              matvec=lambda v: self.jvp(v.reshape(-1)),
                              rmatvec=lambda w: self.vjp(w.reshape(-1)))


def linearize(problem: Problem2D, base: EvalState) -> Linearization:
    _check_state(problem, base)
    grid = problem.grid
    return Linearization(base, (residual_size(grid, problem.geometry_discretization,
                          with_barrier=problem.jacobian_barrier is not None), 2*grid.n_free_scalar), problem)
