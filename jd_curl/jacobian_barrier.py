"""§19: normalized log barrier and the first exit along a Q1 search direction."""

from dataclasses import dataclass

import numpy as np

from .boundary import inject_free
from .types import EvalState, FloatArray, Problem2D
from .validation import InputValidationError, as_float64_array, finite_real, require_finite_result


@dataclass(frozen=True)
class JacobianFeasibilityConfig:
    """Hard Q1 corner domain only; contributes no residual or energy."""
    min_jd: float = .01
    fraction_to_boundary: float = .9

    def __post_init__(self):
        for name in ("min_jd", "fraction_to_boundary"):
            value = finite_real(getattr(self, name), "jacobian_feasibility."+name, positive=True)
            if value >= 1:
                raise InputValidationError("jacobian_feasibility."+name+": must be < 1")
            object.__setattr__(self, name, value)


def validate_feasibility(config, barrier, geometry_discretization):
    if config is not None:
        if not isinstance(config, JacobianFeasibilityConfig):
            raise InputValidationError("jacobian_feasibility: expected JacobianFeasibilityConfig")
        if barrier is not None or geometry_discretization != "q1":
            raise InputValidationError("hard-only feasibility requires q1 and no absolute barrier")


def feasibility_config(problem):
    """Legacy barrier still carries its own domain, with unchanged parameters."""
    return problem.jacobian_feasibility if problem.jacobian_feasibility is not None else problem.jacobian_barrier


@dataclass(frozen=True)
class JacobianBarrierConfig:
    weight: float
    min_jd: float
    fraction_to_boundary: float = .9

    def __post_init__(self):
        for name in ("weight", "min_jd", "fraction_to_boundary"):
            value = finite_real(getattr(self, name), "jacobian_barrier."+name, positive=True)
            if name != "weight" and value >= 1:
                raise InputValidationError("jacobian_barrier."+name+": must be < 1")
            object.__setattr__(self, name, value)


def validate_barrier(config, geometry_discretization: str) -> None:
    if config is not None:
        if not isinstance(config, JacobianBarrierConfig):
            raise InputValidationError("jacobian_barrier: expected JacobianBarrierConfig or None")
        if geometry_discretization != "q1":
            raise InputValidationError("jacobian_barrier requires q1 geometry_discretization")


def barrier_residual(J: FloatArray, config: JacobianBarrierConfig, omega: float):
    """Return weighted signed sqrt(2 psi) and its derivative with respect to J.

    psi(t)=t-1-log(t), t=(J-delta)/(1-delta). Near identity evaluate
    h(x)=psi(1+x)/x^2 by its series, not a cancellation-prone subtraction.
    Neither J nor t is clipped. The caller must reject infeasible candidates.
    """
    J = as_float64_array(J, "barrier.J")
    omega = finite_real(omega, "omega", positive=True)
    if np.any(J <= config.min_jd):
        raise InputValidationError("barrier.J: must exceed min_jd strictly")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        d = 1.-config.min_jd
        x, t = (J-1.)/d, (J-config.min_jd)/d
        near = np.abs(x) < .01
        r, dr = np.empty_like(J), np.empty_like(J)
        xn = x[near]
        h = np.full_like(xn, -1./11.)
        for k in range(8, -1, -1):
            h = h*xn+(-1.)**k/(k+2)
        scale = np.sqrt(2.*h)
        r[near], dr[near] = xn*scale, 1./(d*t[near]*scale)
        far = ~near
        psi = t[far]-1.-np.log(t[far])
        r[far] = np.sign(x[far])*np.sqrt(2.*psi)
        dr[far] = (1.-1./t[far])/(d*r[far])
        weight = np.sqrt(config.weight)*np.sqrt(omega/4.)
        r *= weight
        dr *= weight
    require_finite_result(r, "barrier residual")
    require_finite_result(dr, "barrier derivative")
    if weight == 0 or np.any(dr <= 0):
        raise FloatingPointError("barrier derivative: positive value underflowed")
    return r, dr


def first_positive_root(a: FloatArray, b: FloatArray, c: FloatArray) -> float:
    """Earliest positive zero among c+b*t+a*t^2, with c strictly positive.

    Include tangencies and the first of two positive roots (endpoint-only
    checks miss exit/re-entry). Normalize coefficients in extended precision
    and use the cancellation-safe q formula; infinity denotes no finite exit.
    """
    a = as_float64_array(a, "quadratic.a")
    b = as_float64_array(b, "quadratic.b", a.shape)
    c = as_float64_array(c, "quadratic.c", a.shape)
    if np.any(c <= 0):
        raise InputValidationError("quadratic.c: require a strictly feasible base")
    aa, bb, cc = (v.astype(np.longdouble).ravel() for v in (a, b, c))
    scale = np.maximum.reduce([np.abs(aa), np.abs(bb), cc])
    aa, bb, cc = aa/scale, bb/scale, cc/scale
    roots = np.full(aa.shape, np.inf, dtype=np.longdouble)
    linear = (aa == 0) & (bb < 0)
    roots[linear] = -cc[linear]/bb[linear]
    disc = bb*bb-4*aa*cc
    real = (aa != 0) & (disc >= 0)
    q = -.5*(bb[real]+np.copysign(np.sqrt(disc[real]), bb[real]))
    # c>0 implies q!=0 for real quadratic roots.
    r1, r2 = q/aa[real], cc[real]/q
    roots[real] = np.minimum(np.where(r1 > 0, r1, np.inf), np.where(r2 > 0, r2, np.inf))
    value = np.min(roots)
    return float(value) if value <= np.finfo(float).max else float("inf")


def feasible_step_bound(problem: Problem2D, base: EvalState, direction: FloatArray) -> float:
    """First exit from corner-JD and free-node sampling support, before safety factor.

    This is only a step proposal: evaluate_state rechecks the actual rounded
    candidate. A zero returned bound means no representable safe positive step.
    """
    config = feasibility_config(problem)
    if config is None or base.corner_geometry is None:
        raise InputValidationError("feasible_step_bound requires a feasible corner state")
    v = inject_free(direction, problem.grid)
    ops, A = problem.corner_operators, base.corner_geometry.A
    with np.errstate(over="raise", invalid="raise"):
        xx, xy = ops.dx(v[..., 0]), ops.dy(v[..., 0])
        yx, yy = ops.dx(v[..., 1]), ops.dy(v[..., 1])
        a = xx*yy-xy*yx
        b = A[..., 0, 0]*xx+A[..., 0, 1]*xy+A[..., 1, 0]*yx+A[..., 1, 1]*yy
        c = base.corner_geometry.J-config.min_jd
    for value in (a, b, c):
        require_finite_result(value, "feasible polynomial")
    bound = first_positive_root(a, b, c)
    # Fixed edges do not move; free samples require strict interior support.
    for component, coordinates in enumerate((problem.grid.x, problem.grid.y)):
        p = base.phi[1:-1, 1:-1, component].astype(np.longdouble)
        dv = v[1:-1, 1:-1, component].astype(np.longdouble)
        for mask, distance in ((dv > 0, coordinates[-1]-p), (dv < 0, coordinates[0]-p)):
            if np.any(mask):
                limit = np.min(distance[mask]/dv[mask])
                if limit < bound:
                    bound = float(limit)
    return bound
