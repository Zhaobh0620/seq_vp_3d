"""T26 double-sided PG and BB2 context; not a masked optimization loop."""
from dataclasses import dataclass

import numpy as np

from ...validation import InputValidationError, as_float64_array, finite_real, readonly_copy, require_finite_result
from .poisson_gradient import _step_proposal


@dataclass(frozen=True)
class MaskedPGDirection:
    valid: bool
    reason: str
    p: np.ndarray
    slope: float


def masked_pg_direction(problem, base, space, gradient, apply_inverse):
    space.validate(problem, base)
    g = as_float64_array(gradient, 'gradient', (space.n_full,))
    restricted = space.project(g)
    if not np.any(restricted):
        return MaskedPGDirection(False, 'ZERO_RESTRICTED_GRADIENT', readonly_copy(np.zeros_like(g)), 0.)
    with np.errstate(over='raise', invalid='raise'):
        action = as_float64_array(apply_inverse(restricted), 'inverse action', g.shape, check_finite=False)
        require_finite_result(action, 'masked PG inverse action')
        p = -space.project(action)
        slope = float(g@p)
    require_finite_result(slope, 'masked PG slope')
    return MaskedPGDirection(slope < 0, 'DIRECTION_READY' if slope < 0 else 'NON_DESCENT_DIRECTION',
                             readonly_copy(p), slope)


@dataclass(frozen=True)
class MaskedBB2History:
    problem_id: str
    mask_id: str
    theta: np.ndarray
    gradient: np.ndarray
    direction: np.ndarray
    tau: float


def record_masked_bb2(problem, base, space, gradient, direction, tau):
    """Call ONLY after acceptance of the step from base; no state is mutated."""
    space.validate(problem, base)
    g = readonly_copy(as_float64_array(gradient, 'gradient', base.theta.shape))
    p = readonly_copy(as_float64_array(direction, 'direction', base.theta.shape))
    if np.any(p[space.frozen] != 0):
        raise InputValidationError('BB2 direction updates frozen variables')
    tau = finite_real(tau, 'accepted tau', positive=True)
    return MaskedBB2History(problem.problem_id, space.mask_id, readonly_copy(base.theta), g, p, tau)


def masked_step_proposal(config, problem, base, space, gradient, direction, previous=None):
    space.validate(problem, base)
    g = as_float64_array(gradient, 'gradient', base.theta.shape)
    p = as_float64_array(direction, 'direction', base.theta.shape)
    if np.any(p[space.frozen] != 0):
        raise InputValidationError('BB2 direction updates frozen variables')
    reset = None
    old = None
    if previous is not None:
        if not isinstance(previous, MaskedBB2History):
            raise InputValidationError('previous: expected MaskedBB2History')
        if previous.problem_id != problem.problem_id:
            reset = 'reset_problem_or_reference'
        elif previous.mask_id != space.mask_id:
            reset = 'reset_working_set'
        else:
            old = (previous.theta, previous.gradient, previous.direction, previous.tau)
    proposal, rule, sy, yy = _step_proposal(config, base.theta, g, p, old)
    return proposal, reset or rule, sy, yy
