"""Small shared lifecycle/result helpers; no dimension-specific 2D solver calls."""
from dataclasses import dataclass, replace
from time import perf_counter

import numpy as np
from scipy.linalg import norm

from ...solvers.common import gradient_norm_mass
from ...types import InvalidTrial
from ...validation import finite_real, require_finite_result, as_float64_array, InputValidationError
from ..residuals import AcceptedState3D, evaluate_trial, promote_trial


@dataclass
class SolverCounters3D:
    trials: int = 0
    promotions: int = 0
    gradients: int = 0
    hvp: int = 0
    poisson: int = 0
    cg_iterations: int = 0
    true_residual_checks: int = 0


@dataclass(frozen=True)
class InnerResult3D:
    state: AcceptedState3D | None
    status: str
    reason: str
    gradient_norm: float | None
    target: float
    accepted_steps: int
    history: tuple[dict, ...]
    counters: SolverCounters3D
    elapsed_seconds: float
    precision_evidence: bool = False

    @property
    def forcing_passed(self):
        return bool(self.state is not None and self.gradient_norm is not None and self.gradient_norm <= self.target)


def validate_target(target):
    value = finite_real(target, "inner gradient target")
    if value < 0:
        raise InputValidationError("inner gradient target must be nonnegative")
    return value


def prepare_base(problem, theta, counters, *, backend=None):
    counters.trials += 1
    evaluator = evaluate_trial if backend is None else backend.evaluate_trial
    promoter = promote_trial if backend is None else backend.promote_trial
    try:
        trial = evaluator(problem, theta)
        if isinstance(trial, InvalidTrial):
            return trial
        counters.promotions += 1
        return promoter(problem, trial)
    except FloatingPointError as exc:
        return InvalidTrial("NONFINITE_STATE", "initial evaluation/promotion failed: "+str(exc), 1)


def initial_failure_status(invalid):
    return {"INVALID_DOMAIN": "invalid_initial_domain",
            "FEASIBILITY_UNCERTAIN": "initial_feasibility_uncertain",
            "JACOBIAN_VIOLATION": "initial_jacobian_violation"}.get(invalid.reason, "invalid_initial_state")


def step_ratio(delta, theta, grid):
    delta = as_float64_array(delta, "step", (3*grid.n_free_scalar,))
    theta = as_float64_array(theta, "base theta", delta.shape)
    length = norm([axis[-1]-axis[0] for axis in grid.axes])
    from .. import fast_checks
    if fast_checks.enabled(delta.size):
        # E51 exact parallel max|.| (order-independent); same float division.
        value = float(np.float64(fast_checks.max_abs(delta))/max(length, fast_checks.max_abs(theta)))
    else:
        value = float(np.max(np.abs(delta))/max(length, np.max(np.abs(theta))))
    require_finite_result(value, "relative step")
    return value


def step_ratio_pair(candidate, theta, grid, original):
    """E89 (P03): step_ratio(candidate-theta, theta, grid) bit for bit without forming the difference.
    max|candidate-theta| is the exact compiled scan of the same IEEE differences. A finite maximum
    implies that both inputs are finite (inf-x and inf-inf are not finite), so the original input checks
    would pass and give the same quotient; otherwise original() runs the original expression unchanged."""
    from .. import fast_checks
    n = 3*grid.n_free_scalar
    if (fast_checks.enabled(n) and isinstance(candidate, np.ndarray) and isinstance(theta, np.ndarray)
            and candidate.dtype == np.float64 and theta.dtype == np.float64
            and candidate.shape == theta.shape == (n,)):
        top = fast_checks.max_abs_diff(candidate, theta)
        if np.isfinite(top):
            length = norm([axis[-1]-axis[0] for axis in grid.axes])
            value = float(np.float64(top)/max(length, fast_checks.max_abs(theta)))
            require_finite_result(value, "relative step")
            return value
    return original()


def step_ratio_fast_applies(candidate, theta, grid):
    """E97: whether step_ratio_pair takes its compiled branch for these inputs (its own guard, verbatim)."""
    from .. import fast_checks
    n = 3*grid.n_free_scalar
    return bool(fast_checks.enabled(n) and isinstance(candidate, np.ndarray) and isinstance(theta, np.ndarray)
                and candidate.dtype == np.float64 and theta.dtype == np.float64
                and candidate.shape == theta.shape == (n,))


def step_ratio_from_extrema(top, theta_max, grid):
    """E97: step_ratio_pair's finite branch with top = max|candidate-theta| and theta_max = max|theta| known
    (the same expression). Callers check step_ratio_fast_applies and np.isfinite(top) first: then
    step_ratio_pair would take this very branch with these very values."""
    length = norm([axis[-1]-axis[0] for axis in grid.axes])
    value = float(np.float64(top)/max(length, theta_max))
    require_finite_result(value, "relative step")
    return value


def step_ratio_pair_observed(candidate, theta, grid, original, observe):
    """E97: step_ratio_pair(candidate, theta, grid, original), and observe(top, theta_max) with the two compiled
    maxima whenever its compiled finite branch ran (before the quotient is checked)."""
    from .. import fast_checks
    if step_ratio_fast_applies(candidate, theta, grid):
        top = fast_checks.max_abs_diff(candidate, theta)
        if np.isfinite(top):
            length = norm([axis[-1]-axis[0] for axis in grid.axes])
            theta_max = fast_checks.max_abs(theta)
            observe(top, theta_max)
            value = float(np.float64(top)/max(length, theta_max))
            require_finite_result(value, "relative step")
            return value
    return original()


def finish_inner(state, status, reason, gradient_norm, target, accepted_steps, history,
                 counters, started, *, precision_evidence=False):
    return InnerResult3D(state, status, reason, gradient_norm, target, accepted_steps,
                         tuple(dict(row) for row in history), replace(counters),
                         perf_counter()-started, bool(precision_evidence))
