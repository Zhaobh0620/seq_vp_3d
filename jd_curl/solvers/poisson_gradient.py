"""F12 PG/Armijo with fixed C model, F14 stopping, and explicit failure states."""

from dataclasses import asdict, replace
from time import perf_counter

import numpy as np
from scipy.linalg import norm

from ..linearization import _check_state, energy_gradient
from ..energy_comparison import stable_energy_reduction, precision_stagnation_evidence
from ..jacobian_barrier import feasible_step_bound, feasibility_config
from ..derivative_metric import DerivativeMetricInverse, build_derivative_metric_inverse
from ..q1_metric import Q1MetricInverse, build_q1_metric_inverse
from ..poisson import SPDInverse, build_screened_poisson_inverse
from ..residuals import evaluate_state
from ..types import EvalState, FloatArray, InvalidTrial, Problem2D
from ..validation import InputValidationError, as_float64_array, readonly_copy, require_finite_result
from .common import (
    AcceptedStep, DirectionResult, InitializationFailure, IterationRecord, PoissonGradientConfig,
    RejectedStep, SolverCounters, SolverResult, energy_evaluations, geometry_diagnostics,
    gradient_norm_mass, step_metrics, stopping_reason,
)


def _step_proposal(config, z, gradient, direction, previous):
    """§20 BB2 in the *fixed* SPD metric, using only accepted secant pairs.

    previous=(z_previous, g_previous, p_previous, accepted_tau). Because
    p=-M^-1*g, M^-1*y=p_previous-p: no extra inverse or Hessian action.
    Return proposal, rule, s.T*y, y.T*M^-1*y for an auditable selection.
    """
    if config.step_policy == "fixed":
        return config.initial_step, "fixed", None, None
    def protect(value):
        return min(config.step_max, max(config.step_min, value))
    if previous is None:
        return protect(config.initial_step), "initial", None, None
    old_z, old_g, old_p, tau = previous
    inherited = protect(2.*min(tau, config.step_max/2.))
    if config.step_policy == "inherit":
        return inherited, "inherit", None, None
    sy, yy = None, None
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            s, y, inverse_y = z-old_z, gradient-old_g, old_p-direction
            require_finite_result(s, "PG secant displacement")
            require_finite_result(y, "PG secant gradient")
            require_finite_result(inverse_y, "PG secant inverse-gradient")
            value = float(s@y)
            require_finite_result(value, "PG secant curvature")
            sy = value
            if sy <= 0:
                return inherited, "fallback_nonpositive_curvature", sy, None
            ns, ny = norm(s), norm(y)
            # Normalized test avoids an overflowing product of the two norms.
            cosine = float((s/ns)@(y/ny))
            if cosine <= 1e-12:
                return inherited, "fallback_small_curvature", sy, None
            value = float(y@inverse_y)
            require_finite_result(value, "PG metric curvature")
            yy = value
            if yy <= 0:
                return inherited, "fallback_nonpositive_metric_curvature", sy, yy
            alpha = sy/yy
            require_finite_result(alpha, "PG BB2 proposal")
            if alpha <= 0:
                return inherited, "fallback_nonpositive_step", sy, yy
    except FloatingPointError:
        return inherited, "fallback_nonfinite_secant", sy, yy
    return protect(alpha), "bb2", sy, yy


def poisson_gradient_direction(gradient: FloatArray, inverse: SPDInverse | DerivativeMetricInverse | Q1MetricInverse) -> DirectionResult:
    gradient = as_float64_array(gradient, "gradient", (inverse.shape[0],), check_finite=False)
    if not np.all(np.isfinite(gradient)):
        return DirectionResult(False, None, None, "NONFINITE_GRADIENT")
    try:
        with np.errstate(over="raise", invalid="raise"):
            p = -inverse.apply(gradient)
            p = as_float64_array(p, "direction", gradient.shape, check_finite=False)
            require_finite_result(p, "direction")
            slope = float(gradient@p)
        require_finite_result(slope, "directional derivative")
    except FloatingPointError:
        return DirectionResult(False, None, None, "NONFINITE_DIRECTION")
    if slope >= 0:
        return DirectionResult(False, readonly_copy(p), slope, "NON_DESCENT_DIRECTION")
    return DirectionResult(True, readonly_copy(p), slope, "DESCENT")


def armijo_search(
    problem: Problem2D, base: EvalState, gradient: FloatArray, direction: FloatArray,
    config: PoissonGradientConfig,
) -> AcceptedStep | RejectedStep:
    started = perf_counter()
    _check_state(problem, base)
    if config.jacobian_feasibility != problem.jacobian_feasibility:
        raise InputValidationError("config jacobian_feasibility does not match problem")
    if config.jacobian_barrier != problem.jacobian_barrier:
        raise InputValidationError("config jacobian_barrier does not match problem")
    gradient = as_float64_array(gradient, "gradient", base.z.shape, check_finite=False)
    direction = as_float64_array(direction, "direction", base.z.shape, check_finite=False)
    try:
        G = gradient_norm_mass(gradient, problem.grid.omega)
        require_finite_result(direction, "direction")
        with np.errstate(over="raise", invalid="raise"):
            slope = float(gradient@direction)
        require_finite_result(slope, "directional derivative")
    except FloatingPointError:
        return RejectedStep("nonfinite_evaluation", "NONFINITE_DIRECTION_OR_GRADIENT", 0, 0)
    if slope >= 0:
        return RejectedStep("non_descent_direction", "NON_DESCENT_DIRECTION", 0, 0)
    history = []
    n_domain, n_energy, tau = 0, 0, config.initial_step
    checked, limit = int(feasibility_config(problem) is not None), None
    if checked:
        try:
            raw_bound = feasible_step_bound(problem, base, direction)
            safe_bound = feasibility_config(problem).fraction_to_boundary*raw_bound
        except FloatingPointError:
            return RejectedStep("nonfinite_evaluation", "NONFINITE_FEASIBLE_BOUND", 0, 0,
                                n_feasibility_checks=1)
        if safe_bound <= 0:
            return RejectedStep("feasibility_blocked", "NO_POSITIVE_FEASIBLE_STEP", 0, 0,
                                n_feasibility_checks=1)
        limit = safe_bound if np.isfinite(safe_bound) else None
        tau = min(tau, safe_bound)
    base_diag = geometry_diagnostics(base)
    for j in range(config.max_backtracks+1):
        trial_id = base.eval_id+j+1
        # Pass nonfinite arithmetic candidates to C for classification, not repair.
        with np.errstate(over="ignore", invalid="ignore"):
            z_trial = base.z+tau*direction
            delta = z_trial-base.z
        trial = evaluate_state(problem, z_trial, eval_id=trial_id)
        n_energy += energy_evaluations(trial)
        valid = isinstance(trial, EvalState)
        n_domain += int(not valid and trial.reason == "INVALID_DOMAIN")
        try:
            mass, relative = step_metrics(delta, base.z, problem.grid)
        except FloatingPointError:
            mass, relative = None, None
        accepted = False
        actual = resolution = None
        if valid:
            with np.errstate(over="ignore", invalid="ignore"):
                bound = base.energy.total+config.armijo_c1*tau*slope
                actual = base.energy.total-trial.energy.total
            accepted = bool(relative is not None and np.isfinite(bound) and actual > 0 and trial.energy.total <= bound)
            reason = "ACCEPTED" if accepted else ("NO_STRICT_DECREASE" if actual <= 0 else "ARMIJO_REJECTED")
            if config.energy_comparison == "stable":
                reduction = stable_energy_reduction(problem, base, trial)
                actual, resolution = reduction.actual, reduction.resolution
                accepted = bool(relative is not None and reduction.certified > 0 and
                                reduction.certified >= -config.armijo_c1*tau*slope)
                reason = ("ACCEPTED" if accepted else "ENERGY_RESOLUTION" if abs(actual) <= resolution
                          else "NO_STRICT_DECREASE" if actual < 0 else "ARMIJO_REJECTED")
            if relative is None:
                reason = "NONFINITE_STEP"
        else:
            reason = trial.reason
        trial_diag = geometry_diagnostics(trial) if valid else None
        history.append(IterationRecord(
            iteration=0, attempt=j+1, solver="poisson_gradient", accepted=accepted,
            accepted_count_after=int(accepted), base_eval_id=base.eval_id, trial_eval_id=trial_id,
            energy=base.energy, trial_energy=trial.energy if valid else None, grad_norm_mass=G,
            direction_dot_gradient=slope, step_norm_mass=mass, step_relative=relative, tau=tau,
            gamma_used=None, gamma_next=None, predicted_reduction=None, actual_reduction=actual,
            ratio=None, cg_info=None, cg_iterations=None, cg_relative_residual=None, **base_diag,
            trial_min_jd=trial_diag["min_jd"] if valid else None,
            trial_boundary_max_abs=trial_diag["boundary_max_abs"] if valid else None,
            trial_min_corner_jd=trial_diag["min_corner_jd"] if valid else None,
            feasibility_checked=bool(checked), feasible_step_limit=limit,
            trial_valid=valid, backtracks=j, n_domain_rejections=n_domain,
            total_cg_iterations=0, n_energy_evaluations=n_energy, reason=reason,
            wall_seconds=perf_counter()-started, energy_resolution=resolution))
        if valid and relative is None:
            return RejectedStep("nonfinite_evaluation", reason, j+1, n_domain, tuple(history), n_energy, checked)
        if accepted:
            return AcceptedStep(trial, tau, j+1, n_domain, tuple(history), n_energy, checked)
        tau *= config.backtrack_factor
    status = "domain_blocked" if n_domain == len(history) else "line_search_failed"
    full = limit is None or limit >= config.initial_step
    if config.energy_comparison == "stable" and precision_stagnation_evidence(
            history, untruncated=full):
        status = "precision_stagnated"
    return RejectedStep(status, history[-1].reason, len(history), n_domain, tuple(history), n_energy, checked)


def solve_poisson_gradient(
    problem: Problem2D, z0: FloatArray, config: PoissonGradientConfig,
) -> SolverResult | InitializationFailure:
    started = perf_counter()
    if config.geometry_discretization != problem.geometry_discretization:
        raise InputValidationError("config geometry_discretization does not match problem")
    if config.jacobian_feasibility != problem.jacobian_feasibility:
        raise InputValidationError("config jacobian_feasibility does not match problem")
    if config.jacobian_barrier != problem.jacobian_barrier:
        raise InputValidationError("config jacobian_barrier does not match problem")
    config = replace(config, a=problem.weights.lam if config.a is None else config.a,
                     b=problem.weights.mu/2+problem.weights.nu/2 if config.b is None else config.b)
    resolved = {"solver": "poisson_gradient", **asdict(config)}
    base = evaluate_state(problem, z0, eval_id=0)
    if isinstance(base, InvalidTrial):
        return InitializationFailure("invalid_initial_state", base.reason+": "+base.message,
                                     base.invalid_count, resolved)
    counts = SolverCounters(n_energy_evaluations=energy_evaluations(base))
    history, inverse = [], None
    previous = None
    initial_G, G, threshold = None, None, None
    small_steps = 0

    def finish(status, message):
        return SolverResult(status == "converged_gradient", status, message, base.z, base,
                            tuple(history), **asdict(counts), resolved_config=resolved,
                            initial_gradient_norm_mass=initial_G, final_gradient_norm_mass=G,
                            gradient_threshold=threshold, wall_seconds=perf_counter()-started)

    while True:
        counts.n_gradient_evaluations += 1
        try:
            gradient = energy_gradient(problem, base)
            G = gradient_norm_mass(gradient, problem.grid.omega)
            if initial_G is None:
                initial_G = G
                threshold = config.stop.grad_atol+config.stop.grad_rtol*initial_G
                require_finite_result(threshold, "gradient threshold")
        except FloatingPointError as exc:
            G = None
            return finish("nonfinite_evaluation", str(exc))
        stop = stopping_reason(G, threshold, counts.n_accepted, small_steps, config.stop)
        if stop is not None:
            return finish(*stop)
        if inverse is None:
            factory = {"five_point": build_screened_poisson_inverse,
                       "derivative_gram": build_derivative_metric_inverse,
                       "q1": build_q1_metric_inverse}[config.metric]
            inverse = factory(problem.grid, a=config.a, b=config.b)
        counts.n_preconditioner_applications += 1
        direction = poisson_gradient_direction(gradient, inverse)
        if not direction.valid:
            status = "non_descent_direction" if direction.reason == "NON_DESCENT_DIRECTION" else "nonfinite_evaluation"
            return finish(status, direction.reason)
        proposal, rule, sy, yy = _step_proposal(config, base.z, gradient, direction.p, previous)
        elapsed_before_search = perf_counter()-started
        step = armijo_search(problem, base, gradient, direction.p, replace(config, initial_step=proposal))
        for row in step.history:
            history.append(replace(row, iteration=counts.n_accepted,
                                   accepted_count_after=counts.n_accepted+int(row.accepted),
                                   n_domain_rejections=counts.n_domain_rejections+row.n_domain_rejections,
                                   n_energy_evaluations=counts.n_energy_evaluations+row.n_energy_evaluations,
                                   pg_step_proposal=proposal, pg_step_rule=rule,
                                   pg_secant_curvature=sy, pg_metric_curvature=yy,
                                   wall_seconds=elapsed_before_search+row.wall_seconds))
        counts.n_trial_evaluations += step.n_trials
        counts.n_domain_rejections += step.n_domain_rejections
        counts.n_energy_evaluations += step.n_energy_evaluations
        counts.n_feasibility_checks += step.n_feasibility_checks
        counts.n_jacobian_rejections += sum(row.reason == "JACOBIAN_BOUND_VIOLATION" for row in step.history)
        if isinstance(step, RejectedStep):
            return finish(step.status, step.reason)
        _, relative = step_metrics(step.state.z-base.z, base.z, problem.grid)
        small_steps = small_steps+1 if relative <= config.stop.relative_step_tol else 0
        if config.step_policy != "fixed":
            previous = (base.z, gradient, direction.p, step.tau)
        base = step.state
        counts.n_accepted += 1
