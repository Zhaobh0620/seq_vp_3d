"""F13/F26/F27: frozen LM model, undamped prediction and explicit globalization."""

from dataclasses import asdict, replace
from time import perf_counter

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

from ..linearization import (Linearization, energy_gradient, linearize,
                             _residual_jvp_blocks, _residual_vjp_blocks)
from ..energy_comparison import stable_energy_reduction, precision_stagnation_evidence
from ..jacobian_barrier import feasible_step_bound, feasibility_config
from ..poisson import build_screened_poisson_inverse
from ..residuals import evaluate_state
from ..types import EvalState, FloatArray, Grid2D, InvalidTrial, Problem2D, Weights
from ..validation import InputValidationError, as_float64_array, finite_real, readonly_copy, require_finite_result
from .common import (
    InitializationFailure, IterationRecord, LinearSolveResult, LMConfig, LMDecision,
    SolverCounters, SolverResult, energy_evaluations, geometry_diagnostics,
    gradient_norm_mass, step_metrics, stopping_reason,
)


def model_hvp(linearization: Linearization, v: FloatArray, barrier_curvature="residual_gn") -> FloatArray:
    """F26: replace only the barrier curvature, without changing true derivatives."""
    if barrier_curvature == "residual_gn":
        return linearization.gn_hvp(v)
    problem, base = linearization._problem, linearization.base
    if barrier_curvature != "ggn" or problem.jacobian_barrier is None or problem.geometry_discretization != "q1":
        raise InputValidationError("GGN model requires q1 and an absolute barrier")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        blocks = _residual_jvp_blocks(problem, base, v)
        factor = linearization._barrier_ggn_factor
        barrier = factor*blocks.barrier
        require_finite_result(barrier, "barrier GGN action")
        return _residual_vjp_blocks(problem, base, replace(blocks, barrier=barrier))


def make_lm_operator(linearization: Linearization, *, omega: float, gamma: float,
                     barrier_curvature="residual_gn") -> LinearOperator:
    omega = finite_real(omega, "omega", positive=True)
    gamma = finite_real(gamma, "gamma", positive=True)
    damping = omega*gamma
    if not np.isfinite(damping) or damping <= 0:
        raise InputValidationError("omega*gamma: must be positive finite float64")
    n = linearization.shape[1]
    def action(v):
        v = as_float64_array(v.reshape(-1), "LM vector", (n,), check_finite=False)
        require_finite_result(v, "LM vector")
        with np.errstate(over="raise", invalid="raise"):
            out = model_hvp(linearization, v, barrier_curvature)+damping*v
        require_finite_result(out, "LM operator")
        return out
    return LinearOperator((n, n), dtype=np.float64, matvec=action, rmatvec=action)


def make_lm_preconditioner(grid: Grid2D, weights: Weights, *, gamma: float, b: float,
                           model_diagonal: FloatArray | None = None) -> LinearOperator:
    gamma = finite_real(gamma, "gamma", positive=True)
    inverse = build_screened_poisson_inverse(grid, a=weights.lam+gamma, b=b)
    scale = None
    if model_diagonal is not None:
        diagonal = as_float64_array(model_diagonal, "model_diagonal", (inverse.shape[0],))
        if np.any(diagonal <= 0):
            raise InputValidationError("model_diagonal: must be strictly positive")
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            m = grid.omega*(weights.lam+gamma+b*(2/grid.hx**2+2/grid.hy**2))
            scale = np.sqrt(m)/np.sqrt(diagonal+grid.omega*gamma)
        require_finite_result(scale, "equilibrated Poisson scale")
        if np.any(scale <= 0):
            raise FloatingPointError("equilibrated Poisson scale must be positive")
        scale.flags.writeable = False
    def action(v):
        v = as_float64_array(v.reshape(-1), "PCG residual", (inverse.shape[0],), check_finite=False)
        require_finite_result(v, "PCG residual")
        if scale is None:
            return inverse.apply(v)
        with np.errstate(over="raise", invalid="raise"):
            result = scale*inverse.apply(scale*v)
        require_finite_result(result, "equilibrated Poisson inverse")
        return result
    return LinearOperator(inverse.shape, dtype=np.float64, matvec=action, rmatvec=action)


def solve_lm_direction(linearization: Linearization, gradient: FloatArray,
                       *, gamma: float, config: LMConfig) -> LinearSolveResult:
    gradient = as_float64_array(gradient, "gradient", (linearization.shape[1],), check_finite=False)
    gamma = finite_real(gamma, "gamma", positive=True)
    if not config.gamma_min <= gamma <= config.gamma_max:
        raise InputValidationError("gamma: outside LM bounds")
    problem = linearization._problem
    grid, weights = problem.grid, problem.weights
    b = weights.mu/2+weights.nu/2 if config.preconditioner_b is None else config.preconditioner_b
    iterations, n_hvp, n_preconditioner = 0, 0, 0
    p, Hp, info, relative, slope = None, None, None, None, None

    def finish(valid, reason):
        return LinearSolveResult(valid, readonly_copy(p) if p is not None else None,
            info, iterations, relative, gamma, reason, readonly_copy(Hp) if Hp is not None else None,
            slope, n_hvp, n_preconditioner)

    def callback(_):
        nonlocal iterations
        iterations += 1

    try:
        gnorm = gradient_norm_mass(gradient, 1.)
        if gnorm == 0:
            return finish(False, "ZERO_GRADIENT")
        A = make_lm_operator(linearization, omega=grid.omega, gamma=gamma,
                             barrier_curvature=config.barrier_curvature)
        if config.preconditioner == "ggn_diagonal_poisson":
            inverse = make_lm_preconditioner(grid, weights, gamma=gamma, b=b,
                                             model_diagonal=linearization._barrier_ggn_diagonal)
        else:
            inverse = make_lm_preconditioner(grid, weights, gamma=gamma, b=b)

        def matvec(v):
            nonlocal n_hvp
            require_finite_result(v, "PCG vector")
            n_hvp += 1
            return A@v

        def precondition(v):
            nonlocal n_preconditioner
            require_finite_result(v, "PCG residual")
            n_preconditioner += 1
            return inverse@v

        counted_A = LinearOperator(A.shape, dtype=np.float64, matvec=matvec)
        counted_M = LinearOperator(A.shape, dtype=np.float64, matvec=precondition)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            p, info = cg(counted_A, -gradient, x0=np.zeros_like(gradient), M=counted_M,
                         rtol=config.cg_rtol, atol=config.cg_atol, maxiter=config.cg_maxiter, callback=callback)
            p = as_float64_array(p, "CG direction", gradient.shape, check_finite=False)
            require_finite_result(p, "CG direction")
            # Recompute a true residual and retain the undamped Hp for pred.
            n_hvp += 1
            Hp = model_hvp(linearization, p, config.barrier_curvature)
            require_finite_result(Hp, "GN direction")
            residual = Hp+(grid.omega*gamma)*p+gradient
            rnorm = gradient_norm_mass(residual, 1.)
            candidate_relative = float(rnorm/gnorm)
            tolerance = max(config.cg_atol, config.cg_rtol*gnorm)
            require_finite_result([candidate_relative, tolerance], "CG residual test")
            relative = candidate_relative
            if info != 0:
                return finish(False, "CG_NOT_CONVERGED")
            if rnorm > tolerance:
                return finish(False, "TRUE_RESIDUAL_FAILED")
            candidate_slope = float(gradient@p)
            require_finite_result(candidate_slope, "LM directional derivative")
            slope = candidate_slope
        if slope >= 0:
            return finish(False, "NON_DESCENT_DIRECTION")
        return finish(True, "LINEAR_SOLVE_ACCEPTED")
    except FloatingPointError:
        return finish(False, "NONFINITE_LINEAR_SOLVE")


def predicted_reduction(gradient: FloatArray, p: FloatArray, Hp: FloatArray) -> float:
    gradient = as_float64_array(gradient, "gradient")
    if gradient.ndim != 1:
        raise InputValidationError("gradient: expected a vector")
    p = as_float64_array(p, "p", gradient.shape)
    Hp = as_float64_array(Hp, "Hp", gradient.shape)
    with np.errstate(over="raise", invalid="raise"):
        value = float(-gradient@p-.5*(p@Hp))
    require_finite_result(value, "LM prediction")
    return value


def _increase_gamma(gamma: float, config: LMConfig) -> float:
    return min(config.gamma_max, gamma*config.gamma_increase)


def assess_lm_trial(base_energy: float, trial: EvalState | InvalidTrial,
                    *, pred: float, gamma: float, config: LMConfig, energy_change=None) -> LMDecision:
    base_energy = finite_real(base_energy, "base_energy")
    gamma = finite_real(gamma, "gamma", positive=True)
    if not config.gamma_min <= gamma <= config.gamma_max:
        raise InputValidationError("gamma: outside LM bounds")
    if not isinstance(trial, (EvalState, InvalidTrial)):
        raise InputValidationError("trial: expected EvalState or InvalidTrial")
    pred = float(as_float64_array(pred, "pred", (), check_finite=False))
    next_gamma = _increase_gamma(gamma, config)
    if not np.isfinite(pred):
        return LMDecision(False, "NONFINITE_PREDICTION", None, None, None, next_gamma)
    if pred <= 0:
        return LMDecision(False, "NONPOSITIVE_PREDICTION", None, pred, None, next_gamma)
    if isinstance(trial, InvalidTrial):
        return LMDecision(False, trial.reason, None, pred, None, next_gamma)
    ared = base_energy-trial.energy.total
    ratio = ared/pred
    if config.energy_comparison == "stable":
        if energy_change is None:
            raise InputValidationError("stable LM requires an evaluated energy difference")
        ared, ratio = energy_change.actual, energy_change.certified/pred
        if abs(ared) <= energy_change.resolution:
            return LMDecision(False, "ENERGY_RESOLUTION", ared, pred, ratio, next_gamma)
    if not np.isfinite(ared) or not np.isfinite(ratio):
        return LMDecision(False, "NONFINITE_GAIN", None, pred, None, next_gamma)
    if ared <= 0:
        return LMDecision(False, "NO_STRICT_DECREASE", ared, pred, ratio, next_gamma)
    if ratio < config.accept_ratio:
        return LMDecision(False, "GAIN_REJECTED", ared, pred, ratio, next_gamma)
    if ratio > config.good_ratio:
        next_gamma = max(config.gamma_min, gamma*config.gamma_decrease)
    elif ratio >= config.poor_ratio:
        next_gamma = gamma
    return LMDecision(True, "ACCEPTED", ared, pred, ratio, next_gamma)


def solve_lm(problem: Problem2D, z0: FloatArray, config: LMConfig) -> SolverResult | InitializationFailure:
    started = perf_counter()
    if config.geometry_discretization != problem.geometry_discretization:
        raise InputValidationError("config geometry_discretization does not match problem")
    if config.jacobian_feasibility != problem.jacobian_feasibility:
        raise InputValidationError("config jacobian_feasibility does not match problem")
    if config.jacobian_barrier != problem.jacobian_barrier:
        raise InputValidationError("config jacobian_barrier does not match problem")
    config = replace(config, gamma0=1e-3*max(1., problem.weights.lam) if config.gamma0 is None else config.gamma0,
                     preconditioner_b=problem.weights.mu/2+problem.weights.nu/2 if config.preconditioner_b is None else config.preconditioner_b)
    resolved = {"solver": "lm", **asdict(config)}
    base = evaluate_state(problem, z0, eval_id=0)
    if isinstance(base, InvalidTrial):
        return InitializationFailure("invalid_initial_state", base.reason+": "+base.message, base.invalid_count, resolved)
    counts = SolverCounters(n_energy_evaluations=energy_evaluations(base))
    history, small_steps, gamma = [], 0, config.gamma0
    initial_G, G, threshold = None, None, None

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
        lin = linearize(problem, base)
        base_diag = geometry_diagnostics(base)
        attempt_start = len(history)
        for attempt in range(1, config.max_retries+1):
            direction = solve_lm_direction(lin, gradient, gamma=gamma, config=config)
            counts.total_cg_iterations += direction.iterations
            counts.n_hvp += direction.n_hvp
            counts.n_preconditioner_applications += direction.n_preconditioner_applications
            pred, trial, trial_id, mass, relative = None, None, None, None, None
            reason = direction.reason
            partial = config.step_policy == "feasible_backtracking"
            alpha = 1.
            checked, limit, feasible = False, None, True
            if direction.valid:
                try:
                    pred = predicted_reduction(gradient, direction.p, direction.Hp)
                    require_finite_result(pred, "LM prediction")
                except FloatingPointError:
                    pred, reason = None, "NONFINITE_PREDICTION"
                if pred is not None and pred <= 0:
                    reason = "NONPOSITIVE_PREDICTION"
                if pred is not None and pred > 0 and feasibility_config(problem) is not None:
                    checked = True
                    counts.n_feasibility_checks += 1
                    try:
                        bound = feasibility_config(problem).fraction_to_boundary*feasible_step_bound(problem, base, direction.p)
                        limit = bound if np.isfinite(bound) else None
                        feasible = bound > 0 if partial else bound >= 1.
                        if partial:
                            alpha = min(1., bound)
                        if not feasible:
                            reason = "FEASIBLE_STEP_BOUND"
                    except FloatingPointError:
                        feasible, reason = False, "NONFINITE_FEASIBLE_BOUND"
            for backtracks in range(config.max_backtracks+1 if partial else 1):
                trial, trial_id, mass, relative = None, None, None, None
                if partial and direction.valid and feasible and pred is not None and pred > 0:
                    try:
                        pred = predicted_reduction(gradient, alpha*direction.p, alpha*direction.Hp)
                    except FloatingPointError:
                        pred, reason = None, "NONFINITE_PREDICTION"
                    if pred is not None and pred <= 0:
                        reason = "NONPOSITIVE_PREDICTION"
                if pred is not None and pred > 0 and feasible:
                    trial_id = counts.n_trial_evaluations+1
                    with np.errstate(over="ignore", invalid="ignore"):
                        z_trial = base.z+(alpha*direction.p if partial else direction.p)
                    trial = evaluate_state(problem, z_trial, eval_id=trial_id)
                    counts.n_trial_evaluations += 1
                    counts.n_energy_evaluations += energy_evaluations(trial)
                    counts.n_domain_rejections += int(isinstance(trial, InvalidTrial) and trial.reason == "INVALID_DOMAIN")
                    counts.n_jacobian_rejections += int(isinstance(trial, InvalidTrial) and trial.reason == "JACOBIAN_BOUND_VIOLATION")
                reduction = None
                if config.energy_comparison == "stable" and isinstance(trial, EvalState):
                    reduction = stable_energy_reduction(problem, base, trial)
                if trial is None:
                    next_gamma = _increase_gamma(gamma, config)
                    if (config.linear_failure_recovery == "initial_scale" and not direction.valid
                            and reason in ("CG_NOT_CONVERGED", "TRUE_RESIDUAL_FAILED")):
                        # F28: restore the original automatic initialization scale, not a global floor.
                        reference = min(config.gamma_max, max(config.gamma_min, 1e-3*max(1., problem.weights.lam)))
                        next_gamma = max(next_gamma, reference)
                    decision = LMDecision(False, reason, None, pred, None, next_gamma)
                else:
                    decision = assess_lm_trial(base.energy.total, trial, pred=pred, gamma=gamma, config=config,
                                               energy_change=reduction)
                    if isinstance(trial, EvalState):
                        try:
                            mass, relative = step_metrics(trial.z-base.z, base.z, problem.grid)
                        except FloatingPointError:
                            decision = LMDecision(False, "NONFINITE_STEP", decision.ared, pred, decision.ratio,
                                                  _increase_gamma(gamma, config))
                backtrack = bool(partial and not decision.accepted and trial is not None and
                                 backtracks < config.max_backtracks)
                if backtrack:
                    decision = replace(decision, gamma_next=gamma)
                elif partial and decision.accepted and alpha < 1.:
                    decision = replace(decision, gamma_next=max(gamma, decision.gamma_next))
                valid = isinstance(trial, EvalState)
                trial_diag = geometry_diagnostics(trial) if valid else None
                history.append(IterationRecord(
                    iteration=counts.n_accepted, attempt=attempt, solver="lm", accepted=decision.accepted,
                    accepted_count_after=counts.n_accepted+int(decision.accepted), base_eval_id=base.eval_id,
                    trial_eval_id=trial_id, energy=base.energy, trial_energy=trial.energy if valid else None,
                    grad_norm_mass=G, direction_dot_gradient=direction.directional_derivative,
                    step_norm_mass=mass, step_relative=relative, tau=alpha if partial else None, gamma_used=gamma,
                    gamma_next=decision.gamma_next, predicted_reduction=pred, actual_reduction=decision.ared,
                    ratio=decision.ratio, cg_info=direction.info, cg_iterations=direction.iterations if backtracks == 0 else 0,
                    cg_relative_residual=direction.relative_residual, **base_diag,
                    trial_min_jd=trial_diag["min_jd"] if valid else None,
                    trial_boundary_max_abs=trial_diag["boundary_max_abs"] if valid else None,
                    trial_min_corner_jd=trial_diag["min_corner_jd"] if valid else None,
                    feasibility_checked=checked and backtracks == 0, feasible_step_limit=limit,
                    trial_valid=valid, backtracks=backtracks, n_domain_rejections=counts.n_domain_rejections,
                    total_cg_iterations=counts.total_cg_iterations, n_energy_evaluations=counts.n_energy_evaluations,
                    reason=decision.reason, wall_seconds=perf_counter()-started,
                    energy_resolution=None if reduction is None else reduction.resolution,
                    lm_direction_curvature=float(direction.p@direction.Hp) if partial and direction.valid and pred is not None else None))
                if not backtrack:
                    break
                alpha *= config.backtrack_factor
            if decision.accepted:
                small_steps = small_steps+1 if relative <= config.stop.relative_step_tol else 0
                base, gamma = trial, decision.gamma_next
                counts.n_accepted += 1
                break
            if decision.gamma_next <= gamma:
                if precision_stagnation_evidence(history[attempt_start:],
                        untruncated=history[attempt_start].tau in (None, 1.)):
                    return finish("precision_stagnated", decision.reason)
                return finish("damping_limit", decision.reason)
            gamma = decision.gamma_next
        else:
            if precision_stagnation_evidence(history[attempt_start:],
                        untruncated=history[attempt_start].tau in (None, 1.)):
                return finish("precision_stagnated", history[-1].reason)
            return finish("lm_retry_exhausted", history[-1].reason)
