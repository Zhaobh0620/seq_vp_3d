"""T12 full-step barrier-GGN LM with mass damping and Poisson PCG.

Each base, derivative-ready state and true GGN diagonal remain frozen through
all damping retries. No PG fallback, partial step or implicit reference refresh.
"""
from dataclasses import dataclass
from copy import deepcopy
from time import perf_counter

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

from ...types import InvalidTrial
from ...validation import InputValidationError, as_float64_array, finite_real, require_finite_result
from ..linearization import barrier_ggn_diagonal, barrier_ggn_hvp, build_linearization, energy_gradient
from ..poisson import build_poisson_inverse
from ..residuals import evaluate_trial, promote_trial
from ..validation import integer_at_least
from ..working_set import WorkingSet3D
from .common import (
    SolverCounters3D, finish_inner, gradient_norm_mass, initial_failure_status,
    prepare_base, step_ratio, validate_target,
)
from ..energy_comparison import stable_energy_reduction


@dataclass(frozen=True)
class LMConfig3D:
    max_iterations: int = 500
    cg_rtol: float = 1e-3
    cg_atol: float = 0.
    max_cg_iterations: int = 200
    max_directions: int = 12
    gamma0: float | None = None
    gamma_min: float = 1e-12
    gamma_max: float = 1e12
    preconditioner: str = "poisson"
    recovery: str = "multiply"
    step_tol: float = 1e-12
    stall_steps: int = 5

    def __post_init__(self):
        object.__setattr__(self, "max_iterations", integer_at_least(self.max_iterations, "max_iterations", 0))
        for name in ("max_cg_iterations", "max_directions", "stall_steps"):
            object.__setattr__(self, name, integer_at_least(getattr(self, name), name))
        for name in ("cg_rtol", "gamma_min", "gamma_max"):
            object.__setattr__(self, name, finite_real(getattr(self, name), name, positive=True))
        if self.cg_rtol >= 1:
            raise InputValidationError("cg_rtol: expected < 1")
        object.__setattr__(self, "cg_atol", finite_real(self.cg_atol, "cg_atol"))
        object.__setattr__(self, "step_tol", finite_real(self.step_tol, "step_tol"))
        if self.cg_atol != 0:
            raise InputValidationError("cg_atol: T12 requires zero absolute CG tolerance")
        if self.step_tol < 0:
            raise InputValidationError("step_tol: must be nonnegative")
        if self.gamma_max < self.gamma_min:
            raise InputValidationError("gamma_max: must be >= gamma_min")
        if self.gamma0 is not None:
            object.__setattr__(self, "gamma0", finite_real(self.gamma0, "gamma0", positive=True))
            if not self.gamma_min <= self.gamma0 <= self.gamma_max:
                raise InputValidationError("gamma0: outside damping bounds")
        if self.preconditioner not in ("poisson", "ggn_diagonal_poisson"):
            raise InputValidationError("preconditioner: expected poisson or ggn_diagonal_poisson")
        if self.recovery not in ("multiply", "initial_scale"):
            raise InputValidationError("recovery: expected multiply or initial_scale")


def _initial_gamma(problem, config):
    automatic = min(config.gamma_max, max(config.gamma_min, 1e-3*max(1., problem.weights.lam)))
    return automatic if config.gamma0 is None else config.gamma0


def _increase_gamma(gamma, config, *, problem, reason):
    # Avoid overflow before the configured upper clipping bound.
    increased = config.gamma_max if gamma >= config.gamma_max/4 else 4*gamma
    if config.recovery == "initial_scale" and reason in ("CG_NOT_CONVERGED", "TRUE_RESIDUAL_FAILED"):
        automatic = min(config.gamma_max, max(config.gamma_min, 1e-3*max(1., problem.weights.lam)))
        increased = max(increased, automatic)
    return increased


def build_lm_preconditioner(lin, gamma, *, kind="poisson", diagonal=None):
    """T12 inverse; equilibrated P has diag(P)=diag(H_C)+omega*gamma."""
    gamma = finite_real(gamma, "gamma", positive=True)
    problem = lin.problem
    grid, weights = problem.grid, problem.weights
    inverse = build_poisson_inverse(grid, a=weights.lam+gamma,
                                    b=weights.mu/2+weights.nu/2, metric="seven_point")
    if kind == "poisson":
        return LinearOperator(inverse.shape, dtype=np.float64, matvec=inverse.apply)
    if kind != "ggn_diagonal_poisson":
        raise InputValidationError("unknown LM preconditioner")
    if diagonal is None:
        diagonal = barrier_ggn_diagonal(lin)
    diagonal = as_float64_array(diagonal, "GGN diagonal", (inverse.shape[0],))
    if np.any(diagonal <= 0):
        raise InputValidationError("GGN diagonal: must be positive")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        mass_diagonal = grid.omega*(weights.lam+gamma
                         +(weights.mu+weights.nu)*sum(1/h**2 for h in grid.spacing))
        scale = np.sqrt(mass_diagonal)/np.sqrt(diagonal+grid.omega*gamma)
    require_finite_result(scale, "LM equilibration scale")
    if np.any(scale <= 0):
        raise FloatingPointError("nonpositive LM equilibration scale")
    return LinearOperator(inverse.shape, dtype=np.float64,
                          matvec=lambda v: scale*inverse.apply(scale*v))


@dataclass(frozen=True)
class LMDirection3D:
    valid: bool
    reason: str
    p: np.ndarray | None
    Hp: np.ndarray | None
    info: int | None
    iterations: int
    relative_residual: float | None
    slope: float | None
    n_hvp: int
    n_preconditioner: int
    true_residual_checked: bool


def solve_lm_direction(lin, gradient, *, gamma, config, diagonal=None, subspace=None):
    """T12/T26: optional reduced space; true residual is in that same space.

    No mask in the old solver loop. Explicit callers must still perform the
    full original feasibility/energy acceptance on the returned full p.
    """
    n = 3*lin.problem.grid.n_free_scalar
    gradient = as_float64_array(gradient, "gradient", (n,))
    gamma = finite_real(gamma, "gamma", positive=True)
    if not isinstance(config, LMConfig3D):
        raise InputValidationError("config: expected LMConfig3D")
    if not config.gamma_min <= gamma <= config.gamma_max:
        raise InputValidationError("gamma: outside damping bounds")
    if subspace is not None:
        if not isinstance(subspace, WorkingSet3D):
            raise InputValidationError('subspace: expected WorkingSet3D')
        subspace.validate(lin.problem, lin.state)
        n = subspace.n_free
    rhs_gradient = gradient if subspace is None else subspace.restrict(gradient)
    p = Hp = info = relative = slope = None
    iterations = n_hvp = n_preconditioner = 0
    true_residual_checked = False

    def finish(valid, reason):
        return LMDirection3D(valid, reason, p, Hp, info, iterations, relative, slope,
                             n_hvp, n_preconditioner, true_residual_checked)

    def matvec(v):
        nonlocal n_hvp
        n_hvp += 1
        full = v if subspace is None else subspace.inject(v)
        result = barrier_ggn_hvp(lin, full)+lin.problem.grid.omega*gamma*full
        if subspace is not None:
            require_finite_result(result, 'reduced LM operator action')
        return result if subspace is None else subspace.restrict(result)

    def precondition(v):
        nonlocal n_preconditioner
        n_preconditioner += 1
        full = v if subspace is None else subspace.inject(v)
        result = inverse@full
        if subspace is not None:
            require_finite_result(result, 'reduced LM preconditioner action')
        return result if subspace is None else subspace.restrict(result)

    def callback(_):
        nonlocal iterations
        iterations += 1

    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            gnorm = gradient_norm_mass(rhs_gradient, 1.)
            if gnorm == 0:
                return finish(False, "ZERO_GRADIENT" if subspace is None else "ZERO_RESTRICTED_GRADIENT")
            inverse = build_lm_preconditioner(lin, gamma, kind=config.preconditioner, diagonal=diagonal)
            A = LinearOperator((n, n), dtype=np.float64, matvec=matvec)
            M = LinearOperator((n, n), dtype=np.float64, matvec=precondition)
            p, info = cg(A, -rhs_gradient, x0=np.zeros_like(rhs_gradient), M=M,
                         rtol=config.cg_rtol, atol=config.cg_atol,
                         maxiter=config.max_cg_iterations, callback=callback)
            info = int(info)
            if subspace is not None:
                p = as_float64_array(p, 'reduced CG direction', (n,), check_finite=False)
                require_finite_result(p, 'reduced CG direction')
                p = subspace.inject(p)
            p = as_float64_array(p, "CG direction", gradient.shape, check_finite=False)
            require_finite_result(p, "CG direction")
            n_hvp += 1
            Hp = barrier_ggn_hvp(lin, p)
            true_residual_checked = True
            residual = Hp+lin.problem.grid.omega*gamma*p+gradient
            if subspace is not None:
                require_finite_result(residual, 'reduced LM true residual')
            residual_norm = gradient_norm_mass(residual if subspace is None else subspace.restrict(residual), 1.)
            relative = residual_norm/gnorm
            require_finite_result(relative, "true CG relative residual")
            if info != 0:
                return finish(False, "CG_NOT_CONVERGED")
            if residual_norm > config.cg_rtol*gnorm:
                return finish(False, "TRUE_RESIDUAL_FAILED")
            slope = float(gradient@p)
            require_finite_result(slope, "LM slope")
            if slope >= 0:
                return finish(False, "NON_DESCENT_DIRECTION")
        return finish(True, "LINEAR_SOLVE_ACCEPTED")
    except FloatingPointError:
        return finish(False, "NONFINITE_LINEAR_SOLVE")


def predicted_reduction(gradient, p, Hp):
    """Undamped prediction; gamma belongs to the solve, not this model gain."""
    with np.errstate(over="raise", invalid="raise"):
        value = float(-gradient@p-.5*(p@Hp))
    require_finite_result(value, "LM prediction")
    return value


def _precision_evidence(records):
    """F24b: every original full trial in this failed retry round unresolved."""
    return bool(records) and all(
        row["reason"] == "ENERGY_RESOLUTION" and row["trial_valid"] and not row["accepted"]
        and row["actual_reduction"] is not None and np.isfinite(row["actual_reduction"])
        and abs(row["actual_reduction"]) <= row["energy_resolution"]
        for row in records)


def solve_lm(problem, theta, *, target, config=None, backend=None,
             subspace=None, restart=None, budget=None, certificate_observer=None,
             iteration_observer=None):
    """Return only the last strictly accepted state, never a failed trial."""
    started = perf_counter()
    target = validate_target(target)
    config = LMConfig3D() if config is None else config
    if not isinstance(config, LMConfig3D):
        raise InputValidationError("config: expected LMConfig3D")
    counters = SolverCounters3D()
    evaluation = backend if getattr(backend, "evaluation_backend_id", None) is not None else None
    evaluator = evaluate_trial if evaluation is None else evaluation.evaluate_trial
    promoter = promote_trial if evaluation is None else evaluation.promote_trial
    base = (prepare_base(problem, theta, counters) if evaluation is None else
            prepare_base(problem, theta, counters, backend=evaluation))
    history, accepted, small_steps, G = [], 0, 0, None

    def finish(status, reason, *, precision_evidence=False):
        return finish_inner(None if isinstance(base, InvalidTrial) else base,
                            status, reason, G, target, accepted, history, counters,
                            started, precision_evidence=precision_evidence)

    if isinstance(base, InvalidTrial):
        return finish(initial_failure_status(base), base.reason)
    if subspace is not None:
        if backend is not None:
            raise InputValidationError("masked restart path is CPU only")
        subspace.validate(problem, base)
    gamma = _initial_gamma(problem, config) if restart is None else restart.get('gamma', _initial_gamma(problem, config))
    while True:
        counters.gradients += 1
        lin = build_linearization(problem, base)
        try:
            gradient = energy_gradient(lin) if backend is None else backend.gradient(lin, target)
            G = gradient_norm_mass(gradient, problem.grid.omega)
        except FloatingPointError:
            G = None
            return finish("nonfinite_evaluation", "NONFINITE_GRADIENT")
        # T14: true full gradient precedes budget and small-step tests.
        if G <= target:
            return finish("converged_gradient", "FORCING_GRADIENT")
        if budget is not None and budget():
            return finish("work_budget_exhausted", "WALL_BUDGET")
        if accepted >= config.max_iterations:
            return finish("max_iterations", "ACCEPTED_STEP_BUDGET")
        if small_steps >= config.stall_steps:
            return finish("stagnated", "SMALL_ACCEPTED_STEPS")
        try:
            diagonal = None
            if config.preconditioner == "ggn_diagonal_poisson":
                diagonal = barrier_ggn_diagonal(lin) if backend is None else backend.diagonal(lin)
        except FloatingPointError:
            return finish("nonfinite_evaluation", "NONFINITE_GGN_DIAGONAL")
        attempt_start = len(history)
        for attempt in range(1, config.max_directions+1):
            if budget is not None and budget():
                return finish("work_budget_exhausted", "WALL_BUDGET")
            direction_solve = solve_lm_direction if backend is None else backend.lm_direction
            extra = {} if subspace is None else {'subspace': subspace}
            direction = direction_solve(lin, gradient, gamma=gamma, config=config, diagonal=diagonal, **extra)
            counters.hvp += direction.n_hvp
            counters.poisson += direction.n_preconditioner
            counters.cg_iterations += direction.iterations
            counters.true_residual_checks += int(direction.true_residual_checked)
            reason = direction.reason
            pred = reduction = trial = ratio = relative_step = None
            take = False
            if direction.valid:
                try:
                    pred = predicted_reduction(gradient, direction.p, direction.Hp)
                    if pred <= 0:
                        reason = "NONPOSITIVE_PREDICTION"
                    else:
                        with np.errstate(over="ignore", invalid="ignore"):
                            trial_theta = base.theta+direction.p
                        counters.trials += 1
                        trial = (evaluator(problem, trial_theta) if certificate_observer is None else
                                 evaluator(problem, trial_theta, certificate_observer=certificate_observer))
                        if isinstance(trial, InvalidTrial):
                            reason = trial.reason
                        else:
                            reduction = (stable_energy_reduction(problem, base, trial) if evaluation is None else
                                         evaluation.stable_energy_reduction(problem, base, trial,
                                             required_reduction=1e-4*pred))
                            ratio = reduction.certified/pred
                            require_finite_result(ratio, "LM gain")
                            relative_step = step_ratio(trial.theta-base.theta, base.theta, problem.grid)
                            if abs(reduction.actual) <= reduction.resolution:
                                reason = "ENERGY_RESOLUTION"
                            elif reduction.certified <= 0:
                                reason = "NO_STRICT_DECREASE"
                            elif ratio < 1e-4:
                                reason = "GAIN_REJECTED"
                            else:
                                take, reason = True, "ACCEPTED"
                except FloatingPointError:
                    take, reason = False, "NONFINITE_TRIAL_MODEL"
            gamma_next = _increase_gamma(gamma, config, problem=problem, reason=reason)
            if take:
                counters.promotions += 1
                try:
                    ready = promoter(problem, trial)
                except FloatingPointError:
                    ready = InvalidTrial("NONFINITE_PROMOTION", "nonfinite trial derivative preparation", 0)
                if isinstance(ready, InvalidTrial):
                    take, reason = False, ready.reason
                elif ratio > .75:
                    gamma_next = max(config.gamma_min, gamma/2)
                elif ratio >= .25:
                    gamma_next = gamma
            valid_trial = trial is not None and not isinstance(trial, InvalidTrial)
            history.append({
                "iteration": accepted, "attempt": attempt, "accepted": take, "reason": reason,
                "base_eval_id": base.eval_id, "trial_eval_id": trial.eval_id if valid_trial else None,
                "base_energy": base.energy.total, "trial_energy": trial.energy.total if valid_trial else None,
                "gradient_norm": G, "gamma_used": gamma, "gamma_next": gamma_next,
                "predicted_reduction": pred,
                "actual_reduction": reduction.actual if reduction is not None else None,
                "energy_resolution": reduction.resolution if reduction is not None else None,
                "certified_reduction": reduction.certified if reduction is not None else None,
                "gain": ratio, "step_relative": relative_step, "trial_valid": valid_trial,
                "trial_min_jd_lower_bound": trial.certificate.min_jd_lower_bound if valid_trial else None,
                "cg_info": direction.info, "cg_iterations": direction.iterations,
                "cg_relative_residual": direction.relative_residual,
                "direction_dot_gradient": direction.slope,
                "tau": 1., "untruncated": True, "controls_id": problem.controls.controls_id,
                "elapsed_seconds": perf_counter()-started,
            })
            if evaluation is not None:
                history[-1]['forcing_target'] = target
                history[-1]['comparison_diagnostics'] = dict(evaluation.last_comparison) if reduction is not None else None
            if iteration_observer is not None:
                iteration_observer(deepcopy(history[-1]))
            if take:
                small_steps = small_steps+1 if relative_step <= config.step_tol else 0
                base, gamma = ready, gamma_next
                if restart is not None:
                    restart['gamma'] = gamma
                accepted += 1
                break
            if gamma_next <= gamma:
                evidence = _precision_evidence(history[attempt_start:])
                return finish("precision_stagnated" if evidence else "damping_limit", reason,
                              precision_evidence=evidence)
            gamma = gamma_next
            if restart is not None:
                restart['gamma'] = gamma
        else:
            records = history[attempt_start:]
            evidence = _precision_evidence(records)
            only_linear_failures = all(row["reason"] in (
                "CG_NOT_CONVERGED", "TRUE_RESIDUAL_FAILED", "NONFINITE_LINEAR_SOLVE",
                "ZERO_GRADIENT", "NON_DESCENT_DIRECTION") for row in records)
            status = "precision_stagnated" if evidence else "linear_failed" if only_linear_failures else "lm_retry_exhausted"
            return finish(status, history[-1]["reason"], precision_evidence=evidence)
