"""F22 sequential Q1 self-reference: fixed images, old-reference certificate.

The default F22 branch has no absolute barrier; explicit F23 barrier mode uses
D+B for its certificate and outer stopping. Neither mode composes/resamples
images, averages references or schedules weights.
"""
from dataclasses import asdict, dataclass, replace
from time import perf_counter

import numpy as np

from .boundary import pack_initial_displacement
from .energy_comparison import stable_energy_reduction, precision_stagnation_evidence
from .linearization import image_gradient, image_barrier_gradient, _check_state
from .residuals import build_problem, evaluate_state, with_q1_self_reference
from .solvers import solve_inner
from .solvers.common import (InitializationFailure, StopConfig, SolverResult,
                             energy_evaluations, gradient_norm_mass, step_metrics)
from .types import EvalState, InvalidTrial
from .validation import InputValidationError, require_finite_result, finite_real


@dataclass(frozen=True)
class SequentialResult:
    success: bool
    status: str
    message: str
    initial_state: EvalState | None
    final_state: EvalState | None
    history: tuple[dict, ...]
    initial_image_gradient_norm: float | None
    final_image_gradient_norm: float | None
    image_gradient_threshold: float | None
    counts: dict
    resolved_config: dict
    wall_seconds: float
    outer_objective: str = 'image'
    initial_objective_gradient_norm: float | None = None
    final_objective_gradient_norm: float | None = None
    objective_gradient_threshold: float | None = None


def outer_energy(state):
    """Fixed outer target: D for F22, D+B for F23; not the inner total E_i."""
    return state.energy.image+state.energy.barrier


def assess_outer_candidate(problem, start, candidate):
    """Re-evaluate with OLD references, even if the inner result claims success."""
    trial = evaluate_state(problem, candidate, eval_id=start.eval_id+1)
    if isinstance(trial, InvalidTrial):
        return trial, False, trial.reason
    if np.array_equal(trial.z, start.z):
        return trial, False, "NO_PROGRESS"
    if not (trial.energy.total <= outer_energy(start) and outer_energy(trial) < outer_energy(start)):
        return trial, False, "DESCENT_CERTIFICATE_FAILED"
    return trial, True, "ACCEPTED"


def _blocked(inner):
    """Evidence of a hard-domain obstruction, NOT a constrained KKT test."""
    if inner.status in ('feasibility_blocked', 'domain_blocked'):
        return True
    if inner.status not in ('stagnated', 'line_search_failed', 'lm_retry_exhausted', 'damping_limit'):
        return False
    if not inner.history:
        return False
    row = inner.history[-1]
    return (row.reason in ('FEASIBLE_STEP_BOUND', 'JACOBIAN_BOUND_VIOLATION', 'INVALID_DOMAIN') or
            (row.feasible_step_limit is not None and row.feasible_step_limit < 1e-12))


def solve_sequential(case, weights, inner_config, outer: StopConfig = StopConfig(max_iterations=100),
                     *, observer=None, mode='sequential_self_reference', accept_precision_stagnation=False,
                     inner_forcing_eta0=None):
    """Full existing inner solve per round; only outer references are rebuilt.

    observer(problem, start, inner_result, row, audited_candidate) is read-only
    evidence export, called once per attempted outer round. Arrays are immutable.
    Budgets/stagnation may yield a descending candidate, true failures do not.
    """
    if not isinstance(outer, StopConfig):
        raise InputValidationError('outer: expected StopConfig')
    if mode not in ('sequential_self_reference', 'sequential_self_reference_barrier'):
        raise InputValidationError('Unknown sequential mode')
    barrier_mode = mode == 'sequential_self_reference_barrier'
    if not isinstance(accept_precision_stagnation, bool):
        raise InputValidationError('accept_precision_stagnation must be bool')
    if accept_precision_stagnation and (not barrier_mode or inner_config.energy_comparison != 'stable'):
        raise InputValidationError('precision recovery requires explicit F23 stable comparison')
    if inner_forcing_eta0 is not None:
        inner_forcing_eta0 = finite_real(inner_forcing_eta0, 'inner_forcing_eta0', positive=True)
        if inner_forcing_eta0 >= 1 or not barrier_mode or inner_config.energy_comparison != 'stable':
            raise InputValidationError('forcing requires 0 < eta0 < 1 and explicit F23 stable comparison')
    if barrier_mode:
        if (inner_config.geometry_discretization != 'q1' or inner_config.jacobian_barrier is None
                or inner_config.jacobian_feasibility is not None):
            raise InputValidationError('sequential barrier mode requires q1 and an absolute barrier with its hard domain')
    elif (inner_config.geometry_discretization != 'q1' or inner_config.jacobian_barrier is not None
          or inner_config.jacobian_feasibility is None):
        raise InputValidationError('sequential requires q1, hard feasibility and no absolute barrier')
    started = perf_counter()
    template = build_problem(case, weights, geometry_discretization='q1',
                             jacobian_feasibility=inner_config.jacobian_feasibility,
                             jacobian_barrier=inner_config.jacobian_barrier)
    resolved = dict(mode=mode, weights=asdict(weights),
                    inner_solver=type(inner_config).__name__, inner=asdict(inner_config), outer=asdict(outer))
    if accept_precision_stagnation:
        resolved['accept_precision_stagnation'] = True
    if inner_forcing_eta0 is not None:
        resolved['inner_forcing_eta0'] = inner_forcing_eta0
    count_names = ('n_accepted', 'n_trial_evaluations', 'n_domain_rejections', 'n_energy_evaluations',
                   'n_gradient_evaluations', 'total_cg_iterations', 'n_hvp',
                   'n_preconditioner_applications', 'n_feasibility_checks', 'n_jacobian_rejections')
    counts = dict(outer_accepted=0, inner_calls=0, outer_state_evaluations=0,
                  outer_energy_evaluations=0, image_gradient_evaluations=0,
                  **{'inner_'+key: 0 for key in count_names})
    history = []
    initial = current = None
    G0 = G = threshold = None
    GF0 = GF = objective_threshold = None
    success_status = 'converged_regularized_gradient' if barrier_mode else 'converged_image_gradient'
    objective_name = 'image_barrier' if barrier_mode else 'image'
    if barrier_mode:
        counts['objective_gradient_evaluations'] = 0

    def finish(status, message):
        return SequentialResult(status == success_status, status, message,
            initial, current, tuple(history), G0, G, threshold, dict(counts), resolved, perf_counter()-started,
            objective_name, GF0, GF, objective_threshold)

    def evaluate(problem, z):
        counts['outer_state_evaluations'] += 1
        state = evaluate_state(problem, z, eval_id=0)
        counts['outer_energy_evaluations'] += energy_evaluations(state)
        return state

    def image_G(problem, state):
        counts['image_gradient_evaluations'] += 1
        return gradient_norm_mass(image_gradient(problem, state), case.grid.omega)

    def objective_G(problem, state, image_norm):
        if not barrier_mode:
            return image_norm
        counts['objective_gradient_evaluations'] += 1
        return gradient_norm_mass(image_barrier_gradient(problem, state), case.grid.omega)

    raw = evaluate(template, pack_initial_displacement(case.u0, case.grid))
    if isinstance(raw, InvalidTrial):
        return finish('invalid_initial_state', raw.reason+': '+raw.message)
    problem = with_q1_self_reference(template, raw, include_barrier=barrier_mode)
    initial = current = evaluate(problem, raw.z)
    G0 = G = image_G(problem, current)
    threshold = outer.grad_atol+outer.grad_rtol*G0
    GF0 = GF = objective_G(problem, current, G)
    objective_threshold = outer.grad_atol+outer.grad_rtol*GF0
    require_finite_result(threshold, 'outer image gradient threshold')
    require_finite_result(objective_threshold, 'outer objective gradient threshold')
    small_steps = 0
    while True:
        if GF <= objective_threshold:
            return finish(success_status, 'Feasible interior '+objective_name+' gradient criterion; not global optimality')
        if small_steps >= outer.stagnation_window:
            return finish('outer_stagnated', 'Small accepted outer steps without '+objective_name+' gradient convergence')
        if counts['outer_accepted'] >= outer.max_iterations:
            return finish('outer_max_iterations', 'Outer budget exhausted without '+objective_name+' gradient convergence')
        start = current
        # Only change penalties reset. The absolute barrier must remain intact.
        if (start.energy.proximity, start.energy.jd, start.energy.curl) != (0., 0., 0.):
            raise InputValidationError('self-reference base penalties must be exactly zero')
        controls = problem.controls
        snapshot = tuple(a.copy() for a in (controls.u_ref, controls.f_gauss, controls.g_gauss))
        counts['inner_calls'] += 1
        round_config = inner_config
        if inner_forcing_eta0 is not None:
            eta = inner_forcing_eta0*np.sqrt(min(GF, GF0)/GF0)
            target = eta*GF
            round_config = replace(inner_config, stop=replace(inner_config.stop, grad_atol=target, grad_rtol=0.))
        inner = solve_inner(problem, start.z, round_config)
        for a, b in zip(snapshot, (controls.u_ref, controls.f_gauss, controls.g_gauss)):
            if not np.array_equal(a, b):
                raise InputValidationError('inner solver mutated frozen references')
        candidate, accepted, reason = None, False, 'INNER_INITIALIZATION_FAILED'
        precision_certificate = None
        GD_after, GF_after, relative = None, None, None
        if isinstance(inner, SolverResult):
            _check_state(problem, inner.final_state)
            for name in count_names:
                counts['inner_'+name] += getattr(inner, name)
            counts['outer_state_evaluations'] += 1
            candidate, certificate, reason = assess_outer_candidate(problem, start, inner.final_z)
            counts['outer_energy_evaluations'] += energy_evaluations(candidate)
            if isinstance(candidate, EvalState):
                GD_after = image_G(problem, candidate)
                GF_after = objective_G(problem, candidate, GD_after)
                _, relative = step_metrics(candidate.z-start.z, start.z, case.grid)
            accepted = certificate and inner.status in ('converged_gradient', 'max_iterations', 'stagnated')
            if accept_precision_stagnation and inner.status == 'precision_stagnated' and isinstance(candidate, EvalState):
                tail = [r for r in inner.history if r.iteration == inner.history[-1].iteration] if inner.history else []
                full = (not tail or tail[0].pg_step_proposal is None or tail[0].feasible_step_limit is None
                        or tail[0].feasible_step_limit >= tail[0].pg_step_proposal)
                if tail and tail[0].solver == 'lm':
                    full = full and tail[0].tau in (None, 1.)
                evidence = precision_stagnation_evidence(tail, untruncated=full)
                total = stable_energy_reduction(problem, start, candidate)
                fixed = stable_energy_reduction(problem, start, candidate, outer_only=True)
                accepted = bool(certificate and evidence and total.certified > 0 and fixed.certified > 0)
                precision_certificate = dict(evidence=evidence, total_reduction=total.actual,
                    total_resolution=total.resolution, fixed_reduction=fixed.actual, fixed_resolution=fixed.resolution)
            if not accepted and certificate:
                reason = 'INNER_FAILED_NOT_COMMITTED'
        valid = isinstance(candidate, EvalState)
        row = dict(outer_i=len(history), controls_id=controls.controls_id, problem_id=problem.problem_id,
            D_before=start.energy.image, D_after=candidate.energy.image if valid else None,
            E_i_start=start.energy.total, E_i_candidate=candidate.energy.total if valid else None,
            prox_change=candidate.energy.proximity if valid else None,
            jd_change=candidate.energy.jd if valid else None, curl_change=candidate.energy.curl if valid else None,
            certificate_margin=outer_energy(start)-candidate.energy.total if valid else None,
            GD_before=G, GD_after=GD_after, GD_initial=G0, image_gradient_threshold=threshold,
            inner_status=inner.status, inner_success=isinstance(inner, SolverResult) and inner.success,
            inner_final_gradient_norm=inner.final_gradient_norm_mass if isinstance(inner, SolverResult) else None,
            inner_gradient_threshold=inner.gradient_threshold if isinstance(inner, SolverResult) else None,
            outer_accepted=bool(accepted), reason=reason, relative_step=relative,
            min_corner_jd=float(candidate.corner_geometry.J.min()) if valid else None,
            nonpositive_cell_fraction=float(np.mean(np.any(candidate.corner_geometry.J <= 0, axis=-1))) if valid else None,
            counts_after=dict(counts), wall_seconds=perf_counter()-started)
        if barrier_mode:
            row.update(outer_objective=objective_name, F_before=outer_energy(start),
                F_after=outer_energy(candidate) if valid else None, B_before=start.energy.barrier,
                B_after=candidate.energy.barrier if valid else None, GF_before=GF, GF_after=GF_after,
                GF_initial=GF0, objective_gradient_threshold=objective_threshold,
                image_increased=bool(valid and candidate.energy.image > start.energy.image))
        if accept_precision_stagnation:
            row.update(precision_certificate=precision_certificate,
                       precision_recovered=bool(accepted and inner.status == 'precision_stagnated'))
        if inner_forcing_eta0 is not None:
            row.update(inner_forcing_eta=float(eta), inner_forcing_target=float(target),
                baseline_inner_gradient_threshold=inner_config.stop.grad_atol+inner_config.stop.grad_rtol*GF,
                inner_forcing_passed=bool(isinstance(inner, SolverResult) and inner.final_gradient_norm_mass <= target))
        if accepted:
            counts['outer_accepted'] += 1
            row['counts_after'] = dict(counts)
        history.append(row)
        if observer is not None:
            observer(problem, start, inner, row, candidate)
        if not accepted:
            if inner.status == 'precision_stagnated':
                return finish('outer_precision_stagnated', 'No certified outer progress from precision-limited inner state')
            if isinstance(inner, SolverResult) and _blocked(inner):
                return finish('outer_feasibility_blocked', inner.status+': '+inner.message)
            if reason == 'NO_PROGRESS' and inner.status in ('converged_gradient', 'max_iterations', 'stagnated'):
                return finish('outer_no_progress', 'Inner returned the same state while '+objective_name+' gradient remains nonzero')
            if inner.status not in ('converged_gradient', 'max_iterations', 'stagnated'):
                return finish('outer_inner_failed', inner.status+': '+inner.message)
            return finish('outer_candidate_rejected', reason)
        current, G, GF = candidate, GD_after, GF_after
        if GF <= objective_threshold:
            return finish(success_status, 'Feasible interior '+objective_name+' gradient criterion; not global optimality')
        if _blocked(inner):
            return finish('outer_feasibility_blocked', 'Descending inner stagnation candidate committed, but hard domain blocks further progress')
        small_steps = small_steps+1 if relative <= outer.relative_step_tol else 0
        if counts['outer_accepted'] >= outer.max_iterations or small_steps >= outer.stagnation_window:
            continue
        problem = with_q1_self_reference(problem, current, include_barrier=barrier_mode)
        current = evaluate(problem, current.z)
        # G is unchanged by reference rebuilding; no new initial gradient scale.
