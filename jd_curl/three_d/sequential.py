"""T06/T13/T14 single Sequential driver for both Poisson PG and GGN LM.

Only the last inner accepted state is eligible; candidate re-evaluation and
two guarded OLD-reference reductions precede committing or refreshing controls.
"""
from dataclasses import asdict, dataclass, replace
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from ..energy_comparison import precision_stagnation_evidence
from ..types import InvalidTrial
from ..validation import InputValidationError, StateMismatchError, finite_real, require_finite_result
from .energy_comparison import stable_energy_reduction, stable_energy_reductions
from .linearization import build_linearization, energy_gradient, outer_gradient_vectors
from .problem import Problem3D, ParallelMemoryConfig, array_hash, make_self_reference, with_controls
from .residuals import AcceptedState3D, EnergyTerms3D, validate_state
from .work_control import emit, operation
from .solvers.common import (InnerResult3D, SolverCounters3D, gradient_norm_mass,
                             initial_failure_status, prepare_base, step_ratio, step_ratio_pair,
                             step_ratio_pair_observed)
from .solvers.poisson_gradient import PGConfig3D, solve_poisson_gradient
from .solvers.lm import LMConfig3D, solve_lm
from .validation import integer_at_least


@dataclass(frozen=True)
class SequentialConfig3D:
    max_iterations: int = 3
    eta0: float = .01
    accept_precision_stagnation: bool = False
    step_tol: float = 1e-12
    stall_steps: int = 5

    def __post_init__(self):
        object.__setattr__(self, "max_iterations", integer_at_least(self.max_iterations, "outer.max_iterations", 0))
        object.__setattr__(self, "stall_steps", integer_at_least(self.stall_steps, "outer.stall_steps"))
        object.__setattr__(self, "eta0", finite_real(self.eta0, "eta0", positive=True))
        object.__setattr__(self, "step_tol", finite_real(self.step_tol, "outer.step_tol"))
        if self.eta0 >= 1 or self.step_tol < 0 or not isinstance(self.accept_precision_stagnation, bool):
            raise InputValidationError("outer: require eta0<1, step_tol>=0 and explicit boolean precision policy")


@dataclass(frozen=True)
class SequentialReuseConfig3D(SequentialConfig3D):
    """Explicit evaluation-only optimizations; legacy config/schema unchanged."""
    reuse_initial_reference: bool = True
    joint_outer_gradients: bool = True
    lean_outer: bool = False        # E89 (P03/P08): compiled outer step ratio and equality test (bit-identical)

    def __post_init__(self):
        super().__post_init__()
        if any(not isinstance(getattr(self, name), bool)
               for name in ("reuse_initial_reference", "joint_outer_gradients", "lean_outer")):
            raise InputValidationError("evaluation reuse flags must be explicit booleans")


@dataclass(frozen=True)
class SequentialResult3D:
    success: bool
    status: str
    reason: str
    initial_state: AcceptedState3D | None
    final_state: AcceptedState3D | None
    best_state: AcceptedState3D | None
    final_problem: Problem3D
    history: tuple[dict, ...]
    GF_initial: float | None
    GF_final: float | None
    GF_threshold: float | None
    GD_final: float | None
    GE_final: float | None
    native_mse: float | None
    final_mse: float | None
    mse_ratio: float | None
    best_mse_ratio: float | None
    counts: dict
    elapsed_seconds: float
    last_commit_seconds: float
    best_commit_seconds: float
    resolved_config: dict


# E89 (P08, explicit, off by default): sum q in longdouble per grid object (the same NumPy sum of the
# same freshly built node weights, computed once); keyed by object identity with a weak reference.
_WEIGHT_SUMS = None


def enable_weight_sum_cache(enabled=True):
    global _WEIGHT_SUMS
    _WEIGHT_SUMS = {} if enabled else None


def _weight_sum(grid, q=None):
    memo = _WEIGHT_SUMS
    if memo is not None:
        entry = memo.get(id(grid))
        if entry is not None and entry[0]() is grid:
            return entry[1]
    value = np.sum(grid.node_weights() if q is None else q, dtype=np.longdouble)
    if memo is not None:
        import weakref
        memo[id(grid)] = (weakref.ref(grid), value)
    return value


def mse(state, grid, problem=None):
    """Reported MSE: 2 D / sum q. With T80 data weights (problem.data_weight set) the report stays UNWEIGHTED,
    sum q (W - F)^2 / sum q from the state's warped image, so ratios remain comparable across runs."""
    if problem is not None and getattr(problem, "data_weight", None) is not None:
        q = grid.node_weights()
        value = float(np.sum(q*(np.asarray(state.warped)-problem.fixed_image)**2, dtype=np.longdouble)
                      / _weight_sum(grid, q))
    else:
        value = float(2*np.longdouble(state.energy.image)/_weight_sum(grid))
    require_finite_result(value, "MSE")
    return value


def _same_theta(a, b, lean):
    """np.array_equal(a, b); E89 (P08): the compiled == test with an early exit under lean_outer."""
    if lean:
        from . import host_kernels
        if host_kernels.enabled(a.size) and host_kernels._flat_ok(a, b):
            return host_kernels.equal(a, b)
    return np.array_equal(a, b)


def precision_tail_valid(inner, solver):
    """Independent F24b tail audit; a boolean claim from an inner solver is insufficient."""
    if not inner.precision_evidence or not inner.history:
        return False
    last = inner.history[-1]["iteration"]
    tail = [r for r in inner.history if r["iteration"] == last]
    untruncated = all(r.get("untruncated", False) for r in tail)
    if solver == "lm":
        untruncated = untruncated and all(r.get("tau") == 1. for r in tail)
    return precision_stagnation_evidence([SimpleNamespace(**r) for r in tail], untruncated=untruncated)


def _initial_reference_matches(problem, state):
    """Only after original prepare_base; never trusts an inner candidate cache."""
    validate_state(problem, state)
    return bool(isinstance(state, AcceptedState3D)
                and state.certificate == problem.controls.certificate
                and array_hash(problem.controls.u_ref) == state.certificate.state_hash
                and (state.energy.proximity, state.energy.jd, state.energy.curl) == (0., 0., 0.))


def solve_sequential(problem, theta, **options):
    """See _solve_sequential. E103: a next-metric prestart (throughput-v21) never outlives the solve -- it is drained
    on every exit, a return or an exception (WorkBudgetExceeded included)."""
    try:
        return _solve_sequential(problem, theta, **options)
    finally:
        backend = options.get("backend")
        if getattr(backend, "metric_prestart", False):
            backend.metric_prestart_drain()


REFERENCE_MODES = {"fixed": "fixed-native-start-v1", "identity": "identity-v1"}     # T92 (D2-c / D2-c')


def _solve_sequential(problem, theta, *, solver="pg", inner_config=None, config=None, observer=None, backend=None,
                      mask_config=None, inner_observer=None, baseline_at_zero=False, budget=None, practical_stop=None,
                      weights_schedule=None, outer_gate=None, reference_mode=None):
    """CPU synthetic/array API; observer(problem,start,inner,row,candidate) is read-only.

    Histories contain scalars only, not per-round states or Gauss arrays.
    Initial/latest/best maps are retained; a watchdog must export via observer
    if it needs committed checkpoints across process termination.
    budget() is an optional monotone expiration predicate. Expiry is a work
    limit, never permission to commit a partial inner state or report success.
    """
    config = SequentialConfig3D() if config is None else config
    if not isinstance(config, SequentialConfig3D) or solver not in ("pg", "lm", "jd_qp"):
        raise InputValidationError("expected SequentialConfig3D and solver pg, lm or jd_qp")
    if not isinstance(baseline_at_zero, bool):
        raise InputValidationError("baseline_at_zero must be an explicit boolean")
    if budget is not None and (not callable(budget) or mask_config is not None):
        raise InputValidationError("budget must be callable and cannot be combined with mask_config")
    qp_memory = None
    if solver == 'jd_qp':
        from .solvers.sequential_qp import SequentialQPConfig3D, qp_memory as make_qp_memory, solve_sequential_qp
        if mask_config is not None:raise InputValidationError('jd_qp cannot be combined with mask_config')
        config_type, inner_solve = SequentialQPConfig3D, solve_sequential_qp
    else:
        config_type, inner_solve = (PGConfig3D, solve_poisson_gradient) if solver == "pg" else (LMConfig3D, solve_lm)
    inner_config = config_type() if inner_config is None else inner_config
    if not isinstance(inner_config, config_type):
        raise InputValidationError("inner_config does not match solver")
    if solver == 'jd_qp':qp_memory = make_qp_memory(inner_config)
    if mask_config is not None:
        from .solvers.working_set_driver import MaskConfig3D, solve_working_set, partial_candidate_valid
        if not isinstance(mask_config, MaskConfig3D) or backend is not None:
            raise InputValidationError("mask_config requires MaskConfig3D and CPU backend")
    started = perf_counter()
    history, counts = [], dict(outer_accepted=0, inner_calls=0, reference_certifications=0,
                               **{"inner_"+k: 0 for k in vars(SolverCounters3D())})
    reuse = isinstance(config, SequentialReuseConfig3D)
    lean = reuse and config.lean_outer      # E89
    evaluation = backend if getattr(backend, "evaluation_backend_id", None) is not None else None
    start_exact = current_exact = None                       # T90: float64 energies for the records when screening
    if reuse:
        counts.update(initial_reference_reuses=0, joint_outer_gradient_passes=0)
    if evaluation is not None:
        counts.update(device_outer_gradient_passes=0, cpu_outer_gradient_checks=0)
        if getattr(evaluation, "reuse_reference", False):
            counts["reference_reuses"] = 0
    outer_counts = SolverCounters3D()
    initial = current = best = None
    from .solvers.poisson_gradient import JDStepPGConfig3D
    from .solvers.jd_step import JDStepMemory
    jd_step_memory = JDStepMemory() if solver == 'pg' and isinstance(inner_config, JDStepPGConfig3D) else None
    from .solvers.poisson_gradient import TriggeredQPConfig3D
    if isinstance(inner_config,TriggeredQPConfig3D):
        from .solvers.qp_trigger import TriggerMemory
        jd_step_memory=TriggerMemory()
    G0 = GF = GD = GE = threshold = native = None
    last_commit = best_commit = 0.
    small_steps = 0
    resolved = dict(solver=solver, inner=asdict(inner_config), outer=asdict(config),
                    weights=asdict(problem.weights), memory=asdict(problem.memory),
                    certificate=asdict(problem.certificate_config), grid_id=problem.grid.grid_id,
                    backend=problem.backend_id, outer_objective="image_barrier", GF_atol=1e-8, GF_rtol=1e-6)

    if problem.barrier_activation is not None:
        resolved["barrier_activation"] = problem.barrier_activation
        resolved["barrier_model"] = "localized-c2-v1"
    if problem.barrier_location != "corners8":
        resolved["barrier_location"] = problem.barrier_location
    if mask_config is not None:
        resolved['working_set'] = asdict(mask_config)
        resolved['termination_policy'] = 'original_full_GF_only'
    if backend is not None:
        resolved["direction_backend"] = backend.backend_id
    if evaluation is not None:
        resolved["evaluation_backend"] = evaluation.evaluation_backend_id
    if baseline_at_zero:
        resolved["baseline_policy"] = "level_zero_displacement"
    if budget is not None:
        resolved["budget_policy"] = "safe_boundaries_no_late_outer_commit"
    if getattr(evaluation, "device_forcing", False):
        resolved["device_forcing"] = evaluation.device_forcing_record["policy"]     # E109
    if weights_schedule is not None:
        # T86 (S30-G2, path only): the geometry weights of the inner objective E by outer index; F, E/F, forcing, GF
        # and the certificate do not contain mu/nu. The weights of every outer are recorded in its history row.
        from .weights_schedule import MuSchedule
        if not isinstance(weights_schedule, MuSchedule) or mask_config is not None or solver != "pg":
            raise InputValidationError("weights_schedule must be a MuSchedule on the PG route without a mask")
        if weights_schedule.base.lam != problem.weights.lam or weights_schedule.base.kappa != problem.weights.kappa:
            raise InputValidationError("weights_schedule: lam and kappa must be those of the problem")
        resolved["weights_schedule"] = weights_schedule.record()
    if practical_stop is not None:
        # H17 (a change of the stopping rule, explicit option): never convergence; GF and its threshold unchanged
        from .practical_stop import PracticalStop
        if not isinstance(practical_stop, PracticalStop):
            raise InputValidationError("practical_stop must be a PracticalStop")
        resolved["practical_stop"] = practical_stop.record()
    if outer_gate is not None:
        # FX1 F4 (throughput-v35, path only): see outer_gate.py; never convergence
        from .outer_gate import OuterGate
        if not isinstance(outer_gate, OuterGate) or budget is None:
            raise InputValidationError("outer_gate must be an OuterGate and requires a budget")
        resolved["outer_gate"] = dict(policy=outer_gate.record()["policy"], window=outer_gate.window)
    if reference_mode is not None:
        # T92 (D2-c / D2-c', ablation switch, MODEL CHANGE): a fixed reference -- the start state ('fixed') or the
        # identity map u_ref = 0 ('identity') -- never refreshed; a commit needs only E's certified stable decrease
        # (F is compared and recorded, not required). GF, its threshold and the meaning of convergence are unchanged.
        if reference_mode not in REFERENCE_MODES or solver != "pg" or mask_config is not None \
                or weights_schedule is not None or practical_stop is not None:
            raise InputValidationError("reference_mode is 'fixed' or 'identity' on the PG route, without a mask, T86 or H17")

    def finish(status, reason):
        elapsed = perf_counter()-started
        counters = dict(counts, **{"outer_"+k: v for k, v in vars(outer_counts).items()})
        final_mse = None if current is None else mse(current, problem.grid, problem)
        ratio = None if native in (None, 0.) else final_mse/native
        best_ratio = None if native in (None, 0.) else mse(best, problem.grid, problem)/native
        return SequentialResult3D(status == "converged_regularized_gradient", status, reason,
                                  initial, current, best, problem, tuple(history), G0, GF, threshold,
                                  GD, GE, native, final_mse, ratio, best_ratio, counters, elapsed,
                                  last_commit, best_commit, resolved)

    norms_source = {}                       # E109 (device_forcing): where the last outer norms came from

    def norms(state, *, forcing_target=None):
        norms_source.clear()
        lin = build_linearization(problem, state)
        if evaluation is not None:
            outer_counts.gradients += 3
            counts["device_outer_gradient_passes"] += 1
            if getattr(evaluation, "lean_checks", False):
                # E95: the same norms; the backend knows which of these arrays are proven finite
                values = evaluation.outer_norms(lin, problem.grid.omega)
            else:
                values = tuple(gradient_norm_mass(g, problem.grid.omega)
                               for g in evaluation.outer_gradient_vectors(lin))
            near_threshold = threshold is not None and values[0] <= 1.01*threshold
            if near_threshold or (forcing_target is not None and values[2] <= 1.01*forcing_target):
                def cpu_values():
                    if getattr(evaluation, "fused_recheck", False):
                        # E108: the same three vectors bit for bit (one sampling, each tile's geometry once); counted
                        # once they exist (a failure ends the run with nonfinite_evaluation either way)
                        vectors = evaluation.cpu_outer_recheck(problem, state)
                        counts["cpu_outer_gradient_checks"] += 1
                        outer_counts.gradients += 3
                        return tuple(gradient_norm_mass(g, problem.grid.omega) for g in vectors)
                    cpu = evaluation.cpu_state(problem, state)
                    if isinstance(cpu, InvalidTrial):
                        raise FloatingPointError("CPU outer gradient recheck state invalid: "+cpu.reason)
                    counts["cpu_outer_gradient_checks"] += 1
                    outer_counts.gradients += 3
                    return tuple(gradient_norm_mass(g, problem.grid.omega)
                                 for g in outer_gradient_vectors(build_linearization(problem, cpu)))

                if not near_threshold and getattr(evaluation, "device_forcing", False):
                    # E109 (B14, rounding level): a recheck triggered by the forcing target alone, outside the 1e-6
                    # band, keeps the device norms (first uses per grid are checked against the CPU recheck)
                    values = evaluation.device_forcing_norms(problem, values, forcing_target, cpu_values)
                    norms_source["source"] = evaluation.device_forcing_last
                    return values
                values = cpu_values()
                norms_source["source"] = "cpu_threshold" if near_threshold else "cpu_forcing"
                return values
            norms_source["source"] = "device"
            return values
        if reuse and config.joint_outer_gradients:
            # gradients counts logical vector requests, not geometry passes.
            outer_counts.gradients += 3
            counts["joint_outer_gradient_passes"] += 1
            return tuple(gradient_norm_mass(g, problem.grid.omega) for g in outer_gradient_vectors(lin))
        values = []
        for term in ("outer", "image", "total"):
            outer_counts.gradients += 1
            values.append(gradient_norm_mass(energy_gradient(lin, term=term), problem.grid.omega))
        return tuple(values)

    def scheduled_weights():
        """T86: the weights of the next outer when they differ from the problem's, else None."""
        if weights_schedule is None:
            return None
        w = weights_schedule(counts["outer_accepted"])
        return None if w == problem.weights else w

    def refresh(state):
        with operation("reference_refresh"):
            scheduled = scheduled_weights()
            if scheduled is not None:
                # T86: a new inner objective (new weights -> new problem id); the reference is certified, never reused
                counts["reference_certifications"] += 1
                controls = make_self_reference(problem, state)
                refreshed = with_controls(replace(problem, weights=scheduled), controls)
                base = (prepare_base(refreshed, state.theta, outer_counts) if evaluation is None else
                        prepare_base(refreshed, state.theta, outer_counts, backend=evaluation))
                return refreshed, base
            if evaluation is not None and getattr(evaluation, "reuse_reference", False):
                rebound = evaluation.refresh_reference(problem, state)
                if rebound is not None:
                    counts["reference_reuses"] += 1
                    return rebound
            counts["reference_certifications"] += 1
            refreshed = with_controls(problem, make_self_reference(problem, state))
            base = (prepare_base(refreshed, state.theta, outer_counts) if evaluation is None else
                    prepare_base(refreshed, state.theta, outer_counts, backend=evaluation))
            return refreshed, base

    if reference_mode == "identity":
        # T92 (D2-c'): the reference controls are those of u = 0 (det(I) = 1 and curl 0 exactly); built before the start
        # state is evaluated, and never refreshed (the solver's initial refresh is skipped too)
        identity_state = (prepare_base(problem, np.zeros_like(theta), outer_counts) if evaluation is None else
                          prepare_base(problem, np.zeros_like(theta), outer_counts, backend=evaluation))
        if isinstance(identity_state, InvalidTrial):
            return finish("nonfinite_evaluation", "IDENTITY_REFERENCE_FAILED: "+identity_state.reason)
        counts["reference_certifications"] += 1
        problem = with_controls(problem, make_self_reference(problem, identity_state))
        del identity_state
    raw = (prepare_base(problem, theta, outer_counts) if evaluation is None else
           prepare_base(problem, theta, outer_counts, backend=evaluation))
    if isinstance(raw, InvalidTrial):
        return finish(initial_failure_status(raw), raw.reason)
    if reference_mode == "identity":
        current = raw
    elif reuse and config.reuse_initial_reference and _initial_reference_matches(problem, raw) and scheduled_weights() is None:
        current = raw
        counts["initial_reference_reuses"] += 1
    else:
        problem, current = refresh(raw)
    if isinstance(current, InvalidTrial):
        current = None
        return finish("nonfinite_evaluation", "INITIAL_REFERENCE_REFRESH_FAILED")
    initial = best = current
    if reference_mode is not None:
        resolved["reference_mode"] = dict(policy=REFERENCE_MODES[reference_mode], controls_id=problem.controls.controls_id)
    native = None if baseline_at_zero else mse(current, problem.grid, problem)
    try:
        GF, GD, GE = norms(current)
    except FloatingPointError:
        return finish("nonfinite_evaluation", "INITIAL_GRADIENT_FAILED")
    G0 = GF
    zero = current
    if baseline_at_zero:
        native = mse(current, problem.grid, problem)
    if baseline_at_zero and np.any(current.theta):
        # F=D+B is reference-independent. The probe's soft energy/GE are NOT
        # used as either the starting state or a stopping criterion.
        zero = (prepare_base(problem, np.zeros_like(current.theta), outer_counts) if evaluation is None else
                prepare_base(problem, np.zeros_like(current.theta), outer_counts, backend=evaluation))
        if isinstance(zero, InvalidTrial):
            G0 = native = None
            return finish("zero_baseline_failed", zero.reason)
        try:
            G0 = norms(zero)[0]
            native = mse(zero, problem.grid, problem)
        except FloatingPointError:
            G0 = native = None
            return finish("zero_baseline_failed", "ZERO_BASELINE_GRADIENT_FAILED")
    threshold = 1e-8+1e-6*G0
    if baseline_at_zero and evaluation is not None and zero is not current and G0 <= 1.01*threshold:
        try:
            G0 = norms(zero)[0]  # critical G0 must use the full CPU recheck
        except FloatingPointError:
            G0 = native = threshold = None
            return finish("zero_baseline_failed", "ZERO_BASELINE_CPU_CHECK_FAILED")
        threshold = 1e-8+1e-6*G0
    if evaluation is not None and GF <= 1.01*threshold:
        # Even an initially converged device estimate needs a true CPU check.
        try:
            GF, GD, GE = norms(current)
        except FloatingPointError:
            return finish("nonfinite_evaluation", "INITIAL_CPU_GRADIENT_CHECK_FAILED")
        if zero is current:
            G0 = GF
            threshold = 1e-8+1e-6*G0
    if baseline_at_zero:
        resolved["baseline_state_hash"] = zero.certificate.state_hash
    del zero
    emit("sequential_baseline_ready", grid_id=problem.grid.grid_id, native_mse=native,
         GF_initial=G0, GF_threshold=threshold, baseline_at_zero=baseline_at_zero)
    while True:
        if budget is not None and budget():
            return finish("outer_time_budget", "WALL_BUDGET; last formal state only, not convergence")
        if GF <= threshold:
            return finish("converged_regularized_gradient", "Feasible F=D+B gradient criterion, not global optimality")
        if small_steps >= config.stall_steps:
            return finish("outer_stagnated", "Small accepted steps with nonzero outer gradient")
        if counts["outer_accepted"] >= config.max_iterations:
            return finish("outer_max_iterations", "Outer budget exhausted; not convergence")
        if outer_gate is not None and outer_gate.stop():
            return finish("outer_time_budget", "WALL_BUDGET (FX1 F4): the next outer iteration cannot finish before the "
                                               "solve cutoff; last formal state only, not convergence")
        start = current
        if evaluation is not None and getattr(evaluation, "fp32_screen", False):
            # T90: the outer start's float64 energy for the records -- after a self-reference refresh the soft terms are
            # exact zeros and image/barrier/curvature are those of the committed state (its recorded float64 energy)
            if reference_mode is not None:
                # T92: no refresh -- the start is the committed candidate itself (its float64 energy, all six terms)
                start_exact = current_exact if current_exact is not None else evaluation.exact_energy(problem, start)
            else:
                start_exact = (EnergyTerms3D(current_exact.image, 0., 0., 0., current_exact.barrier, current_exact.curvature)
                               if current_exact is not None else evaluation.exact_energy(problem, start))
            if current_exact is None:
                current_exact = start_exact
        if reference_mode is not None:
            if problem.controls.controls_id != resolved["reference_mode"]["controls_id"]:
                raise StateMismatchError("T92: the fixed reference changed")
        elif (start.energy.proximity, start.energy.jd, start.energy.curl) != (0., 0., 0.):
            raise StateMismatchError("outer round requires exact matched self reference")
        ref_hash = array_hash(problem.controls.u_ref)
        eta = config.eta0 if G0 == 0 else config.eta0*np.sqrt(min(GF, G0)/G0)
        target = float(eta*GF)
        counts["inner_calls"] += 1
        execution = {} if backend is None else {"backend": backend}
        if jd_step_memory is not None:
            execution['jd_step_memory'] = jd_step_memory
        if qp_memory is not None:
            execution['working_set'] = qp_memory
        if inner_observer is not None:
            execution["iteration_observer"] = inner_observer
        if budget is not None:
            execution["budget"] = budget
        if mask_config is not None and mask_config.enabled:
            inner = solve_working_set(problem, start.theta, target=target, config=inner_config,
                                     solver=solver, mask_config=mask_config,
                                     deadline=started+mask_config.solver_seconds)
        else:
            if mask_config is not None:
                execution['budget'] = lambda: perf_counter() >= started+mask_config.solver_seconds
            if getattr(evaluation, "gpu_edge_hull", False):
                # E101 (T84): trial certificates may use the GPU edge-hull margins inside the inner solve only
                with evaluation.gpu_certificate_scope():
                    inner = inner_solve(problem, start.theta, target=target, config=inner_config, **execution)
            else:
                inner = inner_solve(problem, start.theta, target=target, config=inner_config, **execution)
        try:
            if array_hash(problem.controls.u_ref) != ref_hash:
                raise StateMismatchError("inner solver mutated frozen outer reference")
            if not isinstance(inner, InnerResult3D):
                raise InputValidationError("inner solver must return InnerResult3D")
        except BaseException:
            if getattr(evaluation, "commit_recheck", False):
                evaluation.commit_recheck_discard()    # E101: an early plan D recheck never outlives an exception
            raise
        for key, value in vars(inner.counters).items():
            counts["inner_"+key] += value
        candidate = recheck = mismatch = None
        total = fixed = None
        GFa = GDa = GEa = relative = None
        accepted, precision_ok, numeric_failure = False, False, False
        expired = budget is not None and (inner.status == "work_budget_exhausted" or budget())
        reason = "INNER_INITIALIZATION_FAILED"
        checked = inner.state is not None and not expired
        if not checked and getattr(evaluation, "commit_recheck", False):
            evaluation.commit_recheck_discard()       # E101: an early plan D recheck no candidate will use
        if checked:
            try:
                validate_state(problem, inner.state)
                candidate = (prepare_base(problem, inner.state.theta, outer_counts) if evaluation is None else
                             prepare_base(problem, inner.state.theta, outer_counts, backend=evaluation))
            except BaseException:
                if getattr(evaluation, "commit_recheck", False):
                    evaluation.commit_recheck_discard()   # E101 (review): e.g. a deadline in the candidate's promotion
                raise
            if isinstance(candidate, InvalidTrial):
                reason = candidate.reason
                candidate = None
                if getattr(evaluation, "commit_recheck", False):
                    evaluation.commit_recheck_discard()
            else:
                if getattr(evaluation, "commit_recheck", False):
                    # E101 (T84 plan D): the candidate's full CPU certificate, overlapping the outer checks below
                    recheck = evaluation.commit_recheck_start(problem, candidate)
                try:
                    GFa, GDa, GEa = norms(candidate, forcing_target=target)
                    if lean:
                        # E89 (P03): the same quotient without the difference array.
                        if getattr(evaluation, "step_extrema", False):
                            # E97: the same value; the two maxima are kept for the outer comparison's microstep test
                            relative = step_ratio_pair_observed(candidate.theta, start.theta, problem.grid,
                                lambda: step_ratio(candidate.theta-start.theta, start.theta, problem.grid),
                                lambda top, theta_max: evaluation.note_step_extrema(candidate.theta, start.theta,
                                                                                    top, theta_max))
                        else:
                            relative = step_ratio_pair(candidate.theta, start.theta, problem.grid,
                                lambda: step_ratio(candidate.theta-start.theta, start.theta, problem.grid))
                    else:
                        relative = step_ratio(candidate.theta-start.theta, start.theta, problem.grid)
                    if evaluation is not None:
                        total, fixed = evaluation.stable_energy_reductions(problem, start, candidate)
                    elif isinstance(problem.memory, ParallelMemoryConfig):
                        total, fixed = stable_energy_reductions(problem, start, candidate)
                    else:
                        total = stable_energy_reduction(problem, start, candidate)
                        fixed = stable_energy_reduction(problem, start, candidate, outer_only=True)
                    # T92 (D2-c/c'): with a fixed reference only E's certified decrease is required (F recorded)
                    cert = total.certified > 0 and (fixed.certified > 0 or reference_mode is not None)
                    precision_ok = bool(config.accept_precision_stagnation and inner.status == "precision_stagnated"
                                        and precision_tail_valid(inner, solver))
                    eligible = inner.status in ("converged_gradient", "max_iterations", "stagnated") or precision_ok
                    if mask_config is not None and mask_config.enabled and inner.status == 'inner_subspace_stationary':
                        eligible = partial_candidate_valid(problem, inner, target=target)
                        outer_counts.gradients += 1
                    accepted = bool(eligible and cert and not _same_theta(start.theta, candidate.theta, lean))
                    reason = ("ACCEPTED" if accepted else "INNER_FAILED_NOT_COMMITTED" if not eligible
                              else "NO_PROGRESS" if _same_theta(start.theta, candidate.theta, lean)
                              else "DESCENT_CERTIFICATE_FAILED")
                except FloatingPointError:
                    numeric_failure, reason = True, "NONFINITE_OUTER_CHECK"
                except BaseException:
                    if recheck is not None:          # E101 (review): never leave the plan D proof running unobserved
                        evaluation.commit_recheck_drain(recheck)
                    raise
        # Recheck AFTER all expensive certificates/gradients/comparisons (E101: the plan D join below comes later; a
        # join past the deadline cannot commit -- the budget is checked again right before the commit). A
        # completed inner solve is not permission to commit past the deadline.
        expired = expired or (budget is not None and budget())
        # An actual failure takes precedence over a coincident clock expiry;
        # otherwise a failed coarse layer could incorrectly advance.
        time_limited = (expired and not numeric_failure and (not checked or accepted)
                        and (inner.status in ("converged_gradient", "max_iterations", "stagnated",
                                              "work_budget_exhausted") or precision_ok))
        if time_limited:
            accepted, reason = False, "WALL_BUDGET_NOT_COMMITTED"
        elif expired and not checked:
            reason = "INNER_FAILED_NOT_COMMITTED"
        if recheck is not None:
            # E101 (T84 plan D, D-a): a CPU certificate that differs from the candidate's switches the GPU part off; if
            # it does not certify, the candidate is never committed and the run ends (gpu_certificate_mismatch).
            candidate, verdict = evaluation.commit_recheck_join(recheck)
            if verdict == "cpu_not_certified":
                accepted, time_limited = False, False
                mismatch = reason = "GPU_CERTIFICATE_MISMATCH"
        valid = candidate is not None
        # T90 (S30-G5): with the float32 screening the state's energy is a screening value; the recorded candidate
        # energies are the float64 ones (exact_energy), so every record and the audit keep the float64 meaning
        screened = evaluation is not None and getattr(evaluation, "fp32_screen", False)
        ce = (evaluation.exact_energy(problem, candidate) if (valid and screened) else candidate.energy) if valid else None
        se = start_exact if (screened and start_exact is not None) else start.energy
        row = dict(outer_i=len(history), controls_id=problem.controls.controls_id, problem_id=problem.problem_id,
                   start_eval_id=start.eval_id, candidate_eval_id=candidate.eval_id if valid else None,
                   E_start=se.total, E_candidate=ce.total if valid else None,
                   F_start=se.outer, F_candidate=ce.outer if valid else None,
                   D_start=se.image, D_candidate=ce.image if valid else None,
                   B_start=se.barrier, B_candidate=ce.barrier if valid else None,
                   R_start=se.curvature, R_candidate=ce.curvature if valid else None,   # T76
                   penalty_proximity=ce.proximity if valid else None,
                   penalty_jd=ce.jd if valid else None, penalty_curl=ce.curl if valid else None,
                   actual_reduction_E=total.actual if total else None, resolution_E=total.resolution if total else None,
                   actual_reduction_F=fixed.actual if fixed else None, resolution_F=fixed.resolution if fixed else None,
                   GF_initial=G0, GF_threshold=threshold, GF_before=GF, GF_candidate=GFa, GD_candidate=GDa,
                   GE_candidate=GEa, inner_reported_GE=inner.gradient_norm, forcing_eta=float(eta), forcing_target=target,
                   forcing_passed=bool(valid and GEa is not None and GEa <= target), inner_status=inner.status, inner_reason=inner.reason,
                   inner_accepted_steps=inner.accepted_steps, inner_history=inner.history,
                   inner_counts=asdict(inner.counters), inner_elapsed_seconds=inner.elapsed_seconds,
                   outer_accepted=accepted, precision_evidence=precision_ok, reason=reason, step_ratio=relative,
                   candidate_min_jd_lower_bound=candidate.certificate.min_jd_lower_bound if valid else None,
                   candidate_sampled_min_jd=candidate.certificate.sampled_min_jd if valid else None,
                   candidate_mse_ratio=(mse(candidate, problem.grid, problem)/native) if valid and native != 0 else None,
                   elapsed_seconds=perf_counter()-started)
        if mask_config is not None and mask_config.enabled:
            row['working_set'] = inner.diagnostics
        if reference_mode is not None:
            # T92: the mode, and whether F's stable decrease was certified too (recorded, not required)
            row['reference_mode'] = REFERENCE_MODES[reference_mode]
            row['F_decrease_certified'] = bool(fixed is not None and fixed.certified > 0)
        if weights_schedule is not None:
            row['weights'] = asdict(problem.weights)                 # T86: this outer's inner weights
        if getattr(evaluation, "device_forcing", False):
            row["outer_norms_source"] = norms_source.get("source") if GFa is not None else None   # E109
        if evaluation is not None:
            row['comparison_diagnostics'] = dict(evaluation.last_comparison) if total is not None else None
        if accepted and budget is not None and budget():
            # Metadata includes native-volume MSE reductions; those also cost
            # time. Check immediately before the formal state transition.
            accepted, time_limited, reason = False, True, "WALL_BUDGET_NOT_COMMITTED"
            row.update(outer_accepted=False, reason=reason)
        if accepted:
            counts["outer_accepted"] += 1
            if recheck is not None:
                evaluation.gpu_certificate_record["rechecked_commits"] += 1     # E101: plan D before this commit
            current, GF, GD, GE = candidate, GFa, GDa, GEa
            last_commit = row["elapsed_seconds"]
            if current.energy.image < best.energy.image:
                best, best_commit = current, last_commit
            small_steps = small_steps+1 if relative <= config.step_tol else 0
        if accepted and screened:
            current_exact = ce                                   # T90: the committed state's float64 energy
        committed = current_exact if (screened and current_exact is not None) else current.energy
        row.update(committed_mse_ratio=None if native == 0 else mse(current, problem.grid, problem)/native,
                   committed_GF=GF, committed_E=committed.total, committed_F=committed.outer,
                   counts_after=dict(counts))
        stop = None
        if practical_stop is not None and accepted and row["committed_mse_ratio"] is not None:
            stop = row["practical_stop"] = practical_stop.after_commit(row["elapsed_seconds"], row["committed_mse_ratio"])
        history.append(row)
        if observer is not None:
            observer(problem, start, inner, dict(row), candidate)
        if not accepted:
            if mismatch is not None:
                return finish("gpu_certificate_mismatch", "the CPU re-certification of the outer candidate failed "
                                                           "(T84 plan D); last formal state only, not convergence")
            if time_limited:
                return finish("outer_time_budget", reason)
            if numeric_failure:
                return finish("nonfinite_evaluation", reason)
            if inner.status == "precision_stagnated":
                return finish("outer_precision_stagnated", reason)
            if inner.status in ("domain_blocked", "feasibility_blocked", "feasibility_uncertain"):
                return finish("outer_"+inner.status, inner.reason)
            if inner.status not in ("converged_gradient", "max_iterations", "stagnated"):
                return finish("outer_inner_failed", inner.status+": "+inner.reason)
            return finish("outer_no_progress" if reason == "NO_PROGRESS" else "outer_candidate_rejected", reason)
        # Gradient and budget checks take priority over an unnecessary refresh.
        if (GF <= threshold or counts["outer_accepted"] >= config.max_iterations or small_steps >= config.stall_steps
                or (budget is not None and budget())):
            continue
        if reference_mode is not None:
            continue          # T92: the reference stays fixed; the next outer starts from the committed candidate
        if stop is not None and stop["triggered"]:
            # H17: after convergence, iteration, stall and budget checks; the run ends on this formal commit
            return finish("practical_stop", f"H17 projected ratio gain {stop['projected_gain']:.6g} <= epsilon "
                                            f"{stop['epsilon']:g} after formal commit {stop['commit_index']} at "
                                            f"{stop['native_seconds']:.3f} s; a budget-type end, not convergence")
        refreshed_problem, refreshed_state = refresh(current)
        if isinstance(refreshed_state, InvalidTrial):
            return finish("reference_refresh_failed", refreshed_state.reason)
        problem, current = refreshed_problem, refreshed_state
        GE = GF  # exact self-reference change gradients vanish; fixed outer scale is NOT reset.
