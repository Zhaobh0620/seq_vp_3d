"""T11 fixed-metric Poisson PG/BB2, certified trials and stable Armijo."""
from dataclasses import dataclass
from copy import deepcopy
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from scipy.linalg import norm

from ...energy_comparison import precision_stagnation_evidence
from ...types import InvalidTrial
from ...validation import InputValidationError, finite_real, readonly_copy, require_finite_result
from ..energy_comparison import stable_energy_reduction
from ..linearization import build_linearization, energy_gradient
from ..poisson import build_poisson_inverse
from ..residuals import evaluate_trial, promote_trial
from ..validation import integer_at_least
from .common import (SolverCounters3D, finish_inner, gradient_norm_mass,
                     initial_failure_status, prepare_base, step_ratio, step_ratio_pair, validate_target,
                     step_ratio_fast_applies, step_ratio_from_extrema)


@dataclass(frozen=True)
class PGConfig3D:
    max_iterations: int = 500
    metric: str = "q1"
    initial_step: float = 1.
    step_min: float = 1e-8
    step_max: float = 10.
    max_backtracks: int = 25
    c1: float = 1e-4
    step_tol: float = 1e-12
    stall_steps: int = 5

    def __post_init__(self):
        for name, minimum in (("max_iterations", 0), ("max_backtracks", 0), ("stall_steps", 1)):
            object.__setattr__(self, name, integer_at_least(getattr(self, name), name, minimum))
        for name in ("initial_step", "step_min", "step_max", "c1"):
            object.__setattr__(self, name, finite_real(getattr(self, name), name, positive=True))
        object.__setattr__(self, "step_tol", finite_real(self.step_tol, "step_tol"))
        if self.step_min > self.step_max or self.c1 >= 1 or self.step_tol < 0:
            raise InputValidationError("PG step bounds, c1, or step_tol invalid")
        if self.metric not in ("q1", "seven_point"):
            raise InputValidationError("PG metric: expected q1 or seven_point")


@dataclass(frozen=True)
class JDStepPGConfig3D(PGConfig3D):
    """Explicit point predictor; all legacy feasibility/acceptance tests remain."""
    jd_step_safety: float = .9

    def __post_init__(self):
        super().__post_init__()
        value = finite_real(self.jd_step_safety, "jd step safety", positive=True)
        if value >= 1:
            raise InputValidationError("JD step safety must be <1")
        object.__setattr__(self, 'jd_step_safety', value)


@dataclass(frozen=True)
class TriggeredQPConfig3D(JDStepPGConfig3D):
    """E46 PG mainline, bounded geometric trigger; original stopping rules."""
    def __post_init__(self):
        super().__post_init__()
        if self.metric!='q1':raise InputValidationError('triggered QP requires Q1 metric')


@dataclass(frozen=True)
class LocalRepairPGConfig3D(JDStepPGConfig3D):
    """T47 opt-in local increment repair, with original certified fallback."""
    max_local_repairs: int = 9

    def __post_init__(self):
        super().__post_init__()
        value = integer_at_least(self.max_local_repairs, 'max local repairs')
        if value > 9:
            raise InputValidationError('max local repairs must be <=9')
        object.__setattr__(self, 'max_local_repairs', value)


@dataclass(frozen=True)
class PredictiveLocalRepairPGConfig3D(LocalRepairPGConfig3D):
    """T48: repair near the JD predictor, not an arbitrarily large raw step."""
    local_prediction_factor: float = 2.

    def __post_init__(self):
        super().__post_init__()
        value = finite_real(self.local_prediction_factor, 'local prediction factor', positive=True)
        if value < 1:
            raise InputValidationError('local prediction factor must be >=1')
        object.__setattr__(self, 'local_prediction_factor', value)


METRIC_POLICIES = ("image_shift_v1", "ggn_equilibrated_v1", "image_shift_ggn_equilibrated_v1")


@dataclass(frozen=True)
class CurvaturePGConfig3D(JDStepPGConfig3D):
    """T59 (E49/A1): curvature-corrected fixed PG metric, frozen per reference.

    Only the SPD metric P in p=-P^-1 g changes; F/E, certificates, Armijo,
    E/F acceptance, forcing and GF are those of JDStepPGConfig3D.
    """
    metric_policy: str = "image_shift_v1"
    shift_factor: float = 1.
    _POLICIES = METRIC_POLICIES  # class constant, not a dataclass field

    def __post_init__(self):
        super().__post_init__()
        if self.metric_policy not in self._POLICIES:
            raise InputValidationError("unknown PG metric policy")
        if self.metric != "q1":
            raise InputValidationError("T59 metric variants use the Q1 metric")
        object.__setattr__(self, "shift_factor", finite_real(self.shift_factor, "shift factor", positive=True))


T69_POLICY = "image_shift_ggn_equilibrated_jd_affine_v2"   # E59: T62 + triggered face/interior extension
T73_POLICY = "image_shift_ggn_equilibrated_jd_affine_v3"   # E66: T69 extension always on (no trigger)
T62_POLICIES = ("image_shift_ggn_equilibrated_jd_affine_v1", T69_POLICY, T73_POLICY)
T70_POLICY = "legacy_q1_jd_affine_triggered_v1"   # E62: legacy Q1; after an undetermined certificate R M^-1 R
T74_POLICY = "legacy_q1_jd_affine_always_v1"      # E70: legacy Q1 with R M^-1 R always (T70 without trigger)
RESOLUTION_EXIT_POLICY = "next-tau-slope-at-most-15eps-Ebase-v1"
BB2_GUARD_POLICY = "first-bb2-step-ggn-model-cap-1.5-v1"   # T72 (E64)
BB2_GUARD_FACTOR = 1.5
FIRST_TRIAL_CAP_POLICY = "max-node-voxel-displacement-cap-v1"   # T81 (E84)
FIRST_TRIAL_SCOPES = ("initial", "all")                         # T81: first step of each inner solve, or every step
FIRST_TRIAL_CAP_RANGE = (1e-6, 1e3)                             # T81 (review): c in level voxels; keeps c/nu finite and normal
FIRST_TRIAL_FLOOR_FACTOR = 2.**20                               # T81 (review): no cap below 2^20 x the T63 floor 15 eps E
MOMENTUM_POLICY = "outer-increment-first-direction-v1"          # T85 (E111): first inner direction = last outer increment
from .pullback import PULLBACK_MIN_SLOPE_RATIO                 # T87 (S30-G2)


@dataclass(frozen=True)
class ConservativePGConfig3D(CurvaturePGConfig3D):
    """E53: T62 JD-affine A1 metric option and T63 resolution exit.

    'legacy_q1' keeps the E34 Q1 Poisson metric. F/E, certificates, Armijo,
    E/F acceptance, forcing and GF are those of JDStepPGConfig3D.
    """
    metric_policy: str = "legacy_q1"
    jd_affine_activation: float | None = None   # T62/T69/T70 only; None -> .25 for those policies
    resolution_exit: bool = False
    jd_step_global: bool = False                # T66 (E55): global closed27 cubic step prediction
    bb2_ggn_guard: bool = False                 # T72 (E64): GGN-model cap of the first BB2 step
    fast_host: bool = False                     # E76: bit-identical host-side shortcuts (no new arithmetic)
    compiled_extension: bool = False            # E81: compiled T69/T73 face/interior estimator (bit-identical)
    fused_steps: bool = False                   # E89 (P03): compiled fused host step algebra (bit-identical)
    metric_host: bool = False                   # E89 (P07): lean host work of the metric rebuild (bit-identical)
    lean_checks: bool = False                   # E95: no rescans of proven-finite arrays, lazy direction digest (bit-identical)
    first_trial_cap: float | None = None        # T81 (E84): c in level voxels; trial tau <= c/max_i |p_i|_h (path only)
    first_trial_scope: str = "initial"          # T81: "initial" (rule == initial) or "all" (every step proposal)
    momentum_first: bool = False                # T85 (E111): first inner step along the last outer increment (path only)
    pullback: bool = False                      # T87 (S30-G2): direction pulled back to the current map, p~ = p(x + u) (path only)
    pullback_min_slope_ratio: float = .5        # T87: p~ replaces p only when g.p~ < 0 and |g.p~| >= ratio |g.p|
    _POLICIES = ("legacy_q1",)+METRIC_POLICIES+T62_POLICIES+(T70_POLICY, T74_POLICY)

    def __post_init__(self):
        super().__post_init__()
        from .jd_affine import JD_AFFINE_FLOOR, JD_AFFINE_ACTIVATION
        if self.metric_policy in T62_POLICIES+(T70_POLICY, T74_POLICY):
            value = JD_AFFINE_ACTIVATION if self.jd_affine_activation is None else self.jd_affine_activation
            value = finite_real(value, "JD affine activation", positive=True)
            if not JD_AFFINE_FLOOR < value <= 1:
                raise InputValidationError("JD affine activation must be in (1e-12, 1]")
            object.__setattr__(self, "jd_affine_activation", value)
        elif self.jd_affine_activation is not None:
            # Never record a parameter that has no effect on the run.
            raise InputValidationError("jd_affine_activation applies to T62/T70 metric policies only")
        if not isinstance(self.resolution_exit, bool):
            raise InputValidationError("resolution_exit must be boolean")
        if not isinstance(self.jd_step_global, bool):
            raise InputValidationError("jd_step_global must be boolean")
        if not isinstance(self.bb2_ggn_guard, bool):
            raise InputValidationError("bb2_ggn_guard must be boolean")
        if not isinstance(self.fast_host, bool):
            raise InputValidationError("fast_host must be boolean")
        if not isinstance(self.compiled_extension, bool):
            raise InputValidationError("compiled_extension must be boolean")
        if not isinstance(self.fused_steps, bool) or (self.fused_steps and not self.fast_host):
            raise InputValidationError("fused_steps must be boolean and requires fast_host")
        if not isinstance(self.metric_host, bool) or (self.metric_host and not self.fused_steps):
            raise InputValidationError("metric_host must be boolean and requires fused_steps")
        if not isinstance(self.lean_checks, bool) or (self.lean_checks and not self.metric_host):
            raise InputValidationError("lean_checks must be boolean and requires metric_host")
        if self.first_trial_cap is not None:
            if isinstance(self.first_trial_cap, (bool, np.bool_, str, bytes)):
                raise InputValidationError("first_trial_cap must be a real number")
            value = finite_real(self.first_trial_cap, "first trial cap", positive=True)
            if not FIRST_TRIAL_CAP_RANGE[0] <= value <= FIRST_TRIAL_CAP_RANGE[1]:
                raise InputValidationError("first_trial_cap must be in [1e-6, 1e3] level voxels")
            object.__setattr__(self, "first_trial_cap", value)
        if self.first_trial_scope not in FIRST_TRIAL_SCOPES:
            raise InputValidationError("first_trial_scope must be 'initial' or 'all'")
        if self.first_trial_cap is None and self.first_trial_scope != "initial":
            # Never record a parameter that has no effect on the run.
            raise InputValidationError("first_trial_scope applies only with first_trial_cap")
        if not isinstance(self.momentum_first, bool):
            raise InputValidationError("momentum_first must be boolean")
        if not isinstance(self.pullback, bool):
            raise InputValidationError("pullback must be boolean")
        if self.pullback and self.momentum_first:
            raise InputValidationError("pullback and momentum_first are exclusive")
        ratio = self.pullback_min_slope_ratio
        if isinstance(ratio, (bool, str, bytes)) or not isinstance(ratio, (int, float, np.floating)):
            raise InputValidationError("pullback_min_slope_ratio must be a real number")
        ratio = finite_real(ratio, "pullback min slope ratio", positive=True)
        if not ratio <= 1.:
            raise InputValidationError("pullback_min_slope_ratio must be in (0, 1]")
        object.__setattr__(self, "pullback_min_slope_ratio", ratio)
        if not self.pullback and self.pullback_min_slope_ratio != PULLBACK_MIN_SLOPE_RATIO:
            raise InputValidationError("pullback_min_slope_ratio applies only with pullback")   # never record an idle parameter


METRIC_ABLATION_MODES = ("diagonal", "identity")       # T92 (D2-d)


@dataclass(frozen=True)
class MetricAblationPGConfig3D(ConservativePGConfig3D):
    """T92 (D2-d, ablation switch only; docs/04 T92): the native level's metric inverse replaced.
    'diagonal': P^-1 = S diag(M)^-1 S with the T62-family scale S (GGN equilibration and JD affine damping kept, no
    Poisson coupling, no DST); 'identity': P^-1 = I (no metric build; the first trial needs a T81 cap). Everything else
    as ConservativePGConfig3D. Built only by the native-metric-ablation switch."""
    metric_inverse: str = "diagonal"

    def __post_init__(self):
        super().__post_init__()
        if self.metric_inverse not in METRIC_ABLATION_MODES:
            raise InputValidationError("metric_inverse must be 'diagonal' or 'identity'")
        if self.metric_inverse == "diagonal" and self.metric_policy not in T62_POLICIES:
            raise InputValidationError("the diagonal metric ablation keeps the T62-family scale S (T62/T69/T73 policies)")
        if self.metric_inverse == "identity" and self.first_trial_cap is None:
            raise InputValidationError("the identity metric ablation needs a T81 first-trial cap")
        if self.momentum_first or self.pullback:
            raise InputValidationError("the metric ablation is one switch at a time (no T85 or T87)")


def max_node_voxel_displacement(direction, grid):
    """T81 (E84): max over free nodes of |p_i|_h = sqrt(sum_d (p_{d,i}/h_d)^2), in level voxels.

    direction is component-major (3, free nodes) with components xyz (T01), h = grid.spacing (xyz).
    The compiled scan returns the NumPy reference value bit for bit (same per-node IEEE operations,
    order-independent max), see fast_checks.max_node_sq."""
    from .. import fast_checks
    if fast_checks.enabled(direction.size):
        m = fast_checks.max_node_sq(direction, grid.spacing)
    else:
        m = fast_checks.max_node_sq_reference(direction, grid.spacing)
    require_finite_result(m, "T81 maximum node displacement")
    return float(np.sqrt(m))


_NODE_QUOTIENTS = {}   # E89 (P07): grid object -> node_weights()[1:-1,1:-1,1:-1]/omega (used only with metric_host)


def _node_quotient(grid):
    import weakref
    entry = _NODE_QUOTIENTS.get(id(grid))
    if entry is not None and entry[0]() is grid:
        return entry[1]
    q = grid.node_weights()[1:-1, 1:-1, 1:-1]/grid.omega
    q.flags.writeable = False
    if len(_NODE_QUOTIENTS) > 16:
        for k in [k for k, (ref, _) in _NODE_QUOTIENTS.items() if ref() is None]:
            del _NODE_QUOTIENTS[k]
    _NODE_QUOTIENTS[id(grid)] = (weakref.ref(grid), q)
    return q


def image_curvature_shift(problem, base, factor, *, cached=False):
    """T59 alpha = factor * mean over free nodes of (q/omega)|b|^2/3. cached (E89 P07): the same q array,
    built once per grid object by the same expression."""
    grid = problem.grid
    b = base.image_grad[1:-1, 1:-1, 1:-1]
    q = _node_quotient(grid) if cached else grid.node_weights()[1:-1, 1:-1, 1:-1]/grid.omega
    with np.errstate(over="raise", invalid="raise"):
        alpha = factor*float(np.mean(q*np.sum(b*b, axis=-1)))/3
    require_finite_result(alpha, "T59 image curvature shift")
    return alpha


def metric_diagonal(grid, a, b, metric):
    """Constant diagonal m of the scalar metric, times omega (T59/T12)."""
    inv_h2 = sum(1/h**2 for h in grid.spacing)
    return grid.omega*(a+b*(8/9)*inv_h2 if metric == "q1" else a+2*b*inv_h2)


class EquilibratedInverse:
    """P^-1 = S M^-1 S with fixed positive diagonal S (T59 variant 2/3)."""
    def __init__(self, inner, scale):
        self.inner, self.scale = inner, scale

    def apply(self, rhs):
        return self.scale*self.inner.apply(self.scale*rhs)


class ScalarInverse:
    """T92 (D2-d diagonal): diag(M)^-1 for the constant diagonal m of the scalar metric (metric_diagonal): rhs / m."""
    def __init__(self, m):
        self.m = m

    def apply(self, rhs):
        return rhs/self.m


class IdentityInverse:
    """T92 (D2-d identity): P^-1 = I (a fresh copy, so the caller owns the result)."""
    def apply(self, rhs):
        return np.array(rhs, dtype=np.float64, copy=True)


def fused_direction(inverse, gradient):
    """E89 (P03): -inverse.apply(gradient) bit for bit. For P^-1 = S M^-1 S the pre-scale s*g and the
    post-scale -(s*x) are compiled passes into fresh arrays (otherwise only the negation is). Any
    nonfinite intermediate replays the original NumPy expression under the caller's errstate, so the
    same exception (or the same nonfinite values) results; the inner M^-1 is applied once either way."""
    from .. import host_kernels
    if not (host_kernels.enabled(gradient.size) and host_kernels._flat_ok(gradient)):
        return -inverse.apply(gradient)
    if isinstance(inverse, EquilibratedInverse):
        s = inverse.scale
        if not (isinstance(s, np.ndarray) and s.shape == gradient.shape and host_kernels._flat_ok(gradient, s)):
            return -inverse.apply(gradient)
        pre, finite = host_kernels.scaled(s, gradient)
        if not finite:
            return -inverse.apply(gradient)       # the original s*g raises (or yields) the same
        x = inverse.inner.apply(pre)
        if not (isinstance(x, np.ndarray) and x.shape == gradient.shape and host_kernels._flat_ok(x, s)):
            return -(s*x)
        out, finite = host_kernels.neg_scaled(s, x)
        return out if finite else -(s*x)
    x = inverse.apply(gradient)
    if not (isinstance(x, np.ndarray) and x.shape == gradient.shape and host_kernels._flat_ok(x)):
        return -x
    out, finite = host_kernels.negated(x)
    return out if finite else -x


def fused_direction_checked(inverse, gradient, device_scaling=False, between=None):
    """E95 (lean_checks only; E96: fused_direction keeps its own E94 body for v4-v13):
    (fused_direction(inverse, gradient), proven finite). The flag is True only when a compiled pass
    (or the device pass) established that every value of the returned direction is finite.
    device_scaling: for P^-1 = S M^-1 S with a device Poisson that offers apply_equilibrated, both scalings run
    on the device (single IEEE multiplications and a negation, as the compiled passes); a nonfinite
    intermediate there falls back to the host path below, which then behaves exactly as before.
    between: host work (reading none of these values) run exactly once -- while the device computes when the
    device path runs, otherwise first. Every path enters exactly the 'poisson' operations v13 enters."""
    from .. import host_kernels
    prepared = None
    if (device_scaling and isinstance(inverse, EquilibratedInverse) and hasattr(inverse.inner, "prepare_equilibrated")
            and host_kernels.enabled(gradient.size) and host_kernels._flat_ok(gradient)
            and isinstance(inverse.scale, np.ndarray) and inverse.scale.shape == gradient.shape
            and host_kernels._flat_ok(gradient, inverse.scale)):
        prepared = inverse.inner.prepare_equilibrated(inverse.scale, gradient)
    if prepared is not None:
        result = inverse.inner.apply_equilibrated(prepared, between=between)
        if not isinstance(result[0], str):
            return result
        x, s = result[1], inverse.scale
        # v13's tail after its own inner.apply: the compiled negated product, else the NumPy expression
        if not (isinstance(x, np.ndarray) and x.shape == gradient.shape and host_kernels._flat_ok(x, s)):
            return -(s*x), False
        out, finite = host_kernels.neg_scaled(s, x)
        return (out, True) if finite else (-(s*x), False)
    if between is not None:
        between()
    if not (host_kernels.enabled(gradient.size) and host_kernels._flat_ok(gradient)):
        return -inverse.apply(gradient), False
    if isinstance(inverse, EquilibratedInverse):
        s = inverse.scale
        if not (isinstance(s, np.ndarray) and s.shape == gradient.shape and host_kernels._flat_ok(gradient, s)):
            return -inverse.apply(gradient), False
        pre, finite = host_kernels.scaled(s, gradient)
        if not finite:
            return -inverse.apply(gradient), False       # the original s*g raises (or yields) the same
        x = inverse.inner.apply(pre)
        if not (isinstance(x, np.ndarray) and x.shape == gradient.shape and host_kernels._flat_ok(x, s)):
            return -(s*x), False
        out, finite = host_kernels.neg_scaled(s, x)
        return (out, True) if finite else (-(s*x), False)
    x = inverse.apply(gradient)
    if not (isinstance(x, np.ndarray) and x.shape == gradient.shape and host_kernels._flat_ok(x)):
        return -x, False
    out, finite = host_kernels.negated(x)
    return (out, True) if finite else (-x, False)


def pg_metric_inverse(problem, base, lin, config, backend, memory=None):
    """Legacy M for non-T59 configs (unchanged); T59/T62/T69 variants; T70 = legacy or triggered R M^-1 R.
    E100: a node-measure first part the build did not consume (it returned or raised before node_min_jd) is dropped."""
    if not getattr(backend, "estimator_overlap", False):
        return _pg_metric_inverse(problem, base, lin, config, backend, memory)
    try:
        return _pg_metric_inverse(problem, base, lin, config, backend, memory)
    finally:
        backend.discard_node_prepared()
        if getattr(backend, "metric_prestart", False):
            backend.metric_prestart_finish()          # E103: the taken prestart counted used or dropped


def _pg_metric_inverse(problem, base, lin, config, backend, memory=None):
    lam, bp = problem.weights.lam, problem.weights.mu/2+problem.weights.nu/2
    info = None
    a = lam
    policy = config.metric_policy if isinstance(config, CurvaturePGConfig3D) else None
    ablation = config.metric_inverse if isinstance(config, MetricAblationPGConfig3D) else None   # T92 (D2-d)
    if ablation is not None and backend is not None and hasattr(backend, "prefetch_detach"):
        backend.prefetch_detach()          # T92: no device Poisson in this build for E105 to prefetch into
    if ablation == "identity":
        return IdentityInverse(), dict(policy=policy, metric_inverse="identity")
    host = isinstance(config, ConservativePGConfig3D) and config.metric_host     # E89 (P07)
    lean = isinstance(config, ConservativePGConfig3D) and config.lean_checks     # E95
    extrema = None                                                                # E95: (min, max) of the final scale
    ggn = policy in ("ggn_equilibrated_v1", "image_shift_ggn_equilibrated_v1")+T62_POLICIES
    pending = prepared = early = None
    if getattr(backend, "metric_prestart", False):
        # E103: this build's host parts may have been computed from this very base at the previous inner step budget
        early = backend.metric_prestart_take(problem, base, *_prestart_plan(problem, config, backend, memory))
    if host and ggn and backend is not None and hasattr(backend, "diagonal_launch"):
        if policy in T62_POLICIES and getattr(backend, "estimator_overlap", False):
            # E100: the node measure's device part runs now (before the diagonal is queued) and its host estimator
            # runs while the diagonal kernel does; node_min_jd below uses them at its own place, in its own order.
            activation = _extension_activation(problem, policy, config, memory)
            if activation is not None:
                if not (backend.metric_prestart_ready(problem, early, "node") and backend.metric_prestart_adopt(lin, early)):
                    prepared = backend.node_min_prepare(lin, activation, bool(getattr(config, "compiled_extension", False)))
        # E89 (P07): queue the device diagonal first; the shift and Poisson build below run on the host
        # meanwhile. It is joined (same checks, exact download) where the original computed it.
        pending = backend.diagonal_launch(lin)
        if prepared is not None:
            prepared.estimate()
            if early is not None and early.prepared is not None:
                backend.metric_prestart_check(problem, early, "node", prepared)    # E103 first use (v20 values used)
    if policy in ("legacy_q1", T70_POLICY, T74_POLICY):
        info = dict(policy=policy, a=a)
    if policy in ("image_shift_v1", "image_shift_ggn_equilibrated_v1")+T62_POLICIES:
        if early is not None and backend.metric_prestart_ready(problem, early, "alpha"):
            alpha = backend.metric_prestart_alpha(early)          # E103: the same function on the same arrays
        elif host and getattr(backend, "fused_shift", False):
            # E105: the same alpha (NumPy's reduction order, compiled; first uses compared bit for bit)
            alpha = backend.shift_alpha(problem, base, config.shift_factor,
                                        lambda: image_curvature_shift(problem, base, config.shift_factor, cached=True))
        else:
            alpha = image_curvature_shift(problem, base, config.shift_factor, cached=host)
            if early is not None and early.alpha_key is not None:
                backend.metric_prestart_check(problem, early, "alpha", alpha)
        a = lam+alpha
        info = dict(policy=policy, shift_alpha=alpha, shift_factor=config.shift_factor, a=a)
    try:
        # T92 (D2-d diagonal): no Poisson inverse is built (diag(M)^-1 = 1/m below)
        inverse = None if ablation == "diagonal" else build_poisson_inverse(problem.grid, a=a, b=bp, metric=config.metric)
    except InputValidationError as exc:
        # An unrepresentable shifted metric is a numerical failure of this direction.
        raise FloatingPointError("T59 metric not representable: "+str(exc)) from exc
    if backend is not None and pending is None and ablation is None:
        inverse = backend.poisson(inverse)
    if policy in ("ggn_equilibrated_v1", "image_shift_ggn_equilibrated_v1")+T62_POLICIES:
        if pending is not None:
            # E89 (P07): the device checked these exact values (finite, > 0) before the download, so the
            # host repeats of both checks are dropped on this path only.
            diagonal = pending.result()
            if ablation is None:
                inverse = backend.poisson(inverse)
        elif backend is not None and hasattr(backend, "diagonal"):
            diagonal = backend.device.download(backend.diagonal(lin))
        else:
            from ..linearization import barrier_ggn_diagonal
            diagonal = barrier_ggn_diagonal(lin)
        if pending is None:
            require_finite_result(diagonal, "T59 GGN diagonal")
            if np.any(diagonal <= 0):
                raise FloatingPointError("T59 GGN diagonal must be positive")
        m = metric_diagonal(problem.grid, a, bp, config.metric)
        scale = _host_quotient_scale(m, diagonal) if host else None
        scale_proven = scale is not None and lean             # E95: the compiled pass returned it only if finite
        if scale is None:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                scale = np.sqrt(m)/np.sqrt(diagonal)
        if not scale_proven:
            require_finite_result(scale, "T59 equilibration scale")
        affine = None
        if policy in T62_POLICIES:
            from .jd_affine import affine_factors, node_min_jd, JD_AFFINE_FLOOR
            if problem.barrier_location != "closed27_uniform" or problem.barrier_activation is None:
                raise InputValidationError("T62 requires the closed27 local barrier route")
            extension = None
            if policy == T73_POLICY or (policy == T69_POLICY and memory is not None and getattr(memory, "uncertain", 0) > 0):
                # T69: triggered (sticky per level grid) by an undetermined certificate on this level;
                # T73: always on (the damping acts before a valley can reach the certificate's limit).
                from .jd_affine import EXTENSION_POLICY
                extension = dict(policy=EXTENSION_POLICY, triggered=True if policy == T69_POLICY else "always",
                                 uncertain_seen=int(getattr(memory, "uncertain", 0) or 0), candidates=0, lowered_cells=0)
            kwargs = {} if extension is None else dict(extension_activation=config.jd_affine_activation, stats=extension)
            if extension is not None and getattr(config, "compiled_extension", False):
                kwargs["compiled"] = True          # E81: same values, compiled estimator
            tick = perf_counter()
            node = (backend.node_min_jd(lin, **kwargs) if backend is not None and hasattr(backend, "node_min_jd")
                    else node_min_jd(problem, base, **kwargs))
            if extension is not None:
                extension["node_measure_seconds"] = perf_counter()-tick   # T62 part included
            tail = _lean_affine_tail(problem, node, scale, config.jd_affine_activation) if (lean and scale_proven) else None
            if tail is not None:
                # E95: the same r, scale*r and statistics from two compiled passes (no full-size temporaries)
                scale, node_jd_min, damped, factor_min, extrema = tail
                affine = dict(activation=config.jd_affine_activation, floor=JD_AFFINE_FLOOR,
                              node_jd_min=node_jd_min, damped_node_fraction=damped, factor_min=factor_min)
            else:
                try:
                    r = affine_factors(node, problem.grid, config.jd_affine_activation)
                except InputValidationError as exc:
                    raise FloatingPointError("T62 node JD invalid: "+str(exc)) from exc
                product = _host_product(scale, r) if host else None
                scale = scale*r if product is None else product
                require_finite_result(scale, "T62 scaled equilibration")
                if not np.all(scale > 0):
                    raise FloatingPointError("T62 scale must be positive")
                inner = node[1:-1, 1:-1, 1:-1]
                affine = dict(activation=config.jd_affine_activation, floor=JD_AFFINE_FLOOR,
                              node_jd_min=float(inner.min()), damped_node_fraction=float(np.mean(inner < config.jd_affine_activation)),
                              factor_min=float(r.min()))
            if policy in (T69_POLICY, T73_POLICY):
                affine["extension"] = extension if extension is not None else dict(triggered=False)
        if ablation == "diagonal":
            inverse = ScalarInverse(m)             # T92 (D2-d): P^-1 = S diag(M)^-1 S
        inverse = EquilibratedInverse(inverse, scale)
        if host:
            # E89 (P07): the diagnostic median is computed on a helper thread (np.median of the same frozen
            # array) and written into info before the first attempt row records it (resolve_metric_info).
            scale.flags.writeable = False
            if extrema is not None:
                # E95: the lean tail's (min, max) of these very values
                info = dict(info or dict(policy=policy, a=a), mass_diagonal=m,
                            scale_min=extrema[0], scale_median=_PendingMedian(scale), scale_max=extrema[1])
            else:
                info = dict(info or dict(policy=policy, a=a), mass_diagonal=m,
                            scale_min=float(scale.min()), scale_median=_PendingMedian(scale), scale_max=float(scale.max()))
        else:
            info = dict(info or dict(policy=policy, a=a), mass_diagonal=m,
                        scale_min=float(scale.min()), scale_median=float(np.median(scale)), scale_max=float(scale.max()))
        if affine is not None:
            info['jd_affine'] = affine
        if ablation is not None:
            info['metric_inverse'] = ablation      # T92 (D2-d)
    if policy in (T70_POLICY, T74_POLICY):
        # T70 (E62): the legacy Q1 metric until this level has seen an undetermined certificate;
        # then P^-1 = R M^-1 R with the T69 node measure (face/interior extension included).
        # T74 (E70): the same R M^-1 R from the first metric build on (no trigger).
        if problem.barrier_location != "closed27_uniform" or problem.barrier_activation is None:
            raise InputValidationError("T70 requires the closed27 local barrier route")
        if policy == T74_POLICY or (memory is not None and getattr(memory, "uncertain", 0) > 0):
            from .jd_affine import EXTENSION_POLICY, JD_AFFINE_FLOOR, affine_factors, node_min_jd
            extension = dict(policy=EXTENSION_POLICY, triggered=True if policy == T70_POLICY else "always",
                             uncertain_seen=int(getattr(memory, "uncertain", 0) or 0), candidates=0, lowered_cells=0)
            kwargs = dict(extension_activation=config.jd_affine_activation, stats=extension)
            if getattr(config, "compiled_extension", False):
                kwargs["compiled"] = True          # E81: same values, compiled estimator
            tick = perf_counter()
            node = (backend.node_min_jd(lin, **kwargs) if backend is not None and hasattr(backend, "node_min_jd")
                    else node_min_jd(problem, base, **kwargs))
            extension["node_measure_seconds"] = perf_counter()-tick
            try:
                r = affine_factors(node, problem.grid, config.jd_affine_activation)
            except InputValidationError as exc:
                raise FloatingPointError("T70 node JD invalid: "+str(exc)) from exc
            if not np.all(r > 0):
                raise FloatingPointError("T70 factors must be positive")
            inverse = EquilibratedInverse(inverse, r)
            inner = node[1:-1, 1:-1, 1:-1]
            info['jd_affine'] = dict(activation=config.jd_affine_activation, floor=JD_AFFINE_FLOOR,
                                     node_jd_min=float(inner.min()),
                                     damped_node_fraction=float(np.mean(inner < config.jd_affine_activation)),
                                     factor_min=float(r.min()), extension=extension)
        else:
            info['jd_affine'] = dict(extension=dict(triggered=False))
    if getattr(problem, "active_weights", None) is not None:
        inverse, info = _active_set_inverse(problem, inverse, info, host)       # T89 (S30-G4)
    return inverse, info


def _active_set_inverse(problem, inverse, info, host):
    """T89 (S30-G4): P^-1 = S' M^-1 S' with S' = w S (w the problem's active weights on the free degrees of freedom):
    the existing equilibration scale multiplied by w (E89 compiled product when the host passes are on, else the
    NumPy expression), or w alone when the policy has no scale. The direction is then exactly 0 (signed) on the
    frozen nodes (w = 0), so they never move; every value of F is unchanged."""
    from ..active_set import free_weights, metric_record
    w = free_weights(problem)
    if isinstance(inverse, EquilibratedInverse):
        scale = inverse.scale
        if not (isinstance(scale, np.ndarray) and scale.shape == w.shape):
            raise FloatingPointError("T89 active set: unexpected equilibration scale layout")
        product = _host_product(scale, w) if host else None
        if product is None:
            with np.errstate(over="raise", invalid="raise"):
                product = scale*w
        require_finite_result(product, "T89 active-set scale")
        product.flags.writeable = False
        inverse = EquilibratedInverse(inverse.inner, product)
    else:
        inverse = EquilibratedInverse(inverse, w)
    info = dict(info or {})
    info["active_set"] = metric_record(problem)
    return inverse, info


def _extension_activation(problem, policy, config, memory):
    """E100: the extension activation node_min_jd will receive in pg_metric_inverse (T73 always, T69 once this level
    has seen an undetermined certificate), or None (no estimator; the closed27 check below decides as before)."""
    if problem.barrier_location != "closed27_uniform" or problem.barrier_activation is None:
        return None
    if policy == T73_POLICY or (policy == T69_POLICY and memory is not None and getattr(memory, "uncertain", 0) > 0):
        return config.jd_affine_activation
    return None


ALPHA_POLICIES = ("image_shift_v1", "image_shift_ggn_equilibrated_v1")+T62_POLICIES   # the T59 shift alpha (E103)


def _prestart_plan(problem, config, backend, memory):
    """E103: what the next _pg_metric_inverse will compute on the host from the base alone -- alpha (key: shift_factor,
    cached) and the E100 estimator first part (key: activation, compiled), with the same conditions; (None, None)
    otherwise."""
    policy = config.metric_policy if isinstance(config, CurvaturePGConfig3D) else None
    host = isinstance(config, ConservativePGConfig3D) and config.metric_host
    ggn = policy in ("ggn_equilibrated_v1", "image_shift_ggn_equilibrated_v1")+T62_POLICIES
    alpha_key = (config.shift_factor, bool(host)) if policy in ALPHA_POLICIES else None
    node = None
    if (host and ggn and hasattr(backend, "diagonal_launch") and policy in T62_POLICIES
            and getattr(backend, "estimator_overlap", False)):
        activation = _extension_activation(problem, policy, config, memory)
        if activation is not None:
            node = (activation, bool(getattr(config, "compiled_extension", False)))
    return alpha_key, node


def metric_prestart(problem, base, lin, config, backend, memory):
    """E103: start the next metric build's host parts for this base (the outer candidate) in the background."""
    alpha_key, node = _prestart_plan(problem, config, backend, memory)
    alpha_fn = None
    if alpha_key is not None:
        alpha_fn = lambda: image_curvature_shift(problem, base, alpha_key[0], cached=alpha_key[1])
    return backend.metric_prestart_start(problem, base, lin, alpha_fn, alpha_key, node)


def _host_quotient_scale(m, diagonal):
    """E89 (P07): np.sqrt(m)/np.sqrt(diagonal) by a compiled pass, or None (then the original expression
    runs, raising under its errstate exactly as before)."""
    from .. import host_kernels
    if host_kernels.enabled(diagonal.size) and host_kernels._flat_ok(diagonal):
        out, finite = host_kernels.quotient_scale(np.sqrt(m), diagonal)
        if finite:
            return out
    return None


def _lean_affine_tail(problem, node, scale, activation):
    """E95: (scale*r, node_jd_min, damped_node_fraction, factor_min, (min, max) of scale*r) of the T62 tail by two
    compiled passes, or None -- then the original code runs and raises (or not) exactly as before.
    r = affine_factors(node, grid, activation); the statistics are those of the original record."""
    from .. import host_kernels
    from .jd_affine import JD_AFFINE_FLOOR
    grid = problem.grid
    if not (isinstance(node, np.ndarray) and node.dtype == np.float64 and node.flags.c_contiguous
            and node.shape == grid.shape and host_kernels.enabled(scale.size) and host_kernels._flat_ok(scale)
            and scale.size == 3*int(np.prod([n-2 for n in grid.shape]))):
        return None
    if not (isinstance(activation, float) and np.isfinite(activation) and JD_AFFINE_FLOOR < activation <= 1):
        return None                                   # affine_factors' own validation decides
    r, inner_ok, r_ok, inner_min, r_min, below = host_kernels.affine_pass(node, JD_AFFINE_FLOOR, activation)
    if not (inner_ok and r_ok) or inner_min == 0.:
        return None              # (review E95: the sign of a zero minimum is NumPy's reduction order's to pick)
    out, finite, positive, low, high = host_kernels.broadcast_product(scale, r)
    if not (finite and positive):
        return None
    return out, float(inner_min), float(below)/float(r.size), float(r_min), (float(low), float(high))


def _host_product(scale, r):
    """E89 (P07): scale*r by a compiled pass, or None (the original expression then runs)."""
    from .. import host_kernels
    if (host_kernels.enabled(scale.size) and host_kernels._flat_ok(scale, r)
            and scale.shape == r.shape):
        out, finite = host_kernels.product(scale, r)
        if finite:
            return out
    return None


class _PendingMedian:
    """E89 (P07): float(np.median(array)) of a frozen array, computed on a helper thread."""
    _POOL = None

    def __init__(self, array):
        from concurrent.futures import ThreadPoolExecutor
        if _PendingMedian._POOL is None:
            _PendingMedian._POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jd-curl-median")
        self._future = _PendingMedian._POOL.submit(lambda: float(np.median(array)))

    def result(self):
        return self._future.result()


def resolve_metric_info(info):
    """E89 (P07): replace a pending median by its value (in place; the dict is shared by the solve's rows)."""
    if isinstance(info, dict) and isinstance(info.get("scale_median"), _PendingMedian):
        info["scale_median"] = info["scale_median"].result()
    return info


def _global_prediction(problem, base, lin, direction, limit, predicted, info, config, backend):
    """T66: min(historical prediction, safety * global closed27 first root); never a certificate."""
    from .global_jd_step import POLICY, global_root_cpu, summarize
    started = perf_counter()
    info = dict(info)
    try:
        if backend is not None and hasattr(backend, "global_jd_root"):
            root_s, invalid, points = backend.global_jd_root(lin, direction, limit)
            found = summarize(np.array([root_s]), limit, invalid, points, started)
        else:
            found = global_root_cpu(problem, base, direction, limit)
    except (FloatingPointError, OverflowError, MemoryError) as exc:
        info["global"] = dict(policy=POLICY, fallback=str(exc), seconds=perf_counter()-started)
        return predicted, info
    except RuntimeError as exc:   # torch.OutOfMemoryError only; anything else is a programming error
        if type(exc).__name__ != "OutOfMemoryError":
            raise
        info["global"] = dict(policy=POLICY, fallback="device_out_of_memory", seconds=perf_counter()-started)
        return predicted, info
    if found["root"] == 0:
        # Root below the bisection resolution (2^-50 of tau0): never issue tau = 0; the
        # historical prediction and the certificate handle this near-fold direction.
        found["fallback"] = "zero_root_below_resolution"
    elif found["root"] is not None:
        predicted = min(predicted, config.jd_step_safety*found["root"])
    found["predicted_tau"] = float(predicted)
    info["global"] = found
    return predicted, info


def _ggn_curvature(lin, direction, backend):
    """T72: p^T H p with the full frozen GGN H_C at the current base (device or CPU)."""
    if backend is not None and hasattr(backend, "linearize") and hasattr(backend, "device"):
        device = backend.device
        hp = device.download(backend.linearize(lin).hvp(device.upload(direction)))
    else:
        from ..linearization import barrier_ggn_hvp
        hp = barrier_ggn_hvp(lin, direction)
    value = float(direction@hp)
    require_finite_result(value, "T72 GGN curvature")
    return value


# E76: |cos_original - sy/(|s||y|)| <= ~2 n u for any n-term data (u = 2^-53); below this
# screen the original normalized-copy expression decides, so the decision is unchanged.
FAST_COSINE_SCREEN = 1e-6


def _bb2_independent(config, theta, gradient, previous, scratch):
    """E95: the direction-independent part of _step_proposal_fast's BB2 secant work -- s and y into their scratch
    buffers, s@y and the two norms -- computed while the device builds the direction. The values are those the
    proposal computes (same compiled differences, same BLAS calls, same errstate); None unless both differences
    are proven finite and nothing raised, in which case the proposal computes everything itself as before."""
    from .. import host_kernels
    if previous is None or 4*theta.size*np.finfo(float).eps >= FAST_COSINE_SCREEN or theta.dtype != np.float64:
        return None
    old_theta, old_gradient = previous[0], previous[1]
    if not (host_kernels.enabled(theta.size) and host_kernels._flat_ok(theta, old_theta)
            and host_kernels._flat_ok(gradient, old_gradient) and theta.size == gradient.size):
        return None
    def buffer(name):
        b = scratch.get(name)
        if b is None or b.shape != theta.shape:
            b = scratch[name] = np.empty_like(theta)
        return b
    s, y = buffer("s"), buffer("y")
    if not (host_kernels.subtract_into(theta, old_theta, s) and host_kernels.subtract_into(gradient, old_gradient, y)):
        return None
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            sy = float(s@y)
            ns, ny = float(norm(s, check_finite=False)), float(norm(y, check_finite=False))
    except FloatingPointError:
        return None
    return dict(s=s, y=y, sy=sy, ns=ns, ny=ny)


def _step_proposal_fast(config, theta, gradient, direction, previous, scratch, fused=False, lean=False, pre=None,
                        device=None, grid=None):
    """E76: _step_proposal with reused secant buffers and no normalized copies.

    sy and yy are the same float(s@y) / float(y@inverse_y) BLAS products on the same values,
    so the proposal, rule and recorded curvatures are bit-identical to _step_proposal."""
    def protect(value):
        return min(config.step_max, max(config.step_min, value))

    if previous is None:
        return protect(config.initial_step), "initial", None, None
    old_theta, old_gradient, old_direction, old_tau = previous
    fallback = protect(2*min(old_tau, config.step_max/2))
    if 4*theta.size*np.finfo(float).eps >= FAST_COSINE_SCREEN or theta.dtype != np.float64:
        return _step_proposal(config, theta, gradient, direction, previous)
    sy, yy = None, None

    def buffer(name):
        b = scratch.get(name)
        if b is None or b.shape != theta.shape:
            b = scratch[name] = np.empty_like(theta)
        return b
    proven = set()                               # E95: secants a compiled pass proved finite

    def secant(a, b, name):
        out = buffer(name)
        if fused:
            # E89 (P03): the same differences by a compiled pass; a nonfinite one replays np.subtract
            # (which then raises under this errstate, or yields the same values).
            from .. import host_kernels
            if (host_kernels.enabled(a.size) and host_kernels._flat_ok(a, b, out)
                    and host_kernels.subtract_into(a, b, out)):
                if lean:
                    proven.add(name)
                return out
        return np.subtract(a, b, out=out)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            if pre is not None:
                # E95: s, y, s@y and the norms were computed (proven finite) while the device built the direction
                s, y = pre["s"], pre["y"]
                proven.update(("s", "y"))
            else:
                s = secant(theta, old_theta, "s")
                y = secant(gradient, old_gradient, "y")
            # E104: with the device copies, iy's finiteness is decided on the device (exactly: the same values) and
            # y.iy is the device product when clearly positive; iy is formed on the host only for the band.
            tail = (device.device_bb2_tail(gradient, old_gradient, direction, old_direction, grid)
                    if device is not None and pre is not None else None)
            if tail is not None and not tail["finite"]:
                return fallback, "fallback_nonfinite_secant", sy, yy       # (v21: the iy check raises here)
            inverse_y = secant(old_direction, direction, "iy") if tail is None else None
            for name, value in (("s", s), ("y", y), ("iy", inverse_y)):
                if name not in proven and value is not None:
                    require_finite_result(value, "BB2 secants")
            value = pre["sy"] if pre is not None else float(s@y)
            require_finite_result(value, "BB2 secant curvature")
            sy = value
            if sy <= 0:
                return fallback, "fallback_nonpositive_curvature", sy, yy
            # E95: on proven-finite secants scipy's finite check would only rescan them; the same BLAS nrm2.
            if pre is not None:
                ns, ny = pre["ns"], pre["ny"]
            else:
                ns = float(norm(s, check_finite=False) if "s" in proven else norm(s))
                ny = float(norm(y, check_finite=False) if "y" in proven else norm(y))
            # Review E76: away from gradual underflow only (subnormal products void the 2nu bound).
            screened = (ns > 0 and ny > 0 and np.isfinite(ns) and np.isfinite(ny)
                        and np.isfinite(ns*ny) and ns*ny >= 2.**-900 and sy/(ns*ny) > FAST_COSINE_SCREEN)
            if not screened:
                cosine = float((s/norm(s))@(y/norm(y)))
                require_finite_result(cosine, "BB2 curvature cosine")
                if cosine <= 1e-12:
                    return fallback, "fallback_small_curvature", sy, yy
            value = None
            if tail is not None:
                value = device.device_yy_decide(tail, ny, lambda: float(y@np.subtract(old_direction, direction)),
                                                grid, theta.size)
                if value is None:                                         # the band: v21's own product
                    inverse_y = secant(old_direction, direction, "iy")
                    require_finite_result(inverse_y, "BB2 secants")
            if value is None:
                value = float(y@inverse_y)
            require_finite_result(value, "BB2 metric curvature")
            yy = value
            if yy <= 0:
                return fallback, "fallback_nonpositive_metric_curvature", sy, yy
            proposal = sy/yy
            require_finite_result(proposal, "BB2 step")
            if proposal <= 0:
                return fallback, "fallback_nonpositive_step", sy, yy
    except FloatingPointError:
        return fallback, "fallback_nonfinite_secant", sy, yy
    return protect(proposal), "bb2", sy, yy


def sampling_step_bound_fast(problem, base, direction, finite_known=False):
    """E76: sampling_step_bound without the identity/sum/contiguous copies (bit-identical).
    E95: finite_known=True only when the caller proved every value of this direction finite."""
    grid = problem.grid
    from .. import fast_checks
    if fast_checks.enabled(direction.size) and (finite_known or np.all(np.isfinite(direction))):
        return fast_checks.step_bound_xu(direction.reshape((3,)+grid.interior_shape), base.u, grid.axes)
    return sampling_step_bound(problem, base, direction)


def _digest_kw(digest, lazy=False):
    # Only the fast path passes the shared digest; other memories keep their old signature.
    # E95: lazy keeps the digest unresolved in the prediction record until the row is recorded.
    if digest is None:
        return {}
    return {"direction_id": digest, "lazy_digest": True} if lazy else {"direction_id": digest}


def _release_lean(backend, inverses):
    """E96: at the end of a lean inner solve, drop the device copies kept for it -- the scale S of its device
    Poisson (kept for the solve's directions) and the backend's device gradient memo (kept for the next
    direction). Memory only: nothing reads them after the solve; a later direction uploads afresh."""
    for inverse in inverses:
        release = getattr(getattr(inverse, "inner", None), "release", None)
        if release is not None:
            release()
    inverses.clear()
    release = getattr(backend, "release_device_gradient", None)
    if release is not None:
        release()


def _resolve_digests(value):
    """E95: replace every unresolved direction digest (a _DirectionDigest kept by a lazy JD prediction) in a
    record by its hex string, in place, before the record is stored or copied. Idempotent."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "direction_id" and isinstance(item, _DirectionDigest):
                value[key] = item()
            else:
                _resolve_digests(item)
    elif isinstance(value, list):
        for item in value:
            _resolve_digests(item)


class _DirectionDigest:
    """E76: SHA-256 of one direction computed once, in a background thread (hashlib releases
    the GIL), and shared by every JD prediction of that direction. Same hex digest."""
    _POOL = None

    def __init__(self, direction):
        import hashlib
        from concurrent.futures import ThreadPoolExecutor
        if _DirectionDigest._POOL is None:
            _DirectionDigest._POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jd-curl-direction-hash")
        data = np.ascontiguousarray(direction)
        self._future = _DirectionDigest._POOL.submit(lambda: hashlib.sha256(data.view(np.uint8)).hexdigest())

    def __call__(self):
        return self._future.result()


def _fused_axpy_ok(theta, direction):
    from .. import host_kernels
    return (host_kernels.enabled(theta.size) and host_kernels._flat_ok(theta, direction)
            and theta.shape == direction.shape)


def _local_prediction_cap(tau, predicted, factor):
    # Avoid overflow of factor*predicted when no truncation is needed.
    return tau if predicted >= tau/factor else factor*predicted


def _step_proposal(config, theta, gradient, direction, previous):
    """Accepted-only BB2; p_old-p = M^-1 y for the fixed Poisson metric."""
    def protect(value):
        return min(config.step_max, max(config.step_min, value))

    if previous is None:
        return protect(config.initial_step), "initial", None, None
    old_theta, old_gradient, old_direction, old_tau = previous
    fallback = protect(2*min(old_tau, config.step_max/2))
    sy, yy = None, None
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            s, y, inverse_y = theta-old_theta, gradient-old_gradient, old_direction-direction
            for value in (s, y, inverse_y):
                require_finite_result(value, "BB2 secants")
            value = float(s@y)
            require_finite_result(value, "BB2 secant curvature")
            sy = value
            if sy <= 0:
                return fallback, "fallback_nonpositive_curvature", sy, yy
            cosine = float((s/norm(s))@(y/norm(y)))
            require_finite_result(cosine, "BB2 curvature cosine")
            if cosine <= 1e-12:
                return fallback, "fallback_small_curvature", sy, yy
            value = float(y@inverse_y)
            require_finite_result(value, "BB2 metric curvature")
            yy = value
            if yy <= 0:
                return fallback, "fallback_nonpositive_metric_curvature", sy, yy
            proposal = sy/yy
            require_finite_result(proposal, "BB2 step")
            if proposal <= 0:
                return fallback, "fallback_nonpositive_step", sy, yy
    except FloatingPointError:
        return fallback, "fallback_nonfinite_secant", sy, yy
    return protect(proposal), "bb2", sy, yy


def sampling_step_bound(problem, base, direction):
    """Exact linear node-domain upper bound, not a JD or line-segment certificate."""
    grid = problem.grid
    v = direction.reshape((3,)+grid.interior_shape)
    points = (grid.identity()+base.u)[1:-1, 1:-1, 1:-1]
    from .. import fast_checks
    if fast_checks.enabled(direction.size) and np.all(np.isfinite(direction)):
        # E51 exact parallel min of the same IEEE element-wise quotients.
        return fast_checks.step_bound(v, points, grid.axes)
    bound = np.inf
    # Extremely small nonzero directions can have an infinite floating bound.
    with np.errstate(over="ignore", invalid="raise", divide="raise"):
        for r, axis in enumerate(grid.axes):
            pos, neg = v[r] > 0, v[r] < 0
            if np.any(pos):
                bound = min(bound, float(np.min((axis[-1]-points[..., r][pos])/v[r][pos])))
            if np.any(neg):
                bound = min(bound, float(np.min((axis[0]-points[..., r][neg])/v[r][neg])))
    return bound


def _failed_search_status(history):
    reasons = {row["reason"] for row in history}
    if reasons == {"INVALID_DOMAIN"}:
        return "domain_blocked"
    if reasons == {"FEASIBILITY_UNCERTAIN"}:
        return "feasibility_uncertain"
    if reasons == {"JACOBIAN_VIOLATION"}:
        return "feasibility_blocked"
    if reasons and all(reason.startswith("NONFINITE") for reason in reasons):
        return "nonfinite_evaluation"
    return "line_search_failed"


def _momentum_direction(memory, grid, theta):
    """T85 (E111): p = theta - theta_prev, the last accepted outer increment on this grid (theta_prev = the start of the
    previous inner solve on the same grid, kept in the Sequential JD-step memory), or None. Always records this start."""
    from .. import problem as _identity
    held = getattr(memory, "momentum_start", None)
    # review E111 (minor 1): keep a reference only to an owned, frozen array; anything else (a writable array or a
    # read-only view of one) is copied
    memory.momentum_start = (grid.grid_id, theta if _identity._immutable(theta) else readonly_copy(theta))
    if held is None or held[0] != grid.grid_id or held[1].shape != theta.shape:
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        p = theta-held[1]
    if not np.all(np.isfinite(p)) or not np.any(p):
        return None
    p.flags.writeable = False
    return p


def momentum_counts(rows):
    """T85 (E111) summary of attempt rows (each with 'level'; level None when absent): per level, the first steps with a
    momentum record, how many used p, were non-descending, the first momentum trials refused by the JD certificate,
    the accepted momentum steps, and the fallbacks to the Poisson direction after a failed momentum search."""
    out = {}
    for r in rows:
        m, f = r.get('momentum'), r.get('momentum_fallback')
        if m is None and f is None:
            continue
        c = out.setdefault(str(r.get('level')), dict(first_steps=0, used=0, nondescent=0, first_trial_jd_rejected=0,
                                                     accepted=0, fallbacks=0))
        if m is not None and r.get('attempt') == 1:
            c['first_steps'] += 1
            c['used' if m['used'] else 'nondescent'] += 1
            if m['used'] and r.get('reason') == 'JACOBIAN_VIOLATION':
                c['first_trial_jd_rejected'] += 1
        if m is not None and m['used'] and r.get('accepted'):
            c['accepted'] += 1
        if f is not None and r.get('attempt') == 1:
            c['fallbacks'] += 1
    return out


def _t63_stop_screened(decrease, energy, exact_total):
    """E-HS1b: the T63 stop under the T90 float32 screening, equal to the float64 decision.

    Only the soft-geometry (jd, curl) and barrier terms are screened; image, proximity and curvature are float64. T90's
    band bounds the screening error of the total: |E_screened - E_float64| <= band = FP32_EPS*(FP32_BAND_SOFT*(jd+curl) +
    FP32_BAND_BARRIER*barrier) (the band of the T90 comparisons, kept by its sampled float64 checks). With the margin of
    CRITICAL_GUARDS bands, a predicted decrease outside 15 eps (E_screened -/+ CRITICAL_GUARDS*band) gives the same
    decision for the float64 energy; otherwise exact_total() (the backend's exact_energy, once per base state) decides.
    Returns (stop, info)."""
    from ..backends.accelerated import CRITICAL_GUARDS, FP32_BAND_BARRIER, FP32_BAND_SOFT, FP32_EPS
    eps = np.finfo(float).eps
    band = FP32_EPS*(FP32_BAND_SOFT*(max(energy.jd, 0.)+max(energy.curl, 0.))+FP32_BAND_BARRIER*max(energy.barrier, 0.))
    band += 4*eps*abs(energy.total)            # review: an ulp floor so that a vanishing band never decides at a 1-ulp tie
    lower, upper = 15*eps*(energy.total-CRITICAL_GUARDS*band), 15*eps*(energy.total+CRITICAL_GUARDS*band)
    if decrease <= lower:
        return True, dict(band=band, decided="screened")
    if decrease > upper:
        return False, dict(band=band, decided="screened")
    floor = 15*eps*exact_total()
    return bool(decrease <= floor), dict(band=band, decided="float64", float64_floor=floor)


def solve_poisson_gradient(problem, theta, *, target, config=None, backend=None,
                           subspace=None, restart=None, budget=None, certificate_observer=None,
                           iteration_observer=None, jd_step_memory=None):
    """Solve one frozen-reference subproblem; only GE <= target means forcing passed."""
    started = perf_counter()
    target = validate_target(target)
    config = PGConfig3D() if config is None else config
    if not isinstance(config, PGConfig3D):
        raise InputValidationError("config: expected PGConfig3D")
    counters, history = SolverCounters3D(), []
    evaluation = backend if getattr(backend, "evaluation_backend_id", None) is not None else None
    # E-HS1b: with the T90 float32 screening the
    # base energy may hold screening values; the T63 floor is then decided as from the float64 energy (_t63_stop_screened).
    screen = evaluation is not None and bool(getattr(evaluation, "fp32_screen", False))
    exit_exact = {}
    evaluator = evaluate_trial if evaluation is None else evaluation.evaluate_trial
    promoter = promote_trial if evaluation is None else evaluation.promote_trial
    prediction = isinstance(config, JDStepPGConfig3D)
    hybrid = isinstance(config, TriggeredQPConfig3D)
    local_repair = isinstance(config, LocalRepairPGConfig3D)
    predictive_local = isinstance(config, PredictiveLocalRepairPGConfig3D)
    resolution_exit = isinstance(config, ConservativePGConfig3D) and config.resolution_exit
    global_step = isinstance(config, ConservativePGConfig3D) and config.jd_step_global
    bb2_guard = isinstance(config, ConservativePGConfig3D) and config.bb2_ggn_guard
    fast_host = isinstance(config, ConservativePGConfig3D) and config.fast_host
    fused = isinstance(config, ConservativePGConfig3D) and config.fused_steps   # E89 (P03)
    lean = isinstance(config, ConservativePGConfig3D) and config.lean_checks     # E95
    first_trial_cap = config.first_trial_cap if isinstance(config, ConservativePGConfig3D) else None   # T81 (E84)
    momentum = isinstance(config, ConservativePGConfig3D) and config.momentum_first                  # T85 (E111)
    pullback_on = isinstance(config, ConservativePGConfig3D) and config.pullback                    # T87 (S30-G2)
    if pullback_on and (not prediction or subspace is not None or restart is not None or hybrid or local_repair):
        raise InputValidationError("pullback requires the JD-step PG route and is not combined with a mask, a restart, the hybrid or the local-repair path")
    if momentum and (not prediction or subspace is not None or restart is not None):
        # review E111 round 2: a restart's 'previous' would turn the first step's proposal into a BB2 along p
        raise InputValidationError("momentum_first is not combined with a mask or a restart (masked restart path)")
    scratch = {}
    proven_cells = set()
    if prediction and subspace is not None:
        raise InputValidationError("JD prediction is not combined with mask")
    if prediction:
        from .jd_step import JDStepMemory
        if hybrid:
            from .qp_trigger import TriggerMemory, geometry_progress, run_triggered_qp
            jd_step_memory = TriggerMemory() if jd_step_memory is None else jd_step_memory
            if not isinstance(jd_step_memory,TriggerMemory):raise InputValidationError('trigger requires its own memory')
        else:jd_step_memory = JDStepMemory() if jd_step_memory is None else jd_step_memory
        jd_step_memory.bind(problem.grid)
    def certificate_seen(proof):
        if (local_repair and proof.status == 'infeasible'
                and proof.reason == 'JACOBIAN_VIOLATION' and proof.witness_cell is not None):
            proven_cells.add(proof.witness_cell)
        if prediction:
            jd_step_memory.certificate(proof)
        if certificate_observer is not None:
            certificate_observer(proof)
    def witness_seen(cell, point):
        jd_step_memory.witness(cell, point)
        if local_repair:
            proven_cells.add(cell)
    base = (prepare_base(problem, theta, counters) if evaluation is None else
            prepare_base(problem, theta, counters, backend=evaluation))
    accepted_steps, G = 0, None

    lean_inverses = [] if lean else None      # E96: this solve's metric inverse (its device copies end with it)

    def finish(status, reason, *, evidence=False):
        if lean:
            _release_lean(backend, lean_inverses)
        return finish_inner(base, status, reason, G, target, accepted_steps, history,
                            counters, started, precision_evidence=evidence)

    if isinstance(base, InvalidTrial):
        invalid, base = base, None
        return finish(initial_failure_status(invalid), invalid.reason)
    momentum_p = _momentum_direction(jd_step_memory, problem.grid, base.theta) if momentum else None   # T85
    momentum_fallback = None               # T85: set when a failed momentum search hands the first step back
    if subspace is not None:
        if backend is not None:
            raise InputValidationError("masked restart path is CPU only")
        subspace.validate(problem, base)
        from .masked import masked_pg_direction, masked_step_proposal, record_masked_bb2
    inverse, previous, small_steps = None, None if restart is None else restart.get('previous'), 0
    cached_gradient=None;qp_used=False
    digest = None
    # E95: the digest is only recorded, so on the plain route its join moves from the prediction to the row.
    lazy_digest = lean and not global_step and not hybrid and not local_repair
    if lean and prediction:
        jd_step_memory.cached_tables = True       # E95: the prediction's D tables and cell nodes are reused
        if getattr(evaluation, "root_screen", False):
            jd_step_memory.root_screen = True     # E97: margin screen of rootless rows
    extrema = lean and getattr(evaluation, "step_extrema", False)            # E97
    device_candidate = lean and getattr(evaluation, "device_candidate", False)   # E97
    device_scalars = lean and getattr(backend, "device_scalars", False)          # E104 (round-off level)
    fp32_guard = backend.fp32_descent_guard if lean and getattr(backend, "fp32_poisson", False) else None   # E106
    while True:
        momentum_row = None                    # T85 (E111): set on the first inner step of a solve with momentum
        momentum_poisson = None                # T85 (review B1): the replaced Poisson direction, for the next BB2
        pullback_row = pullback_poisson = None  # T87: the pull-back record and the replaced Poisson direction
        try:
            lin = build_linearization(problem, base)
            if cached_gradient is None:
                counters.gradients += 1
                if (backend is not None and accepted_steps >= config.max_iterations
                        and getattr(backend, "boundary_reuse", False)):
                    if getattr(backend, "commit_recheck", False):
                        # E101 (T84 plan D): every exit below returns this base as the outer candidate; its CPU
                        # re-certification starts now, overlapping this gradient and the outer checks
                        backend.commit_recheck_prestart(problem, base)
                    if getattr(backend, "metric_prestart", False) and isinstance(config, CurvaturePGConfig3D):
                        # E103 (M1): so is the next metric build's base; its host parts start now in the background
                        metric_prestart(problem, base, lin, config, backend, jd_step_memory if prediction else None)
                    # E89 (P05b): this gradient ends the solve (every exit below returns); the same vector,
                    # computed with the joint outer triple that the outer norms then reuse.
                    gradient = backend.final_gradient(lin, target)
                else:
                    gradient = energy_gradient(lin) if backend is None else backend.gradient(lin, target)
            else:
                gradient=cached_gradient;cached_gradient=None
            G = None
            if fused and hasattr(backend, "known_gradient_norm"):
                # E89 (P03): the norm the backend already computed on this same (unmodified) array.
                G = backend.known_gradient_norm(gradient, problem.grid.omega)
            if G is None:
                G = gradient_norm_mass(gradient, problem.grid.omega)
        except FloatingPointError as exc:
            G = None
            return finish("nonfinite_evaluation", str(exc))
        # This ordering is part of T14: tiny steps never hide an achieved target.
        if G <= target:
            return finish("converged_gradient", "GRADIENT_TARGET")
        if budget is not None and budget():
            return finish("work_budget_exhausted", "WALL_BUDGET")
        if small_steps >= config.stall_steps:
            return finish("stagnated", "SMALL_ACCEPTED_STEPS")
        if accepted_steps >= config.max_iterations:
            return finish("max_iterations", "ACCEPTED_STEP_BUDGET")
        if inverse is None:
            if isinstance(config, CurvaturePGConfig3D):
                try:
                    inverse, metric_info = pg_metric_inverse(problem, base, lin, config, backend,
                                                              memory=jd_step_memory if prediction else None)
                except FloatingPointError as exc:
                    return finish("nonfinite_evaluation", str(exc))
                if lean:
                    lean_inverses.append(inverse)
            else:
                inverse = build_poisson_inverse(problem.grid, a=problem.weights.lam,
                                                b=problem.weights.mu/2+problem.weights.nu/2,
                                                metric=config.metric)
                if backend is not None:
                    inverse = backend.poisson(inverse)
        counters.poisson += 1
        try:
            with np.errstate(over="raise", invalid="raise"):
                direction_finite = False
                if subspace is None:
                    if lean:
                        bb2_pre = {}
                        if fast_host and previous is not None:
                            between = lambda: bb2_pre.update(value=_bb2_independent(config, base.theta, gradient, previous, scratch))
                        else:
                            between = None
                        direction, direction_finite = fused_direction_checked(inverse, gradient, device_scaling=True,
                                                                              between=between)
                    else:
                        direction = fused_direction(inverse, gradient) if fused else -inverse.apply(gradient)
                else:
                    proposed = masked_pg_direction(problem, base, subspace, gradient, inverse.apply)
                    if not proposed.valid:
                        return finish("non_descent_direction", proposed.reason)
                    direction = proposed.p
                if not direction_finite:
                    require_finite_result(direction, "PG direction")
                slope = float(gradient@direction)
                if fp32_guard is not None:
                    # E106: a float32 direction not clearly descending is replaced by this gradient's float64 one
                    replaced = fp32_guard(gradient, direction, slope)
                    if replaced is not None:
                        direction = replaced
                        require_finite_result(direction, "PG direction")      # as v23's check of its direction
                        direction_finite = True
                        slope = float(gradient@direction)
                if momentum_p is not None and accepted_steps == 0:
                    # T85 (E111): the last outer increment replaces this first direction when it descends; the step
                    # proposal (initial_step), domain bound, JD prediction, Armijo and backtracking are unchanged.
                    with np.errstate(over="ignore", invalid="ignore"):
                        momentum_slope = float(gradient@momentum_p)
                    used = bool(np.isfinite(momentum_slope) and momentum_slope < 0)
                    # review E111 round 2: a nonfinite slope is recorded as None (rows are strict JSON)
                    momentum_row = dict(policy=MOMENTUM_POLICY, used=used,
                                        slope=momentum_slope if np.isfinite(momentum_slope) else None,
                                        poisson_slope=slope, norm_p=float(norm(momentum_p)))
                    if used:
                        momentum_poisson = direction
                        direction, slope, direction_finite = momentum_p, momentum_slope, True
                if pullback_on:
                    # T87 (S30-G2): p~ = p(x + u) on the same nodes; the step proposal, domain bound, JD prediction,
                    # Armijo, backtracking and acceptance are unchanged; a not clearly descending p~ keeps p.
                    from .pullback import pullback_direction, PULLBACK_POLICY
                    tick = perf_counter()
                    p_tilde, p_finite = pullback_direction(direction, base.u, problem.grid)
                    with np.errstate(over="ignore", invalid="ignore"):
                        tilde_slope = float(gradient@p_tilde) if p_finite else float("nan")
                    used = bool(p_finite and np.isfinite(tilde_slope) and tilde_slope < 0
                                and abs(tilde_slope) >= config.pullback_min_slope_ratio*abs(slope))
                    pullback_row = dict(policy=PULLBACK_POLICY, used=used, finite=bool(p_finite),
                                        slope=tilde_slope if np.isfinite(tilde_slope) else None, poisson_slope=slope,
                                        min_slope_ratio=config.pullback_min_slope_ratio, norm_poisson=float(norm(direction)),
                                        norm_pullback=float(norm(p_tilde)) if p_finite else None, seconds=perf_counter()-tick)
                    if used:
                        pullback_poisson = direction
                        direction, slope, direction_finite = p_tilde, tilde_slope, True
            digest = _DirectionDigest(direction) if (fast_host and prediction) else None
            require_finite_result(slope, "PG directional derivative")
            if slope >= 0:
                return finish("non_descent_direction", "NON_DESCENT_DIRECTION")
            guard_info = cap_info = None
            if subspace is None:
                if fast_host:
                    proposal, rule, sy, yy = _step_proposal_fast(config, base.theta, gradient, direction, previous, scratch,
                                                                 fused=fused, lean=lean,
                                                                 pre=bb2_pre.get("value") if lean else None,
                                                                 device=backend if device_scalars else None,
                                                                 grid=problem.grid)
                else:
                    proposal, rule, sy, yy = _step_proposal(config, base.theta, gradient, direction, previous)
                if bb2_guard and rule == "bb2" and accepted_steps == 1:
                    # T72: the GGN quadratic model along p decreases only for tau < 2 tau*, tau* = -g.p/p.Hp.
                    tick = perf_counter()
                    curvature = _ggn_curvature(lin, direction, backend)
                    guard_info = dict(policy=BB2_GUARD_POLICY, bb2=proposal, curvature=curvature, applied=False,
                                      hvp_seconds=perf_counter()-tick)
                    if curvature > 0:
                        model_tau = -slope/curvature
                        guard_info["model_tau"] = model_tau
                        if proposal > BB2_GUARD_FACTOR*model_tau:
                            proposal = min(config.step_max, max(config.step_min, model_tau))
                            rule, guard_info["applied"] = "bb2_ggn_guard", True
                if first_trial_cap is not None and (rule == "initial" or config.first_trial_scope == "all"):
                    # T81 (E84): trial upper bound tau <= c/max_i |p_i|_h (level voxels). Only the proposal
                    # shrinks; the domain bound, JD prediction, backtracking and acceptance are unchanged.
                    tick = perf_counter()
                    nu = max_node_voxel_displacement(direction, problem.grid)
                    cap_tau = first_trial_cap/nu if nu > 0 else None
                    # Review T81: never let the cap push the predicted decrease tau|g.p| below 2^20 times the
                    # T63 resolution floor 15 eps E(base); there the uncapped search proceeds unchanged.
                    floor = FIRST_TRIAL_FLOOR_FACTOR*15*np.finfo(float).eps*abs(base.energy.total)
                    guarded = cap_tau is not None and cap_tau < proposal and cap_tau*abs(slope) < floor
                    before = proposal
                    if cap_tau is not None and cap_tau < proposal and not guarded:
                        proposal = cap_tau
                    cap_info = dict(policy=FIRST_TRIAL_CAP_POLICY, c=first_trial_cap, scope=config.first_trial_scope,
                                    max_node_voxels=nu, cap_tau=cap_tau, proposal_before=before, floor_guard=bool(guarded),
                                    applied=bool(proposal != before), seconds=perf_counter()-tick)
            else:
                proposal, rule, sy, yy = masked_step_proposal(config, problem, base, subspace, gradient, direction, previous)
            if momentum_row is not None and momentum_row["used"]:
                rule = "momentum"                      # T85: proposal = initial_step (previous is None on this step)
            if fast_host and subspace is None:
                bound = sampling_step_bound_fast(problem, base, direction, **(dict(finite_known=True) if direction_finite else {}))
            else:
                bound = sampling_step_bound(problem, base, direction)
        except FloatingPointError as exc:
            return finish("nonfinite_evaluation", str(exc))
        if bound <= 0:
            return finish("domain_blocked", "NO_POSITIVE_DOMAIN_STEP")
        tau = min(proposal, .9*bound)
        jd_prediction = None
        if prediction:
            predicted, jd_prediction = jd_step_memory.predict(problem, base, direction, tau, config.jd_step_safety, **_digest_kw(digest, lazy_digest))
            if global_step:
                predicted, jd_prediction = _global_prediction(problem, base, lin, direction, tau, predicted,
                                                              jd_prediction, config, backend)
            if not local_repair:
                tau = predicted
            elif predictive_local:
                tau = _local_prediction_cap(tau, predicted, config.local_prediction_factor)
        trigger_info=None
        if hybrid:
            trigger_info=geometry_progress(jd_prediction,tau,proposal,bound,slope)
            eligible=(trigger_info['low_geometry_progress'] and tau<=.5*proposal
                      and any(jd_step_memory.recent) and jd_step_memory.pg_since_qp>=2 and not qp_used)
            trigger_info['eligible']=eligible
            if eligible:
                qp_used=True
                def qp_observe(row):
                    row=dict(row,iteration=accepted_steps,hybrid_event='qp_attempt',
                             hybrid_call_start=row['attempt']==1,trigger=trigger_info)
                    if lazy_digest:
                        _resolve_digests(row)
                    history.append(row)
                    if iteration_observer is not None:iteration_observer(deepcopy(row))
                result,ctx=run_triggered_qp(problem,base,gradient,target,backend,budget,jd_step_memory,qp_observe)
                for key,value in vars(result.counters).items():setattr(counters,key,getattr(counters,key)+value)
                if result.accepted_steps:
                    relative=step_ratio(result.state.theta-base.theta,base.theta,problem.grid)
                    small_steps=small_steps+1 if relative<=config.step_tol else 0
                    base=result.state;accepted_steps+=1;G=result.gradient_norm
                    cached_gradient=ctx.gradient;previous=None
                    if restart is not None:restart['previous']=None
                    continue
                if result.status!='trial_budget_exhausted':
                    return finish(result.status,result.reason)
                trigger_info['fallback']='same_base_PG_after_bounded_QP_exhaustion'
        # Review T81: a proposal lowered by the cap is a truncation (precision evidence needs the natural proposal).
        untruncated = tau == proposal and not (cap_info is not None and cap_info['applied'])
        attempts = []
        repair, pending = None, None
        repairs_used, global_attempts = 0, 0
        # T63: 15*eps*E(base) < every comparison resolution of this line search (the
        # 16/15 margin covers device-vs-CPU rounding of E(base), see the contract).
        exit_floor = 15*np.finfo(float).eps*base.energy.total if resolution_exit else None
        extra_attempts = config.max_local_repairs if local_repair else 0
        for attempt in range(config.max_backtracks+1+extra_attempts):
            if budget is not None and budget():
                return finish("work_budget_exhausted", "WALL_BUDGET")
            repair_info = None
            known = None
            if pending is None:
                if fused and not local_repair and _fused_axpy_ok(base.theta, direction):
                    # E89 (P03): the same two roundings, compiled; frozen so that the backend may take
                    # this fresh owned array as the trial theta without a copy (E89 P01).
                    from .. import host_kernels
                    if lean:
                        # E95: the same values; the pass also tells whether all are finite, recorded as a fact
                        # of this frozen array (the backend's trial entry check then needs no rescan).
                        if extrema:
                            # E97: the same pass also yields max|candidate-theta| and max|theta|
                            candidate, candidate_finite, top, theta_max = host_kernels.axpy_extrema(base.theta, tau,
                                                                                                    direction)
                        else:
                            candidate, candidate_finite = host_kernels.axpy_checked(base.theta, tau, direction)
                        candidate.flags.writeable = False
                        if candidate_finite:
                            from .. import problem as _identity
                            _identity.store_fact(candidate, _identity.FACT_FINITE)
                            if extrema and np.isfinite(top):
                                # (a finite candidate does not imply a finite max|candidate-theta|: v-b can overflow)
                                known = (top, theta_max)
                                evaluation.note_step_extrema(candidate, base.theta, top, theta_max)
                            if device_candidate:
                                evaluation.note_axpy(candidate, base.theta, tau, direction)
                    else:
                        candidate = host_kernels.axpy(base.theta, tau, direction)
                        candidate.flags.writeable = False
                else:
                    with np.errstate(over="ignore", invalid="ignore"):
                        candidate = base.theta+tau*direction
            else:
                candidate, repair_info = pending
                pending = None
                repairs_used += 1
            if fused and not local_repair:
                delta = None                          # E89 (P03): only the relative step needs it, see below
                increment_slope = tau*slope
            else:
                with np.errstate(over="ignore", invalid="ignore"):
                    delta = candidate-base.theta
                    increment_slope = float(gradient@delta) if local_repair else tau*slope
            required = -config.c1*increment_slope if local_repair else -config.c1*tau*slope
            counters.trials += 1
            proven_cells.clear()
            if local_repair and not np.isfinite(increment_slope):
                trial = InvalidTrial('NONFINITE_LOCAL_INCREMENT', 'nonfinite actual increment slope', 0)
            elif local_repair and increment_slope >= 0:
                trial = InvalidTrial('NON_DESCENT_INCREMENT', 'actual increment must be descent', 0)
            elif prediction:
                extra = {"witness_observer": witness_seen} if evaluation is not None else {}
                trial = evaluator(problem, candidate, certificate_observer=certificate_seen, **extra)
            else:
                trial = (evaluator(problem, candidate) if certificate_observer is None else
                         evaluator(problem, candidate, certificate_observer=certificate_observer))
            valid = not isinstance(trial, InvalidTrial)
            actual, resolution, certified, relative = None, None, None, None
            accepted = False
            if valid:
                try:
                    if delta is None:
                        def original(candidate=candidate, theta=base.theta):
                            with np.errstate(over="ignore", invalid="ignore"):
                                difference = candidate-theta
                            return step_ratio(difference, theta, problem.grid)
                        if known is not None and step_ratio_fast_applies(candidate, base.theta, problem.grid):
                            relative = step_ratio_from_extrema(known[0], known[1], problem.grid)   # E97
                        else:
                            relative = step_ratio_pair(candidate, base.theta, problem.grid, original)
                    else:
                        relative = step_ratio(delta, base.theta, problem.grid)
                    reduction = (stable_energy_reduction(problem, base, trial) if evaluation is None else
                                 evaluation.stable_energy_reduction(problem, base, trial,
                                     required_reduction=required))
                    actual, resolution, certified = reduction.actual, reduction.resolution, reduction.certified
                    accepted = certified > 0 and certified >= required
                    reason = ("ACCEPTED" if accepted else "ENERGY_RESOLUTION" if abs(actual) <= resolution
                              else "NO_STRICT_DECREASE" if actual < 0 else "ARMIJO_REJECTED")
                except FloatingPointError:
                    reason = "NONFINITE_COMPARISON"
            else:
                reason = trial.reason
            if accepted:
                counters.promotions += 1
                try:
                    promoted = promoter(problem, trial)
                except FloatingPointError:
                    promoted = InvalidTrial("NONFINITE_PROMOTION", "nonfinite trial derivative preparation", 0)
                if isinstance(promoted, InvalidTrial):
                    accepted, reason = False, promoted.reason
            scale_gate = None
            if predictive_local and reason == 'JACOBIAN_VIOLATION' and proven_cells:
                # A newly discovered witness can make even the capped trial
                # grossly oversized. Recenter before constructing local arrays.
                predicted, updated_prediction = jd_step_memory.predict(
                    problem, base, direction, tau, config.jd_step_safety, **_digest_kw(digest, lazy_digest))
                next_tau = _local_prediction_cap(tau, predicted, config.local_prediction_factor)
                # Cubic root recomputation can move the same cap by a few ulps.
                # Do not burn the line-search budget recentering that roundoff.
                recenter = next_tau < tau*(1-32*np.finfo(float).eps)
                scale_gate = dict(action='recenter' if recenter else 'allow_repair',
                    next_tau=next_tau, prediction=updated_prediction)
            exit_info = None
            if (resolution_exit and not accepted and not reason.startswith("NONFINITE")
                    and global_attempts+1 <= config.max_backtracks):
                # T63: the unchanged T11 next step, computed once here and reused below.
                if reason == 'JACOBIAN_VIOLATION':
                    predicted, next_prediction = jd_step_memory.predict(problem, base, direction, tau, config.jd_step_safety, **_digest_kw(digest, lazy_digest))
                    next_tau = min(.5*tau, predicted)
                else:
                    next_tau, next_prediction = .5*tau, jd_prediction
                decrease = next_tau*abs(slope)
                exit_info = dict(policy=RESOLUTION_EXIT_POLICY, next_tau=next_tau, predicted_decrease=decrease,
                                 resolution_floor=exit_floor, stop=bool(decrease <= exit_floor))
                if screen:
                    def exact_total(state=base):
                        if state.eval_id not in exit_exact:
                            exit_exact[state.eval_id] = float(evaluation.exact_energy(problem, state).total)
                        return exit_exact[state.eval_id]
                    exit_info['stop'], exit_info['screen'] = _t63_stop_screened(decrease, base.energy, exact_total)
            row = dict(iteration=accepted_steps, attempt=attempt+1, accepted=bool(accepted),
                       reason=reason, trial_valid=valid, energy_resolution=resolution,
                       actual_reduction=actual, certified_reduction=certified,
                       energy_base=base.energy.total, energy_trial=trial.energy.total if valid else None,
                       **({"energy_screened": True} if screen else {}),   # E-HS1b: T90 screening values possible
                       base_eval_id=base.eval_id, trial_eval_id=trial.eval_id if valid else None,
                       controls_id=problem.controls.controls_id, tau=tau, step_proposal=proposal,
                       step_rule=rule, secant_curvature=sy, metric_curvature=yy,
                       untruncated=untruncated and repair_info is None, domain_step_bound=bound if np.isfinite(bound) else None,
                       gradient_norm=G, step_ratio=relative, gradient_dot_direction=slope,
                       certificate_min_margin=trial.certificate.min_margin_lower_bound if valid else None,
                       certificate_min_jd=trial.certificate.min_jd_lower_bound if valid else None,
                       elapsed_seconds=perf_counter()-started)
            if hybrid:row['hybrid_trigger']=trigger_info
            if guard_info is not None:
                row['bb2_guard'] = guard_info
            if cap_info is not None:
                row['first_trial_cap'] = cap_info
            if momentum_row is not None:
                row['momentum'] = momentum_row
            elif momentum_fallback is not None and accepted_steps == 0:
                row['momentum_fallback'] = momentum_fallback
            if pullback_row is not None:
                row['pullback'] = pullback_row                 # T87
            if isinstance(config, CurvaturePGConfig3D):
                row['metric'] = resolve_metric_info(metric_info)
            if local_repair:
                row.update(local_repair=repair_info, local_repairs_used=repairs_used,
                           proven_violation_cells=sorted(proven_cells),
                           gradient_dot_increment=increment_slope, required_reduction=required,
                           increment_max=float(np.max(np.abs(delta))), increment_unit=problem.grid.unit,
                           increment_l2=float(norm(delta)))
            if predictive_local:
                row['local_scale_gate'] = scale_gate
                row['local_prediction_factor'] = config.local_prediction_factor
            if prediction:
                row['jd_prediction'] = jd_prediction
                row['jd_witnesses_after_trial'] = len(jd_step_memory.points)
            if exit_info is not None:
                row['resolution_exit'] = exit_info
            if evaluation is not None:
                row['forcing_target'] = target
                row['comparison_diagnostics'] = dict(evaluation.last_comparison) if actual is not None else None
            if lazy_digest:
                _resolve_digests(row)
            history.append(row)
            if iteration_observer is not None:
                iteration_observer(deepcopy(row))
            attempts.append(row)
            if reason.startswith("NONFINITE"):
                return finish("nonfinite_evaluation", reason)
            if accepted:
                if hybrid:
                    observed=geometry_progress(jd_prediction,tau,proposal,bound,slope)
                    energy_blocked=any(h['reason'] in ('ENERGY_RESOLUTION','NO_STRICT_DECREASE','ARMIJO_REJECTED') for h in attempts[:-1])
                    jd_step_memory.record_pg(observed['low_geometry_progress'] and not energy_blocked)
                small_steps = small_steps+1 if relative <= config.step_tol else 0
                # T85 (review B1): after a momentum step the next BB2 needs this base's Poisson direction (its
                # secant y = g1 - g0 is mapped by old_direction - direction = P^-1 y); s comes from the thetas.
                # T87: after a pulled-back step the next BB2 likewise needs this base's Poisson direction.
                poisson_for_bb2 = (pullback_poisson if pullback_poisson is not None else
                                   direction if momentum_poisson is None else momentum_poisson)
                previous = ((base.theta, gradient, poisson_for_bb2, tau)
                            if subspace is None else record_masked_bb2(problem, base, subspace, gradient, direction, tau))
                if restart is not None:
                    restart['previous'] = previous
                base = promoted
                accepted_steps += 1
                break
            if scale_gate is not None and scale_gate['action'] == 'recenter':
                global_attempts += 1
                if global_attempts > config.max_backtracks:
                    break
                tau = scale_gate['next_tau']
                jd_prediction = scale_gate['prediction']
                repair = None
                untruncated = False
                continue
            # Only certified violations can start a repair. Uncertain proofs do
            # not become dangerous cells, including within an existing chain.
            if (local_repair and repairs_used < config.max_local_repairs
                    and reason != 'FEASIBILITY_UNCERTAIN'):
                if repair is None and reason == 'JACOBIAN_VIOLATION' and proven_cells:
                    from .local_increment import LocalIncrementRepair
                    repair = LocalIncrementRepair(problem.grid, base.theta, candidate)
                if repair is not None:
                    try:
                        pending = repair.next(proven_cells)
                    except FloatingPointError as exc:
                        return finish('nonfinite_evaluation', str(exc))
                    if pending is not None:
                        continue
            global_attempts += 1
            if global_attempts > config.max_backtracks:
                break
            if exit_info is not None:
                tau, jd_prediction = exit_info['next_tau'], next_prediction
                if exit_info['stop']:
                    # Same order as T11: an expired budget at the next attempt wins.
                    if budget is not None and budget():
                        return finish("work_budget_exhausted", "WALL_BUDGET")
                    break  # T63: tau_next*|g.p| <= 15*eps*E(base); not evaluated
            elif prediction and (reason == 'JACOBIAN_VIOLATION' or repair is not None):
                predicted, jd_prediction = jd_step_memory.predict(problem, base, direction, tau, config.jd_step_safety, **_digest_kw(digest, lazy_digest))
                tau = min(.5*tau, predicted)
            else:
                tau *= .5
            repair = None
        if not accepted and momentum_row is not None and momentum_row["used"]:
            # T85 (E111): a failed momentum line search falls back, on the same base and gradient, to the Poisson
            # direction of this step (the rows above keep the failed attempts). Review S1: the JD-step memory keeps
            # the witnesses those attempts produced, so the steps from here need not repeat the plain solver's.
            momentum_p, cached_gradient = None, gradient
            momentum_fallback = dict(policy=MOMENTUM_POLICY, after_attempts=len(attempts), last_reason=attempts[-1]["reason"])
            continue
        if not accepted:
            evidence = precision_stagnation_evidence(
                [SimpleNamespace(**row) for row in attempts], untruncated=untruncated and repairs_used == 0)
            status = "precision_stagnated" if evidence else _failed_search_status(attempts)
            return finish(status, attempts[-1]["reason"], evidence=evidence)
