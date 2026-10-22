"""Shared configuration, diagnostics and records; no direction/acceptance policy."""

from dataclasses import dataclass, field
from numbers import Integral

import numpy as np
from scipy.linalg import norm

from ..jacobian_barrier import (JacobianBarrierConfig, JacobianFeasibilityConfig,
                                validate_barrier, validate_feasibility)
from ..types import EnergyTerms, EvalState, FloatArray, Grid2D, InvalidTrial
from ..validation import (InputValidationError, as_float64_array, finite_real, require_finite_result,
                          validate_geometry_discretization)


def _integer(value, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise InputValidationError(f"{name}: expected integer >= {minimum}")
    return int(value)


def _nonnegative(value, name: str) -> float:
    value = finite_real(value, name)
    if value < 0:
        raise InputValidationError(f"{name}: must be nonnegative")
    return value


@dataclass(frozen=True)
class StopConfig:
    max_iterations: int = 500
    grad_atol: float = 1e-8
    grad_rtol: float = 1e-6
    relative_step_tol: float = 1e-12
    stagnation_window: int = 5

    def __post_init__(self):
        for name in ("grad_atol", "grad_rtol", "relative_step_tol"):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        for name, minimum in (("max_iterations", 0), ("stagnation_window", 1)):
            object.__setattr__(self, name, _integer(getattr(self, name), name, minimum))


@dataclass(frozen=True)
class PoissonGradientConfig:
    a: float | None = None
    b: float | None = None
    armijo_c1: float = 1e-4
    backtrack_factor: float = .5
    initial_step: float = 1.
    max_backtracks: int = 25
    stop: StopConfig = field(default_factory=StopConfig)
    metric: str = "five_point"
    geometry_discretization: str = "nodal"
    jacobian_barrier: JacobianBarrierConfig | None = None
    step_policy: str = "fixed"
    step_min: float = 1e-8
    step_max: float = 10.
    jacobian_feasibility: JacobianFeasibilityConfig | None = None
    energy_comparison: str = "absolute"

    def __post_init__(self):
        if self.energy_comparison not in ("absolute", "stable"):
            raise InputValidationError("energy_comparison: expected absolute or stable")
        validate_geometry_discretization(self.geometry_discretization)
        validate_barrier(self.jacobian_barrier, self.geometry_discretization)
        validate_feasibility(self.jacobian_feasibility, self.jacobian_barrier, self.geometry_discretization)
        if not isinstance(self.step_policy, str) or self.step_policy not in ("fixed", "inherit", "bb2"):
            raise InputValidationError("step_policy: expected fixed, inherit or bb2")
        for name in ("step_min", "step_max"):
            object.__setattr__(self, name, finite_real(getattr(self, name), name, positive=True))
        if self.step_min > self.step_max:
            raise InputValidationError("step_min must be <= step_max")
        if not isinstance(self.metric, str) or self.metric not in ("five_point", "derivative_gram", "q1"):
            raise InputValidationError("metric: expected five_point, derivative_gram or q1")
        if self.a is not None:
            object.__setattr__(self, "a", _nonnegative(self.a, "a"))
        if self.b is not None:
            object.__setattr__(self, "b", finite_real(self.b, "b", positive=True))
        for name in ("armijo_c1", "backtrack_factor", "initial_step"):
            value = finite_real(getattr(self, name), name, positive=True)
            if name != "initial_step" and value >= 1:
                raise InputValidationError(f"{name}: must be < 1")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "max_backtracks", _integer(self.max_backtracks, "max_backtracks", 0))
        if not isinstance(self.stop, StopConfig):
            raise InputValidationError("stop: expected StopConfig")


@dataclass(frozen=True)
class LMConfig:
    gamma0: float | None = None
    gamma_min: float = 1e-12
    gamma_max: float = 1e12
    cg_rtol: float = 1e-3
    cg_atol: float = 0.
    cg_maxiter: int = 200
    max_retries: int = 12
    accept_ratio: float = 1e-4
    poor_ratio: float = .25
    good_ratio: float = .75
    gamma_increase: float = 4.
    gamma_decrease: float = .5
    preconditioner_b: float | None = None
    stop: StopConfig = field(default_factory=StopConfig)
    geometry_discretization: str = "nodal"
    jacobian_barrier: JacobianBarrierConfig | None = None
    jacobian_feasibility: JacobianFeasibilityConfig | None = None
    energy_comparison: str = "absolute"
    barrier_curvature: str = "residual_gn"
    step_policy: str = "full"
    max_backtracks: int = 25
    backtrack_factor: float = .5
    linear_failure_recovery: str = "multiply"
    preconditioner: str = "poisson"

    def __post_init__(self):
        if self.preconditioner not in ("poisson", "ggn_diagonal_poisson"):
            raise InputValidationError("LM preconditioner: expected poisson or ggn_diagonal_poisson")
        if self.preconditioner == "ggn_diagonal_poisson" and (
                self.barrier_curvature != "ggn" or self.geometry_discretization != "q1" or self.jacobian_barrier is None):
            raise InputValidationError("GGN diagonal Poisson requires q1, absolute barrier and ggn curvature")
        if self.linear_failure_recovery not in ("multiply", "initial_scale"):
            raise InputValidationError("linear_failure_recovery: expected multiply or initial_scale")
        if self.step_policy not in ("full", "feasible_backtracking"):
            raise InputValidationError("LM step_policy: expected full or feasible_backtracking")
        if self.step_policy == "feasible_backtracking" and (self.geometry_discretization != "q1" or
                (self.jacobian_barrier is None and self.jacobian_feasibility is None)):
            raise InputValidationError("feasible LM steps require q1 and a hard Jacobian domain")
        object.__setattr__(self, "max_backtracks", _integer(self.max_backtracks, "max_backtracks", 0))
        factor = finite_real(self.backtrack_factor, "backtrack_factor", positive=True)
        if factor >= 1:
            raise InputValidationError("backtrack_factor: must be < 1")
        object.__setattr__(self, "backtrack_factor", factor)
        if self.barrier_curvature not in ("residual_gn", "ggn"):
            raise InputValidationError("barrier_curvature: expected residual_gn or ggn")
        if self.barrier_curvature == "ggn" and (self.geometry_discretization != "q1" or self.jacobian_barrier is None):
            raise InputValidationError("barrier GGN requires q1 and an absolute barrier")
        if self.energy_comparison not in ("absolute", "stable"):
            raise InputValidationError("energy_comparison: expected absolute or stable")
        validate_geometry_discretization(self.geometry_discretization)
        validate_barrier(self.jacobian_barrier, self.geometry_discretization)
        validate_feasibility(self.jacobian_feasibility, self.jacobian_barrier, self.geometry_discretization)
        for name in ("gamma_min", "gamma_max", "accept_ratio", "poor_ratio", "good_ratio",
                     "gamma_increase", "gamma_decrease"):
            object.__setattr__(self, name, finite_real(getattr(self, name), name, positive=True))
        if self.gamma_max < self.gamma_min:
            raise InputValidationError("gamma_max: must be >= gamma_min")
        if not 0 < self.accept_ratio <= self.poor_ratio < self.good_ratio < 1:
            raise InputValidationError("ratios: require 0 < accept_ratio <= poor_ratio < good_ratio < 1")
        if self.gamma_increase <= 1 or self.gamma_decrease >= 1:
            raise InputValidationError("gamma factors: require increase > 1 and 0 < decrease < 1")
        if self.gamma0 is not None:
            value = finite_real(self.gamma0, "gamma0", positive=True)
            if not self.gamma_min <= value <= self.gamma_max:
                raise InputValidationError("gamma0: must be within [gamma_min,gamma_max]")
            object.__setattr__(self, "gamma0", value)
        if self.preconditioner_b is not None:
            object.__setattr__(self, "preconditioner_b", finite_real(self.preconditioner_b, "preconditioner_b", positive=True))
        for name in ("cg_rtol", "cg_atol"):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        if self.cg_rtol == self.cg_atol == 0:
            raise InputValidationError("cg tolerances: at least one must be positive")
        for name in ("cg_maxiter", "max_retries"):
            object.__setattr__(self, name, _integer(getattr(self, name), name, 1))
        if not isinstance(self.stop, StopConfig):
            raise InputValidationError("stop: expected StopConfig")


def stopping_reason(G: float, threshold: float, n_accepted: int, small_steps: int,
                    config: StopConfig) -> tuple[str, str] | None:
    """Shared F14 priority; only the gradient condition is success."""
    if G <= threshold:
        return "converged_gradient", "Mass-scaled gradient reached the fixed threshold"
    if small_steps >= config.stagnation_window:
        return "stagnated", "Consecutive accepted small steps without gradient convergence"
    if n_accepted >= config.max_iterations:
        return "max_iterations", "Accepted-step limit reached without gradient convergence"
    return None


def gradient_norm_mass(gradient: FloatArray, omega: float) -> float:
    gradient = as_float64_array(gradient, "gradient", check_finite=False)
    require_finite_result(gradient, "gradient")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        # BLAS nrm2 scales before squaring; numpy's direct dot can underflow.
        value = float(norm(gradient.ravel(), check_finite=False)/np.sqrt(omega))
    require_finite_result(value, "mass gradient norm")
    if value == 0. and np.any(gradient != 0):
        raise FloatingPointError("mass gradient norm: nonzero result underflowed")
    return value


def step_metrics(delta: FloatArray, z_base: FloatArray, grid: Grid2D) -> tuple[float, float]:
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        length = np.hypot((grid.nx-1)*grid.hx, (grid.ny-1)*grid.hy)
        relative = float(np.linalg.norm(delta, ord=np.inf)/max(length, np.linalg.norm(z_base, ord=np.inf)))
        require_finite_result(delta, "step")
        mass = float(np.sqrt(grid.omega)*norm(delta.ravel(), check_finite=False))
    require_finite_result([mass, relative], "step metrics")
    if mass == 0. and np.any(delta != 0):
        raise FloatingPointError("step mass norm: nonzero result underflowed")
    return mass, relative


def geometry_diagnostics(state: EvalState) -> dict:
    u, J = state.u, state.geometry.J
    boundary = max(np.max(np.abs(u[[0, -1], :, :])), np.max(np.abs(u[:, [0, -1], :])))
    return dict(min_jd=float(np.min(J)), negative_jd_fraction=float(np.mean(J < 0)),
                nonpositive_jd_fraction=float(np.mean(J <= 0)), boundary_max_abs=float(boundary),
                min_corner_jd=None if state.corner_geometry is None else float(np.min(state.corner_geometry.J)))


def energy_evaluations(state: EvalState | InvalidTrial) -> int:
    """C evaluates energy once iff it returns a state or NONFINITE_ENERGY.

    Count after each actual evaluation, including an overflowing energy call;
    this is verified against a spy on C's energy function in real PG runs.
    """
    return int(isinstance(state, EvalState) or state.reason == "NONFINITE_ENERGY")


@dataclass(frozen=True)
class IterationRecord:
    iteration: int
    attempt: int
    solver: str
    accepted: bool
    accepted_count_after: int
    base_eval_id: int
    trial_eval_id: int | None
    energy: EnergyTerms
    trial_energy: EnergyTerms | None
    grad_norm_mass: float
    direction_dot_gradient: float | None
    step_norm_mass: float | None
    step_relative: float | None
    tau: float | None
    gamma_used: float | None
    gamma_next: float | None
    predicted_reduction: float | None
    actual_reduction: float | None
    ratio: float | None
    cg_info: int | None
    cg_iterations: int | None
    cg_relative_residual: float | None
    min_jd: float
    trial_min_jd: float | None
    negative_jd_fraction: float
    nonpositive_jd_fraction: float
    boundary_max_abs: float
    trial_boundary_max_abs: float | None
    trial_valid: bool
    backtracks: int
    n_domain_rejections: int
    total_cg_iterations: int
    n_energy_evaluations: int
    reason: str
    wall_seconds: float
    min_corner_jd: float | None = None
    trial_min_corner_jd: float | None = None
    feasibility_checked: bool = False
    feasible_step_limit: float | None = None
    pg_step_proposal: float | None = None
    pg_step_rule: str | None = None
    pg_secant_curvature: float | None = None
    pg_metric_curvature: float | None = None
    energy_resolution: float | None = None
    lm_direction_curvature: float | None = None


@dataclass(frozen=True)
class SolverResult:
    success: bool
    status: str
    message: str
    final_z: FloatArray
    final_state: EvalState
    history: tuple[IterationRecord, ...]
    n_accepted: int
    n_trial_evaluations: int
    n_domain_rejections: int
    n_gradient_evaluations: int
    n_energy_evaluations: int
    total_cg_iterations: int
    n_hvp: int
    n_preconditioner_applications: int
    resolved_config: dict
    initial_gradient_norm_mass: float | None
    final_gradient_norm_mass: float | None
    gradient_threshold: float | None
    wall_seconds: float
    n_feasibility_checks: int = 0
    n_jacobian_rejections: int = 0


@dataclass(frozen=True)
class InitializationFailure:
    status: str
    message: str
    invalid_count: int
    resolved_config: dict


@dataclass(frozen=True)
class DirectionResult:
    valid: bool
    p: FloatArray | None
    directional_derivative: float | None
    reason: str


@dataclass(frozen=True)
class LinearSolveResult:
    valid: bool
    p: FloatArray | None
    info: int | None
    iterations: int
    relative_residual: float | None
    gamma_used: float
    reason: str
    Hp: FloatArray | None = None
    directional_derivative: float | None = None
    n_hvp: int = 0
    n_preconditioner_applications: int = 0


@dataclass(frozen=True)
class LMDecision:
    accepted: bool
    reason: str
    ared: float | None
    pred: float | None
    ratio: float | None
    gamma_next: float


@dataclass(frozen=True)
class AcceptedStep:
    state: EvalState
    tau: float
    n_trials: int
    n_domain_rejections: int
    history: tuple[IterationRecord, ...] = ()
    n_energy_evaluations: int = 0
    n_feasibility_checks: int = 0


@dataclass(frozen=True)
class RejectedStep:
    status: str
    reason: str
    n_trials: int
    n_domain_rejections: int
    history: tuple[IterationRecord, ...] = ()
    n_energy_evaluations: int = 0
    n_feasibility_checks: int = 0


@dataclass
class SolverCounters:
    n_accepted: int = 0
    n_trial_evaluations: int = 0
    n_domain_rejections: int = 0
    n_gradient_evaluations: int = 0
    n_energy_evaluations: int = 0
    total_cg_iterations: int = 0
    n_hvp: int = 0
    n_preconditioner_applications: int = 0
    n_feasibility_checks: int = 0
    n_jacobian_rejections: int = 0
