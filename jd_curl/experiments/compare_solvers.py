"""Same-case, same-objective comparisons with no automatic retuning or warm start."""

from ..boundary import pack_initial_displacement
from ..residuals import build_problem
from ..solvers import SolveOutcome, SolverConfig, solve_inner
from ..solvers.common import LMConfig, PoissonGradientConfig
from ..types import RegistrationCase2D, Weights
from ..validation import InputValidationError


def _iter_solutions(case: RegistrationCase2D, weights: Weights, configs: tuple[SolverConfig, ...]):
    if not configs or any(not isinstance(c, (PoissonGradientConfig, LMConfig)) for c in configs):
        raise InputValidationError("configs: require at least one supported solver config")
    if any(c.stop != configs[0].stop for c in configs):
        raise InputValidationError("comparison: all solvers must use the same StopConfig")
    if any(c.geometry_discretization != configs[0].geometry_discretization for c in configs):
        raise InputValidationError("comparison: geometry_discretization must match")
    if any(c.jacobian_barrier != configs[0].jacobian_barrier for c in configs):
        raise InputValidationError("comparison: jacobian_barrier must match")
    if any(c.jacobian_feasibility != configs[0].jacobian_feasibility for c in configs):
        raise InputValidationError("comparison: jacobian_feasibility must match")
    problem = build_problem(case, weights, geometry_discretization=configs[0].geometry_discretization,
                            jacobian_barrier=configs[0].jacobian_barrier,
                            jacobian_feasibility=configs[0].jacobian_feasibility)
    z0 = pack_initial_displacement(case.u0, case.grid)
    for config in configs:
        yield solve_inner(problem, z0.copy(), config)


def compare_solvers(case: RegistrationCase2D, weights: Weights,
                    configs: tuple[SolverConfig, ...]) -> tuple[SolveOutcome, ...]:
    return tuple(_iter_solutions(case, weights, configs))
