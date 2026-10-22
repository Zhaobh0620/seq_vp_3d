"""One inner-solve dispatch; numerical kernels retain their own acceptance rules."""

from ..types import FloatArray, Problem2D
from ..validation import InputValidationError
from .common import InitializationFailure, LMConfig, PoissonGradientConfig, SolverResult
from .lm import solve_lm
from .poisson_gradient import solve_poisson_gradient

SolverConfig = PoissonGradientConfig | LMConfig
SolveOutcome = SolverResult | InitializationFailure


def solve_inner(problem: Problem2D, z0: FloatArray, config: SolverConfig) -> SolveOutcome:
    if isinstance(config, PoissonGradientConfig):
        return solve_poisson_gradient(problem, z0, config)
    if isinstance(config, LMConfig):
        return solve_lm(problem, z0, config)
    raise InputValidationError("config: expected PoissonGradientConfig or LMConfig")
