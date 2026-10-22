"""F24: stable differences of the existing energy, not a new objective."""
from dataclasses import dataclass
import numpy as np

from .types import EvalState, RESIDUAL_NAMES
from .validation import InputValidationError, require_finite_result


@dataclass(frozen=True)
class EnergyReduction:
    actual: float
    resolution: float

    @property
    def certified(self):
        return self.actual-self.resolution


def stable_energy_reduction(problem, base, trial, *, outer_only=False):
    """Use longdouble accumulation, with a conservative float64 resolution guard.

    The guard is not a rigorous error bound for spline/residual evaluation.
    No uphill or unresolved step is accepted by relaxing the energy inequality.
    """
    for state in (base, trial):
        if not isinstance(state, EvalState) or state.problem_id != problem.problem_id or state.controls_id != problem.controls.controls_id:
            raise InputValidationError('energy comparison requires states of the same frozen problem')
    names = ('image',) if outer_only else RESIDUAL_NAMES
    delta = np.longdouble(0)
    for name in names:
        a = getattr(base.residuals, name).astype(np.longdouble)
        b = getattr(trial.residuals, name).astype(np.longdouble)
        delta += np.sum((b-a)*(b+a), dtype=np.longdouble)/2
    if problem.jacobian_barrier is not None:
        cfg = problem.jacobian_barrier
        a = base.corner_geometry.J.astype(np.longdouble)
        b = trial.corner_geometry.J.astype(np.longdouble)
        if np.any(a <= cfg.min_jd) or np.any(b <= cfg.min_jd):
            raise InputValidationError('barrier difference requires strictly feasible corners')
        x = (b-a)/(a-cfg.min_jd)
        small = np.abs(x) < .01
        remainder = np.empty_like(x)
        xs = x[small]
        polynomial = np.full_like(xs, np.longdouble(1)/14)
        for k in range(13, 1, -1):
            # Preserve the longdouble multiply/add order without allocating
            # two new full-corner arrays at every Horner stage.
            np.multiply(xs, polynomial, out=polynomial)
            np.add(np.longdouble((-1)**k)/k, polynomial, out=polynomial)
        remainder[small] = xs*xs*polynomial
        far = (~small) & (x >= -.5)
        remainder[far] = x[far]-np.log1p(x[far])
        negative = x < -.5
        remainder[negative] = x[negative]-np.log((b[negative]-cfg.min_jd)/(a[negative]-cfg.min_jd))
        difference = x*(a-1)/(1-np.longdouble(cfg.min_jd))+remainder
        delta += np.longdouble(cfg.weight)*np.longdouble(problem.grid.omega)/4*np.sum(difference)
    energies = [s.energy.image+s.energy.barrier if outer_only else s.energy.total for s in (base, trial)]
    actual = float(-delta)
    resolution = float(16*np.finfo(float).eps*max(energies))
    require_finite_result([actual, resolution], 'stable energy reduction')
    return EnergyReduction(actual, resolution)


def precision_stagnation_evidence(history, *, untruncated=True):
    """ALL original-proposal trials are unresolved; not a stationary-point test.

    A prediction is a model, not an error bound for actual energy differences.
    Never inspect only the tiny-step suffix of a failed line search.
    """
    if not history or not untruncated:
        return False
    if any(r.reason != 'ENERGY_RESOLUTION' or not r.trial_valid or r.accepted or
           r.energy_resolution is None for r in history):
        return False
    return all(r.actual_reduction is not None and np.isfinite(r.actual_reduction) and
               np.isfinite(r.energy_resolution) and r.energy_resolution >= 0 and
               abs(r.actual_reduction) <= r.energy_resolution for r in history)
