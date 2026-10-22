"""T13: streamed stable differences of the unchanged frozen 3D objective."""
import numpy as np

from ..energy_comparison import EnergyReduction
from ..validation import InputValidationError, StateMismatchError, as_float64_array, finite_real, require_finite_result
from .geometry import _compute_cell_geometry as compute_cell_geometry
from .bernstein_barrier import barrier_values
from .problem import ParallelMemoryConfig
from .residuals import (_residual_block_pairs, _node_residual_blocks,
                        _geometry_residual_block_pairs, validate_state)
from .tile_execution import ordered_tile_map


def _barrier_difference(base_j, trial_j, *, delta, activation=None):
    """Return unweighted psi(J_trial)-psi(J_base), without clipping corners.

    The small relative-gap branch separates the linear term from x-log1p(x).
    Outside that branch use the original difference: separating two O(x)
    terms would lose accuracy when the base is close to the hard threshold.
    """
    a = as_float64_array(base_j, "barrier base J")
    b = as_float64_array(trial_j, "barrier trial J", a.shape)
    delta = finite_real(delta, "delta")
    if delta < 0 or delta >= 1 or np.any(a <= delta) or np.any(b <= delta):
        raise InputValidationError("barrier difference requires 0 <= delta < 1 and J > delta")
    a, b, d = a.astype(np.longdouble), b.astype(np.longdouble), np.longdouble(delta)
    if activation is not None:
        activation = np.longdouble(finite_real(activation, "barrier activation"))
        if not d < activation <= 1:
            raise InputValidationError("require delta < activation <= 1")
        # Upper truncation closes the inactive potential, never clips J<=delta.
        a, b = np.minimum(a, activation), np.minimum(b, activation)
        scale = activation-d
        ra, rb = (activation-a)/scale, (activation-b)/scale
        h = (b-a)/scale
        loga = np.log((a-d)/scale)
        near = ra < .5
        loga[near] = np.log1p(-ra[near])
        x = (b-a)/(a-d)
        ratio = np.empty_like(x)
        near = x >= -.5
        ratio[near] = np.log1p(x[near])
        ratio[~near] = np.log((b[~near]-d)/(a[~near]-d))
        difference = h*(ra+rb)*loga-rb*rb*ratio
        require_finite_result(difference, "localized barrier difference")
        return difference
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        change, gap = b-a, a-d
        x = change/gap
        small = np.abs(x) < .01
        difference = np.empty_like(x)
        xs = x[small]
        polynomial = np.full_like(xs, np.longdouble(1)/14)
        for k in range(13, 1, -1):
            polynomial = np.longdouble((-1)**k)/k+xs*polynomial
        difference[small] = xs*(a[small]-1)/(1-d)+xs*xs*polynomial
        far = (~small) & (x >= -.5)
        difference[far] = change[far]/(1-d)-np.log1p(x[far])
        negative = x < -.5
        difference[negative] = change[negative]/(1-d)-np.log((b[negative]-d)/gap[negative])
    require_finite_result(difference, "barrier energy difference")
    return difference


def _squared_change(base_block, trial_block):
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        a = base_block.values.astype(np.longdouble)
        b = trial_block.values.astype(np.longdouble)
        return np.sum((b-a)*(b+a), dtype=np.longdouble)/2


def _residual_changes(problem, base, trial, outer_only):
    if not isinstance(problem.memory, ParallelMemoryConfig):
        for a, b in _residual_block_pairs(problem, base, trial, outer_only=outer_only):
            yield a.name, _squared_change(a, b)
        return
    for a, b in zip(_node_residual_blocks(problem, base, outer_only=outer_only),
                    _node_residual_blocks(problem, trial, outer_only=outer_only), strict=True):
        yield a.name, _squared_change(a, b)
    if outer_only:
        return
    del a, b

    def work(tile):
        return tuple((a.name, _squared_change(a, b))
                     for a, b in _geometry_residual_block_pairs(problem, base, trial, tile))

    for changes in ordered_tile_map(problem, work):
        yield from changes


def _differences(problem, base, trial, *, outer_only=False):
    for state in (base, trial):
        validate_state(problem, state)
        if getattr(state, "evaluation_backend_id", None) is not None:
            raise StateMismatchError("device states require paired stable comparison or CPU re-evaluation")
    difference = image_difference = np.longdouble(0)
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        for name, change in _residual_changes(problem, base, trial, outer_only):
            difference += change
            if name in ("image", "curvature"):      # the outer F blocks (T76 adds curvature)
                image_difference += change
        weight = np.longdouble(problem.weights.kappa)*np.longdouble(problem.grid.omega)/len(problem.barrier_ops.points)

        def barrier_change(tile):
            a = barrier_values(problem, compute_cell_geometry(base.u, problem.barrier_ops, tile=tile).J)
            b = barrier_values(problem, compute_cell_geometry(trial.u, problem.barrier_ops, tile=tile).J)
            return weight*np.sum(_barrier_difference(a, b, delta=problem.certificate_config.delta, activation=problem.barrier_activation),
                                  dtype=np.longdouble)

        for change in ordered_tile_map(problem, barrier_change):
            difference += change
            image_difference += change
    return difference, image_difference


def _reduction(base, trial, difference, outer_only):
    energies = [s.energy.outer if outer_only else s.energy.total for s in (base, trial)]
    actual = float(-difference)
    resolution = float(16*np.finfo(float).eps*max(energies))
    require_finite_result([actual, resolution], "stable energy reduction")
    return EnergyReduction(actual, resolution)


def stable_energy_reduction(problem, base, trial, *, outer_only=False):
    """Return E(base)-E(trial), or F(base)-F(trial), with the T13 guard.

    Longdouble accumulation and 16*eps64*max(Eb,Et) are unchanged; this is
    not a rigorous spline/residual error bound or a feasibility certificate.
    """
    difference, _ = _differences(problem, base, trial, outer_only=outer_only)
    return _reduction(base, trial, difference, outer_only)


def stable_energy_reductions(problem, base, trial):
    """Joint E/F comparison, sharing barrier work but not sum order or guards."""
    total, outer = _differences(problem, base, trial)
    return _reduction(base, trial, total, False), _reduction(base, trial, outer, True)
