"""E27 fused, independently sampled CPU critical comparison (T40).

Reuse only an unchanged map's whole-cell proof. No GPU image/energy is used
in the CPU reduction or its original 16-eps guard. No cache outlives a call.
"""
from dataclasses import replace
import hashlib
from types import SimpleNamespace

import numpy as np

from ...types import InvalidTrial
from ...validation import StateMismatchError, require_finite_result
from ..barrier import barrier_terms
from ..bernstein_barrier import barrier_values
from ..energy_comparison import _barrier_difference, _reduction, _squared_change
from ..curvature import curvature_energy
from ..data_weight import data_mass
from ..feasibility import certificate_version
from ..geometry import _compute_cell_geometry as geometry
from ..interpolation import sample_points
from ..problem import QUADRATURE_ID, _problem_id, array_hash, prefetched_hashes
from ..residuals import EnergyTerms3D, _node_residual_blocks, validate_state
from ..tile_execution import ordered_tile_map
from ..validation import zero_boundary
from ..work_control import checkpoint, operation

POLICY = "fused-cpu-certified-maps-v1"


def _map_binding(problem, u, certificate):
    version = certificate_version(problem.certificate_config)
    if (u.dtype != np.float64 or u.shape != problem.grid.shape+(3,) or u.flags.writeable
            or not _finite_zero_boundary(u)
            or certificate.status != "certified_feasible"
            or certificate.grid_id != problem.grid.grid_id
            or certificate.config != problem.certificate_config
            or certificate.delta_hex != float(problem.certificate_config.delta).hex()
            or certificate.certificate_version != version
            or certificate.n_cells_checked != problem.grid.n_cells
            or certificate.n_uncertain != 0 or certificate.n_infeasible != 0
            or certificate.min_jd_lower_bound is None
            or not np.isfinite(certificate.min_jd_lower_bound)
            or certificate.min_jd_lower_bound <= problem.certificate_config.delta
            or certificate.state_hash != array_hash(u)):
        raise StateMismatchError("CPU proof reuse requires an exact readonly whole-map certificate binding")


def _finite_zero_boundary(u):
    """np.all(np.isfinite(u)) and zero_boundary(u). E89: a fact already proven on these exact immutable
    bytes (problem.seed_certified_map, or an earlier full check here) is reused; the memo is off by default."""
    from ..problem import has_fact, store_fact, FACT_FINITE_ZERO_BOUNDARY
    if has_fact(u, FACT_FINITE_ZERO_BOUNDARY):
        return True
    result = bool(np.all(np.isfinite(u))) and bool(zero_boundary(u))
    if result:
        store_fact(u, FACT_FINITE_ZERO_BOUNDARY)
    return result


def _validate_bindings(problem, states):
    # E48: the digests below are computed concurrently once; semantics unchanged.
    with prefetched_hashes([problem.controls.u_ref, problem.fixed_image]+[s.u for s in states]):
        return _validate_bindings_unscoped(problem, states)


# E76 (explicit, off by default): theta/u pairs already proven byte-identical. Both arrays must be
# immutable (problem._immutable) and alive (weak references re-checked), so the result cannot change.
_THETA_BINDINGS = None


def enable_binding_memo(enabled=True):
    global _THETA_BINDINGS
    _THETA_BINDINGS = {} if enabled else None


def _theta_matches(theta, free, u):
    memo = _THETA_BINDINGS
    if memo is not None:
        from ..problem import _immutable
        key = (id(theta), id(u))
        entry = memo.get(key)
        if (entry is not None and entry[0]() is theta and entry[1]() is u
                and _immutable(theta) and _immutable(u)):
            return True
    same = np.array_equal(theta.reshape(free.shape).view(np.uint64), free.view(np.uint64))
    if same and memo is not None and _immutable(theta) and _immutable(u):
        import weakref
        if len(memo) > 512:
            for k in [k for k, (a, b) in memo.items() if a() is None or b() is None]:
                del memo[k]
        memo[key] = (weakref.ref(theta), weakref.ref(u))
    return same


def bind_theta_u(theta, u):
    """E89: record the theta/u binding where u is built from theta (u = inject_free(theta), bytewise:
    the interior is a copy of theta, the faces are +0). Only immutable pairs, only with the E76 memo on;
    a later _theta_matches of exactly these two objects then needs no byte comparison."""
    memo = _THETA_BINDINGS
    if memo is None:
        return
    from ..problem import _immutable
    if not (_immutable(theta) and _immutable(u)):
        return
    import weakref
    if len(memo) > 512:
        for k in [k for k, (a, b) in memo.items() if a() is None or b() is None]:
            del memo[k]
    memo[(id(theta), id(u))] = (weakref.ref(theta), weakref.ref(u))


def _validate_bindings_unscoped(problem, states):
    ref = problem.controls
    _map_binding(problem, ref.u_ref, ref.certificate)
    expected_ref = hashlib.sha256((problem.grid.grid_id+QUADRATURE_ID+ref.certificate.state_hash).encode()).hexdigest()
    if (ref.controls_id != expected_ref or ref.grid_id != problem.grid.grid_id
            or ref.quadrature_id != QUADRATURE_ID or problem.fixed_image.flags.writeable
            or _problem_id(problem.grid, problem.fixed_image, problem.sampler, ref,
                           problem.weights, problem.memory, problem.certificate_config,
                           problem.barrier_location, problem.barrier_activation,
                           problem.barrier_basis, problem.curvature_weight, problem.data_weight,
                           getattr(problem, "active_weights", None)) != problem.problem_id):   # T89
        raise StateMismatchError("CPU comparison model/reference identity mismatch")
    for state in states:
        validate_state(problem, state)
        _map_binding(problem, state.u, state.certificate)
        theta = state.theta
        free = np.moveaxis(state.u[1:-1, 1:-1, 1:-1], -1, 0)
        if (theta.dtype != np.float64 or theta.shape != (3*problem.grid.n_free_scalar,)
                or theta.flags.writeable
                or not _theta_matches(theta, free, state.u)):
            raise StateMismatchError("CPU comparison theta does not match certified displacement bytes")
        policy = getattr(state, "evaluation_backend_id", "")
        expected = hashlib.sha256((problem.problem_id+state.certificate.state_hash+policy).encode()).hexdigest()
        if expected != state.eval_id:
            raise StateMismatchError("CPU comparison evaluation identity mismatch")


def _change(a, b):
    return _squared_change(SimpleNamespace(values=a), SimpleNamespace(values=b))


def fused_cpu_reductions(problem, base, trial, *, outer_only=False, joint=False):
    """Same float64 operands, longdouble sum order and CPU guards as T13.

    F-only skips the soft geometry entirely. Full E/F shares geometry within
    each tile, deferring only scalar barrier deltas to retain T13 sum order.
    """
    with operation("fused_cpu_comparison"):
        checkpoint("cpu_comparison_binding")
        _validate_bindings(problem, (base, trial))
        strict = np.zeros(problem.grid.shape, dtype=bool)
        strict[1:-1, 1:-1, 1:-1] = True
        states = []
        for state in (base, trial):
            sampled = sample_points(problem.sampler, problem.grid.identity()+state.u,
                strict=strict, with_gradient=False, chunk_size=problem.memory.spline_query_chunk)
            if isinstance(sampled, InvalidTrial):
                raise FloatingPointError("CPU resampling of certified map failed: "+sampled.reason)
            states.append(replace(state, warped=sampled.value))
        return _accumulate(problem, *states, outer_only=outer_only, joint=joint)


def _accumulate(problem, base, trial, *, outer_only, joint):
    q, w = problem.grid.node_weights(), problem.weights
    accum = np.zeros((2, 6), dtype=np.longdouble)   # image, proximity, jd, curl, barrier, curvature (T76)
    total = outer = np.longdouble(0)
    barrier_changes = []  # scalars, not a full-volume geometry cache
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for i, s in enumerate((base, trial)):
            accum[i, 0] = .5*np.sum(data_mass(problem)*(s.warped-problem.fixed_image)**2, dtype=np.longdouble)   # T80
            if not outer_only:
                accum[i, 1] = .5*w.lam*np.sum(q[..., None]*(s.u-problem.controls.u_ref)**2, dtype=np.longdouble)
            if problem.curvature_weight:
                accum[i, 5] = curvature_energy(problem, s.u)
        for a, b in zip(_node_residual_blocks(problem, base, outer_only=outer_only),
                        _node_residual_blocks(problem, trial, outer_only=outer_only), strict=True):
            change = _squared_change(a, b)
            total += change
            if a.name in ("image", "curvature"):
                outer += change
        # Keep the original extended precision product, not barrier_weight's
        # rounded float64 product, for the stable difference only.
        weight = np.longdouble(w.kappa)*np.longdouble(problem.grid.omega)/len(problem.barrier_ops.points)

        def work(tile):
            values = np.zeros((2, 3), dtype=np.longdouble)
            changes = []
            if not outer_only:
                a, b = [geometry(s.u, problem._gauss, tile=tile) for s in (base, trial)]
                ref = problem.controls._gauss_reference(tile)
                for i, geo in enumerate((a, b)):
                    values[i, 0] = .5*w.mu*np.sum(problem._gauss.weights*(geo.J-ref.J)**2, dtype=np.longdouble)
                    values[i, 1] = .5*w.nu*np.sum(problem._gauss.weights[None, :, None]*(geo.curl-ref.curl)**2, dtype=np.longdouble)
                changes.append(_change(np.sqrt(w.mu*problem._gauss.weights)*(a.J-ref.J),
                                       np.sqrt(w.mu*problem._gauss.weights)*(b.J-ref.J)))
                factor = np.sqrt(w.nu*problem._gauss.weights)[None, :, None]
                changes.append(_change(factor*(a.curl-ref.curl), factor*(b.curl-ref.curl)))
            a, b = [barrier_values(problem, geometry(s.u, problem.barrier_ops, tile=tile).J) for s in (base, trial)]
            for i, J in enumerate((a, b)):
                if np.any(J <= problem.certificate_config.delta):
                    raise FloatingPointError("CPU barrier points contradict reused whole-cell proof")
                values[i, 2] = np.sum(barrier_terms(J, delta=problem.certificate_config.delta,
                    weight=problem.barrier_weight, activation=problem.barrier_activation)[0], dtype=np.longdouble)
            change = weight*np.sum(_barrier_difference(a, b, delta=problem.certificate_config.delta,
                                                      activation=problem.barrier_activation), dtype=np.longdouble)
            return values, changes, change

        for values, changes, barrier_change in ordered_tile_map(problem, work):
            accum[:, 2:5] += values
            for change in changes:
                total += change
            barrier_changes.append(barrier_change)
        for change in barrier_changes:
            total += change
            outer += change
    energies = [EnergyTerms3D(*(float(x) for x in row)) for row in accum]
    require_finite_result([e.total for e in energies], "fused CPU energies")
    a, b = [SimpleNamespace(energy=e) for e in energies]
    checkpoint("cpu_comparison_complete")
    return ((_reduction(a, b, total, False), _reduction(a, b, outer, True)) if joint else
            _reduction(a, b, total, outer_only))
