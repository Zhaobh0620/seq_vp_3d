"""T08/T09 matrix-free residual derivatives, full gradient and barrier-GGN.

Base/controls remain frozen; only tile geometry is recomputed. No K/H or
whole-volume Gauss cofactor/control cache is stored in Linearization3D.
"""
from dataclasses import dataclass
import itertools

import numpy as np

from ..validation import InputValidationError, StateMismatchError, as_float64_array, readonly_copy, require_finite_result
from .barrier import barrier_terms
from .bernstein_barrier import barrier_values, point_first
from .boundary import inject_free
from .cell_q1 import cell_tiles, cell_node_indices
from .data_weight import data_mass
from .curvature import (curvature_diagonal_nodes, curvature_gradient_nodes, curvature_hvp_nodes, curvature_residual,
                        residual_weight, transpose_laplacian_nodes)
from .geometry import _compute_cell_geometry as compute_cell_geometry
from .problem import Problem3D, QUADRATURE_ID
from .residuals import AcceptedState3D, ResidualBlock, block, validate_state
from .tile_execution import ordered_tile_map


@dataclass(frozen=True)
class Linearization3D:
    problem: Problem3D
    state: AcceptedState3D

    def __post_init__(self):
        validate_state(self.problem, self.state)
        if not isinstance(self.state, AcceptedState3D):
            raise InputValidationError("linearization requires a derivative-ready promoted base")


def build_linearization(problem, state):
    return Linearization3D(problem, state)


def _geometry_jvp(ops, A, v, tile):
    V = np.stack([np.stack([ops.derivative(v[..., r], axis, tile=tile) for axis in ("x", "y", "z")], axis=-1)
                  for r in range(3)], axis=-2)
    jd = np.sum(A*V, axis=(-2, -1))
    curl = np.stack((V[..., 2, 1]-V[..., 1, 2], V[..., 0, 2]-V[..., 2, 0], V[..., 1, 0]-V[..., 0, 1]), axis=-1)
    return jd, curl


def _geometry_flux(A, jd, curl):
    flux = A*jd[..., None, None]
    if curl is not None:
        flux[..., 2, 1] += curl[..., 0]
        flux[..., 1, 2] -= curl[..., 0]
        flux[..., 0, 2] += curl[..., 1]
        flux[..., 2, 0] -= curl[..., 1]
        flux[..., 1, 0] += curl[..., 2]
        flux[..., 0, 1] -= curl[..., 2]
    return flux


def _scatter_geometry(ops, A, jd, curl, tile, out, *, also_into=None):
    flux = _geometry_flux(A, jd, curl)
    indices = cell_node_indices(ops.grid, tile)
    for r in range(3):
        for j in range(3):
            ops._transpose_add(flux[..., r, j], ops.coefficients[j], indices, out[r],
                               also_into=None if also_into is None else also_into[r])


def _scatter_local(ops, A, jd, curl, tile):
    """E108: the per-tile half of _scatter_geometry (its flux, node indices and nine local products, in r/j order),
    for a worker thread; _scatter_local_into adds them in _scatter_geometry's r/j/vertex order."""
    flux = _geometry_flux(A, jd, curl)
    indices = cell_node_indices(ops.grid, tile)
    return indices, [ops._transpose_local(flux[..., r, j], ops.coefficients[j]) for r in range(3) for j in range(3)]


def _scatter_local_into(ops, parts, out, *, also_into=None):
    indices, local = parts
    for r in range(3):
        for j in range(3):
            ops._transpose_scatter(local[3*r+j], indices, out[r], also_into=None if also_into is None else also_into[r])


def _finish(out):
    require_finite_result(out, "complete derivative accumulator")
    result = out[:, 1:-1, 1:-1, 1:-1].copy().ravel(order="C")
    require_finite_result(result, "free derivative result")
    return result


def residual_jvp_blocks(lin, p):
    problem, state = lin.problem, lin.state
    v = inject_free(p, problem.grid)
    q, w = problem.grid.node_weights(), problem.weights
    with np.errstate(over="raise", invalid="raise"):
        image = np.sqrt(data_mass(problem))*np.sum(state.image_grad*v, axis=-1)   # T80
        proximity = np.sqrt(w.lam*q)[..., None]*v
    yield block(state, "image", image)
    yield block(state, "proximity", proximity)
    if problem.curvature_weight:
        yield block(state, "curvature", curvature_residual(problem, v))   # T76: linear in v
    del image, proximity, q
    for tile in cell_tiles(problem.grid, problem.memory.cell_tile_size):
        geo = compute_cell_geometry(state.u, problem._gauss, tile=tile)
        with np.errstate(over="raise", invalid="raise"):
            jd, curl = _geometry_jvp(problem._gauss, geo.A, v, tile)
            jd *= np.sqrt(w.mu*problem._gauss.weights)
            curl *= np.sqrt(w.nu*problem._gauss.weights)[None, :, None]
        yield block(state, "jd", jd, tile)
        yield block(state, "curl", curl, tile)


def residual_vjp_blocks(lin, blocks):
    """Consume exactly one canonical stream; missing/repeated/foreign blocks fail."""
    problem, state = lin.problem, lin.state
    iterator = iter(blocks)

    def read(name, shape, tile=None):
        try:
            item = next(iterator)
        except StopIteration as exc:
            raise StateMismatchError("missing residual block: "+name) from exc
        key = None if tile is None else (tile.start, tile.stop)
        if (not isinstance(item, ResidualBlock) or item.name != name or item.tile != key
                or item.problem_id != problem.problem_id or item.eval_id != state.eval_id
                or item.quadrature_id != QUADRATURE_ID):
            raise StateMismatchError("residual block order/identity mismatch: "+name)
        return as_float64_array(item.values, name+" VJP input", shape)

    q, w = problem.grid.node_weights(), problem.weights
    with np.errstate(over="raise", invalid="raise"):
        image = read("image", problem.grid.shape)*np.sqrt(data_mass(problem))   # T80
        proximity = read("proximity", problem.grid.shape+(3,))*np.sqrt(w.lam*q)[..., None]
        out = np.moveaxis(state.image_grad*image[..., None]+proximity, -1, 0).copy(order="C")
        if problem.curvature_weight:
            z = read("curvature", problem.grid.shape+(3,))*residual_weight(problem)
            out += np.moveaxis(transpose_laplacian_nodes(z, problem.grid), -1, 0)   # T76: L^T (interior rows only)
        for tile in cell_tiles(problem.grid, problem.memory.cell_tile_size):
            shape = (tile.stop-tile.start, 27)
            jd = read("jd", shape, tile)*np.sqrt(w.mu*problem._gauss.weights)
            curl = read("curl", shape+(3,), tile)*np.sqrt(w.nu*problem._gauss.weights)[None, :, None]
            geo = compute_cell_geometry(state.u, problem._gauss, tile=tile)
            _scatter_geometry(problem._gauss, geo.A, jd, curl, tile, out)
    sentinel = object()
    if next(iterator, sentinel) is not sentinel:
        raise StateMismatchError("unexpected extra residual block")
    return _finish(out)


def _image_gradient_nodes(problem, state, q):
    qd = data_mass(problem)   # T80: q m (q itself without data weights)
    return np.moveaxis(qd[..., None]*(state.warped-problem.fixed_image)[..., None]*state.image_grad, -1, 0)


def _proximity_gradient_nodes(problem, state, q):
    return np.moveaxis(problem.weights.lam*q[..., None]*(state.u-problem.controls.u_ref), -1, 0)


def _gauss_gradient_data(problem, state, tile, term, memo=None):
    geo = (compute_cell_geometry(state.u, problem._gauss, tile=tile) if memo is None
           else memo.geometry(state.u, problem._gauss))      # E108: the tile's shared geometry
    ref = problem.controls._gauss_reference(tile) if memo is None else memo.reference()
    w = problem.weights
    jd = w.mu*problem._gauss.weights*(geo.J-ref.J) if term != "curl" else np.zeros_like(geo.J)
    curl = w.nu*problem._gauss.weights[None, :, None]*(geo.curl-ref.curl) if term != "jd" else None
    return geo.A, jd, curl


def _barrier_gradient_data(problem, state, tile, memo=None):
    corner = (compute_cell_geometry(state.u, problem.barrier_ops, tile=tile) if memo is None
              else memo.geometry(state.u, problem.barrier_ops))
    if memo is not None and memo.barrier is not None:
        _, first, _ = memo.barrier      # E108: the energy's barrier_terms of the same barrier values
    else:
        _, first, _ = barrier_terms(barrier_values(problem, corner.J), delta=problem.certificate_config.delta,
                                     weight=problem.barrier_weight, activation=problem.barrier_activation)
    return corner.A, point_first(problem, first)


def energy_gradient(lin, *, term="total"):
    """Full, outer F or one energy term; all use the same local adjoint actions."""
    if term not in ("total", "outer", "image", "proximity", "jd", "curl", "barrier", "curvature"):
        raise InputValidationError("unknown energy gradient term")
    problem, state = lin.problem, lin.state
    q = problem.grid.node_weights()
    out = np.zeros((3,)+problem.grid.shape)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        if term in ("total", "outer", "image"):
            out += _image_gradient_nodes(problem, state, q)
        if term in ("total", "proximity"):
            out += _proximity_gradient_nodes(problem, state, q)
        if term in ("total", "outer", "curvature") and problem.curvature_weight:
            out += np.moveaxis(curvature_gradient_nodes(problem, state.u), -1, 0)   # T76
        for tile in cell_tiles(problem.grid, problem.memory.cell_tile_size):
            if term in ("total", "jd", "curl"):
                A, jd, curl = _gauss_gradient_data(problem, state, tile, term)
                _scatter_geometry(problem._gauss, A, jd, curl, tile, out)
            if term in ("total", "outer", "barrier"):
                A, first = _barrier_gradient_data(problem, state, tile)
                _scatter_geometry(problem.barrier_ops, A, first, None, tile, out)
    return _finish(out)


def _outer_node_gradients(problem, state):
    """The node terms of outer_gradient_vectors, in its order (caller's error state): F so far, gD, E so far."""
    q = problem.grid.node_weights()
    outer = np.zeros((3,)+problem.grid.shape)
    outer += _image_gradient_nodes(problem, state, q)
    image = _finish(outer)
    if problem.curvature_weight:
        outer += np.moveaxis(curvature_gradient_nodes(problem, state.u), -1, 0)   # T76: in F, not in D
    total = outer.copy()
    total += _proximity_gradient_nodes(problem, state, q)
    return outer, image, total


def outer_gradient_vectors(lin):
    """Return gF, gD, gE with shared work, preserving each scatter's order.

    Workers only compute local geometry/flux data. Global scatter stays on
    the caller thread, in the same tile/r/j/vertex order as energy_gradient.
    Barrier transpose products are reused, never summed with a soft field
    before scattering. This is not gE minus a separately rounded penalty.
    """
    problem, state = lin.problem, lin.state
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        outer, image, total = _outer_node_gradients(problem, state)

        def work(tile):
            return (tile, _gauss_gradient_data(problem, state, tile, "total"),
                    _barrier_gradient_data(problem, state, tile))

        for tile, soft, barrier in ordered_tile_map(problem, work):
            A, jd, curl = soft
            _scatter_geometry(problem._gauss, A, jd, curl, tile, total)
            A, first = barrier
            _scatter_geometry(problem.barrier_ops, A, first, None, tile, total, also_into=outer)
    return _finish(outer), image, _finish(total)


class _RecheckFields:
    """E108: the arrays of a CPU state that the outer gradient vectors read."""
    __slots__ = ("u", "warped", "image_grad")

    def __init__(self, u, warped, image_grad):
        self.u, self.warped, self.image_grad = u, warped, image_grad


def cpu_outer_recheck(problem, theta):
    """E108 (throughput v26, fused_recheck): sequential.norms' outer CPU gradient recheck in one tile pass.

    The default computes backend.cpu_state -- residuals.evaluate_trial (certificate, value sampling, energy) and
    residuals.promote_trial (value and gradient sampling, whose values must equal the first ones) -- and then
    outer_gradient_vectors. The recheck uses only the three vectors; the energy matters only through its failure
    checks. Here the same certificate, one sampling (the value-and-gradient call: its values come from the same spline
    calls on the same queries as the value-only call), the energy with all its checks and the three vectors are
    computed with each tile's Gauss and barrier-point geometry, Gauss reference and barrier terms evaluated once
    (residuals.TileGeometry), the tiles in the canonical order: every array is the same expression of the same inputs,
    so gF, gD, gE are the default's bit for bit. Every numerical failure raises FloatingPointError, as cpu_state does
    (the outer loop records NONFINITE_OUTER_CHECK for both)."""
    from ..types import InvalidTrial
    from .bernstein_barrier import BarrierDomainError
    from .feasibility import certify_map
    from .interpolation import sample_value_gradient
    from .residuals import TileGeometry, _cell_energy, _energy
    failed = "CPU re-evaluation of certified state failed: "
    caller = np.geterr()          # review E108 (S1): the sampling runs under the caller's error state, as in v25
    theta = as_float64_array(theta, "theta", (3*problem.grid.n_free_scalar,), check_finite=False)
    if not np.all(np.isfinite(theta)):
        raise FloatingPointError(failed+"nonfinite free displacement")
    theta = readonly_copy(theta)
    u = inject_free(theta, problem.grid)
    certificate = certify_map(u, problem.grid, config=problem.certificate_config)
    if certificate.status != "certified_feasible":
        raise FloatingPointError(failed+"trial not certified: "+certificate.reason)
    fields = readonly_copy(u)
    geometry_state = _RecheckFields(fields, None, None)   # the tile threads read u only (review E108, M4)
    sampled = []

    def work(tile):
        memo = TileGeometry(problem, fields, tile)
        energy = _cell_energy(problem, fields, tile, memo=memo)
        soft = _gauss_gradient_data(problem, geometry_state, tile, "total", memo=memo)
        memo.release(problem._gauss, reference=True)       # review E108 (S2): the job's peak memory
        soft = _scatter_local(problem._gauss, *soft, tile)
        A, derivative = _barrier_gradient_data(problem, geometry_state, tile, memo=memo)
        del memo
        return energy, soft, _scatter_local(problem.barrier_ops, A, derivative, None, tile)

    def sample():
        # on the caller thread (its work-budget checkpoints), while the first tiles run (they read u only)
        with np.errstate(**caller):
            result = sample_value_gradient(problem.sampler, problem.grid.identity()+u,
                                           chunk_size=problem.memory.spline_query_chunk)
        if isinstance(result, InvalidTrial):
            raise FloatingPointError(failed+result.reason)
        sampled.append(result)

    items = None
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            items = ordered_tile_map(problem, work, after_submit=sample)
            first = next(items)
            state = _RecheckFields(fields, sampled[0].value, sampled[0].gradient)
            outer, image, total = _outer_node_gradients(problem, state)

            def cells():
                # outer_gradient_vectors' scatter, in its tile/r/j/vertex order, while the energy sums its terms
                for energy, soft, barrier in itertools.chain((first,), items):
                    _scatter_local_into(problem._gauss, soft, total)
                    _scatter_local_into(problem.barrier_ops, barrier, total, also_into=outer)
                    yield energy

            _energy(problem, fields, state.warped, cells=cells())
    except BarrierDomainError as exc:
        raise FloatingPointError(failed+str(exc)) from exc
    finally:
        if items is not None:
            items.close()       # an early failure never leaves tile jobs running
    return _finish(outer), image, _finish(total)


def image_barrier_gradient(lin):
    return energy_gradient(lin, term="outer")


def barrier_ggn_hvp(lin, p):
    """Full H_C=K0.T K0+L_C.T diag(b_B'') L_C (not only the B block).

    Fuse each local JVP/VJP to avoid computing the base cofactor twice.
    No reference geometry is needed by this frozen GGN action.
    """
    problem, state = lin.problem, lin.state
    v = inject_free(p, problem.grid)
    q, w = problem.grid.node_weights(), problem.weights
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        node = (data_mass(problem)[..., None]*state.image_grad*np.sum(state.image_grad*v, axis=-1)[..., None]
                + w.lam*q[..., None]*v)   # T80: data weights on the image block only
        out = np.moveaxis(node, -1, 0).copy(order="C")
        if problem.curvature_weight:
            out += np.moveaxis(curvature_hvp_nodes(problem, v), -1, 0)   # T76: exact Hessian of R
        for tile in cell_tiles(problem.grid, problem.memory.cell_tile_size):
            geo = compute_cell_geometry(state.u, problem._gauss, tile=tile)
            jd, curl = _geometry_jvp(problem._gauss, geo.A, v, tile)
            _scatter_geometry(problem._gauss, geo.A, w.mu*problem._gauss.weights*jd,
                               w.nu*problem._gauss.weights[None, :, None]*curl, tile, out)
            corner = compute_cell_geometry(state.u, problem.barrier_ops, tile=tile)
            _, _, curvature = barrier_terms(barrier_values(problem, corner.J), delta=problem.certificate_config.delta,
                                             weight=problem.barrier_weight, activation=problem.barrier_activation)
            lj, _ = _geometry_jvp(problem.barrier_ops, corner.A, v, tile)
            # T68: T^T diag(psi'') T acts on the point JVP; identity for the sample basis.
            _scatter_geometry(problem.barrier_ops, corner.A,
                              point_first(problem, curvature*barrier_values(problem, lj)), None, tile, out)
    return _finish(out)


def barrier_ggn_diagonal(lin):
    """Exact diagonal of the frozen full GGN; no squared averaged nodal JD."""
    problem, state = lin.problem, lin.state
    q, w = problem.grid.node_weights(), problem.weights
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        out = np.moveaxis(data_mass(problem)[..., None]*state.image_grad**2+w.lam*q[..., None], -1, 0).copy(order="C")   # T80
        if problem.curvature_weight:
            out += np.moveaxis(curvature_diagonal_nodes(problem), -1, 0)   # T76
        for tile in cell_tiles(problem.grid, problem.memory.cell_tile_size):
            indices = cell_node_indices(problem.grid, tile)
            geo = compute_cell_geometry(state.u, problem._gauss, tile=tile)
            corner = compute_cell_geometry(state.u, problem.barrier_ops, tile=tile)
            _, _, curvature = barrier_terms(barrier_values(problem, corner.J), delta=problem.certificate_config.delta,
                                             weight=problem.barrier_weight, activation=problem.barrier_activation)
            D, Dc = problem._gauss.coefficients, problem.barrier_ops.coefficients
            for r in range(3):
                for a in range(8):
                    lj = sum(geo.A[..., r, j]*D[j, :, a] for j in range(3))
                    lc = sum(corner.A[..., r, j]*Dc[j, :, a] for j in range(3))
                    curl_square = sum(D[j, :, a]**2 for j in range(3) if j != r)
                    value = np.sum(problem._gauss.weights*(w.mu*lj**2+w.nu*curl_square), axis=-1)
                    value += np.sum(curvature*barrier_values(problem, lc)**2, axis=-1)
                    out[r].ravel()[indices[:, a]] += value
    return _finish(out)
