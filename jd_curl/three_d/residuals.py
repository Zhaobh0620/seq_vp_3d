"""T04/T06/T08 certified trial energy and streamed non-barrier R0 blocks."""
from dataclasses import dataclass
import hashlib

import numpy as np

from ..types import InvalidTrial
from ..validation import InputValidationError, StateMismatchError, as_float64_array, readonly_copy, require_finite_result
from .barrier import barrier_terms
from .bernstein_barrier import BarrierDomainError, barrier_values
from .boundary import inject_free
from .cell_q1 import cell_tiles
from .data_weight import data_mass
from .curvature import curvature_energy, curvature_residual
from .feasibility import FeasibilityResult, certify_map
from .geometry import _compute_cell_geometry as compute_cell_geometry
from .interpolation import sample_points, sample_value_gradient
from .problem import QUADRATURE_ID
from .tile_execution import ordered_tile_map


@dataclass(frozen=True)
class EnergyTerms3D:
    image: float
    proximity: float
    jd: float
    curl: float
    barrier: float
    curvature: float = 0.   # T76 (E78): absolute curvature regularizer; part of the outer F

    @property
    def total(self):
        return self.image+self.proximity+self.jd+self.curl+self.barrier+self.curvature

    @property
    def outer(self):
        return self.image+self.barrier+self.curvature


@dataclass(frozen=True)
class TrialState3D:
    theta: np.ndarray
    u: np.ndarray
    warped: np.ndarray
    energy: EnergyTerms3D
    certificate: FeasibilityResult
    problem_id: str
    eval_id: str


@dataclass(frozen=True)
class AcceptedState3D(TrialState3D):
    """Derivative-ready base. Promotion itself does NOT imply optimizer acceptance."""
    image_grad: np.ndarray


@dataclass(frozen=True)
class ResidualBlock:
    name: str
    tile: tuple[int, int] | None
    values: np.ndarray
    problem_id: str
    eval_id: str
    quadrature_id: str = QUADRATURE_ID


def validate_state(problem, state):
    if not isinstance(state, TrialState3D):
        raise InputValidationError("state: evaluated TrialState3D required")
    if (state.problem_id != problem.problem_id or state.certificate.grid_id != problem.grid.grid_id
            or state.certificate.config != problem.certificate_config
            or state.certificate.status != "certified_feasible"):
        raise StateMismatchError("state does not belong to this certified problem/reference")


def block(state, name, values, tile=None):
    require_finite_result(values, name+" residual block")
    key = None if tile is None else (tile.start, tile.stop)
    return ResidualBlock(name, key, readonly_copy(values), state.problem_id, state.eval_id)


def _node_residual_blocks(problem, state, *, outer_only=False):
    """image, then proximity (E only), then the T76 curvature block (only when alpha > 0)."""
    q, w = problem.grid.node_weights(), problem.weights
    with np.errstate(over="raise", invalid="raise"):
        image = np.sqrt(data_mass(problem))*(state.warped-problem.fixed_image)   # T80: q m on the image term
    yield block(state, "image", image)
    if not outer_only:
        with np.errstate(over="raise", invalid="raise"):
            proximity = np.sqrt(w.lam*q)[..., None]*(state.u-problem.controls.u_ref)
        yield block(state, "proximity", proximity)
    if problem.curvature_weight:
        yield block(state, "curvature", curvature_residual(problem, state.u))


def _geometry_residual_blocks(problem, state, tile, ref):
    geo = compute_cell_geometry(state.u, problem._gauss, tile=tile)
    w = problem.weights
    with np.errstate(over="raise", invalid="raise"):
        jd = np.sqrt(w.mu*problem._gauss.weights)*(geo.J-ref.J)
        curl = np.sqrt(w.nu*problem._gauss.weights)[None, :, None]*(geo.curl-ref.curl)
    yield block(state, "jd", jd, tile)
    yield block(state, "curl", curl, tile)


def residual_blocks(problem, state):
    """Canonical stream: node image/proximity, then JD/curl for each cell tile."""
    validate_state(problem, state)
    yield from _node_residual_blocks(problem, state)
    for tile in cell_tiles(problem.grid, problem.memory.cell_tile_size):
        ref = problem.controls._gauss_reference(tile)
        yield from _geometry_residual_blocks(problem, state, tile, ref)


def _residual_block_pairs(problem, base, trial, *, outer_only=False):
    """Same canonical blocks; share only this tile's immutable reference.

    No cache survives the call/tile. Both states must have the same frozen
    problem identity; F comparisons never evaluate proximity or Gauss terms.
    """
    for state in (base, trial):
        validate_state(problem, state)
    for pair in zip(_node_residual_blocks(problem, base, outer_only=outer_only),
                    _node_residual_blocks(problem, trial, outer_only=outer_only), strict=True):
        yield pair
    if outer_only:
        return
    del pair
    for tile in cell_tiles(problem.grid, problem.memory.cell_tile_size):
        yield from _geometry_residual_block_pairs(problem, base, trial, tile)


def _geometry_residual_block_pairs(problem, base, trial, tile):
    ref = problem.controls._gauss_reference(tile)
    yield from zip(_geometry_residual_blocks(problem, base, tile, ref),
                   _geometry_residual_blocks(problem, trial, tile, ref), strict=True)


class TileGeometry:
    """E108 (fused_recheck): one tile's Gauss and barrier-point geometry of one u, its Gauss reference and its barrier
    terms, each computed once by the same call and shared by the energy and the gradients of that u on that tile."""

    def __init__(self, problem, u, tile):
        self._problem, self._u, self._tile = problem, u, tile
        self._geometry, self._reference, self.barrier = {}, None, None

    def geometry(self, u, ops):
        if u is not self._u:
            raise StateMismatchError("tile geometry shared by a different u")
        if id(ops) not in self._geometry:
            self._geometry[id(ops)] = (ops, compute_cell_geometry(u, ops, tile=self._tile))
        return self._geometry[id(ops)][1]

    def reference(self):
        if self._reference is None:
            self._reference = self._problem.controls._gauss_reference(self._tile)
        return self._reference

    def release(self, ops, *, reference=False):
        """Drop this tile's geometry for ops (and the Gauss reference): arrays a caller kept stay alive."""
        self._geometry.pop(id(ops), None)
        if reference:
            self._reference = None


def _cell_energy(problem, u, tile, memo=None):
    w = problem.weights
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        geo = compute_cell_geometry(u, problem._gauss, tile=tile) if memo is None else memo.geometry(u, problem._gauss)
        ref = problem.controls._gauss_reference(tile) if memo is None else memo.reference()
        jd = .5*w.mu*np.sum(problem._gauss.weights*(geo.J-ref.J)**2, dtype=np.longdouble)
        curl = .5*w.nu*np.sum(problem._gauss.weights[None, :, None]*(geo.curl-ref.curl)**2, dtype=np.longdouble)
        corner = (compute_cell_geometry(u, problem.barrier_ops, tile=tile) if memo is None
                  else memo.geometry(u, problem.barrier_ops))
        if np.any(corner.J <= problem.certificate_config.delta):
            raise FloatingPointError("floating barrier-point evaluation contradicts feasible certificate")
        values = barrier_values(problem, corner.J)
        if problem.barrier_basis != "samples" and np.any(values <= problem.certificate_config.delta):
            # T68: J > delta on the cell is certified, but a Bernstein coefficient is not: outside
            # the (stricter) barrier domain; a regular trial rejection, not a numerical failure.
            raise BarrierDomainError("Bernstein barrier coefficient <= delta")
        terms = barrier_terms(values, delta=problem.certificate_config.delta, weight=problem.barrier_weight, activation=problem.barrier_activation)
        if memo is not None:
            memo.barrier = terms     # E108: barrier_terms(barrier_values(problem, corner.J)) for the gradient
        values = terms[0]
        return jd, curl, np.sum(values, dtype=np.longdouble)


def _energy(problem, u, warped, *, cells=None):
    """cells (E108): the per-tile (jd, curl, barrier) stream; default _cell_energy over ordered_tile_map."""
    q, w = problem.grid.node_weights(), problem.weights
    accum = dict(jd=np.longdouble(0), curl=np.longdouble(0), barrier=np.longdouble(0))
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        image = .5*np.sum(data_mass(problem)*(warped-problem.fixed_image)**2, dtype=np.longdouble)   # T80
        proximity = .5*w.lam*np.sum(q[..., None]*(u-problem.controls.u_ref)**2, dtype=np.longdouble)
        if cells is None:
            cells = ordered_tile_map(problem, lambda tile: _cell_energy(problem, u, tile))
        for jd, curl, barrier in cells:
            accum["jd"] += jd
            accum["curl"] += curl
            accum["barrier"] += barrier
        curvature = curvature_energy(problem, u) if problem.curvature_weight else np.longdouble(0)
    result = EnergyTerms3D(float(image), float(proximity), *(float(accum[k]) for k in ("jd", "curl", "barrier")),
                           float(curvature))
    require_finite_result([result.total, result.outer], "energy")
    return result


def evaluate_trial(problem, theta, *, certificate_observer=None):
    theta = as_float64_array(theta, "theta", (3*problem.grid.n_free_scalar,), check_finite=False)
    if not np.all(np.isfinite(theta)):
        return InvalidTrial("NONFINITE_STATE", "nonfinite free displacement", int(np.sum(~np.isfinite(theta))))
    theta = readonly_copy(theta)
    u = inject_free(theta, problem.grid)
    certificate = certify_map(u, problem.grid, config=problem.certificate_config)
    if certificate_observer is not None:
        certificate_observer(certificate)
    if certificate.status != "certified_feasible":
        reason = "FEASIBILITY_UNCERTAIN" if certificate.status == "uncertain" else certificate.reason
        return InvalidTrial(reason, "trial not certified: "+certificate.reason,
                            certificate.n_uncertain+certificate.n_infeasible)
    strict = np.zeros(problem.grid.shape, dtype=bool)
    strict[1:-1, 1:-1, 1:-1] = True
    sampled = sample_points(problem.sampler, problem.grid.identity()+u, strict=strict, with_gradient=False,
                            chunk_size=problem.memory.spline_query_chunk)
    if isinstance(sampled, InvalidTrial):
        return sampled
    try:
        energy = _energy(problem, u, sampled.value)
    except BarrierDomainError as exc:
        return InvalidTrial("BARRIER_DOMAIN", str(exc), 1)
    except FloatingPointError as exc:
        return InvalidTrial("NONFINITE_STATE", str(exc), 1)
    identity = hashlib.sha256((problem.problem_id+certificate.state_hash).encode()).hexdigest()
    return TrialState3D(theta, readonly_copy(u), sampled.value, energy, certificate, problem.problem_id, identity)


def promote_trial(problem, trial):
    """Prepare same-map derivatives only; C must supply the acceptance decision."""
    validate_state(problem, trial)
    if getattr(trial, "evaluation_backend_id", None) is not None:
        raise StateMismatchError("device-evaluated trial requires its paired promotion backend")
    sampled = sample_value_gradient(problem.sampler, problem.grid.identity()+trial.u,
                                    chunk_size=problem.memory.spline_query_chunk)
    if isinstance(sampled, InvalidTrial):
        return sampled
    if not np.array_equal(sampled.value, trial.warped):
        raise StateMismatchError("promoted sampler value changed from certified trial")
    return AcceptedState3D(trial.theta, trial.u, trial.warped, trial.energy, trial.certificate,
                            trial.problem_id, trial.eval_id, sampled.gradient)
