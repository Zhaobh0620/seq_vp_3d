"""T06 frozen direct-Gauss controls and immutable problem identity (no outer loop)."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
import hashlib
import threading

import numpy as np

from ..validation import InputValidationError, StateMismatchError, as_float64_array, finite_real, readonly_copy
from .cell_q1 import CellQ1Operators3D
from .feasibility import CertificateConfig, FeasibilityResult, certify_map
from .geometry import compute_cell_geometry, _compute_cell_geometry
from .grid import Grid3D
from .interpolation import ImageSampler3D, build_image_sampler
from .validation import integer_at_least

QUADRATURE_ID = "q1-gauss27-xyz-v1"


_HASH_POOL = None
_PREFETCH = threading.local()


def _hash_key(value):
    return (id(value), value.__array_interface__["data"][0], value.shape, value.strides, value.dtype.str)


# E76 (explicit, off by default): memo of SHA-256 digests of immutable arrays. An array qualifies
# only if it and every ndarray base are non-writeable and the chain ends in an owned buffer, so
# its bytes cannot change while it is alive; entries hold a weak reference and are re-checked.
_IMMUTABLE_DIGESTS = None
_IMMUTABLE_LOCK = threading.Lock()


def enable_immutable_hash_cache(enabled=True):
    global _IMMUTABLE_DIGESTS
    with _IMMUTABLE_LOCK:
        _IMMUTABLE_DIGESTS = {} if enabled else None


def _immutable(value):
    a = value
    while isinstance(a, np.ndarray):
        if a.flags.writeable:
            return False
        a = a.base
    return a is None


def _cached_digest(value):
    memo = _IMMUTABLE_DIGESTS
    if memo is None or not isinstance(value, np.ndarray) or not _immutable(value):
        return None
    with _IMMUTABLE_LOCK:
        entry = memo.get(_hash_key(value))
    if entry is not None and entry[0]() is value:
        return entry[1]
    return None


def _store_digest(value, digest):
    import weakref
    memo = _IMMUTABLE_DIGESTS
    if memo is None or not isinstance(value, np.ndarray) or not _immutable(value):
        return
    with _IMMUTABLE_LOCK:
        if len(memo) > 512:
            for key in [k for k, (ref, _) in memo.items() if ref() is None]:
                del memo[key]
        memo[_hash_key(value)] = (weakref.ref(value), digest)


def _array_hash_memo(value):
    digest = _cached_digest(value)
    if digest is None:
        digest = _array_hash(value)
        _store_digest(value, digest)
    return digest


# E89 (explicit, off by default): facts proven on the exact bytes of an immutable array, with the
# same qualification, keying and weak-reference re-check as the digest memo. "finite": every value
# is finite; "finite_zero_boundary": finite and the six boundary faces are exactly zero (what a
# certified whole-map certificate establishes on its byte-identical copy).
_IMMUTABLE_FACTS = None
FACT_FINITE = "finite"
FACT_FINITE_ZERO_BOUNDARY = "finite_zero_boundary"


def enable_identity_facts(enabled=True):
    global _IMMUTABLE_FACTS
    with _IMMUTABLE_LOCK:
        _IMMUTABLE_FACTS = {} if enabled else None


def has_fact(value, name):
    memo = _IMMUTABLE_FACTS
    if memo is None or not isinstance(value, np.ndarray) or not _immutable(value):
        return False
    with _IMMUTABLE_LOCK:
        entry = memo.get(_hash_key(value))
    return entry is not None and entry[0]() is value and name in entry[1]


def store_fact(value, name):
    import weakref
    memo = _IMMUTABLE_FACTS
    if memo is None or not isinstance(value, np.ndarray) or not _immutable(value):
        return
    key = _hash_key(value)
    with _IMMUTABLE_LOCK:
        if len(memo) > 512:
            for k in [k for k, (ref, _) in memo.items() if ref() is None]:
                del memo[k]
        entry = memo.get(key)
        names = entry[1] if entry is not None and entry[0]() is value else frozenset()
        memo[key] = (weakref.ref(value), names | {name})


def seed_certified_map(u, certificate):
    """E89: after certify_map(u) returned a certified result for this exact immutable u, record what it
    proved on its byte-identical copy: SHA-256 = certificate.state_hash, finite values, zero boundary.
    No-op unless the memos are enabled and u is immutable (so its bytes cannot have changed)."""
    if certificate.status != "certified_feasible" or not isinstance(u, np.ndarray) or not _immutable(u):
        return
    _store_digest(u, certificate.state_hash)
    store_fact(u, FACT_FINITE_ZERO_BOUNDARY)


def submit_hash(value):
    """E89: a future of array_hash(value) for an immutable array, on the shared hash pool (one task per
    array, never nested); the memo is filled when it completes. The caller joins it before any use."""
    global _HASH_POOL
    if _HASH_POOL is None:
        _HASH_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="jd-curl-hash")
    hasher = _array_hash if _IMMUTABLE_DIGESTS is None else _array_hash_memo
    return _HASH_POOL.submit(hasher, value)


def array_hashes(values):
    """SHA-256 of several arrays concurrently (hashlib releases the GIL); same digests."""
    global _HASH_POOL
    values = list(values)
    hasher = _array_hash if _IMMUTABLE_DIGESTS is None else _array_hash_memo
    if len(values) < 2:
        return [hasher(v) for v in values]
    if _HASH_POOL is None:
        _HASH_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="jd-curl-hash")
    return list(_HASH_POOL.map(hasher, values))


@contextmanager
def prefetched_hashes(values):
    """Scoped memo: inside, array_hash of exactly these unchanged buffers reuses digests
    computed concurrently on entry. The caller must not mutate them within the scope."""
    arrays = [v for v in values if isinstance(v, np.ndarray)]
    digests = array_hashes(arrays)
    previous = getattr(_PREFETCH, "map", None)
    _PREFETCH.map = {_hash_key(v): d for v, d in zip(arrays, digests)}
    try:
        yield
    finally:
        _PREFETCH.map = previous


def array_hash(value):
    memo = getattr(_PREFETCH, "map", None)
    if memo is not None and isinstance(value, np.ndarray):
        digest = memo.get(_hash_key(value))
        if digest is not None:
            return digest
    if _IMMUTABLE_DIGESTS is not None:
        return _array_hash_memo(value)
    return _array_hash(value)


def _array_hash(value):
    # E48: a C-contiguous buffer IS its tobytes(order="C") byte string; hashing
    # the zero-copy view gives the identical SHA-256 without a full-array copy.
    if isinstance(value, np.ndarray) and value.flags.c_contiguous:
        try:
            return hashlib.sha256(memoryview(value).cast("B")).hexdigest()
        except (TypeError, ValueError):
            pass  # dtype without a plain buffer format: use the original bytes
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class Weights3D:
    lam: float
    mu: float
    nu: float
    kappa: float

    def __post_init__(self):
        for name in ("lam", "mu", "nu", "kappa"):
            object.__setattr__(self, name, finite_real(getattr(self, name), name, positive=True))


@dataclass(frozen=True)
class AblationWeights3D(Weights3D):
    """T92 (D2-a/b, ablation switch only): Weights3D with the JD or the curl change weight set to zero. Exactly one of
    mu, nu is 0 and the other positive; lam and kappa stay positive. Built only by the native-geometry-weights switch;
    every other route keeps Weights3D (equality and repr differ by class, so problem ids differ from Weights3D's)."""

    def __post_init__(self):
        for name in ("lam", "kappa"):
            object.__setattr__(self, name, finite_real(getattr(self, name), name, positive=True))
        for name in ("mu", "nu"):
            value = finite_real(getattr(self, name), name)
            object.__setattr__(self, name, 0. if value == 0. else value)     # -0.0 recorded as 0.0
        if not (min(self.mu, self.nu) == 0. and max(self.mu, self.nu) > 0.):
            raise InputValidationError("AblationWeights3D: exactly one of mu, nu is 0 and the other positive")


@dataclass(frozen=True)
class MemoryConfig:
    cell_tile_size: int = 8192
    spline_query_chunk: int = 65536

    def __post_init__(self):
        for name in ("cell_tile_size", "spline_query_chunk"):
            object.__setattr__(self, name, integer_at_least(getattr(self, name), name))


@dataclass(frozen=True)
class ParallelMemoryConfig(MemoryConfig):
    """Explicit ordered CPU tile evaluation; legacy MemoryConfig is unchanged."""
    cpu_workers: int = 2

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "cpu_workers", integer_at_least(self.cpu_workers, "cpu_workers"))


@dataclass(frozen=True)
class GaussReference3D:
    J: np.ndarray
    curl: np.ndarray
    controls_id: str
    grid_id: str
    quadrature_id: str
    tile: tuple[int, int]


@dataclass(frozen=True)
class Controls3D:
    """Factory-owned displacement; no nodal JD/curl controls or mutable provider."""
    u_ref: np.ndarray
    certificate: FeasibilityResult
    controls_id: str
    grid_id: str
    quadrature_id: str
    _ops: CellQ1Operators3D = field(repr=False)

    def gauss_reference(self, tile):
        geo = compute_cell_geometry(self.u_ref, self._ops, tile=tile)
        return self._reference(geo, tile)

    def _gauss_reference(self, tile):
        # Complete-volume internal streams validate every local tile.
        geo = _compute_cell_geometry(self.u_ref, self._ops, tile=tile)
        return self._reference(geo, tile)

    def _reference(self, geo, tile):
        return GaussReference3D(geo.J, geo.curl, self.controls_id, self.grid_id,
                                self.quadrature_id, (tile.start, tile.stop))


def make_controls(grid, u_ref, *, certificate_config=None, certificate_observer=None):
    config = CertificateConfig() if certificate_config is None else certificate_config
    u = readonly_copy(as_float64_array(u_ref, "u_ref", grid.shape+(3,)))
    certificate = certify_map(u, grid, config=config)
    if certificate_observer is not None:
        certificate_observer(certificate)
    if certificate.status != "certified_feasible":
        raise InputValidationError(f"u_ref: not certified feasible ({certificate.reason})")
    identity = hashlib.sha256((grid.grid_id+QUADRATURE_ID+certificate.state_hash).encode()).hexdigest()
    return Controls3D(u, certificate, identity, grid.grid_id, QUADRATURE_ID, CellQ1Operators3D(grid))


@dataclass(frozen=True)
class Problem3D:
    """Use factories; private operators and sampler must not be manually mutated."""
    grid: Grid3D
    fixed_image: np.ndarray
    sampler: ImageSampler3D
    controls: Controls3D
    weights: Weights3D
    memory: MemoryConfig
    certificate_config: CertificateConfig
    problem_id: str
    _gauss: CellQ1Operators3D = field(repr=False)
    _corners: CellQ1Operators3D = field(repr=False)
    backend_id: str = "numpy-float64-cpu"
    barrier_location: str = "corners8"
    _barrier: CellQ1Operators3D | None = field(default=None, repr=False)
    barrier_activation: float | None = None
    barrier_basis: str = "samples"   # T68 (E58): "bernstein" = barrier on the 27 Bernstein coefficients
    curvature_weight: float = 0.     # T76 (E78): alpha of R(u) = alpha/2 sum q |L u|^2 in the outer F (0 = absent)
    data_weight: np.ndarray | None = None   # T80 (E80): per-node m_i of the data term (None = 1 everywhere)
    active_weights: np.ndarray | None = None   # T89 (S30-G4): node weights w in [0, 1] of the PG direction (None = 1 everywhere)

    @property
    def barrier_ops(self):
        return self._corners if self.barrier_location == "corners8" else self._barrier

    @property
    def barrier_weight(self):
        return self.weights.kappa*self.grid.omega/len(self.barrier_ops.points)


def _problem_id(grid, fixed, sampler, controls, weights, memory, config, barrier_location="corners8", barrier_activation=None,
                barrier_basis="samples", curvature_weight=0., data_weight=None, active_weights=None):
    text = repr((grid.grid_id, array_hash(fixed), sampler.image_id, controls.controls_id,
                 weights, memory, config, "numpy-float64-cpu"))
    if barrier_location != "corners8":
        text += repr(("barrier-points-v1", barrier_location))
    if barrier_activation is not None:
        text += repr(("localized-c2-v1", barrier_activation))
    if barrier_basis != "samples":
        text += repr(("barrier-basis-v1", barrier_basis))   # default identity unchanged
    if curvature_weight != 0.:
        text += repr(("curvature-v1", float(curvature_weight)))   # T76 (E78); alpha = 0 keeps the identity
    if data_weight is not None:
        text += repr(("data-weight-v1", array_hash(data_weight)))   # T80 (E80); None keeps the identity
    if active_weights is not None:
        text += repr(("active-set-v1", array_hash(active_weights)))   # T89 (S30-G4); None keeps the identity
    return hashlib.sha256(text.encode()).hexdigest()


def make_problem(*, grid, fixed_image, moving_image, u_ref, weights, memory=None, certificate_config=None,
                 barrier_location="corners8", reference_certificate_observer=None, barrier_activation=None,
                 barrier_basis="samples", curvature_weight=0., data_weight=None, active_weights=None):
    if not isinstance(weights, Weights3D):
        raise InputValidationError("weights: Weights3D required")
    if data_weight is not None:
        # T80 (E80): per-node data-term weights, finite and nonnegative, on the grid nodes.
        data_weight = readonly_copy(as_float64_array(data_weight, "data_weight", grid.shape))
        if np.any(data_weight < 0.):
            raise InputValidationError("data_weight must be nonnegative")
    memory = MemoryConfig() if memory is None else memory
    config = CertificateConfig() if certificate_config is None else certificate_config
    if not isinstance(memory, MemoryConfig) or not isinstance(config, CertificateConfig):
        raise InputValidationError("memory/certificate_config: wrong config type")
    if barrier_location not in ("corners8", "closed27_uniform"):
        raise InputValidationError("barrier_location: expected corners8 or closed27_uniform")
    if barrier_activation is not None:
        barrier_activation = finite_real(barrier_activation, "barrier activation")
        if not config.delta < barrier_activation <= 1:
            raise InputValidationError("require delta < barrier_activation <= 1")
    if barrier_basis not in ("samples", "bernstein"):
        raise InputValidationError("barrier_basis: expected samples or bernstein")
    if barrier_basis == "bernstein" and (barrier_location != "closed27_uniform" or barrier_activation is None):
        raise InputValidationError("T68 Bernstein barrier requires the closed27 localized barrier")
    if active_weights is not None:
        # T89 (S30-G4): per-node weights of the PG direction, in [0, 1], on the grid nodes (never part of F)
        from .active_set import validate_active_weights
        active_weights = validate_active_weights(active_weights, grid)
    curvature_weight = finite_real(curvature_weight, "curvature_weight")   # T76 (E78): alpha >= 0
    if curvature_weight < 0:
        raise InputValidationError("curvature_weight must be nonnegative")
    fixed = readonly_copy(as_float64_array(fixed_image, "fixed_image", grid.shape))
    controls = make_controls(grid, u_ref, certificate_config=config,
                             certificate_observer=reference_certificate_observer)
    sampler = build_image_sampler(moving_image, grid)
    key = _problem_id(grid, fixed, sampler, controls, weights, memory, config, barrier_location, barrier_activation,
                      barrier_basis, curvature_weight, data_weight, active_weights)
    return Problem3D(grid, fixed, sampler, controls, weights, memory, config, key,
                     CellQ1Operators3D(grid), CellQ1Operators3D(grid, location="corners8"),
                     barrier_location=barrier_location,
                     barrier_activation=barrier_activation,
                     barrier_basis=barrier_basis,
                     curvature_weight=curvature_weight,
                     data_weight=data_weight,
                     active_weights=active_weights,
                     _barrier=None if barrier_location == "corners8" else CellQ1Operators3D(grid, location=barrier_location))


def with_controls(problem, controls):
    """Reuse image/spline/operator objects; a new inner objective gets a new ID."""
    if not isinstance(controls, Controls3D):
        raise InputValidationError("controls: Controls3D required")
    cert = controls.certificate
    if (controls.grid_id != problem.grid.grid_id or controls.quadrature_id != QUADRATURE_ID
            or cert.config != problem.certificate_config or cert.status != "certified_feasible"
            or cert.state_hash != array_hash(controls.u_ref)):
        raise StateMismatchError("controls/grid/quadrature/certificate mismatch")
    key = _problem_id(problem.grid, problem.fixed_image, problem.sampler, controls,
                      problem.weights, problem.memory, problem.certificate_config, problem.barrier_location,
                      problem.barrier_activation, problem.barrier_basis, problem.curvature_weight, problem.data_weight,
                      problem.active_weights)
    return replace(problem, controls=controls, problem_id=key)


def make_self_reference(problem, state):
    """Direct Gauss refresh from an evaluated base; not an outer acceptance rule."""
    from .residuals import AcceptedState3D, validate_state
    validate_state(problem, state)
    if not isinstance(state, AcceptedState3D):
        raise InputValidationError("self reference requires a promoted base state")
    return make_controls(problem.grid, state.u, certificate_config=problem.certificate_config)
