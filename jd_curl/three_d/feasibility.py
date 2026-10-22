"""T05 whole-cell physical JD certification; finite samples never prove safety."""
from dataclasses import dataclass, replace
from functools import partial
from itertools import product
import hashlib
import heapq

import numpy as np

from ..validation import InputValidationError, as_float64_array, finite_real
from .cell_q1 import VERTICES, cell_node_indices, cell_tiles
from .geometry import determinant
from .interpolation import check_map_domain
from . import interval as iv
from .validation import integer_at_least, zero_boundary
from .work_control import checkpoint

CERTIFICATE_VERSION = "q1-3d-interval-bernstein-v1"
POINTS = np.array([(x, y, z) for z, y, x in product((0., .5, 1.), repeat=3)])


@dataclass(frozen=True)
class CertificateConfig:
    delta: float = .01
    max_depth: int = 12
    max_boxes: int = 255
    tile_size: int = 8192

    def __post_init__(self):
        d = finite_real(self.delta, "delta")
        if d < 0 or d >= 1:
            raise InputValidationError("delta: require 0 <= delta < 1")
        if d == 0:
            d = 0.0  # Canonical identity for -0.0; the inequality stays strict.
        object.__setattr__(self, "delta", d)
        for key, minimum in (("max_depth", 0), ("max_boxes", 1), ("tile_size", 1)):
            object.__setattr__(self, key, integer_at_least(getattr(self, key), key, minimum))


@dataclass(frozen=True)
class AdaptiveCertificateConfig(CertificateConfig):
    """Explicit T24 bounded lowest-lower-bound-first search; legacy unchanged."""
    max_depth: int = 48
    policy: str = "lowest_bound_first_v1"

    def __post_init__(self):
        super().__post_init__()
        if self.policy != "lowest_bound_first_v1":
            raise InputValidationError("unsupported adaptive certificate policy")
        # Longest-axis splits give at most 40 exact dyadic levels per axis.
        if self.max_depth > 120:
            raise InputValidationError("adaptive max_depth must be <= 120")


@dataclass(frozen=True)
class EnclosureCertificateConfig(AdaptiveCertificateConfig):
    """T30 edge-hull whole-cell preproof, then the unchanged adaptive proof."""
    prepass: str = "edge_hull_v1"

    def __post_init__(self):
        super().__post_init__()
        if self.prepass != "edge_hull_v1":
            raise InputValidationError("unsupported enclosure prepass")


@dataclass(frozen=True)
class CompiledEnclosureCertificateConfig(EnclosureCertificateConfig):
    """Explicit E32 execution, with the same outward edge-hull operations."""
    kernel: str = "numba-directed-edge-hull-v1"

    def __post_init__(self):
        super().__post_init__()
        if self.kernel not in ("numba-directed-edge-hull-v1", "numba-hoisted-edge-hull-v1"):
            raise InputValidationError("unsupported compiled enclosure kernel")


@dataclass(frozen=True)
class ParallelEnclosureCertificateConfig(CompiledEnclosureCertificateConfig):
    """Explicit E50 execution: the E33 hoisted per-cell arithmetic over the whole
    map in parallel, plus threaded Bernstein fallback per tile. Endpoints, cell
    coverage, witnesses and counts equal the hoisted kernel's; only the recorded
    execution identity (kernel/version/workers) differs. Legacy classes unchanged."""
    kernel: str = "numba-parallel-hoisted-edge-hull-v1"
    fallback_workers: int = 8

    def __post_init__(self):
        EnclosureCertificateConfig.__post_init__(self)
        if self.kernel != "numba-parallel-hoisted-edge-hull-v1":
            raise InputValidationError("parallel certificate requires the parallel hoisted kernel")
        object.__setattr__(self, "fallback_workers", integer_at_least(self.fallback_workers, "fallback_workers"))


@dataclass(frozen=True)
class CompiledFallbackCertificateConfig(ParallelEnclosureCertificateConfig):
    """Explicit E54 (T65) execution: E50 plus the compiled Bernstein-fallback arrays
    (compiled_fallback.fallback_arrays), bitwise equal to _fallback_arrays including
    the whole-job arithmetic-failure rule. Decisions, endpoints and counts unchanged."""
    fallback_kernel: str = "numba-bernstein-fallback-v1"

    def __post_init__(self):
        super().__post_init__()
        if self.fallback_kernel != "numba-bernstein-fallback-v1":
            raise InputValidationError("unsupported compiled fallback kernel")


@dataclass(frozen=True)
class PrescreenCertificateConfig(CompiledFallbackCertificateConfig):
    """Explicit E71 (T75) execution: E54 plus a rigorous float64 Bernstein prescreen of the
    cells the edge-hull prepass did not prove (certificate_prescreen: min beta_fl minus an
    a-priori rounding bound). Proven cells skip the interval fallback; the others take it
    unchanged. The prescreen only adds proofs; it never declares a cell infeasible."""
    prescreen_kernel: str = "numba-float-bernstein-apriori-v1"

    def __post_init__(self):
        super().__post_init__()
        if self.prescreen_kernel != "numba-float-bernstein-apriori-v1":
            raise InputValidationError("unsupported prescreen kernel")


def certificate_version(config):
    if isinstance(config, PrescreenCertificateConfig):
        return "q1-3d-parallel-hoisted-edge-hull-float-bernstein-prescreen-compiled-bernstein-v1"
    if isinstance(config, CompiledFallbackCertificateConfig):
        return "q1-3d-parallel-hoisted-edge-hull-compiled-bernstein-v1"
    if isinstance(config, ParallelEnclosureCertificateConfig):
        return "q1-3d-parallel-hoisted-edge-hull-bernstein-v1"
    if isinstance(config, CompiledEnclosureCertificateConfig):
        if config.kernel == "numba-hoisted-edge-hull-v1":
            return "q1-3d-hoisted-edge-hull-bernstein-v1"
        return "q1-3d-compiled-edge-hull-bernstein-v1"
    if isinstance(config, EnclosureCertificateConfig):
        return "q1-3d-edge-hull-bernstein-v1"
    return CERTIFICATE_VERSION


@dataclass(frozen=True)
class CellCertificate:
    status: str
    min_jd_lower_bound: float | None
    min_margin_lower_bound: float | None
    sampled_min_jd: float | None
    n_boxes: int
    max_depth: int
    witness_reference_point: tuple | None
    reason: str


@dataclass(frozen=True)
class FeasibilityResult:
    status: str
    min_jd_lower_bound: float | None
    min_margin_lower_bound: float | None
    sampled_min_jd: float | None
    n_cells_checked: int
    n_uncertain: int
    n_infeasible: int
    n_boxes: int
    max_depth: int
    witness_cell: int | None
    witness_reference_point: tuple | None
    reason: str
    state_hash: str
    grid_id: str
    delta_hex: str
    config: CertificateConfig
    certificate_version: str = CERTIFICATE_VERSION


@dataclass(frozen=True)
class EnclosureFeasibilityResult(FeasibilityResult):
    certificate_version: str = "q1-3d-edge-hull-bernstein-v1"
    n_fast_cells: int = 0
    n_fallback_cells: int = 0


@dataclass(frozen=True)
class PrescreenFeasibilityResult(EnclosureFeasibilityResult):
    """E71: n_fallback_cells counts only the cells the prescreen left to the interval fallback."""
    n_prescreen_cells: int = 0


def _spacing(spacing):
    if len(spacing) != 3:
        raise InputValidationError("spacing must contain hx,hy,hz")
    return tuple(finite_real(h, "spacing", positive=True) for h in spacing)


def jacobian_enclosure(vertices, spacing):
    """Whole-cell physical JD enclosure, NOT a finite-sample check.

    Every Q1 derivative is a convex combination of four parallel edge
    differences. Hull their outward intervals before interval determinant
    evaluation. Dependency loss can be large; failure to prove is inconclusive.
    """
    u = as_float64_array(vertices, "vertices")
    if u.ndim != 3 or u.shape[1:] != (8, 3):
        raise InputValidationError("vertices: expected (cells,8,3)")
    spacing = _spacing(spacing)
    values = iv.Interval.point(u)
    F = [[None]*3 for _ in range(3)]
    for j in range(3):
        edges = []
        for a, bits in enumerate(VERTICES):
            if bits[j]:
                continue
            other = list(bits)
            other[j] = 1
            b = VERTICES.index(tuple(other))
            edges.append((values[:, b, :] - values[:, a, :]) / spacing[j])
        hull = iv.Interval(np.minimum.reduce([e.lo for e in edges]),
                           np.maximum.reduce([e.hi for e in edges]))
        for r in range(3):
            F[r][j] = hull[:, r] + (1. if r == j else 0.)
    a = F[1][1]*F[2][2] - F[1][2]*F[2][1]
    b = F[1][2]*F[2][0] - F[1][0]*F[2][2]
    c = F[1][0]*F[2][1] - F[1][1]*F[2][0]
    return F[0][0]*a + F[0][1]*b + F[0][2]*c


def compiled_jacobian_enclosure(vertices, spacing, *, kernel="numba-directed-edge-hull-v1"):
    """Same public guards; only the explicitly selected kernel imports Numba."""
    u = as_float64_array(vertices, "vertices")
    if u.ndim != 3 or u.shape[1:] != (8, 3):
        raise InputValidationError("vertices: expected (cells,8,3)")
    spacing = _spacing(spacing)
    from .compiled_enclosure import enclosure
    return enclosure(u, spacing, kernel=kernel)


def _enclosure_kernel(config):
    if isinstance(config, ParallelEnclosureCertificateConfig):
        # Single cells use the identical per-cell hoisted arithmetic.
        return partial(compiled_jacobian_enclosure, kernel="numba-hoisted-edge-hull-v1")
    if isinstance(config, CompiledEnclosureCertificateConfig) and config.kernel == "numba-hoisted-edge-hull-v1":
        return partial(compiled_jacobian_enclosure, kernel=config.kernel)
    return compiled_jacobian_enclosure if isinstance(config, CompiledEnclosureCertificateConfig) else jacobian_enclosure


def jacobian_intervals(vertices, spacing):
    """Interval J at {0,.5,1}^3, for tiles of local *displacement* vertices.

    Shapes: (cells,8,3) -> (cells,3,3,3), trailing axes reference z/y/x.
    Edge differences then bilinear interpolation implement the exact Q1
    derivative. All weights here are exact dyadic constants (0,.25,.5,1).
    """
    u = as_float64_array(vertices, "vertices")
    if u.ndim != 3 or u.shape[1:] != (8, 3):
        raise InputValidationError("vertices: expected (cells,8,3)")
    spacing = _spacing(spacing)
    values = iv.Interval.point(u)
    F = [[None]*3 for _ in range(3)]
    for j in range(3):
        derivative = iv.Interval.point(np.zeros((len(u), 27, 3)))
        for a, bits in enumerate(VERTICES):
            if bits[j]:
                continue
            other = list(bits)
            other[j] = 1
            b = VERTICES.index(tuple(other))
            weight = np.ones(27)
            for k in range(3):
                if k != j:
                    weight *= POINTS[:, k] if bits[k] else 1-POINTS[:, k]
            edge = (values[:, b, :] - values[:, a, :]) / spacing[j]
            derivative = derivative + edge[:, None, :].scale_nonnegative(weight[None, :, None])
        for r in range(3):
            F[r][j] = derivative[..., r] + (1. if r == j else 0.)
    # Explicit determinant, each elementary operation encloses its exact result.
    a = F[1][1]*F[2][2] - F[1][2]*F[2][1]
    b = F[1][2]*F[2][0] - F[1][0]*F[2][2]
    c = F[1][0]*F[2][1] - F[1][1]*F[2][0]
    J = F[0][0]*a + F[0][1]*b + F[0][2]*c
    return iv.Interval(J.lo.reshape(-1, 3, 3, 3), J.hi.reshape(-1, 3, 3, 3))


def _sample_jacobians(vertices, spacing):
    """Ordinary float64 diagnostic only; not used for certification decisions."""
    F = np.broadcast_to(np.eye(3), (len(vertices), 27, 3, 3)).copy()
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for j in range(3):
            for a, bits in enumerate(VERTICES):
                if bits[j]:
                    continue
                other = list(bits)
                other[j] = 1
                b = VERTICES.index(tuple(other))
                weight = np.ones(27)
                for k in range(3):
                    if k != j:
                        weight *= POINTS[:, k] if bits[k] else 1-POINTS[:, k]
                F[..., j] += ((vertices[:, b]-vertices[:, a])/spacing[j])[:, None, :]*weight[None, :, None]
        return determinant(F).reshape(-1, 3, 3, 3)


def _classify(coefficients, p_samples, delta, sampled_min, config, point_observer=None):
    root_lower = float(coefficients.lo.min())

    def result(status, margin, boxes, depth, witness, reason):
        try:
            lower = margin if delta == 0 else float((iv.Interval.point(margin)+delta).lo)
        except FloatingPointError:
            return CellCertificate("uncertain", None, None, sampled_min, boxes, depth, None, "arithmetic_uncertain")
        return CellCertificate(status, lower, margin, sampled_min, boxes, depth, witness, reason)

    witness = np.argwhere(p_samples.hi <= 0)
    if len(witness):
        point = tuple(float(v)/2 for v in witness[0][::-1])
        return result("infeasible", root_lower, 1, 0, point, "sample_witness")
    adaptive = isinstance(config, AdaptiveCertificateConfig)
    root = (coefficients, np.zeros(3), np.ones(3), 0)
    pending = [(root_lower, 0, root)] if adaptive else [root]
    serial = 0
    boxes, depth_used, uncertain = 0, 0, False
    certified_lowers = []
    while pending:
        checkpoint("jd_subdivision")
        if boxes >= config.max_boxes:
            return result("uncertain", root_lower, boxes, depth_used, None, "box_budget")
        current, low, high, depth = heapq.heappop(pending)[2] if adaptive else pending.pop()
        boxes += 1
        depth_used = max(depth_used, depth)
        minimum = float(current.lo.min())
        if point_observer is not None:
            # Local Bernstein CORNERS equal point values. Emit coordinates and
            # bounds only; a diagnostic must re-evaluate physical J/derivatives.
            for iz, iy, ix in product((0, 1), repeat=3):
                point = tuple(np.where((ix, iy, iz), high, low).tolist())
                point_observer(point, float(current.lo[2*iz,2*iy,2*ix]),
                               float(current.hi[2*iz,2*iy,2*ix]))
        if minimum > 0:
            certified_lowers.append(minimum)
            continue
        # Corner coefficients equal actual polynomial values, unlike interior coefficients.
        corners = current.hi[::2, ::2, ::2]
        witnesses = np.argwhere(corners <= 0)
        if len(witnesses):
            point = tuple(np.where(witnesses[0][::-1], high, low).tolist())
            return result("infeasible", root_lower, boxes, depth_used, point, "subdivision_witness")
        if depth >= config.max_depth:
            uncertain = True
            continue
        axis = int(np.argmax(high-low))  # reference xyz tie order
        try:
            left, right = iv.split_bernstein(current, axis)
        except FloatingPointError:
            return result("uncertain", root_lower, boxes, depth_used, None, "arithmetic_uncertain")
        midpoint = (low[axis]+high[axis])*.5
        left_high, right_low = high.copy(), low.copy()
        left_high[axis], right_low[axis] = midpoint, midpoint
        if adaptive:
            # Keep both children, including difficult/depth-limited regions.
            # The tie counter avoids comparing interval arrays in the heap.
            for child in ((left, low, left_high, depth+1), (right, right_low, high, depth+1)):
                serial += 1
                heapq.heappush(pending, (float(child[0].lo.min()), serial, child))
        else:
            pending.append((right, right_low, high, depth+1))
            pending.append((left, low, left_high, depth+1))
    if uncertain:
        return result("uncertain", root_lower, boxes, depth_used, None, "depth_budget")
    return result("certified_feasible", min(certified_lowers), boxes, depth_used, None, "positive_bernstein_cover")


def certify_cell(vertices, spacing, *, config=None, point_observer=None):
    """Local geometric certificate; does not assert global boundary or sampling validity."""
    config = CertificateConfig() if config is None else config
    if not isinstance(config, CertificateConfig):
        raise InputValidationError("config: CertificateConfig required")
    u = as_float64_array(vertices, "vertices", (8, 3), check_finite=False).copy()
    spacing = _spacing(spacing)
    if not iv.arithmetic_supported() or not np.all(np.isfinite(u)):
        return CellCertificate("uncertain", None, None, None, 0, 0, None, "unsupported_arithmetic_or_nonfinite")
    # An observer requests point diagnostics: keep the full legacy path.
    if isinstance(config, EnclosureCertificateConfig) and point_observer is None:
        try:
            margin = float((_enclosure_kernel(config)(u[None], spacing)-config.delta).lo[0])
            if margin > 0:
                lower = margin if config.delta == 0 else float((iv.Interval.point(margin)+config.delta).lo)
                return CellCertificate("certified_feasible", lower, margin, None, 1, 0, None,
                                       "positive_edge_hull")
        except FloatingPointError:
            pass  # Inconclusive preproof must fall back, not reject.
    try:
        J = jacobian_intervals(u[None], spacing)[0]
        p = J-config.delta
        coeff = iv.lagrange_to_bernstein(p)
        sampled = float(_sample_jacobians(u[None], spacing).min())
        return _classify(coeff, p, config.delta, sampled, config, point_observer)
    except FloatingPointError:
        return CellCertificate("uncertain", None, None, None, 0, 0, None, "arithmetic_uncertain")


def certify_map(u, grid, *, config=None):
    """All cells plus six faces and sampling domain; no global-injectivity claim.

    Own the input snapshot before hashing/certifying. Positive root boxes are
    classified in batches; only unresolved cells enter the subdivision loop.
    """
    config = CertificateConfig() if config is None else config
    if not isinstance(config, CertificateConfig):
        raise InputValidationError("config: CertificateConfig required")
    u = np.array(as_float64_array(u, "u", grid.shape+(3,), check_finite=False), copy=True, order="C")
    # The owned C-order float64 copy's buffer equals u.tobytes(order="C") (E48 zero-copy hash).
    digest = hashlib.sha256(memoryview(u).cast("B")).hexdigest()
    enclosure = isinstance(config, EnclosureCertificateConfig)
    result_type = (PrescreenFeasibilityResult if isinstance(config, PrescreenCertificateConfig) else
                   EnclosureFeasibilityResult if enclosure else FeasibilityResult)
    base = result_type("uncertain", None, None, None, 0, grid.n_cells, 0, 0, 0,
                             None, None, "not_checked", digest, grid.grid_id, config.delta.hex(), config)
    if isinstance(config, CompiledEnclosureCertificateConfig):
        base = replace(base, certificate_version=certificate_version(config))
    if not np.all(np.isfinite(u)):
        return replace(base, reason="NONFINITE_STATE")
    if not zero_boundary(u):
        return replace(base, status="infeasible", reason="NONZERO_BOUNDARY")
    margins_future = None
    if isinstance(config, ParallelEnclosureCertificateConfig):
        # E50: the nogil per-cell kernel runs while the sampling-domain check runs here.
        # Decisions keep the legacy order: domain/arithmetic failures return first.
        from .compiled_enclosure import parallel_margins
        margins_future = _kernel_pool().submit(parallel_margins, u, grid, config.delta)
    domain = check_map_domain(u, grid)
    if domain is not None:
        _drain(margins_future)
        return replace(base, status="infeasible", reason=domain.reason)
    if not iv.arithmetic_supported():
        _drain(margins_future)
        return replace(base, reason="unsupported_arithmetic")
    if isinstance(config, ParallelEnclosureCertificateConfig):
        return _certify_map_parallel(u, grid, config, base, margins_future)
    checked = unknown = bad = boxes = max_depth = 0
    fast = fallback = 0
    margins, sampled_min = [], np.inf
    witness_cell = witness_point = None
    arithmetic_failure = False
    for tile in cell_tiles(grid, config.tile_size):
        checkpoint("whole_cell_certificate", completed=tile.start, total=grid.n_cells)
        vertices = u.reshape(-1, 3)[cell_node_indices(grid, tile)]
        count = len(vertices)
        checked += count
        local_ids = np.arange(count)
        if enclosure:
            try:
                enclosure_margin = (_enclosure_kernel(config)(vertices, grid.spacing)-config.delta).lo
                proven = enclosure_margin > 0
            except FloatingPointError:
                proven = np.zeros(count, dtype=bool)
            n_proven = int(np.count_nonzero(proven))
            if n_proven:
                margins.append(float(enclosure_margin[proven].min()))
            fast += n_proven
            boxes += n_proven
            local_ids = local_ids[~proven]
            vertices = vertices[~proven]
            count = len(vertices)
            fallback += count
            if not count:
                continue
        try:
            p = jacobian_intervals(vertices, grid.spacing)-config.delta
            coefficients = iv.lagrange_to_bernstein(p)
            samples = _sample_jacobians(vertices, grid.spacing)
        except FloatingPointError:
            arithmetic_failure = True
            unknown += count
            continue
        sampled_min = min(sampled_min, float(samples.min()))
        minima = coefficients.lo.min(axis=(-3, -2, -1))
        safe = minima > 0
        boxes += count
        if np.any(safe):
            margins.append(float(minima[safe].min()))
        for local in np.flatnonzero(~safe):
            try:
                result = _classify(coefficients[local], p[local], config.delta, float(samples[local].min()), config)
            except FloatingPointError:
                arithmetic_failure = True
                unknown += 1
                continue
            if result.min_margin_lower_bound is not None:
                margins.append(result.min_margin_lower_bound)
            if result.reason == "arithmetic_uncertain":
                arithmetic_failure = True
            boxes += result.n_boxes-1
            max_depth = max(max_depth, result.max_depth)
            unknown += result.status == "uncertain"
            bad += result.status == "infeasible"
            if result.status == "infeasible" and witness_cell is None:
                witness_cell, witness_point = tile.start+int(local_ids[local]), result.witness_reference_point
    status = "infeasible" if bad else ("uncertain" if unknown else "certified_feasible")
    margin = min(margins) if margins and not arithmetic_failure else None
    lower = (margin if config.delta == 0 else float((iv.Interval.point(margin)+config.delta).lo)) if margin is not None else None
    reason = "JACOBIAN_VIOLATION" if bad else ("arithmetic_uncertain" if arithmetic_failure else
              ("subdivision_budget" if unknown else "positive_bernstein_cover"))
    if enclosure:
        base = replace(base, n_fast_cells=fast, n_fallback_cells=fallback)
        if fast:
            sampled_min = np.inf  # Incomplete samples cannot be a global sample minimum.
            if status == "certified_feasible":
                reason = "positive_enclosure_cover"
    return replace(base, status=status, min_jd_lower_bound=lower, min_margin_lower_bound=margin,
                   sampled_min_jd=float(sampled_min) if np.isfinite(sampled_min) else None,
                   n_cells_checked=checked, n_uncertain=int(unknown), n_infeasible=int(bad),
                   n_boxes=boxes, max_depth=max_depth, witness_cell=witness_cell,
                   witness_reference_point=witness_point, reason=reason)


def _node_indices_for(grid, cells):
    """cell_node_indices for an arbitrary increasing array of cell ids (same formula)."""
    cz, cy, cx = np.unravel_index(cells, grid.cell_shape)
    _, ny, nx = grid.shape
    return np.stack([((cz + z) * ny + cy + y) * nx + cx + x for x, y, z in VERTICES], axis=-1)


def _fallback_arrays(vertices, spacing, delta, errors):
    with np.errstate(**errors):
        try:
            p = jacobian_intervals(vertices, spacing)-delta
            coefficients = iv.lagrange_to_bernstein(p)
            samples = _sample_jacobians(vertices, spacing)
        except FloatingPointError:
            return None
    return p, coefficients, samples


_KERNEL_POOL = None


def _kernel_pool():
    global _KERNEL_POOL
    if _KERNEL_POOL is None:
        from concurrent.futures import ThreadPoolExecutor
        _KERNEL_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jd-curl-edge-hull")
    return _KERNEL_POOL


def _drain(future):
    """Wait for (and discard) a speculative kernel run before returning."""
    if future is not None:
        try:
            future.result()
        except Exception:
            pass


LEAN_CERTIFICATE_RECORD = dict(policy="owned-certificate-front-v1", owned=0, legacy=0)


def lean_certificate_record(since=None):
    """E98 (record only): owned/legacy counts of certify_map_owned, process-wide since import or, with `since` (an
    earlier record), the differences since then."""
    record = dict(LEAN_CERTIFICATE_RECORD)
    for name in ("owned", "legacy"):
        record[name] -= (since or {}).get(name, 0)
    return record


def certify_map_owned(u, grid, *, config, finite=False, domain_ok=False, margins=None, digest=None, digest_record=None):
    """E98 (throughput-v16): certify_map(u, grid, config=config) -- every result field identical -- without its
    defensive copy, with the edge-hull margins submitted before the SHA-256 (which then runs during the kernel),
    and without repeating checks the caller has already passed on this very array and grid:
      finite=True: every value of u is finite (the backend's trial u, injected from a finite theta);
      domain_ok=True: check_map_domain(u, grid) returned None.
    Only for a frozen, owned, C-contiguous float64 u of the grid's shape (a plain ndarray, not a subclass: a masked
    array would change zero_boundary) and a parallel enclosure config; any other input runs certify_map itself. The legacy early returns that can still occur (nonzero boundary,
    unsupported arithmetic) return the same results.
    margins (T84, E101): a callable with parallel_margins' signature and result, bit for bit (the GPU edge-hull of
    backends.edge_hull_gpu); it runs where parallel_margins would. None: parallel_margins.
    digest (v41, H2): a future of problem.array_hash(u) started when u was injected -- the same SHA-256 of the same
    bytes. The proof does not read the digest, so it runs while the hash finishes and the digest is joined into the
    result at the end (every field as without the future). A failed or malformed future falls back to computing the
    hash here (digest_record['errors'] counts that)."""
    from .problem import _immutable
    if not (finite and domain_ok and isinstance(config, ParallelEnclosureCertificateConfig)
            and type(u) is np.ndarray and u.dtype == np.float64 and u.flags.c_contiguous
            and u.shape == grid.shape+(3,) and _immutable(u)):
        LEAN_CERTIFICATE_RECORD["legacy"] += 1
        return certify_map(u, grid, config=config)
    LEAN_CERTIFICATE_RECORD["owned"] += 1
    from .compiled_enclosure import parallel_margins
    margins_future = _kernel_pool().submit(parallel_margins if margins is None else margins, u, grid, config.delta)
    try:
        # u is frozen and owns its C-order buffer: these bytes are u.tobytes(order="C") (as certify_map's copy).
        early = digest
        # v41 (H2): with a queued digest the proof runs first; the digest is joined into the result at the end
        digest = hashlib.sha256(memoryview(u).cast("B")).hexdigest() if early is None else _PENDING_DIGEST
        result_type = PrescreenFeasibilityResult if isinstance(config, PrescreenCertificateConfig) else EnclosureFeasibilityResult
        base = result_type("uncertain", None, None, None, 0, grid.n_cells, 0, 0, 0,
                           None, None, "not_checked", digest, grid.grid_id, config.delta.hex(), config)
        base = replace(base, certificate_version=certificate_version(config))
        if not zero_boundary(u):
            _drain(margins_future)
            return _joined(replace(base, status="infeasible", reason="NONZERO_BOUNDARY"), early, u, digest_record)
        if not iv.arithmetic_supported():
            _drain(margins_future)
            return _joined(replace(base, reason="unsupported_arithmetic"), early, u, digest_record)
    except BaseException:
        _drain(margins_future)
        raise
    return _joined(_certify_map_parallel(u, grid, config, base, margins_future), early, u, digest_record)


_PENDING_DIGEST = "pending-sha256-of-u"      # v41 (H2): never leaves certify_map_owned (replaced by _joined)
_HEX_DIGITS = frozenset("0123456789abcdef")


def _joined(result, early, u, record):
    """v41 (H2): the result with the queued SHA-256 of u as state_hash (identity when no digest was queued); a failed
    or malformed future is replaced by hashing u here, counted in record['errors']."""
    if early is None:
        return result
    try:
        digest = early.result()
    except Exception:                                          # noqa: BLE001 - recomputed below, counted
        digest = None
    if not (isinstance(digest, str) and len(digest) == 64 and _HEX_DIGITS.issuperset(digest)):
        digest = hashlib.sha256(memoryview(u).cast("B")).hexdigest()
        if record is not None:
            record["errors"] += 1
    return replace(result, state_hash=digest)


def _certify_map_parallel(u, grid, config, base, margins_future=None):
    """E50: identical per-cell decisions/aggregates to the tiled hoisted loop.

    1) all edge-hull margins at once (parallel, per-cell independent);
    2) the legacy whole-tile rule: any invalid cell step disables that tile's
       preproof; 3) each tile's Bernstein fallback arrays in a thread pool,
       consumed strictly in tile order (witness/first-failure order unchanged).
    """
    from concurrent.futures import ThreadPoolExecutor
    from .compiled_enclosure import parallel_margins
    try:
        margin, flag = (parallel_margins(u, grid, config.delta) if margins_future is None
                        else margins_future.result())
        all_unproven = False
    except FloatingPointError:
        margin = flag = None
        all_unproven = True  # legacy: every tile's preproof raised -> fallback
    checked = unknown = bad = boxes = max_depth = 0
    fast = fallback = 0
    margins, sampled_min = [], np.inf
    witness_cell = witness_point = None
    arithmetic_failure = False
    jobs = []
    for tile in cell_tiles(grid, config.tile_size):
        checkpoint("whole_cell_certificate", completed=tile.start, total=grid.n_cells)
        count = tile.stop-tile.start
        checked += count
        if all_unproven or bool(flag[tile].any()):
            proven = np.zeros(count, dtype=bool)
        else:
            tile_margin = margin[tile]
            proven = tile_margin > 0
        n_proven = int(np.count_nonzero(proven))
        if n_proven:
            margins.append(float(tile_margin[proven].min()))
        fast += n_proven
        boxes += n_proven
        local_ids = np.flatnonzero(~proven)
        fallback += len(local_ids)
        if len(local_ids):
            jobs.append((tile, local_ids))
    prescreened = 0
    if isinstance(config, PrescreenCertificateConfig) and jobs:
        # E71 (T75): rigorous float Bernstein prescreen of every unproven cell; a cell is proven
        # when (min beta_fl - B) - delta > 0, rounded down. Job order (tile order) is unchanged.
        from .cell_q1 import CellQ1Operators3D
        from .certificate_prescreen import prescreen_lower_bounds
        checkpoint("whole_cell_certificate_prescreen", completed=0, total=grid.n_cells)
        cells = np.concatenate([tile.start+local_ids for tile, local_ids in jobs])
        try:
            table = CellQ1Operators3D(grid, location="closed27_uniform").coefficients
        except FloatingPointError:
            table = None    # review E71: extreme spacing; no prescreen, the old path decides as before
        lower = (prescreen_lower_bounds(u, grid, table, cells) if table is not None
                 else np.full(len(cells), np.nan))
        with np.errstate(invalid="ignore"):
            margin_pre = lower if config.delta == 0 else np.nextafter(lower-config.delta, -np.inf)
            ok = margin_pre > 0                                   # NaN (not finite) is never proven
        kept, start = [], 0
        for tile, local_ids in jobs:
            stop = start+len(local_ids)
            proven_here = ok[start:stop]
            n_here = int(np.count_nonzero(proven_here))
            if n_here:
                margins.append(float(margin_pre[start:stop][proven_here].min()))
                prescreened += n_here
                boxes += n_here
            rest = local_ids[~proven_here]
            if len(rest):
                kept.append((tile, rest))
            start = stop
        jobs = kept
        fallback -= prescreened
    errors = np.geterr()
    flat = u.reshape(-1, 3)
    from collections import deque
    window = 2*config.fallback_workers   # bounded in-flight jobs: memory O(window tiles), not O(all)
    if isinstance(config, CompiledFallbackCertificateConfig):
        from .compiled_fallback import fallback_arrays as arrays_fn   # E54: bitwise-equal compiled arrays
    else:
        arrays_fn = _fallback_arrays
    pool = ThreadPoolExecutor(max_workers=config.fallback_workers, thread_name_prefix="jd-curl-cert")
    pending = deque()
    upcoming = iter(jobs)

    def submit_next():
        job = next(upcoming, None)
        if job is not None:
            tile, local_ids = job
            pending.append((job, pool.submit(arrays_fn, flat[_node_indices_for(grid, tile.start+local_ids)],
                                             grid.spacing, config.delta, errors)))

    try:
        for _ in range(window):
            submit_next()
        while pending:
            (tile, local_ids), future = pending.popleft()
            checkpoint("whole_cell_certificate_fallback", completed=tile.start, total=grid.n_cells)
            arrays = future.result()
            submit_next()
            count = len(local_ids)
            if arrays is None:
                arithmetic_failure = True
                unknown += count
                continue
            p, coefficients, samples = arrays
            sampled_min = min(sampled_min, float(samples.min()))
            minima = coefficients.lo.min(axis=(-3, -2, -1))
            safe = minima > 0
            boxes += count
            if np.any(safe):
                margins.append(float(minima[safe].min()))
            for local in np.flatnonzero(~safe):
                try:
                    result = _classify(coefficients[local], p[local], config.delta, float(samples[local].min()), config)
                except FloatingPointError:
                    arithmetic_failure = True
                    unknown += 1
                    continue
                if result.min_margin_lower_bound is not None:
                    margins.append(result.min_margin_lower_bound)
                if result.reason == "arithmetic_uncertain":
                    arithmetic_failure = True
                boxes += result.n_boxes-1
                max_depth = max(max_depth, result.max_depth)
                unknown += result.status == "uncertain"
                bad += result.status == "infeasible"
                if result.status == "infeasible" and witness_cell is None:
                    witness_cell, witness_point = tile.start+int(local_ids[local]), result.witness_reference_point
    finally:
        # A deadline/exception must not wait for queued, never-started fallback jobs.
        pool.shutdown(wait=True, cancel_futures=True)
    status = "infeasible" if bad else ("uncertain" if unknown else "certified_feasible")
    lowest = min(margins) if margins and not arithmetic_failure else None
    lower = (lowest if config.delta == 0 else float((iv.Interval.point(lowest)+config.delta).lo)) if lowest is not None else None
    reason = "JACOBIAN_VIOLATION" if bad else ("arithmetic_uncertain" if arithmetic_failure else
              ("subdivision_budget" if unknown else "positive_bernstein_cover"))
    base = replace(base, n_fast_cells=fast, n_fallback_cells=fallback)
    if isinstance(config, PrescreenCertificateConfig):
        base = replace(base, n_prescreen_cells=prescreened)
    if fast or prescreened:
        sampled_min = np.inf
        if status == "certified_feasible":
            reason = "positive_enclosure_cover"
    return replace(base, status=status, min_jd_lower_bound=lower, min_margin_lower_bound=lowest,
                   sampled_min_jd=float(sampled_min) if np.isfinite(sampled_min) else None,
                   n_cells_checked=checked, n_uncertain=int(unknown), n_infeasible=int(bad),
                   n_boxes=boxes, max_depth=max_depth, witness_cell=witness_cell,
                   witness_reference_point=witness_point, reason=reason)
