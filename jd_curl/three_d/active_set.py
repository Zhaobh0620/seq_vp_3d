"""T89 (S30-G4): the native-level active set -- a fixed tapered node weight w in [0, 1] from the distance to the
support union of the two images; the PG direction becomes p = -(wS) M^-1 (wS) g (poisson_gradient), so nodes with
w = 0 never move, and the device backend may skip their cells (backends.triton_q1 / torch_ops, `active_set_skip`).
F, the barrier, the certificate, E/F, GF and the ratio denominator are unchanged; see docs/contract_draft_T89_active_set_S30.md.
"""
from dataclasses import dataclass

import numpy as np

from ..validation import InputValidationError, as_float64_array, finite_real, readonly_copy
from .problem import array_hash

ACTIVE_SET_POLICY = "native-active-set-taper-v1"


@dataclass(frozen=True)
class ActiveSetSpec:
    """d1 < d2 in voxels: w = 1 within d1 of the support union, linear to 0 at d2, 0 beyond."""
    d1: float = 3.
    d2: float = 8.

    def __post_init__(self):
        d1 = finite_real(self.d1, "active set d1")
        d2 = finite_real(self.d2, "active set d2", positive=True)
        if d1 < 0 or not d1 < d2:
            raise InputValidationError("active set: require 0 <= d1 < d2")
        object.__setattr__(self, "d1", d1)
        object.__setattr__(self, "d2", d2)

    def record(self):
        return dict(policy=ACTIVE_SET_POLICY, d1=self.d1, d2=self.d2)


def parse_active_set(value):
    """None, an ActiveSetSpec, a dict {d1, d2[, policy]} or 'd1,d2' -> ActiveSetSpec or None."""
    if value is None or isinstance(value, ActiveSetSpec):
        return value
    if isinstance(value, str):
        parts = value.split(",")
        if len(parts) != 2:
            raise InputValidationError("active set: 'd1,d2'")
        try:
            return ActiveSetSpec(float(parts[0]), float(parts[1]))
        except ValueError:
            raise InputValidationError("active set: d1, d2 must be numbers") from None
    if isinstance(value, dict):
        if value.get("policy", ACTIVE_SET_POLICY) != ACTIVE_SET_POLICY:
            raise InputValidationError("active set: policy "+ACTIVE_SET_POLICY)
        extra = set(value)-{"policy", "d1", "d2"}
        if extra:
            raise InputValidationError("active set: unknown keys "+", ".join(sorted(extra)))
        return ActiveSetSpec(value.get("d1", 3.), value.get("d2", 8.))
    raise InputValidationError("active set: expected 'd1,d2', a dict or an ActiveSetSpec")


def support_union(fixed, moving, grid):
    fixed = as_float64_array(fixed, "fixed image", grid.shape)
    moving = as_float64_array(moving, "moving image", grid.shape)
    return (fixed > 0) | (moving > 0)


def active_weights(fixed, moving, grid, spec):
    """(w, record): the T89 node weights on the grid nodes (read-only float64, shape grid.shape) and the record."""
    from scipy.ndimage import distance_transform_edt
    spec = parse_active_set(spec)
    if spec is None:
        raise InputValidationError("active set: a specification is required")
    union = support_union(fixed, moving, grid)
    if not union.any():
        raise InputValidationError("active set: the images have no positive support")
    distance = distance_transform_edt(~union) if not union.all() else np.zeros(grid.shape)
    w = np.clip((spec.d2-distance)/(spec.d2-spec.d1), 0., 1.)
    w = readonly_copy(np.ascontiguousarray(w, dtype=np.float64))
    record = dict(spec.record(), support_hash=array_hash(union.astype(np.uint8)), weights_hash=array_hash(w),
                  **fractions(w, grid))
    return w, record


def validate_active_weights(w, grid):
    w = as_float64_array(w, "active_weights", grid.shape)
    if np.any(w < 0.) or np.any(w > 1.):
        raise InputValidationError("active_weights must lie in [0, 1]")
    return readonly_copy(w)


def active_nodes(w):
    return w > 0.


def frozen_nodes(w):
    return w == 0.


def _dilate(mask, rings):
    """Chebyshev dilation by `rings` nodes (binary_dilation with a full (2r+1)^3 structuring element)."""
    from scipy.ndimage import binary_dilation
    if rings <= 0:
        return mask.copy()
    size = 2*rings+1
    return binary_dilation(mask, structure=np.ones((size, size, size), bool))


def cells_with_any(node_mask):
    """Cells (nz-1, ny-1, nx-1) having at least one node in node_mask."""
    m = node_mask
    return (m[:-1, :-1, :-1] | m[1:, :-1, :-1] | m[:-1, 1:, :-1] | m[1:, 1:, :-1]
            | m[:-1, :-1, 1:] | m[1:, :-1, 1:] | m[:-1, 1:, 1:] | m[1:, 1:, 1:])


def index_sets(w, grid, *, gradient_rings=2):
    """The host-side index sets of the device skipping (T89 section 3), from the node weights:
      active_nodes: w > 0 (sampled afresh);  frozen: w == 0;
      n1_nodes: active nodes dilated by gradient_rings (the gradient may change there; 2 rings cover the curvature
      term, 1 ring the soft/barrier/image/proximity terms);
      c1_cells: cells with a node in n1 (their locals feed the n1 gathers);
      active_cells: cells with an active node (their energy contributions may change).
    All as boolean arrays; the callers build their flat index tensors from them."""
    w = validate_active_weights(w, grid)
    active = active_nodes(w)
    n1 = _dilate(active, int(gradient_rings))
    return dict(active_nodes=active, frozen_nodes=~active, n1_nodes=n1, c1_cells=cells_with_any(n1),
                active_cells=cells_with_any(active))


def block_mask(cell_mask, block):
    """Blocks of `block` consecutive cells (C order) that hold at least one marked cell; blocks x (count)."""
    flat = np.ascontiguousarray(cell_mask).reshape(-1)
    k = (flat.size+block-1)//block
    padded = np.zeros(k*block, bool)
    padded[:flat.size] = flat
    return padded.reshape(k, block).any(axis=1)


def fractions(w, grid):
    sets = index_sets(w, grid)
    return dict(node_active_fraction=float(sets["active_nodes"].mean()), node_frozen_fraction=float(sets["frozen_nodes"].mean()),
                node_n1_fraction=float(sets["n1_nodes"].mean()), cell_active_fraction=float(sets["active_cells"].mean()),
                cell_c1_fraction=float(sets["c1_cells"].mean()),
                block64_active_fraction=float(block_mask(sets["active_cells"], 64).mean()),
                block128_active_fraction=float(block_mask(sets["active_cells"], 128).mean()))


_FREE_CACHE = {}


def free_weights(problem):
    """The weights of the free degrees of freedom (component-major, as restrict_free), one array per weight
    array object (cached while that object lives): w_free[c*n_interior + i] = w[interior node i]."""
    w = problem.active_weights
    key = id(w)
    hit = _FREE_CACHE.get(key)
    if hit is not None and hit[0] is w:
        return hit[1]
    interior = np.ascontiguousarray(w[1:-1, 1:-1, 1:-1]).reshape(-1)
    out = readonly_copy(np.concatenate([interior, interior, interior]))
    _FREE_CACHE.clear()
    _FREE_CACHE[key] = (w, out)
    return out


def metric_record(problem):
    """The small per-row record of the active set (T89 section 4)."""
    w = problem.active_weights
    return dict(policy=ACTIVE_SET_POLICY, weights_hash=array_hash(w), frozen_fraction=float(np.mean(w == 0.)))
