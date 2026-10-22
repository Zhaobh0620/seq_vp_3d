"""T26 immutable Q1 working spaces; no risk selection or optimizer controller."""
from dataclasses import dataclass
import hashlib

import numpy as np

from ..validation import InputValidationError, StateMismatchError, as_float64_array, finite_real, readonly_copy
from .boundary import restrict_free
from .cell_q1 import cell_node_indices
from .residuals import AcceptedState3D, validate_state
from .validation import integer_at_least


def _readonly(value):
    out = np.array(value, copy=True)
    out.flags.writeable = False
    return out


@dataclass(frozen=True)
class WorkingSet3D:
    """Factory-owned space. Frozen values are anchored, never reset to zero.

    Creating a proposed release does not install it or accept a trial. The
    caller must preserve the previous object until its ordinary step passes.
    """
    problem_id: str
    grid_id: str
    controls_id: str
    mask_id: str
    generation: int
    created_eval_id: str
    cells: tuple[int, ...]
    components: tuple[tuple[int, ...], ...]
    frozen: np.ndarray
    free_indices: np.ndarray
    anchors: np.ndarray

    @property
    def n_full(self):
        return len(self.frozen)

    @property
    def n_free(self):
        return len(self.free_indices)

    @property
    def frozen_fraction(self):
        return float(np.count_nonzero(self.frozen)/self.n_full)

    def validate(self, problem, base):
        validate_state(problem, base)
        if not isinstance(base, AcceptedState3D):
            raise InputValidationError('working set requires a derivative-ready base')
        if (self.problem_id != problem.problem_id or self.grid_id != problem.grid.grid_id
                or self.controls_id != problem.controls.controls_id):
            raise StateMismatchError('working set belongs to a different problem/reference')
        if not np.array_equal(base.theta[self.frozen], self.anchors):
            raise StateMismatchError('frozen values moved; stale working set or unaccepted release')

    def restrict(self, vector):
        return as_float64_array(vector, 'full vector', (self.n_full,))[self.free_indices].copy()

    def inject(self, vector):
        q = as_float64_array(vector, 'reduced vector', (self.n_free,))
        out = np.zeros(self.n_full)
        out[self.free_indices] = q
        return out

    def project(self, vector):
        return self.inject(self.restrict(vector))


def make_working_set(problem, base, cells, *, previous=None, max_frozen_fraction=.5):
    """Propose a pure transition at an already accepted, certified base.

    All selected cells freeze their eight nodes' three components, excluding
    the original fixed boundary. Empty cells means no artificial mask.
    The caller, not promotion or this factory, decides optimizer acceptance.
    """
    validate_state(problem, base)
    if not isinstance(base, AcceptedState3D):
        raise InputValidationError('working set requires a derivative-ready base')
    cap = finite_real(max_frozen_fraction, 'max_frozen_fraction', positive=True)
    if cap >= 1:
        raise InputValidationError('max_frozen_fraction must be < 1; no all-frozen success')
    cells = tuple(sorted({integer_at_least(c, 'cell_id', 0) for c in cells}))
    if cells and cells[-1] >= problem.grid.n_cells:
        raise InputValidationError('cell_id outside grid')
    if previous is not None:
        if not isinstance(previous, WorkingSet3D):
            raise InputValidationError('previous: expected WorkingSet3D')
        previous.validate(problem, base)
        if cells == previous.cells and previous.frozen_fraction <= cap:
            return previous
    grid = problem.grid
    marked = np.zeros(grid.shape)
    nodes = {c:cell_node_indices(grid, slice(c, c+1))[0] for c in cells}
    # Components use shared Q1 control nodes, including original fixed nodes.
    parent = {c:c for c in cells}
    def root(c):
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c
    owner = {}
    for c in cells:
        marked.ravel()[nodes[c]] = 1
        for node in nodes[c]:
            if int(node) in owner:
                a, b = root(c), root(owner[int(node)])
                parent[max(a, b)] = min(a, b)
            else:
                owner[int(node)] = c
    groups = {}
    for c in cells:
        groups.setdefault(root(c), []).append(c)
    frozen = restrict_free(np.repeat(marked[..., None], 3, axis=-1), grid).astype(bool)
    fraction = np.count_nonzero(frozen)/len(frozen)
    if fraction > cap:
        raise InputValidationError('mask exceeds frozen fraction cap; recovery required')
    generation = 0 if previous is None else previous.generation+1
    anchors = readonly_copy(base.theta[frozen])
    identity = hashlib.sha256((repr((problem.problem_id, generation, cells, base.eval_id,
                              None if previous is None else previous.mask_id))).encode()+anchors.tobytes()).hexdigest()
    return WorkingSet3D(problem.problem_id, grid.grid_id, problem.controls.controls_id,
                        identity, generation, base.eval_id, cells,
                        tuple(tuple(groups[k]) for k in sorted(groups)), _readonly(frozen),
                        _readonly(np.flatnonzero(~frozen)), anchors)
