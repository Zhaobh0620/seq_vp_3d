"""T47 bounded candidate-only Q1 increment repair, not a feasibility proof."""
import numpy as np
from scipy.ndimage import distance_transform_edt

from ...validation import InputValidationError, as_float64_array, finite_real
from ..validation import integer_at_least
from ..work_control import checkpoint


SCHEDULE = tuple((radius, beta) for radius in (1, 2, 4) for beta in (.5, .25, 0.))


def nodal_scale(grid, cells, radius, beta_min):
    """Union of all eight vertices; distances in node-index units, not mm."""
    radius = integer_at_least(radius, 'repair radius')
    beta_min = finite_real(beta_min, 'core beta')
    if not 0 <= beta_min <= 1:
        raise InputValidationError('core beta must be in [0,1]')
    cells = sorted({integer_at_least(c, 'dangerous cell', 0) for c in cells})
    if not cells or cells[-1] >= grid.n_cells:
        raise InputValidationError('nonempty in-grid dangerous cells required')
    coords = np.array(np.unravel_index(cells, grid.cell_shape)).T
    lower = np.maximum(coords.min(axis=0)-radius, 0)
    upper = np.minimum(coords.max(axis=0)+2+radius, grid.shape)
    window = tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))
    core = np.zeros(tuple(upper-lower), dtype=bool)
    for z, y, x in coords-lower:
        core[z:z+2, y:y+2, x:x+2] = True
    # Outside this window all nodes are at least radius from every core node.
    t = np.minimum(distance_transform_edt(~core)/radius, 1.)
    smooth = t*t*t*(10.+t*(-15.+6.*t))
    scale = np.ones(grid.shape)
    scale[window] = np.clip(beta_min+(1.-beta_min)*smooth, beta_min, 1.)
    return scale, int(core.sum())


def repair_candidate(grid, base_theta, raw_candidate, cells, radius, beta_min):
    """Always rebuild from the original candidate, never shrink accumulated u."""
    checkpoint('local_increment_repair')
    shape = (3*grid.n_free_scalar,)
    base = as_float64_array(base_theta, 'repair base', shape)
    raw = as_float64_array(raw_candidate, 'raw candidate', shape)
    beta, core_nodes = nodal_scale(grid, cells, radius, beta_min)
    free_beta = beta[1:-1, 1:-1, 1:-1].ravel()
    b, v = base.reshape(3, -1), raw.reshape(3, -1)
    out = v.copy()
    modified = free_beta < 1.
    with np.errstate(over='raise', invalid='raise'):
        out[:, modified] = b[:, modified]+free_beta[modified]*(v[:, modified]-b[:, modified])
    # Exact restoration/retention even when subtract-then-add would round.
    out[:, free_beta == 0.] = b[:, free_beta == 0.]
    checkpoint('local_increment_repair_finished')
    return out.ravel(), dict(radius_cells=radius, core_beta=beta_min,
        dangerous_cells=sorted(set(cells)), core_nodes=core_nodes,
        modified_free_nodes=int(modified.sum()),
        modified_free_fraction=float(modified.mean()), distance='node-index-euclidean')


class LocalIncrementRepair:
    """Lifetime: one raw candidate. Only proven cell IDs enter this union."""
    def __init__(self, grid, base_theta, raw_candidate):
        self.grid, self.base, self.raw = grid, base_theta, raw_candidate
        self.cells = set()
        self.index = 0

    def next(self, proven_cells):
        self.cells.update(proven_cells)
        if not self.cells or self.index >= len(SCHEDULE):
            return None
        radius, beta = SCHEDULE[self.index]
        self.index += 1
        candidate, info = repair_candidate(self.grid, self.base, self.raw, self.cells, radius, beta)
        info['schedule_index'] = self.index
        return candidate, info
