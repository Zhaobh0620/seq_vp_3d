"""E26 pointwise cubic step prediction. NEVER a whole-cell certificate."""
from collections import OrderedDict
import hashlib
from time import perf_counter

import numpy as np

from ...validation import InputValidationError
from ..cell_q1 import CellQ1Operators3D, VERTICES, cell_node_indices
from ..geometry import cofactor


def determinant_coefficients(A, V, delta=0.):
    """det(A+alpha V)-delta, ascending powers, physical 3x3 matrices."""
    return np.stack((np.linalg.det(A)-delta, np.sum(cofactor(A)*V, axis=(-2, -1)),
                     np.sum(A*cofactor(V), axis=(-2, -1)), np.linalg.det(V)), axis=-1)


def first_root_before(coefficients, limit):
    """Floating root heuristic on (0,limit]; None does NOT mean feasible."""
    c = np.asarray(coefficients, dtype=float)
    with np.errstate(over='ignore', invalid='ignore'):
        c = c*np.array([1., limit, limit**2, limit**3])
    if not np.all(np.isfinite(c)) or c[0] <= 0:
        raise FloatingPointError('invalid cubic predictor base/coefficients')
    if c[0] > np.abs(c[1:]).sum():
        return None  # no root in this sampled point's prefix, not a cell proof
    polynomial = np.trim_zeros(c[::-1], trim='f')
    if len(polynomial) == 1:
        return None
    try:
        roots = np.roots(polynomial/np.max(np.abs(polynomial)))
    except np.linalg.LinAlgError as error:
        raise FloatingPointError('cubic root failure') from error
    candidates = [float(r.real) for r in roots if 0 < r.real <= 1
                  and abs(r.imag) <= 1e-8*max(1., abs(r.real))]
    return None if not candidates else min(candidates)*limit


# E97 (root_screen): S' <= fl(S*(1+2^-49)) for every float sum S' of three nonnegative doubles, S = fl(fl(s1+s2)+s3)
# (u = 2^-53). For normal S: S' <= S*(1+u)^2/(1-u)^2 < S*(1+16u)*(1-u) <= fl(S*(1+16u)). For S = 0 or subnormal S:
# every partial sum is exact, so S' = S <= fl(S*(1+16u)). See docs/04 E97 item 4. The margin is exactly representable.
ROOT_SCREEN_MARGIN = 1.+2.**-49
_SCREEN_STATS = dict(predictions=0, rows=0, screened=0, solved=0, reruns=0)


def root_screen_record(since=None):
    """E97 (record only): counts of the root screen -- process-wide since import, or since a previous record."""
    counts = {k: v-(since or {}).get(k, 0) for k, v in _SCREEN_STATS.items()}
    return dict(policy='jd-predict-root-screen-v1', **counts)


def _screen_rows(C, scale):
    """E97: the rows of C (n, 4) for which first_root_before(C[k], limit) certainly returns None without raising:
    all four cs = C*scale finite, the margin bound finite and cs0 > bound >= the original's sum of |cs1..3|."""
    with np.errstate(all='ignore'):
        cs = C*scale
        a = np.abs(cs[:, 1:])
        bound = ((a[:, 0]+a[:, 1])+a[:, 2])*ROOT_SCREEN_MARGIN
        return np.isfinite(cs).all(axis=1) & np.isfinite(bound) & (cs[:, 0] > bound)


def _cached_cell(tables, grid, points, cell):
    """E95 (cached_tables): predict's D table of this point set and the cell's node arrays, computed by the same
    expressions as predict and kept (D read-only); the key holds the exact bits of the point set."""
    key = (points.shape, np.ascontiguousarray(points, dtype=np.float64).tobytes())
    D = tables.get(('D', key))
    if D is None:
        D = np.empty((3, len(points), 8))
        for a, vertex in enumerate(VERTICES):
            factors = [points[:, j] if vertex[j] else 1-points[:, j] for j in range(3)]
            for j in range(3):
                other = [k for k in range(3) if k != j]
                D[j, :, a] = ((1 if vertex[j] else -1)/grid.spacing[j]
                               *factors[other[0]]*factors[other[1]])
        D.flags.writeable = False
        tables[('D', key)] = D
    cached = tables.get(('cell', cell))
    if cached is None:
        nodes = cell_node_indices(grid, slice(cell, cell+1))[0]
        z, y, x = np.unravel_index(nodes, grid.shape)
        nz, ny, nx = grid.shape
        free = (z > 0) & (z < nz-1) & (y > 0) & (y < ny-1) & (x > 0) & (x < nx-1)
        cached = tables[('cell', cell)] = (nodes, z, y, x, free)
    return (D,)+cached


class JDStepMemory:
    """Bounded recent physical witnesses; only locations survive reference changes."""
    def __init__(self):
        self.grid_id = None
        self.points = OrderedDict()
        self.uncertain = 0      # T69 trigger: undetermined certificates seen on this grid

    def bind(self, grid):
        if self.grid_id != grid.grid_id:
            self.grid_id = grid.grid_id
            self.points.clear()
            self.uncertain = 0
            self.grid = grid
            self.ops = CellQ1Operators3D(grid, location='closed27_uniform')

    def witness(self, cell, point):
        point = tuple(float(x) for x in point)
        if (not 0 <= cell < self.grid.n_cells or len(point) != 3
                or not all(np.isfinite(x) and 0 <= x <= 1 for x in point)):
            raise InputValidationError('invalid physical JD witness location')
        key = (int(cell), point)
        self.points[key] = None
        self.points.move_to_end(key)
        if len(self.points) > 128:
            self.points.popitem(last=False)

    def _lean_tables(self):
        """E95/E96 (cached_tables only): the D tables and cell nodes of the bound grid, replaced when bind() binds
        another grid (the bound grid object is the tag)."""
        entry = self.__dict__.get('_tables')
        if entry is None or entry[0] is not self.grid:
            entry = self._tables = (self.grid, {})
        return entry[1]

    def _layout(self, tables, grid, cells):
        """E97 (root_screen): per cell (cell, number of points, D, nodes, z, y, x, free), built by predict's own
        expressions; kept for the last witness list (keyed by its exact bits, in order)."""
        key = np.array([(c,)+p for c, p in self.points], dtype=np.float64).tobytes()
        held = tables.get(('layout',))
        if held is not None and held[0] == key:
            return held[1]
        layout = []
        for cell in cells:
            points = list(map(tuple, self.ops.points))
            points += [p for c, p in self.points if c == cell and p not in points]
            points = np.asarray(points)
            layout.append((cell, len(points))+_cached_cell(tables, grid, points, cell))
        tables[('layout',)] = (key, layout)
        return layout

    def _screened_minimum(self, problem, base, v, cells, tables, limit, info):
        """E97 (root_screen): predict's minimum root over the cells, or None after resetting info['points'] when
        anything raised (the caller then runs the original loop, which raises or not exactly as before).
        Same per-cell A, V and coefficients; rows the margin screen proves rootless are skipped; every other
        row goes through first_root_before in the original order."""
        try:
            delta = problem.certificate_config.delta
            scale, minimum = None, None
            for cell, npts, D, nodes, z, y, x, free in self._layout(tables, problem.grid, cells):
                local_v = np.zeros((8, 3))
                local_v[free] = v[:, z[free]-1, y[free]-1, x[free]-1].T
                A = np.einsum('jqa,ar->qrj', D, base.u.reshape(-1, 3)[nodes])+np.eye(3)
                V = np.einsum('jqa,ar->qrj', D, local_v)
                C = determinant_coefficients(A, V, delta)
                if scale is None:
                    with np.errstate(over='ignore', invalid='ignore'):     # as first_root_before computes it
                        scale = np.array([1., limit, limit**2, limit**3])
                safe = _screen_rows(C, scale)
                _SCREEN_STATS['rows'] += len(C)
                _SCREEN_STATS['screened'] += int(safe.sum())
                for k in np.flatnonzero(~safe):
                    _SCREEN_STATS['solved'] += 1
                    root = first_root_before(C[k], limit)
                    if root is not None:
                        minimum = root if minimum is None else min(root, minimum)
                info['points'] += npts
            return (minimum,)
        except Exception:
            _SCREEN_STATS['reruns'] += 1
            info['points'] = 0
            return None

    def certificate(self, proof):
        if proof.status == 'uncertain':
            self.uncertain += 1
        if (proof.status == 'infeasible' and proof.witness_cell is not None
                and proof.witness_reference_point is not None):
            self.witness(proof.witness_cell, proof.witness_reference_point)

    def predict(self, problem, base, direction, limit, safety=.9, direction_id=None, lazy_digest=False):
        self.bind(problem.grid)
        started = perf_counter()
        # E76: direction_id may be a shared digest callable (same SHA-256, computed once per direction).
        # E95: lazy_digest keeps that callable in the record; the PG solver resolves it before the row is stored
        # (the prediction itself never reads the digest).
        if callable(direction_id) and lazy_digest:
            digest = direction_id
        else:
            digest = (direction_id() if callable(direction_id) else
                      hashlib.sha256(np.ascontiguousarray(direction).view(np.uint8)).hexdigest())
        info = dict(policy='historical-cell-closed27-cubic-v1', witnesses=len(self.points),
                    points=0, initial_tau=float(limit), root=None, fallback=None,
                    base_eval_id=base.eval_id, controls_id=problem.controls.controls_id,
                    grid_id=self.grid_id, safety=safety, direction_id=digest)
        result = limit
        try:
            if self.points:
                v = direction.reshape((3,)+problem.grid.interior_shape)
                cells = sorted({c for c, _ in self.points})
                minimum = None
                tables = self._lean_tables() if getattr(self, 'cached_tables', False) else None
                screened = None
                if tables is not None and getattr(self, 'root_screen', False):
                    _SCREEN_STATS['predictions'] += 1
                    screened = self._screened_minimum(problem, base, v, cells, tables, limit, info)   # E97
                for cell in (cells if screened is None else ()):
                    points = list(map(tuple, self.ops.points))
                    points += [p for c, p in self.points if c == cell and p not in points]
                    points = np.asarray(points)
                    if tables is None:
                        D = np.empty((3, len(points), 8))
                        for a, vertex in enumerate(VERTICES):
                            factors = [points[:, j] if vertex[j] else 1-points[:, j] for j in range(3)]
                            for j in range(3):
                                other = [k for k in range(3) if k != j]
                                D[j, :, a] = ((1 if vertex[j] else -1)/problem.grid.spacing[j]
                                               *factors[other[0]]*factors[other[1]])
                        nodes = cell_node_indices(problem.grid, slice(cell, cell+1))[0]
                        z, y, x = np.unravel_index(nodes, problem.grid.shape)
                        nz, ny, nx = problem.grid.shape
                        free = (z > 0) & (z < nz-1) & (y > 0) & (y < ny-1) & (x > 0) & (x < nx-1)
                    else:
                        D, nodes, z, y, x, free = _cached_cell(tables, problem.grid, points, cell)
                    local_v = np.zeros((8, 3))
                    local_v[free] = v[:, z[free]-1, y[free]-1, x[free]-1].T
                    A = np.einsum('jqa,ar->qrj', D, base.u.reshape(-1, 3)[nodes])+np.eye(3)
                    V = np.einsum('jqa,ar->qrj', D, local_v)
                    for c in determinant_coefficients(A, V, problem.certificate_config.delta):
                        root = first_root_before(c, limit)
                        if root is not None:
                            minimum = root if minimum is None else min(root, minimum)
                    info['points'] += len(points)
                if screened is not None:
                    minimum = screened[0]
                if minimum is not None:
                    info['root'] = minimum
                    result = min(limit, safety*minimum)
        except (FloatingPointError, OverflowError) as error:
            info['fallback'] = str(error)
            result = limit  # original certified line search remains authoritative
        info.update(predicted_tau=float(result), seconds=perf_counter()-started)
        return result, info
