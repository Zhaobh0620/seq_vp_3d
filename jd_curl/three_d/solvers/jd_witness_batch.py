"""E39 bounded diverse physical witnesses; nominations never certify a map."""
from time import perf_counter

import numpy as np

from ...validation import as_float64_array
from ..cell_q1 import CellQ1Operators3D, cell_node_indices, cell_tiles
from ..geometry import _compute_cell_geometry
from ..stationarity import point_jd_interval
from ..validation import integer_at_least
from ..work_control import checkpoint


def collect_witnesses(problem, u, *, exclude=(), limit=8, backend=None):
    """Scan closed27, nominate bounded per-tile minima, verify CPU intervals.

    Two selection passes prefer separated cells, then fill with adjacent ones.
    This is a bounded discovery heuristic, not an exhaustive list or proof.
    GPU float64 only nominates; no nonfinite/uncertain value is a constraint.
    """
    limit = integer_at_least(limit, 'witness batch limit')
    u = as_float64_array(u, 'witness candidate', problem.grid.shape+(3,))
    started = perf_counter(); grid = problem.grid; delta = problem.certificate_config.delta
    ops = CellQ1Operators3D(grid, location='closed27_uniform')
    excluded = {(int(c), tuple(p)) for c, p in exclude}
    pool = []; capacity = 32*limit; nominated = 0

    def retain(records):
        nonlocal pool, nominated
        for value, cell, q in records:
            if np.isfinite(value) and value <= delta:
                key = (int(cell), tuple(float(x) for x in ops.points[int(q)]))
                if key not in excluded:
                    pool.append((float(value), *key)); nominated += 1
        # Limit host memory across the full scan; most severe nominations first.
        pool = sorted(pool)[:capacity]

    if backend is None:
        for tile in cell_tiles(grid, problem.certificate_config.tile_size):
            checkpoint('constraint_witness_scan', completed=tile.start, total=grid.n_cells)
            J = _compute_cell_geometry(u, ops, tile=tile).J
            q = np.argmin(J, axis=1); values = J[np.arange(len(q)), q]
            chosen = np.argsort(values, kind='stable')[:min(capacity, len(q))]
            retain((values[i], tile.start+i, q[i]) for i in chosen)
    else:
        import torch
        model = backend._model_for(problem); device = model.device
        uploaded = device.upload(u)
        D = (model._Dc if problem.barrier_location == 'closed27_uniform'
             else device.upload(ops.coefficients))
        pending = []
        for (start, _), indices in model.tiles():
            _, _, J, _ = model.geometry(uploaded.reshape(-1, 3)[indices], D,
                                        cofactor=False, with_curl=False)
            values, q = torch.where(torch.isfinite(J), J, torch.full_like(J, torch.inf)).min(dim=1)
            values, cells = torch.topk(values, min(capacity, len(values)), largest=False, sorted=True)
            pending.append(torch.stack((values, (cells+start).to(torch.float64),
                                        q[cells].to(torch.float64)), dim=1))
            if len(pending) == 8:
                retain(device.download(torch.cat(pending))); pending.clear()
        if pending:
            retain(device.download(torch.cat(pending)))

    selected = []; verified = []; rejected = set(); checks = 0
    for separation in (2, 0):
        for _, cell, point in pool:
            key = (cell, point)
            if key in rejected or key in selected:
                continue
            xyz = np.asarray(np.unravel_index(cell, grid.cell_shape))
            if separation and any(np.max(np.abs(xyz-other)) < separation for other in verified):
                continue
            checkpoint('constraint_witness_verify', checked=checks)
            nodes = cell_node_indices(grid, slice(cell, cell+1))[0]
            checks += 1
            try:
                upper = float(point_jd_interval(u.reshape(-1, 3)[nodes], grid.spacing, point).hi)
            except FloatingPointError:
                rejected.add(key); continue
            if not np.isfinite(upper) or upper > delta:
                rejected.add(key); continue
            selected.append(key); verified.append(xyz)
            if len(selected) == limit:
                break
        if len(selected) == limit:
            break
    return selected, dict(seconds=perf_counter()-started, nominated=nominated,
        retained=len(pool), interval_checks=checks, proved_points=len(selected),
        scope='bounded closed27 discovery; NOT full-cell certification')
