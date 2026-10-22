"""T51 bounded discovery of *relative compression*, not a JD certificate."""
from time import perf_counter

import numpy as np

from ...validation import InputValidationError, as_float64_array
from ..cell_q1 import CellQ1Operators3D, cell_node_indices, cell_tiles
from ..geometry import _compute_cell_geometry
from ..residuals import validate_state
from ..stationarity import point_jd_interval
from ..validation import integer_at_least
from ..work_control import checkpoint


def collect_pressure(problem, base, u, *, limit=8, backend=None):
    """Nominate J_trial<a and c_trial<.5*c_base, verify physical intervals.

    Both positive and negative trial JD can trigger this direction-model trust
    policy; they are explicitly distinguished. Uncertain nominations are not
    evidence. The bounded pool and closed27 sampling do NOT bound global GB.
    Existing support is deliberately NOT excluded: persistent nonlinear
    compression at an already modelled point must shrink the trust radius.
    """
    validate_state(problem, base)
    if (problem.barrier_activation is None or problem.barrier_location != 'closed27_uniform'
            or problem.weights.kappa <= 0):
        raise InputValidationError('positive pressure requires active localized closed27 barrier')
    limit = integer_at_least(limit, 'pressure batch limit')
    u = as_float64_array(u, 'pressure candidate', problem.grid.shape+(3,))
    started = perf_counter(); grid = problem.grid
    delta, activation = problem.certificate_config.delta, problem.barrier_activation
    ops = CellQ1Operators3D(grid, location='closed27_uniform')
    capacity = 32*limit; pool = []; nominated = 0

    def retain(records):
        nonlocal pool, nominated
        for ratio, cell, q in records:
            if np.isfinite(ratio) and ratio < .5:
                pool.append((float(ratio), int(cell), tuple(float(x) for x in ops.points[int(q)])))
                nominated += 1
        pool = sorted(pool)[:capacity]

    if backend is None:
        for tile in cell_tiles(grid, problem.certificate_config.tile_size):
            checkpoint('positive_pressure_scan', completed=tile.start, total=grid.n_cells)
            J0 = _compute_cell_geometry(base.u, ops, tile=tile).J
            J1 = _compute_cell_geometry(u, ops, tile=tile).J
            ratios = np.full(J0.shape, np.inf)
            np.divide(J1-delta, J0-delta, out=ratios,
                      where=np.isfinite(J1) & (J1 < activation) & (J0 > delta))
            q = ratios.argmin(axis=1); values = ratios[np.arange(len(q)), q]
            chosen = np.argsort(values, kind='stable')[:min(capacity, len(q))]
            retain((values[i], tile.start+i, q[i]) for i in chosen)
    else:
        import torch
        model = backend._model_for(problem); device = model.device
        current, trial = device.upload(base.u), device.upload(u)
        pending = []
        for (start, _), indices in model.tiles():
            checkpoint('positive_pressure_scan', completed=start, total=grid.n_cells)
            _, _, J0, _ = model.geometry(current.reshape(-1, 3)[indices], model._Dc,
                                         cofactor=False, with_curl=False)
            _, _, J1, _ = model.geometry(trial.reshape(-1, 3)[indices], model._Dc,
                                         cofactor=False, with_curl=False)
            ratios = torch.where(torch.isfinite(J1) & (J1 < activation) & (J0 > delta),
                                 (J1-delta)/(J0-delta), torch.full_like(J0, torch.inf))
            values, q = ratios.min(dim=1)
            values, cells = torch.topk(values, min(capacity, len(values)), largest=False, sorted=True)
            pending.append(torch.stack((values, (cells+start).to(torch.float64),
                                        q[cells].to(torch.float64)), dim=1))
            if len(pending) == 8:
                retain(device.download(torch.cat(pending))); pending.clear()
        if pending:
            retain(device.download(torch.cat(pending)))

    selected = []; locations = []; records = []; rejected = set(); checks = 0
    for separation in (2, 0):
        for _, cell, point in pool:
            key = (cell, point)
            if key in selected or key in rejected:
                continue
            xyz = np.asarray(np.unravel_index(cell, grid.cell_shape))
            if separation and any(np.max(np.abs(xyz-v)) < separation for v in locations):
                continue
            checkpoint('positive_pressure_verify', checked=checks)
            nodes = cell_node_indices(grid, slice(cell, cell+1))[0]; checks += 1
            try:
                J0 = point_jd_interval(base.u.reshape(-1, 3)[nodes], grid.spacing, point)
                J1 = point_jd_interval(u.reshape(-1, 3)[nodes], grid.spacing, point)
                c0, c1 = J0-delta, J1-delta
                floor = c0*.5
                proved = float(c0.lo) > 0 and float(J1.hi) < activation and float(c1.hi) < float(floor.lo)
            except FloatingPointError:
                proved = False
            if not proved:
                rejected.add(key); continue
            selected.append(key); locations.append(xyz)
            records.append(dict(cell=cell, point=list(point),
                base_jd_interval=[float(J0.lo), float(J0.hi)],
                trial_jd_interval=[float(J1.lo), float(J1.hi)],
                slack_ratio_upper=float((c1/c0).hi),
                kind=('positive_compression' if float(J1.lo)>delta else
                      'proved_jd_violation' if float(J1.hi)<=delta else 'jd_sign_uncertain')))
            if len(selected) == limit:
                break
        if len(selected) == limit:
            break
    return selected, dict(seconds=perf_counter()-started, nominated=nominated,
        retained=len(pool), interval_checks=checks, proved_pressure_points=len(selected),
        records=records, retention=.5, base_eval_id=base.eval_id,
        scope='bounded closed27 relative compression; NOT full-cell proof or stationarity')
