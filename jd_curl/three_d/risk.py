"""E8 current-base conservative risk scan; never an acceptance certificate."""
from dataclasses import asdict
from time import perf_counter

import numpy as np

from ..validation import finite_real
from .cell_q1 import cell_node_indices, cell_tiles
from .feasibility import jacobian_intervals, certify_cell
from .interval import lagrange_to_bernstein
from .residuals import validate_state


def scan_risk_cells(problem, base, *, band=1e-4):
    """Root bounds screen tiles, same-cap local certificates refine suspects.

    A small lower bound is conservative risk, not a true near-threshold point.
    No global bound is assigned to individual cells and no extra depth is used.
    """
    validate_state(problem, base)
    band = finite_real(band, 'risk band', positive=True)
    started = perf_counter()
    grid, config = problem.grid, problem.certificate_config
    records, local_boxes = [], 0
    for tile in cell_tiles(grid, config.tile_size):
        nodes = cell_node_indices(grid, tile)
        vertices = base.u.reshape(-1, 3)[nodes]
        coefficients = lagrange_to_bernstein(jacobian_intervals(vertices, grid.spacing)-config.delta)
        lower = coefficients.lo.min(axis=(-3,-2,-1))
        for k in np.flatnonzero(lower <= band):
            certificate = certify_cell(vertices[k], grid.spacing, config=config)
            local_boxes += certificate.n_boxes
            margin = certificate.min_margin_lower_bound
            if certificate.status != 'certified_feasible' or margin is None or margin <= band:
                records.append(dict(cell=tile.start+int(k), source='conservative_current_cell_bound',
                                    certificate=asdict(certificate)))
    return dict(problem_id=problem.problem_id, base_eval_id=base.eval_id,
                controls_id=problem.controls.controls_id, cells=[r['cell'] for r in records],
                records=records, cells_scanned=grid.n_cells, local_boxes=local_boxes,
                elapsed_seconds=perf_counter()-started)
