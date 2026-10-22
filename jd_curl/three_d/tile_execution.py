"""Bounded CPU tile work; ordered consumption preserves scalar reductions."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .cell_q1 import cell_tiles
from .work_control import checkpoint


def ordered_tile_map(problem, function, *, after_submit=None):
    """Yield in canonical order, with at most workers submitted futures.

    Workers must not mutate shared state/accumulators. Caller error settings
    are thread-local in NumPy and therefore explicitly restored for each job.
    No executor/cache outlives this complete evaluation.
    after_submit (E108): caller-thread work run once, after the first jobs are
    submitted and before the first result is awaited (first with one worker).
    """
    tiles = iter(cell_tiles(problem.grid, problem.memory.cell_tile_size))
    workers = getattr(problem.memory, "cpu_workers", 1)
    if workers == 1:
        if after_submit is not None:
            after_submit()
        for tile in tiles:
            checkpoint("cpu_cell_tiles", completed=tile.start, total=problem.grid.n_cells)
            yield function(tile)
        return
    errors = np.geterr()

    def run(tile):
        with np.errstate(**errors):
            return function(tile)

    checkpoint("cpu_parallel_tiles")
    pending = deque()
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="jd-curl-tile") as pool:
        try:
            for _, tile in zip(range(workers), tiles):
                pending.append(pool.submit(run, tile))
            if after_submit is not None:
                after_submit()
            while pending:
                # Observe a completed numerical failure before an expiry can
                # relabel it as a budget exit. At most workers jobs are live.
                value = pending.popleft().result()
                checkpoint("cpu_parallel_tiles")
                yield value
                tile = next(tiles, None)
                if tile is not None:
                    pending.append(pool.submit(run, tile))
        finally:
            for future in pending:
                future.cancel()
