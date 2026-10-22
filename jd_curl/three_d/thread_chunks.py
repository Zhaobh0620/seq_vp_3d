"""E54: run a GIL-releasing kernel over disjoint chunks of range(n) in a shared thread pool.

Kernels write disjoint slices (or return order-independent partial results), so the
thread count and completion order cannot change any value. All chunks finish before the
first exception (if any) is re-raised. A call made from inside a pool worker runs inline
(serially), so nested use cannot deadlock.
"""
import threading
from concurrent.futures import ThreadPoolExecutor

_PREFIX = "jd-curl-scan"
_LOCK = threading.Lock()
_POOLS = {}   # one persistent pool per worker count; never shut down, so callers never race


def scan_workers():
    try:
        from numba import config as _config
        return max(1, int(_config.NUMBA_NUM_THREADS))
    except ImportError:  # pragma: no cover
        return 1


def _pool(workers):
    with _LOCK:
        pool = _POOLS.get(workers)
        if pool is None:
            pool = _POOLS[workers] = ThreadPoolExecutor(max_workers=workers, thread_name_prefix=_PREFIX)
        return pool


def run_chunks(kernel, n, *, workers=None, chunks_per_worker=4):
    """Return [kernel(lo, hi) for each non-empty chunk], in chunk order."""
    workers = scan_workers() if workers is None else workers
    parts = max(1, min(n, workers*chunks_per_worker))
    bounds = [n*i//parts for i in range(parts+1)]
    ranges = [(bounds[i], bounds[i+1]) for i in range(parts) if bounds[i] < bounds[i+1]]
    if workers == 1 or threading.current_thread().name.startswith(_PREFIX):
        return [kernel(lo, hi) for lo, hi in ranges]
    pool = _pool(workers)
    futures = [pool.submit(kernel, lo, hi) for lo, hi in ranges]
    results, first = [], None
    for f in futures:            # wait for every chunk before raising
        try:
            results.append(f.result())
        except BaseException as exc:  # noqa: BLE001 - re-raised below after all chunks end
            first = exc if first is None else first
            results.append(None)
    if first is not None:
        raise first
    return results
