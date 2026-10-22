"""E89 (P02): glibc malloc retention for large host arrays.

glibc serves blocks above its mmap threshold (dynamic, capped at 32 MiB) with a fresh mmap and unmaps them on
free, so every new native-size array (153-165 MB) pays page faults and kernel zeroing on first touch. Raising the
mmap threshold to 1 GiB and the trim threshold to 2^31 - 1 makes such blocks come from the heap and keeps freed
pages mapped for reuse. This is allocator policy only: no value, operation or reduction order changes.

The setting is process-wide and cannot be undone within the process; it is applied once, by the worker or the probe,
under the explicit throughput profile v11, and recorded: the glibc version, the mallopt return codes, and a self-check
that allocates a 64 MiB array and reads glibc's mmapped-bytes counter (mallinfo2 hblkhd) around it.
"""
import ctypes
import ctypes.util

import numpy as np

POLICY = "glibc-malloc-retention-v1"
M_TRIM_THRESHOLD = -1          # glibc mallopt parameter numbers (malloc.h)
M_MMAP_THRESHOLD = -3
MMAP_THRESHOLD = 1 << 30       # 1 GiB: every native-size array is served from the heap
TRIM_THRESHOLD = 2**31 - 1     # never hand freed heap pages back to the OS within the run
SELF_CHECK_BYTES = 64 << 20    # above glibc's dynamic mmap cap (32 MiB), below the new threshold

_STATE = {}


class _Mallinfo2(ctypes.Structure):
    _fields_ = [(name, ctypes.c_size_t) for name in
                ("arena", "ordblks", "smblks", "hblks", "hblkhd", "usmblks", "fsmblks", "uordblks", "fordblks", "keepcost")]


def _libc():
    return ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)


def _probe(libc):
    """Bytes newly mmapped by glibc for one fresh 64 MiB array (0: served from the heap), and its page offset."""
    before = int(libc.mallinfo2().hblkhd)
    probe = np.empty(SELF_CHECK_BYTES//8, dtype=np.float64)
    try:
        return dict(mmapped_bytes_delta=int(libc.mallinfo2().hblkhd)-before, page_offset=int(probe.ctypes.data % 4096))
    finally:
        del probe


def enable_malloc_retention():
    """Apply the retention policy once per process and return its record (the same record on later calls)."""
    if _STATE:
        return dict(_STATE)
    record = dict(policy=POLICY, mmap_threshold=MMAP_THRESHOLD, trim_threshold=TRIM_THRESHOLD)
    try:
        libc = _libc()
        libc.gnu_get_libc_version.restype = ctypes.c_char_p
        record["glibc"] = libc.gnu_get_libc_version().decode()
        libc.mallopt.argtypes = (ctypes.c_int, ctypes.c_int)
        libc.mallopt.restype = ctypes.c_int
        libc.mallinfo2.restype = _Mallinfo2
        record["probe_before"] = _probe(libc)          # depends on the process's history (free heap space)
        record["mmap_threshold_set"] = libc.mallopt(M_MMAP_THRESHOLD, MMAP_THRESHOLD) == 1
        record["trim_threshold_set"] = libc.mallopt(M_TRIM_THRESHOLD, TRIM_THRESHOLD) == 1
        record["probe_after"] = _probe(libc)
    except (OSError, AttributeError) as exc:      # not glibc (e.g. musl), or no mallinfo2: run without the policy
        record.update(error=repr(exc), mmap_threshold_set=False, trim_threshold_set=False)
    record["active"] = bool(record.get("mmap_threshold_set") and record.get("trim_threshold_set"))
    # Self-check: after the setting, a fresh 64 MiB array is not mmapped (glibc's mmapped-bytes counter unchanged).
    record["self_check_heap"] = (record.get("probe_after") or {}).get("mmapped_bytes_delta") == 0
    _STATE.update(record)
    return dict(record)
