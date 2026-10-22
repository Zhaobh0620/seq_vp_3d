"""Opt-in CPU edge-hull proof; every elementary operation rounds outward.

No parallelism, fast-math, FMA contraction, or disk JIT cache. The original
Bernstein proof remains responsible for any tile this preproof cannot enclose.
Numba is imported only when the explicitly selected certificate needs it.
"""
import numpy as np
from importlib.metadata import version

from ..validation import InputValidationError
from .interval import ArithmeticUncertain, Interval
from .thread_chunks import run_chunks  # E54 thread-chunk execution

try:
    from numba import njit, config as numba_config
except ImportError as exc:
    raise InputValidationError("compiled certificate requires the optional numba==0.65.0 dependency") from exc

if numba_config.DISABLE_JIT:
    raise InputValidationError("compiled certificate requires Numba JIT; NUMBA_DISABLE_JIT is set")


LEGACY_KERNEL = 'numba-directed-edge-hull-v1'
HOISTED_KERNEL = 'numba-hoisted-edge-hull-v1'


def runtime_info(kernel=LEGACY_KERNEL):
    if kernel not in (LEGACY_KERNEL, HOISTED_KERNEL, 'numba-parallel-hoisted-edge-hull-v1'):
        raise InputValidationError("unsupported compiled enclosure kernel")
    parallel = kernel == 'numba-parallel-hoisted-edge-hull-v1'
    info = dict(numba=version('numba'), llvmlite=version('llvmlite'),
                fastmath=False, parallel=parallel, disk_cache=False,
                kernel=kernel)
    if parallel:
        # E54: 'parallel' now means Python threads over disjoint chunks of a serial nogil
        # kernel (thread_chunks.run_chunks); no Numba parallel region/threading layer is used.
        from .thread_chunks import scan_workers as _workers
        info.update(numba_threads=_workers(), per_cell_arithmetic=HOISTED_KERNEL,
                    execution='nogil-thread-chunks-v1')
    return info


@njit(cache=False, fastmath=False, inline='always', error_model='numpy')
def _outward(lo, hi):
    lo, hi = np.nextafter(lo, -np.inf), np.nextafter(hi, np.inf)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo > hi:
        raise ValueError("nonfinite compiled interval")
    return lo, hi


@njit(cache=False, fastmath=False, inline='always', error_model='numpy')
def _add(a, b):
    return _outward(a[0]+b[0], a[1]+b[1])


@njit(cache=False, fastmath=False, inline='always', error_model='numpy')
def _sub(a, b):
    return _outward(a[0]-b[1], a[1]-b[0])


@njit(cache=False, fastmath=False, inline='always', error_model='numpy')
def _mul(a, b):
    # Same four rounded products and ordered min/max as Interval.__mul__.
    x, y, z, w = a[0]*b[0], a[0]*b[1], a[1]*b[0], a[1]*b[1]
    return _outward(np.minimum(np.minimum(np.minimum(x, y), z), w),
                    np.maximum(np.maximum(np.maximum(x, y), z), w))


@njit(cache=False, fastmath=False, inline='always', error_model='numpy')
def _determinant_bounds(fl, fh):
    # Shared by both kernels, including the original left-associated sum.
    a = _sub(_mul((fl[1, 1], fh[1, 1]), (fl[2, 2], fh[2, 2])),
             _mul((fl[1, 2], fh[1, 2]), (fl[2, 1], fh[2, 1])))
    b = _sub(_mul((fl[1, 2], fh[1, 2]), (fl[2, 0], fh[2, 0])),
             _mul((fl[1, 0], fh[1, 0]), (fl[2, 2], fh[2, 2])))
    c = _sub(_mul((fl[1, 0], fh[1, 0]), (fl[2, 1], fh[2, 1])),
             _mul((fl[1, 1], fh[1, 1]), (fl[2, 0], fh[2, 0])))
    return _add(_add(_mul((fl[0, 0], fh[0, 0]), a),
                     _mul((fl[0, 1], fh[0, 1]), b)),
                _mul((fl[0, 2], fh[0, 2]), c))


@njit(cache=False, fastmath=False, error_model='numpy')
def arithmetic_supported(x):
    """Dynamic input prevents the rounding/underflow checks being constants."""
    sub = np.nextafter(x, 1.)
    tiny = np.finfo(np.float64).tiny
    return (sub > x and sub*1. == sub and (tiny+x)/2. > x
            and (1.+x)+2.**-53 == 1. and np.nextafter(x, -1.) < x)


@njit(cache=False, fastmath=False, error_model='numpy')
def _bounds(vertices, spacing):
    lo, hi = np.empty(len(vertices)), np.empty(len(vertices))
    # Reuse just 18 scalar slots; never allocate a matrix per cell/full volume.
    fl, fh = np.empty((3, 3)), np.empty((3, 3))
    for cell in range(len(vertices)):
        for j in range(3):
            for r in range(3):
                lower, upper = np.inf, -np.inf
                for a in range(8):
                    if a & (1 << j):
                        continue
                    b = a | (1 << j)  # Q1 vertices use x/y/z bits 0/1/2.
                    av, bv = vertices[cell, a, r], vertices[cell, b, r]
                    if not np.isfinite(av) or not np.isfinite(bv):
                        raise ValueError("nonfinite compiled vertex")
                    edge = _sub((bv, bv), (av, av))
                    # Positive point divisor: do not replace division by 1/h.
                    edge = _outward(edge[0]/spacing[j], edge[1]/spacing[j])
                    lower, upper = np.minimum(lower, edge[0]), np.maximum(upper, edge[1])
                identity = 1. if r == j else 0.
                fl[r, j], fh[r, j] = _add((lower, upper), (identity, identity))
        lo[cell], hi[cell] = _determinant_bounds(fl, fh)
    return lo, hi


@njit(cache=False, fastmath=False, inline='always', error_model='numpy')
def _edge_hull(vertices, j, r, h):
    # T46: positive division and directed rounding commute with finite extrema.
    lower, upper = np.inf, -np.inf
    for a in range(8):
        if a & (1 << j):
            continue
        b = a | (1 << j)
        d = vertices[b, r]-vertices[a, r]
        if not np.isfinite(d):
            raise ValueError("nonfinite compiled edge difference")
        lower, upper = np.minimum(lower, d), np.maximum(upper, d)
    lower, upper = _outward(lower, upper)  # Subtraction's original outward step.
    return _outward(lower/h, upper/h)     # Division's original outward step.


@njit(cache=False, fastmath=False, error_model='numpy')
def _hoisted_bounds(vertices, spacing):
    lo, hi = np.empty(len(vertices)), np.empty(len(vertices))
    fl, fh = np.empty((3, 3)), np.empty((3, 3))
    for cell in range(len(vertices)):
        for j in range(3):
            for r in range(3):
                edge = _edge_hull(vertices[cell], j, r, spacing[j])
                identity = 1. if r == j else 0.
                fl[r, j], fh[r, j] = _add(edge, (identity, identity))
        lo[cell], hi[cell] = _determinant_bounds(fl, fh)
    return lo, hi


# ---------------------------------------------------------------------------
# E50/S2: whole-map parallel execution of the SAME hoisted per-cell arithmetic.
# Each cell's operation sequence (raw edge differences, hull, two outward
# steps, identity add, left-associated cofactor determinant, then the caller's
# outward "- delta") is identical to _hoisted_bounds + Interval.__sub__.
# Exceptions are not raised inside the kernel; a per-cell flag records any
# nonfinite/invalid intermediate and the caller restores the legacy
# whole-tile fallback. Cells are independent, so thread count/order cannot
# change any endpoint; only the global min is formed afterwards (exact).
# ---------------------------------------------------------------------------
PARALLEL_KERNEL = 'numba-parallel-hoisted-edge-hull-v1'



@njit(cache=False, fastmath=False, inline='always', error_model='numpy')
def _nr_outward(lo, hi):
    lo, hi = np.nextafter(lo, -np.inf), np.nextafter(hi, np.inf)
    return lo, hi, np.isfinite(lo) and np.isfinite(hi) and lo <= hi


@njit(cache=False, fastmath=False, inline='always', error_model='numpy')
def _nr_add(alo, ahi, blo, bhi):
    return _nr_outward(alo+blo, ahi+bhi)


@njit(cache=False, fastmath=False, inline='always', error_model='numpy')
def _nr_sub(alo, ahi, blo, bhi):
    return _nr_outward(alo-bhi, ahi-blo)


@njit(cache=False, fastmath=False, inline='always', error_model='numpy')
def _nr_mul(alo, ahi, blo, bhi):
    x, y, z, w = alo*blo, alo*bhi, ahi*blo, ahi*bhi
    return _nr_outward(np.minimum(np.minimum(np.minimum(x, y), z), w),
                       np.maximum(np.maximum(np.maximum(x, y), z), w))


@njit(cache=False, fastmath=False, inline='always', error_model='numpy')
def _nr_entry(u, base, ny, nx, j, r, h):
    """F_rj interval of one cell: same steps as _edge_hull then the identity add."""
    ok = True
    lower, upper = np.inf, -np.inf
    for a in range(8):
        if a & (1 << j):
            continue
        b = a | (1 << j)
        na = base+(a >> 2)*ny*nx+((a >> 1) & 1)*nx+(a & 1)
        nb = base+(b >> 2)*ny*nx+((b >> 1) & 1)*nx+(b & 1)
        d = u[nb, r]-u[na, r]
        if not np.isfinite(d):
            ok = False
        lower, upper = np.minimum(lower, d), np.maximum(upper, d)
    lo, hi, good = _nr_outward(lower, upper)
    ok = ok and good
    lo, hi, good = _nr_outward(lo/h, hi/h)
    ok = ok and good
    identity = 1. if r == j else 0.
    lo, hi, good = _nr_add(lo, hi, identity, identity)
    return lo, hi, ok and good


@njit(cache=False, fastmath=False, nogil=True, error_model='numpy')
def _chunk_hoisted_margins(u, ny, nx, cy, cx, spacing, delta, start, stop, margin, flag):
    """margin[k] = lo(J - delta) for cell k in [start, stop); flag[k]=1 marks an invalid step.

    E54: serial nogil kernel over one chunk (no prange/parfors: Numba 0.65's parfor
    pass failed nondeterministically in compilation). Threads run disjoint chunks;
    each cell's arithmetic is unchanged, so every endpoint is bitwise the same.
    """
    hx, hy, hz = spacing[0], spacing[1], spacing[2]
    for c in range(start, stop):
        k = c
        x0 = c % cx
        y0 = (c//cx) % cy
        z0 = c//(cx*cy)
        base = (z0*ny+y0)*nx+x0
        # Entries F[r, j] in the legacy (j outer, r inner) evaluation order.
        f00l, f00h, e1 = _nr_entry(u, base, ny, nx, 0, 0, hx)
        f10l, f10h, e2 = _nr_entry(u, base, ny, nx, 0, 1, hx)
        f20l, f20h, e3 = _nr_entry(u, base, ny, nx, 0, 2, hx)
        f01l, f01h, e4 = _nr_entry(u, base, ny, nx, 1, 0, hy)
        f11l, f11h, e5 = _nr_entry(u, base, ny, nx, 1, 1, hy)
        f21l, f21h, e6 = _nr_entry(u, base, ny, nx, 1, 2, hy)
        f02l, f02h, e7 = _nr_entry(u, base, ny, nx, 2, 0, hz)
        f12l, f12h, e8 = _nr_entry(u, base, ny, nx, 2, 1, hz)
        f22l, f22h, e9 = _nr_entry(u, base, ny, nx, 2, 2, hz)
        # Same left-associated cofactor expansion as _determinant_bounds.
        p1l, p1h, g1 = _nr_mul(f11l, f11h, f22l, f22h)
        p2l, p2h, g2 = _nr_mul(f12l, f12h, f21l, f21h)
        al, ah, g3 = _nr_sub(p1l, p1h, p2l, p2h)
        p1l, p1h, g4 = _nr_mul(f12l, f12h, f20l, f20h)
        p2l, p2h, g5 = _nr_mul(f10l, f10h, f22l, f22h)
        bl, bh, g6 = _nr_sub(p1l, p1h, p2l, p2h)
        p1l, p1h, g7 = _nr_mul(f10l, f10h, f21l, f21h)
        p2l, p2h, g8 = _nr_mul(f11l, f11h, f20l, f20h)
        ccl, cch, g9 = _nr_sub(p1l, p1h, p2l, p2h)
        t1l, t1h, g10 = _nr_mul(f00l, f00h, al, ah)
        t2l, t2h, g11 = _nr_mul(f01l, f01h, bl, bh)
        sl, sh, g12 = _nr_add(t1l, t1h, t2l, t2h)
        t3l, t3h, g13 = _nr_mul(f02l, f02h, ccl, cch)
        jl, jh, g14 = _nr_add(sl, sh, t3l, t3h)
        # Interval.__sub__(point delta): outward(lo-delta, hi-delta), then validated.
        ml, mh, g15 = _nr_outward(jl-delta, jh-delta)
        ok = (e1 and e2 and e3 and e4 and e5 and e6 and e7 and e8 and e9 and g1 and g2 and g3
              and g4 and g5 and g6 and g7 and g8 and g9 and g10 and g11 and g12 and g13 and g14 and g15)
        margin[k] = ml
        flag[k] = 0 if ok else 1


# E98 (throughput-v16): process-level choice of the margins kernel. The certificate configuration (kernel name,
# certificate_version, problem_id) is unchanged; only the execution differs, bit for bit for finite u (with
# non-finite input, NaN payloads in already flagged cells may differ: the order of two NaN operands is not fixed by
# the compiler; callers pass finite u). Until a grid is verified, every call also runs this module's libm kernel and
# compares margins and flags bit for bit; a match on an informative state (see _informative) verifies the grid; a
# mismatch, or any failure of the SoA path, switches the row-SoA kernel off for the rest of the process (recorded).
_MARGINS_EXECUTION = ["libm"]
_SOA_VERIFIED = set()
SOA_CERTIFICATE_RECORD = dict(policy="row-soa-bitstep-margins-v1", calls=0, compared=0, verified_grids=0,
                              mismatches=0, errors=0, disabled=False)
_SOA_COUNTS = ("calls", "compared", "verified_grids", "mismatches", "errors")


def select_margins_execution(name):
    """'libm' (every profile up to v15) or 'soa' (v16); set from the run's profile before any certificate."""
    if name not in ("libm", "soa"):
        raise ValueError("margins execution: 'libm' or 'soa'")
    _MARGINS_EXECUTION[0] = name


def margins_execution():
    return _MARGINS_EXECUTION[0]


def soa_certificate_record(since=None):
    """E98 (record only): the SoA dispatch counts, process-wide since import or, with `since` (an earlier record),
    the differences since then; `policy` and `disabled` (the process state now) are not differenced."""
    record = dict(SOA_CERTIFICATE_RECORD)
    for name in _SOA_COUNTS:
        record[name] -= (since or {}).get(name, 0)
    return record


def _informative(u, margin):
    """E98 (review): a first-use match verifies a grid only on a state that exercises the per-cell arithmetic --
    every displacement component nonzero somewhere and the margins not all equal. At u = 0 every cell has F = I,
    where a transposed entry, a wrong spacing or a wrong row decomposition would go unseen."""
    flat = np.asarray(u).reshape(-1, 3)
    return bool(np.all(np.any(flat != 0, axis=0)) and np.any(margin != margin[0]))


def _soa_off(error):
    SOA_CERTIFICATE_RECORD["errors" if error else "mismatches"] += 1
    SOA_CERTIFICATE_RECORD["disabled"] = True


def parallel_margins(u, grid, delta):
    """Whole-map per-cell margin lower bounds and invalid-step flags (E50; E54 thread chunks)."""
    if not arithmetic_supported(np.float64(0.)):
        raise ArithmeticUncertain("compiled binary64/gradual-underflow prerequisites failed")
    if _MARGINS_EXECUTION[0] == "soa" and not SOA_CERTIFICATE_RECORD["disabled"]:
        from .soa_enclosure import soa_margins
        # E98 (review): a failure (e.g. a MemoryError for the component-major copy, the second result or the
        # comparison's temporaries) switches the SoA kernel off; the fallback runs after the except block, when the
        # failed call's buffers (kept alive by its traceback inside the block) are released.
        result = None
        try:
            result = soa_margins(u, grid, delta)
        except Exception:
            pass
        if result is None:
            _soa_off(error=True)
            return _libm_margins(u, grid, delta)
        SOA_CERTIFICATE_RECORD["calls"] += 1
        key = (grid.grid_id, float(delta).hex())
        if key in _SOA_VERIFIED:
            return result
        reference = verdict = None
        try:
            reference = _libm_margins(u, grid, delta)
            same = (np.array_equal(result[0].view(np.uint64), reference[0].view(np.uint64))
                    and np.array_equal(result[1], reference[1]))
            verdict = (same, same and _informative(u, reference[0]))
        except Exception:
            pass
        if verdict is None:
            result = None                         # the libm result if it exists, else libm alone, as v15
            _soa_off(error=True)
            return reference if reference is not None else _libm_margins(u, grid, delta)
        SOA_CERTIFICATE_RECORD["compared"] += 1
        if not verdict[0]:
            _soa_off(error=False)
            return reference
        if verdict[1]:
            _SOA_VERIFIED.add(key)
            SOA_CERTIFICATE_RECORD["verified_grids"] += 1
        return result
    return _libm_margins(u, grid, delta)


def _libm_margins(u, grid, delta):
    nz, ny, nx = grid.shape
    flat = np.ascontiguousarray(u, dtype=np.float64).reshape(-1, 3)
    n = grid.n_cells
    margin = np.empty(n); flag = np.empty(n, dtype=np.uint8)
    spacing = np.asarray(grid.spacing, dtype=np.float64)
    run_chunks(lambda lo, hi: _chunk_hoisted_margins(flat, ny, nx, ny-1, nx-1, spacing, float(delta),
                                                     lo, hi, margin, flag), n)
    return margin, flag.astype(bool)




def enclosure(vertices, spacing, *, kernel=LEGACY_KERNEL):
    """Internal validated float64 tile entry, with fail-closed runtime checks."""
    if kernel not in (LEGACY_KERNEL, HOISTED_KERNEL):
        raise InputValidationError("unsupported compiled enclosure kernel")
    if not arithmetic_supported(np.float64(0.)):
        raise ArithmeticUncertain("compiled binary64/gradual-underflow prerequisites failed")
    try:
        bounds = _hoisted_bounds if kernel == HOISTED_KERNEL else _bounds
        lo, hi = bounds(vertices, np.asarray(spacing, dtype=np.float64))
    except ValueError as exc:
        # Preserve the legacy all-or-nothing prepass at this tile boundary.
        raise ArithmeticUncertain(str(exc)) from exc
    return Interval(lo, hi)
