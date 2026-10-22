"""E81 (throughput v10): compiled T69/T73 face/interior estimator, bit for bit the NumPy one.

`jd_affine.face_interior_minima` spends its time in Python-level loops over batched 27-term
evaluations (the bounded Newton's derivative sums and its backtracking values). Here those two
parts run in serial nogil Numba kernels that reproduce NumPy's float64 arithmetic exactly:
- products in NumPy's order: row = (Z_k * Y_j) * X_i, then C * row;
- the 27-term sum in NumPy's add-reduce order for a contiguous length-27 axis: eight
  accumulators seeded with a[0..7], blocks a[8..15], a[16..23], combined as
  ((r0+r1)+(r2+r3))+((r4+r5)+(r6+r7)), then a[24], a[25], a[26] in turn, added to 0.0;
- np.clip(v, 0, 1) as (v > 0 ? v : 0) then (w < 1 ? w : 1), NaN passed through.
The 3x3 eigen-decompositions (LAPACK; NumPy and Numba link different builds), the einsum of the
Newton step, the outer iteration loop, `_monomial`, the 9^3 grid product and the argsort stay the
original NumPy code, so the result equals face_interior_minima exactly whenever the NumPy estimator
returns (tests and an independent review check this bitwise). Where the NumPy version raises (only for
|J| of order 1e306 and above, where its errstate catches an overflow), this version raises
FloatingPointError too: a derivative or a trial value that is not finite is exactly an overflow there.
Metric only: like the NumPy estimator this is never a bound, a certificate or an acceptance input.
"""
import numpy as np

from ...validation import InputValidationError, require_finite_result
from .jd_affine import PARALLEL_ROWS, _G9, _V9, _monomial, _value

try:
    from numba import njit
except ImportError as exc:  # pragma: no cover
    raise InputValidationError("compiled T69 estimator requires the optional numba dependency") from exc

COMPILED_ESTIMATOR = "numba-face-interior-v1"
_KW = dict(cache=False, fastmath=False, nogil=True, error_model="numpy")


@njit(**_KW)
def _sum27(a):
    r0, r1, r2, r3, r4, r5, r6, r7 = a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]
    for i in range(8, 24, 8):
        r0 += a[i]
        r1 += a[i+1]
        r2 += a[i+2]
        r3 += a[i+3]
        r4 += a[i+4]
        r5 += a[i+5]
        r6 += a[i+6]
        r7 += a[i+7]
    res = ((r0+r1)+(r2+r3))+((r4+r5)+(r6+r7))
    res += a[24]
    res += a[25]
    res += a[26]
    return 0.0+res


@njit(**_KW)
def _powers(t, out):
    """out[0] = (1, t, t*t), out[1] = (0, 1, 2t), out[2] = (0, 0, 2) as in jd_affine._powers."""
    out[0, 0], out[0, 1], out[0, 2] = 1.0, t, t*t
    out[1, 0], out[1, 1], out[1, 2] = 0.0, 1.0, 2.0*t
    out[2, 0], out[2, 1], out[2, 2] = 0.0, 0.0, 2.0


@njit(**_KW)
def _e(c, Z, Y, X, tmp):
    for k in range(3):
        for j in range(3):
            zy = Z[k]*Y[j]
            for i in range(3):
                tmp[9*k+3*j+i] = c[9*k+3*j+i]*(zy*X[i])
    return _sum27(tmp)


@njit(**_KW)
def _derivatives_kernel(C, x, g, H):
    px, py, pz = np.empty((3, 3)), np.empty((3, 3)), np.empty((3, 3))
    tmp = np.empty(27)
    for n in range(C.shape[0]):
        _powers(x[n, 0], px)
        _powers(x[n, 1], py)
        _powers(x[n, 2], pz)
        c = C[n]
        g[n, 0] = _e(c, pz[0], py[0], px[1], tmp)
        g[n, 1] = _e(c, pz[0], py[1], px[0], tmp)
        g[n, 2] = _e(c, pz[1], py[0], px[0], tmp)
        H[n, 0, 0] = _e(c, pz[0], py[0], px[2], tmp)
        H[n, 1, 1] = _e(c, pz[0], py[2], px[0], tmp)
        H[n, 2, 2] = _e(c, pz[2], py[0], px[0], tmp)
        v = _e(c, pz[0], py[1], px[1], tmp)
        H[n, 0, 1] = v
        H[n, 1, 0] = v
        v = _e(c, pz[1], py[0], px[1], tmp)
        H[n, 0, 2] = v
        H[n, 2, 0] = v
        v = _e(c, pz[1], py[1], px[0], tmp)
        H[n, 1, 2] = v
        H[n, 2, 1] = v


@njit(**_KW)
def _clip01(v):
    if v != v:
        return v
    w = v if v > 0.0 else 0.0
    return w if w < 1.0 else 1.0


@njit(**_KW)
def _backtrack_kernel(C, x, d, best, xs, bs, done):
    """Per row: steps 1, 1/2, ... (at most 40) until the clipped point strictly lowers the value.
    Returns True if any trial value was not finite (the NumPy loop would have raised)."""
    px, py, pz = np.empty((3, 3)), np.empty((3, 3)), np.empty((3, 3))
    tmp = np.empty(27)
    xt = np.empty(3)
    overflow = False
    for n in range(C.shape[0]):
        step = 1.0
        done[n] = False
        for _ in range(40):
            for a in range(3):
                xt[a] = _clip01(x[n, a]+step*d[n, a])
            _powers(xt[0], px)
            _powers(xt[1], py)
            _powers(xt[2], pz)
            ft = _e(C[n], pz[0], py[0], px[0], tmp)
            if not (ft - ft == 0.0):
                overflow = True
            if ft < best[n]:
                xs[n, 0], xs[n, 1], xs[n, 2] = xt[0], xt[1], xt[2]
                bs[n] = ft
                done[n] = True
                break
            step = .5*step
    return overflow


def _bounded_newton(C, x, best, iterations):
    """jd_affine._bounded_newton with the derivative sums and the backtracking compiled."""
    x, best = x.copy(), best.copy()
    live = np.ones(len(x), dtype=bool)
    for _ in range(iterations):
        idx = np.flatnonzero(live)
        if idx.size == 0:
            break
        xi, Ci = x[idx], C[idx]
        g, H = np.empty((idx.size, 3)), np.empty((idx.size, 3, 3))
        _derivatives_kernel(Ci, xi, g, H)
        if not (np.all(np.isfinite(g)) and np.all(np.isfinite(H))):
            raise FloatingPointError("T69 face/interior derivatives overflowed")     # NumPy's errstate would raise
        free = ~(((xi <= 0.) & (g > 0)) | ((xi >= 1.) & (g < 0)))
        gf = np.where(free, g, 0.)
        Hf = H*free[:, :, None]*free[:, None, :]+np.eye(3)*(~free)[:, :, None]
        w, Q = np.linalg.eigh(Hf)
        scale = np.maximum(1., np.abs(w).max(axis=1))
        ok = np.all(w > 1e-12*scale[:, None], axis=1)
        newton = -np.einsum("nij,nj,nkj,nk->ni", Q, 1/np.where(ok[:, None], w, 1.), Q, gf)
        d = np.where(ok[:, None], newton, -gf/scale[:, None])
        xs, bs, done = np.empty((idx.size, 3)), np.empty(idx.size), np.empty(idx.size, dtype=np.bool_)
        if _backtrack_kernel(Ci, xi, np.ascontiguousarray(d), best[idx], xs, bs, done):
            raise FloatingPointError("T69 face/interior trial value overflowed")      # NumPy's errstate would raise
        x[idx[done]], best[idx[done]] = xs[done], bs[done]
        live[idx[~done]] = False          # no strict decrease: converged to rounding (or stuck)
    return best


TOP4_POLICY = "numba-stable-top-k-v1"
TOP4_VERIFY_CALLS = 3          # informative calls compared with the argsort before the selection is trusted
_TOP4_LOCK = __import__("threading").Lock()


@njit(**_KW)
def _less(v, w):
    """NumPy's float sort order (npy_sort LT): NaN after every number, -0 == +0."""
    return v < w or (w != w and v == v)


@njit(**_KW)
def _stable_top(G, k, out):
    """out[i] = the first k indices of a stable ascending sort of row G[i] (k <= G.shape[1]): an insertion into the
    k smallest (value, index) pairs; an equal value goes after the ones already kept, so ties stay in index order."""
    n, m = G.shape
    for i in range(n):
        cnt = 0
        for j in range(m):
            v = G[i, j]
            pos = cnt
            while pos > 0 and _less(v, G[i, out[i, pos-1]]):
                pos -= 1
            if pos < k:
                q = cnt if cnt < k else k-1
                while q > pos:
                    out[i, q] = out[i, q-1]
                    q -= 1
                out[i, pos] = j
                if cnt < k:
                    cnt += 1


def new_top4_record():
    """E100 (throughput-v18): the record of the compiled stable top-k selection (one per backend)."""
    return dict(policy=TOP4_POLICY, selfcheck=None, calls=0, compared=0, informative=0, verified=False,
                mismatches=0, disabled=False)


def _argsort_top(grid_values, starts):
    return np.argsort(grid_values, axis=1, kind="stable")[:, :starts]


def top_select(grid_values, starts, record):
    """E100: np.argsort(grid_values, axis=1, kind="stable")[:, :starts], bit for bit (int64 indices).
    The compiled selection is trusted after the synthetic self-check and TOP4_VERIFY_CALLS informative calls whose
    result equals the argsort; until then the argsort's own array is returned. A mismatch switches it off."""
    n = grid_values.shape[0]
    usable = (record is not None and not record["disabled"] and record["selfcheck"] is True
              and isinstance(grid_values, np.ndarray) and grid_values.dtype == np.float64 and grid_values.ndim == 2
              and grid_values.flags.c_contiguous and 0 < starts <= grid_values.shape[1])
    if not usable:
        return _argsort_top(grid_values, starts)
    out = np.empty((n, starts), dtype=np.int64)
    _stable_top(grid_values, starts, out)
    with _TOP4_LOCK:
        record["calls"] += 1
        verified = record["verified"]
    if verified:
        return out
    reference = _argsort_top(grid_values, starts)
    same = reference.dtype == out.dtype and reference.shape == out.shape and np.array_equal(reference, out)
    with _TOP4_LOCK:
        record["compared"] += 1
        if not same:
            record["mismatches"] += 1
            record["disabled"] = True
        elif n > 0:
            record["informative"] += 1
            if record["informative"] >= TOP4_VERIFY_CALLS:
                record["verified"] = True
    return reference


def _top4_cases():
    """Synthetic rows for the self-check: ties, NaN (payloads and signs), +-0, +-inf, subnormals, equal rows,
    and random rows of the production width 729."""
    rng = np.random.default_rng(20261008)
    tiny, nan2 = 5e-324, np.frombuffer(np.uint64(0xfff8000000000001).tobytes(), dtype=np.float64)[0]
    special = np.array([0., -0., np.inf, -np.inf, np.nan, -np.nan, nan2, tiny, -tiny, 1., -1., 2.2250738585072014e-308])
    cases = [rng.choice(special, size=(64, 40)),
             rng.integers(0, 3, size=(64, 40)).astype(np.float64),
             np.zeros((4, 40)), np.full((4, 40), np.nan), np.ones((4, 729)),
             np.where(rng.random((32, 729)) < .3, rng.choice(special, size=(32, 729)), rng.standard_normal((32, 729))),
             rng.standard_normal((64, 729)), rng.integers(-2, 3, size=(64, 729)).astype(np.float64)*.5]
    return [np.ascontiguousarray(c) for c in cases]


def top4_selfcheck(record, starts=4):
    """E100: the compiled selection against the argsort on synthetic rows (before any production call)."""
    same = True
    for G in _top4_cases():
        out = np.empty((G.shape[0], starts), dtype=np.int64)
        _stable_top(G, starts, out)
        same = same and np.array_equal(out, _argsort_top(G, starts))
    record["selfcheck"] = bool(same)
    if not same:
        record["disabled"] = True
    return same


def prewarm_top4():
    """Compile the E100 selection outside any task clock; returns the seconds spent."""
    from time import perf_counter
    tick = perf_counter()
    out = np.empty((2, 4), dtype=np.int64)
    _stable_top(np.arange(10., dtype=np.float64).reshape(2, 5), 4, out)
    return perf_counter()-tick


def face_interior_minima(J27, *, starts=4, iterations=30, top4=None):
    """Same contract and result as jd_affine.face_interior_minima (bitwise).
    top4 (E100): a new_top4_record() to select the Newton starts by top_select (the argsort's indices, bit for bit)."""
    J27 = np.asarray(J27, dtype=np.float64)
    if J27.ndim != 2 or J27.shape[1] != 27:
        raise InputValidationError("closed27 JD values must have shape (cells, 27)")
    require_finite_result(J27, "T69 closed27 JD")
    n = len(J27)
    if n == 0:
        return np.empty(0)
    with np.errstate(over="raise", invalid="raise"):
        C = _monomial(J27)
        grid_values = J27 @ _V9.T
        order = (np.argsort(grid_values, axis=1, kind="stable")[:, :starts] if top4 is None
                 else top_select(grid_values, starts, top4))
        best = grid_values.min(axis=1)
        x0 = _G9[order.T.ravel()]                                  # start-major (starts*n, 3)
        Cs = np.ascontiguousarray(np.tile(C, (order.shape[1], 1)))
        found = _bounded_newton(Cs, x0, _value(Cs, x0), iterations).reshape(order.shape[1], n)
        result = np.minimum(best, found.min(axis=0))
    require_finite_result(result, "T69 face/interior minima")
    return result


def face_interior_minima_parallel(J27, top4=None):
    """Same row chunking as jd_affine.face_interior_minima_parallel (hence the same per-chunk GEMM rounding)."""
    from ..thread_chunks import run_chunks
    J27 = np.asarray(J27, dtype=np.float64)
    extra = {} if top4 is None else dict(top4=top4)       # E100
    if len(J27) < PARALLEL_ROWS:
        return face_interior_minima(J27, **extra)
    if top4 is not None and not top4["verified"]:
        # E100 (review): pool-thread chunks never take part in the check (its counts stay independent of thread
        # timing); before verification they use the argsort itself
        extra = {}
    parts = run_chunks(lambda lo, hi: face_interior_minima(J27[lo:hi], **extra), len(J27), chunks_per_worker=1)
    return np.concatenate(parts)


def prewarm():
    """Compile the kernels outside any task clock; returns the seconds spent."""
    from time import perf_counter
    tick = perf_counter()
    rng = np.random.default_rng(0)
    face_interior_minima(1.+.1*rng.standard_normal((3, 27)))
    return perf_counter()-tick
