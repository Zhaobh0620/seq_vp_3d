"""E51 exact parallel scans for large CPU arrays (order-independent results only).

Every function here returns exactly what its NumPy reference returns: booleans,
integer counts, IEEE element-wise quotients and their min/max. Only
reductions whose result cannot depend on evaluation order are parallelized
(min, max, all, integer sums); no floating-point sum or dot product is here.
Used only when Numba is importable and the array is large; otherwise callers
keep the original NumPy code. Equivalence is tested in test_fast_checks_e51.py.
"""
import numpy as np

MIN_SIZE = 1 << 20

try:
    from numba import njit, config as _numba_config
    _NUMBA = not _numba_config.DISABLE_JIT   # same rule as the compiled certificate modules
except ImportError:  # pragma: no cover
    _NUMBA = False

# E54: serial nogil kernels over disjoint chunks, run in threads (thread_chunks.run_chunks).
# No prange/parfors: Numba 0.65's parfor pass failed nondeterministically in compilation.
# Only order-independent reductions are combined across chunks, so results are unchanged.


def enabled(size):
    return _NUMBA and size >= MIN_SIZE


if _NUMBA:
    @njit(cache=False, fastmath=False, nogil=True, error_model="numpy")
    def _domain_counts(u, ax, ay, az, z0, z1):
        nz, ny, nx = u.shape[0], u.shape[1], u.shape[2]
        nonfinite = 0
        invalid = 0
        lo = (ax[0], ay[0], az[0])
        hi = (ax[ax.shape[0]-1], ay[ay.shape[0]-1], az[az.shape[0]-1])
        for z in range(z0, z1):
            bad_nf = 0
            bad = 0
            for y in range(ny):
                for x in range(nx):
                    finite = np.isfinite(u[z, y, x, 0]) and np.isfinite(u[z, y, x, 1]) and np.isfinite(u[z, y, x, 2])
                    if not finite:
                        bad_nf += 1
                    interior = 0 < z < nz-1 and 0 < y < ny-1 and 0 < x < nx-1
                    b = False
                    for j in range(3):
                        c = ax[x] if j == 0 else (ay[y] if j == 1 else az[z])
                        v = u[z, y, x, j]+c
                        if not np.isfinite(v) or v < lo[j] or v > hi[j]:
                            b = True
                        if interior and (v <= lo[j] or v >= hi[j]):
                            b = True
                    if b:
                        bad += 1
            nonfinite += bad_nf
            invalid += bad
        return nonfinite, invalid

    @njit(cache=False, fastmath=False, nogil=True, error_model="numpy")
    def _step_bound(v, p, ax, ay, az, z0, z1):
        # v: (3, mz, my, mx) direction; p: (mz, my, mx, 3) interior identity+u.
        mz, my, mx = v.shape[1], v.shape[2], v.shape[3]
        best = np.full(mz, np.inf)
        for z in range(z0, z1):
            b = np.inf
            for y in range(my):
                for x in range(mx):
                    for r in range(3):
                        d = v[r, z, y, x]
                        if d > 0:
                            top = ax[ax.shape[0]-1] if r == 0 else (ay[ay.shape[0]-1] if r == 1 else az[az.shape[0]-1])
                            q = (top-p[z, y, x, r])/d
                            if q < b:
                                b = q
                        elif d < 0:
                            bottom = ax[0] if r == 0 else (ay[0] if r == 1 else az[0])
                            q = (bottom-p[z, y, x, r])/d
                            if q < b:
                                b = q
            best[z] = b
        return best[z0:z1].min()

    @njit(cache=False, fastmath=False, nogil=True, error_model="numpy")
    def _step_bound_xu(v, u, ax, ay, az, z0, z1):
        # E76: the same quotients as _step_bound with p = fl(axis + u) formed on the fly
        # (grid.identity() is the axes broadcast), so no identity/sum/contiguous copies.
        mz, my, mx = v.shape[1], v.shape[2], v.shape[3]
        best = np.full(mz, np.inf)
        for z in range(z0, z1):
            b = np.inf
            for y in range(my):
                for x in range(mx):
                    for r in range(3):
                        d = v[r, z, y, x]
                        if d > 0 or d < 0:
                            base = ax[x+1] if r == 0 else (ay[y+1] if r == 1 else az[z+1])
                            p = base+u[z+1, y+1, x+1, r]
                            if d > 0:
                                top = ax[ax.shape[0]-1] if r == 0 else (ay[ay.shape[0]-1] if r == 1 else az[az.shape[0]-1])
                                q = (top-p)/d
                            else:
                                bottom = ax[0] if r == 0 else (ay[0] if r == 1 else az[0])
                                q = (bottom-p)/d
                            if q < b:
                                b = q
            best[z] = b
        return best[z0:z1].min()

    @njit(cache=False, fastmath=False, nogil=True, error_model="numpy")
    def _max_abs(a, i0, i1):
        m = 0.
        for i in range(i0, i1):
            x = abs(a[i])
            if x > m or x != x:
                m = x
        return m

    @njit(cache=False, fastmath=False, nogil=True, error_model="numpy")
    def _max_abs_diff(a, b, i0, i1):
        m = 0.
        for i in range(i0, i1):
            x = abs(a[i]-b[i])
            if x > m or x != x:
                m = x
        return m

    @njit(cache=False, fastmath=False, nogil=True, error_model="numpy")
    def _all_finite(a, i0, i1):
        for i in range(i0, i1):
            if not np.isfinite(a[i]):
                return False
        return True

    @njit(cache=False, fastmath=False, nogil=True, error_model="numpy")
    def _max_node_sq(v, hx, hy, hz, i0, i1):
        # E84 T81: per node the reference's IEEE operations in its order, (a*a+b*b)+c*c with
        # a = v0/hx, b = v1/hy, c = v2/hz (no fastmath, so no contraction); the max is order independent.
        m = 0.
        for i in range(i0, i1):
            a = v[0, i]/hx
            b = v[1, i]/hy
            c = v[2, i]/hz
            x = a*a+b*b+c*c
            if x > m or x != x:
                m = x
        return m


def _chunks(kernel, n):
    from .thread_chunks import run_chunks
    return run_chunks(kernel, n)


def domain_counts(u, axes):
    """(#nodes with a nonfinite component, #nodes violating the domain rule)."""
    ax, ay, az = (np.ascontiguousarray(a, dtype=np.float64) for a in axes)
    u = np.ascontiguousarray(u)
    parts = _chunks(lambda z0, z1: _domain_counts(u, ax, ay, az, z0, z1), u.shape[0])
    return sum(p[0] for p in parts), sum(p[1] for p in parts)


def step_bound(direction, points, axes):
    ax, ay, az = (np.ascontiguousarray(a, dtype=np.float64) for a in axes)
    v, p = np.ascontiguousarray(direction), np.ascontiguousarray(points)
    return float(min(_chunks(lambda z0, z1: _step_bound(v, p, ax, ay, az, z0, z1), v.shape[1])))


def step_bound_xu(direction, u, axes):
    """E76: equals step_bound(direction, (identity+u)[1:-1,1:-1,1:-1], axes) bit for bit."""
    ax, ay, az = (np.ascontiguousarray(a, dtype=np.float64) for a in axes)
    v, u = np.ascontiguousarray(direction), np.ascontiguousarray(u)
    return float(min(_chunks(lambda z0, z1: _step_bound_xu(v, u, ax, ay, az, z0, z1), v.shape[1])))


def max_abs(a):
    a = np.ascontiguousarray(a).reshape(-1)
    m = 0.
    for x in _chunks(lambda i0, i1: _max_abs(a, i0, i1), a.shape[0]):
        if x > m or x != x:   # same NaN-propagating rule as within a chunk
            m = x
    return float(m)


def max_node_sq_reference(direction, spacing):
    """E84 T81 NumPy reference: max_i sum_d (p_{d,i}/h_d)^2 of a component-major (3, n) direction."""
    p = np.asarray(direction, dtype=np.float64).reshape(3, -1)
    a, b, c = p[0]/float(spacing[0]), p[1]/float(spacing[1]), p[2]/float(spacing[2])
    return float(np.max(a*a+b*b+c*c))


def max_node_sq(direction, spacing):
    """E84 T81: max_node_sq_reference bit for bit (elementwise IEEE quotients/products/sums, exact max)."""
    v = np.ascontiguousarray(direction, dtype=np.float64).reshape(3, -1)
    hx, hy, hz = (float(h) for h in spacing)
    m = 0.
    for x in _chunks(lambda i0, i1: _max_node_sq(v, hx, hy, hz, i0, i1), v.shape[1]):
        if x > m or x != x:   # same NaN-propagating rule as within a chunk
            m = x
    return float(m)


def max_abs_diff(a, b):
    """E76: np.max(np.abs(a-b)) without temporaries; max/abs are exact, NaN propagates as in np.max."""
    a = np.ascontiguousarray(a).reshape(-1)
    b = np.ascontiguousarray(b).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("max_abs_diff: shapes differ")
    m = 0.
    for x in _chunks(lambda i0, i1: _max_abs_diff(a, b, i0, i1), a.shape[0]):
        if x > m or x != x:
            m = x
    return float(m)


def all_finite(a):
    a = np.ascontiguousarray(a).reshape(-1)
    return bool(all(_chunks(lambda i0, i1: _all_finite(a, i0, i1), a.shape[0])))
