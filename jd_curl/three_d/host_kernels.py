"""E89 (throughput v11): compiled host passes that reproduce their NumPy expressions bit for bit.

Every elementwise kernel performs, per element, exactly the IEEE operations of its NumPy reference in
the same order (fastmath off: no FMA contraction, no reassociation, no reduction of floating sums) and
writes a fresh owned C-contiguous array. Work is split over disjoint chunks (E54 run_chunks, serial
nogil kernels), so the thread count cannot change any value. The equality test has the == semantics of
np.array_equal (+0 == -0, NaN != NaN) and stops at the first difference. Kernels that can meet a
nonfinite value report it, and callers then replay the original NumPy expression, so every error path
(and its message) is the original one. Used only under the explicit throughput profile v11; callers keep
the NumPy code when Numba is unavailable or the array is small.
"""
import numpy as np

from .fast_checks import _NUMBA
from .thread_chunks import run_chunks

MIN_SIZE = 1 << 16          # below this the NumPy expression is used (tests may lower it to 0)
SHIFT_BUFFER = 8192         # E105: NumPy's default reduction buffer (np.getbufsize()); checked by shift_mean
SHIFT_NUMPY = ("2.0.",)      # E105 (review S1): NumPy versions whose reduction order was verified (2.0.2)


def enabled(size):
    return _NUMBA and size >= MIN_SIZE


def _flat_ok(*arrays):
    """C-contiguous float64 arrays of one size: the kernels read them as flat vectors."""
    return all(isinstance(a, np.ndarray) and a.dtype == np.float64 and a.flags.c_contiguous
               and a.size == arrays[0].size for a in arrays)


if _NUMBA:
    from numba import njit

    @njit(cache=False, fastmath=False, nogil=True, error_model="numpy")
    def _inject(theta, out, z0, z1):
        # out[z, y, x, c] = theta[c, z-1, y-1, x-1] on interior nodes, +0.0 on the six faces
        # (np.zeros then a strided assignment: pure copies, so signed zeros and payloads are kept).
        nz, ny, nx = out.shape[0], out.shape[1], out.shape[2]
        for z in range(z0, z1):
            for y in range(ny):
                for x in range(nx):
                    if z == 0 or z == nz-1 or y == 0 or y == ny-1 or x == 0 or x == nx-1:
                        out[z, y, x, 0] = 0.
                        out[z, y, x, 1] = 0.
                        out[z, y, x, 2] = 0.
                    else:
                        out[z, y, x, 0] = theta[0, z-1, y-1, x-1]
                        out[z, y, x, 1] = theta[1, z-1, y-1, x-1]
                        out[z, y, x, 2] = theta[2, z-1, y-1, x-1]
        return 0

    @njit(cache=False, fastmath=False, nogil=True, error_model="numpy")
    def _equal(a, b):
        for i in range(a.shape[0]):
            if not a[i] == b[i]:
                return False
        return True

    @njit(cache=False, fastmath=False, nogil=True, error_model="numpy")
    def _scaled(s, x, out, i0, i1):
        ok = True
        for i in range(i0, i1):
            v = s[i]*x[i]
            out[i] = v
            if not np.isfinite(v):
                ok = False
        return ok

    @njit(cache=False, fastmath=False, nogil=True, error_model="numpy")
    def _neg_scaled(s, x, out, i0, i1):
        ok = True
        for i in range(i0, i1):
            v = -(s[i]*x[i])
            out[i] = v
            if not np.isfinite(v):
                ok = False
        return ok

    @njit(cache=False, fastmath=False, nogil=True, error_model="numpy")
    def _negated(x, out, i0, i1):
        ok = True
        for i in range(i0, i1):
            v = -x[i]
            out[i] = v
            if not np.isfinite(v):
                ok = False
        return ok

    @njit(cache=False, fastmath=False, nogil=True, error_model="numpy")
    def _axpy(theta, tau, d, out, i0, i1):
        # theta + tau*d as NumPy forms it: the product is rounded, then the sum (two roundings).
        for i in range(i0, i1):
            t = tau*d[i]
            out[i] = theta[i]+t
        return 0

    @njit(cache=False, fastmath=False, nogil=True, error_model="numpy")
    def _affine_pass(node, floor, act, r, z0, z1):
        # E95: r = np.minimum(1., np.maximum(inner, floor)/act) on the interior nodes (C order), with the
        # statistics of affine_factors and of the T62 record: all inner finite, all r finite, min inner,
        # count(inner < act), min r. Per element the IEEE operations of the NumPy expression, in its order.
        nz, ny, nx = node.shape[0], node.shape[1], node.shape[2]
        iy, ix = ny-2, nx-2
        inner_ok, r_ok = True, True
        inner_min, r_min, below = np.inf, np.inf, 0
        for z in range(z0, z1):
            for y in range(1, ny-1):
                for x in range(1, nx-1):
                    v = node[z, y, x]
                    if not np.isfinite(v):
                        inner_ok = False
                    m = v if v > floor else floor
                    q = m/act
                    if not np.isfinite(q):
                        r_ok = False          # NumPy warns on this overflow: the caller replays the original
                    w = q if q < 1. else 1.
                    r[((z-1)*iy+(y-1))*ix+(x-1)] = w
                    if not np.isfinite(w):
                        r_ok = False
                    if v < inner_min:
                        inner_min = v
                    if w < r_min:
                        r_min = w
                    if v < act:
                        below += 1
        return inner_ok, r_ok, inner_min, r_min, below

    @njit(cache=False, fastmath=False, nogil=True, error_model="numpy")
    def _broadcast_product(scale, r, out, i0, i1):
        # E95: out[k] = scale[k]*r[k mod n] (scale*broadcast(r) raveled), with all finite, all > 0, min, max.
        n = r.shape[0]
        ok, positive = True, True
        lo, hi = np.inf, -np.inf
        for k in range(i0, i1):
            v = scale[k]*r[k % n]
            out[k] = v
            if not np.isfinite(v):
                ok = False
            if not v > 0:
                positive = False
            if v < lo:
                lo = v
            if v > hi:
                hi = v
        return ok, positive, lo, hi

    @njit(cache=False, fastmath=False, nogil=True, error_model="numpy")
    def _axpy_checked(theta, tau, d, out, i0, i1):
        # E95: _axpy's two roundings, and whether every result is finite.
        ok = True
        for i in range(i0, i1):
            t = tau*d[i]
            v = theta[i]+t
            out[i] = v
            if not np.isfinite(v):
                ok = False
        return ok

    @njit(cache=False, fastmath=False, nogil=True, error_model="numpy")
    def _axpy_extrema(theta, tau, d, out, i0, i1):
        # E97: _axpy_checked's two roundings and finite flag, plus max|out-theta| and max|theta| with the
        # fast_checks max rule (each difference rounded as in max_abs_diff(out, theta)).
        ok = True
        m1 = 0.
        m2 = 0.
        for i in range(i0, i1):
            b = theta[i]
            t = tau*d[i]
            v = b+t
            out[i] = v
            if not np.isfinite(v):
                ok = False
            x = abs(v-b)
            if x > m1 or x != x:
                m1 = x
            y = abs(b)
            if y > m2 or y != y:
                m2 = y
        return ok, m1, m2

    @njit(cache=False, fastmath=False, nogil=True, error_model="numpy")
    def _subtract(a, b, out, i0, i1):
        ok = True
        for i in range(i0, i1):
            v = a[i]-b[i]
            out[i] = v
            if not np.isfinite(v):
                ok = False
        return ok

    @njit(cache=False, fastmath=False, nogil=True, error_model="numpy")
    def _quotient_scale(root_m, diag, out, i0, i1):
        # np.sqrt(m)/np.sqrt(diag) with the scalar np.sqrt(m) given (IEEE sqrt and division).
        ok = True
        for i in range(i0, i1):
            v = root_m/np.sqrt(diag[i])
            out[i] = v
            if not np.isfinite(v):
                ok = False
        return ok

    @njit(cache=False, fastmath=False, nogil=True, error_model="numpy")
    def _product(a, b, out, i0, i1):
        ok = True
        for i in range(i0, i1):
            v = a[i]*b[i]
            out[i] = v
            if not np.isfinite(v):
                ok = False
        return ok

    @njit(cache=False, fastmath=False, nogil=True, error_model="numpy")
    def _pairwise(a, lo, n):
        # NumPy's pairwise_sum (loops_utils.h.src) of a[lo:lo+n]: below 8 in order, up to 128 with eight
        # accumulators combined ((r0+r1)+(r2+r3))+((r4+r5)+(r6+r7)) then the rest in order, else split at
        # n2 = n//2 rounded down to a multiple of 8. Type-inferred self-recursion (the n < 8 branch returns first),
        # so it compiles on first use (review E105 M5: no eager compile at import).
        if n < 8:
            res = -0.
            for i in range(n):
                res += a[lo+i]
            return res
        if n <= 128:
            r0, r1, r2, r3 = a[lo], a[lo+1], a[lo+2], a[lo+3]
            r4, r5, r6, r7 = a[lo+4], a[lo+5], a[lo+6], a[lo+7]
            m = n-n % 8
            i = 8
            while i < m:
                r0 += a[lo+i]
                r1 += a[lo+i+1]
                r2 += a[lo+i+2]
                r3 += a[lo+i+3]
                r4 += a[lo+i+4]
                r5 += a[lo+i+5]
                r6 += a[lo+i+6]
                r7 += a[lo+i+7]
                i += 8
            res = ((r0+r1)+(r2+r3))+((r4+r5)+(r6+r7))
            while i < n:
                res += a[lo+i]
                i += 1
            return res
        n2 = n//2
        n2 -= n2 % 8
        return _pairwise(a, lo, n2)+_pairwise(a, lo+n2, n-n2)

    @njit(cache=False, fastmath=False, nogil=True, error_model="numpy")
    def _shift_chunks(image, q, sums, c0, c1, size):
        # sums[c] = NumPy's pairwise sum of buffer c (size values, C order) of q*((b0*b0+b1*b1)+b2*b2) over the
        # interior nodes, b = image[1:-1, 1:-1, 1:-1] (np.sum over the last axis: ((0+a0)+a1)+a2, a_k >= +0).
        nz, ny, nx = q.shape
        n = nz*ny*nx
        plane = ny*nx
        buf = np.empty(size)
        ok = True
        for c in range(c0, c1):
            lo = c*size
            m = min(size, n-lo)
            z = lo//plane
            r = lo-z*plane
            y = r//nx
            x = r-y*nx
            for k in range(m):
                b0 = image[z+1, y+1, x+1, 0]
                b1 = image[z+1, y+1, x+1, 1]
                b2 = image[z+1, y+1, x+1, 2]
                buf[k] = q[z, y, x]*((b0*b0+b1*b1)+b2*b2)
                x += 1
                if x == nx:
                    x = 0
                    y += 1
                    if y == ny:
                        y = 0
                        z += 1
            s = _pairwise(buf, 0, m)
            sums[c] = s
            if not np.isfinite(s):
                ok = False
        return ok


def inject(theta, grid):
    """inject_free(theta, grid) for a finite C-contiguous float64 theta, into a fresh owned array."""
    out = np.empty(grid.shape+(3,))
    t = theta.reshape((3,)+grid.interior_shape)
    run_chunks(lambda z0, z1: _inject(t, out, z0, z1), grid.shape[0])
    return out


def equal(a, b):
    """np.array_equal(a, b) for C-contiguous float64 arrays (== semantics), stopping at the first difference."""
    if a.shape != b.shape:
        return False
    return bool(_equal(a.reshape(-1), b.reshape(-1)))


def _elementwise(kernel, out, *args):
    flags = run_chunks(lambda i0, i1: kernel(*args, out, i0, i1), out.shape[0])
    return all(flags)


def scaled(s, x):
    """(s*x, all finite) as fresh arrays; s and x flat C-contiguous float64 of one size."""
    out = np.empty(x.size)
    return out, _elementwise(_scaled, out, s.reshape(-1), x.reshape(-1))


def neg_scaled(s, x):
    """(-(s*x), all finite)."""
    out = np.empty(x.size)
    return out, _elementwise(_neg_scaled, out, s.reshape(-1), x.reshape(-1))


def negated(x):
    """(-x, all finite)."""
    out = np.empty(x.size)
    return out, _elementwise(_negated, out, x.reshape(-1))


def axpy(theta, tau, d):
    """theta + tau*d (float tau) into a fresh array; nonfinite values are kept, as NumPy keeps them."""
    out = np.empty(theta.size)
    run_chunks(lambda i0, i1: _axpy(theta.reshape(-1), float(tau), d.reshape(-1), out, i0, i1), out.shape[0])
    return out


def axpy_checked(theta, tau, d):
    """E95: (axpy(theta, tau, d), every value finite) -- the same values as axpy."""
    out = np.empty(theta.size)
    return out, _elementwise(_axpy_checked, out, theta.reshape(-1), float(tau), d.reshape(-1))


def axpy_extrema(theta, tau, d):
    """E97: (axpy_checked(theta, tau, d)[0], every value finite, max|out-theta|, max|theta|). The two maxima are
    fast_checks.max_abs_diff(out, theta) and fast_checks.max_abs(theta) bit for bit whenever both are finite (the
    maximum of exact absolute values does not depend on the chunking). A finite out does not make max|out-theta|
    finite (out-theta can overflow), so callers use the maxima only after checking that."""
    out = np.empty(theta.size)
    t, dd = theta.reshape(-1), d.reshape(-1)
    parts = run_chunks(lambda i0, i1: _axpy_extrema(t, float(tau), dd, out, i0, i1), out.shape[0])
    ok, m1, m2 = True, 0., 0.
    for flag, x, y in parts:
        ok = ok and bool(flag)
        if x > m1 or x != x:
            m1 = x
        if y > m2 or y != y:
            m2 = y
    return out, ok, float(m1), float(m2)


def affine_pass(node, floor, act):
    """E95: (r over the interior nodes (flat, C order), inner all finite, r all finite, min inner,
    min r, count inner < act) for a C-contiguous float64 node array of the grid shape."""
    nz, ny, nx = node.shape
    r = np.empty((nz-2)*(ny-2)*(nx-2))
    parts = run_chunks(lambda z0, z1: _affine_pass(node, float(floor), float(act), r, z0+1, z1+1), nz-2)
    inner_ok = all(p[0] for p in parts)
    r_ok = all(p[1] for p in parts)
    return r, inner_ok, r_ok, min(p[2] for p in parts), min(p[3] for p in parts), sum(p[4] for p in parts)


def broadcast_product(scale, r):
    """E95: (scale*broadcast_to(r, (3,)+...).ravel(), all finite, all > 0, min, max); scale flat of size 3*r.size."""
    out = np.empty(scale.size)
    parts = run_chunks(lambda i0, i1: _broadcast_product(scale.reshape(-1), r, out, i0, i1), out.shape[0])
    return (out, all(p[0] for p in parts), all(p[1] for p in parts),
            min(p[2] for p in parts), max(p[3] for p in parts))


def subtract_into(a, b, out):
    """np.subtract(a, b, out=out) for flat arrays; returns whether every difference is finite."""
    return _elementwise(_subtract, out, a.reshape(-1), b.reshape(-1))


def quotient_scale(root_m, diag):
    """(root_m/np.sqrt(diag), all finite) with the scalar root_m = np.sqrt(m)."""
    out = np.empty(diag.size)
    return out, _elementwise(_quotient_scale, out, float(root_m), diag.reshape(-1))


def product(a, b):
    """(a*b, all finite) for flat arrays of one size."""
    out = np.empty(a.size)
    return out, _elementwise(_product, out, a.reshape(-1), b.reshape(-1))


def shift_mean(image, q):
    """E105: float(np.mean(q*np.sum(b*b, axis=-1))) for b = image[1:-1, 1:-1, 1:-1], bit for bit, or None.

    NumPy 2.0 evaluates the mean of the C-contiguous product as its reduction iterator does: buffers of
    np.getbufsize() = 8192 values in C order, each summed by pairwise_sum, the buffer sums added in order to the
    accumulator (from 0; every value here is >= +0, so the start's sign cannot show). The buffers are
    independent, so they are computed in threads (run_chunks) and added in order afterwards. None (the caller
    then evaluates the NumPy expression, which raises as before) when the inputs are not the expected layout,
    NumPy's buffer size is not 8192, NumPy is not a verified version (SHIFT_NUMPY: the order is a property of its
    reduction iterator, not of its API), or any buffer sum or the total is not finite (an overflow anywhere makes them
    infinite: every term is >= 0)."""
    import math
    if not (isinstance(image, np.ndarray) and isinstance(q, np.ndarray) and image.dtype == np.float64
            and q.dtype == np.float64 and q.ndim == 3 and image.shape == tuple(s+2 for s in q.shape)+(3,)
            and image.flags.c_contiguous and q.flags.c_contiguous and q.size > 0 and np.getbufsize() == SHIFT_BUFFER
            and np.__version__.startswith(SHIFT_NUMPY)):
        return None
    size = SHIFT_BUFFER
    chunks = -(-q.size//size)
    sums = np.empty(chunks)
    if not all(run_chunks(lambda c0, c1: _shift_chunks(image, q, sums, c0, c1, size), chunks)):
        return None
    total = 0.
    for s in sums.tolist():
        total += s
    if not math.isfinite(total):
        return None
    return total/q.size


def prewarm(lean_checks=False, step_extrema=False, fused_shift=False):
    """Compile every kernel for each read-only/writeable combination of its inputs (outside a task clock).
    lean_checks: also the E95 kernels (axpy_checked, broadcast_product, affine_pass), used by throughput-v14 on.
    step_extrema: also the E97 axpy_extrema kernel (throughput-v15).
    fused_shift: also the E105 shift_mean kernel (throughput-v23)."""
    if not _NUMBA:
        return
    from itertools import product as combinations
    from .grid import make_grid_3d
    g = make_grid_3d(nx=5, ny=5, nz=5, hx=1., hy=1., hz=1.)
    n = 3*g.n_free_scalar

    def inputs(*frozen):
        arrays = [np.full(n, 1.+k) for k in range(len(frozen))]
        for a, f in zip(arrays, frozen):
            a.flags.writeable = not f
        return arrays
    for f in (False, True):
        inject(*inputs(f), g)
        negated(*inputs(f))
        quotient_scale(2., *inputs(f))
    for f in combinations((False, True), repeat=2):
        a, b = inputs(*f)
        equal(a, b); scaled(a, b); neg_scaled(a, b); axpy(a, .5, b); product(a, b)
        subtract_into(a, b, np.empty(n))
        if lean_checks:
            axpy_checked(a, .5, b)
            broadcast_product(a, b[:n//3])
        if step_extrema:
            axpy_extrema(a, .5, b)
    if lean_checks:
        for f in (False, True):
            node = np.full(g.shape, .5)
            node.flags.writeable = not f
            affine_pass(node, 1e-12, .25)
    if fused_shift:
        for f in combinations((False, True), repeat=2):
            image, q = np.full(g.shape+(3,), .5), np.full(g.interior_shape, 2.)
            image.flags.writeable, q.flags.writeable = not f[0], not f[1]
            shift_mean(image, q)
