"""E54 (T65): compiled Bernstein-fallback arrays, bitwise equal to feasibility._fallback_arrays.

Per cell and in the same order as the NumPy interval code:
  jacobian_intervals: point-interval edge differences (outward), outward division by h,
  outward nonnegative scaling by the exact dyadic Q1 weight, outward accumulation from a
  point zero, outward identity add (also for +0), left-associated cofactor determinant;
  then the caller's outward "- delta";
  lagrange_to_bernstein: axes x, y, z in turn, mid = 2*b - (a+c)*.5, all outward;
  _sample_jacobians: float64 F = I + sum (diff/h)*w, cofactor, det = ((0+f0a0)+f1a1)+f2a2
  (NumPy's small-n pairwise sum order).
Failure semantics: the NumPy path raises (-> None for the whole job) when any interval is
nonfinite or inverted, or when the float64 samples overflow/produce NaN. Here every such
event clears one flag for the job. No parallel region (no prange/parfors): callers run jobs
in threads; the kernel releases the GIL.
"""
from itertools import product

import numpy as np

from ..validation import InputValidationError, as_float64_array
from .cell_q1 import VERTICES
from .interval import Interval

try:
    from numba import njit, config as numba_config
except ImportError as exc:  # pragma: no cover
    raise InputValidationError("compiled fallback requires the optional numba dependency") from exc

if numba_config.DISABLE_JIT:
    raise InputValidationError("compiled fallback requires Numba JIT; NUMBA_DISABLE_JIT is set")

from .compiled_enclosure import _nr_add, _nr_mul, _nr_outward, _nr_sub

FALLBACK_KERNEL = "numba-bernstein-fallback-v1"
_POINTS = np.array([(x, y, z) for z, y, x in product((0., .5, 1.), repeat=3)])


def _edge_tables():
    """(3,4,2) vertex pairs and (3,4,27) weights, in jacobian_intervals' loop order."""
    pairs = np.empty((3, 4, 2), dtype=np.int64)
    weights = np.empty((3, 4, 27))
    for j in range(3):
        e = 0
        for a, bits in enumerate(VERTICES):
            if bits[j]:
                continue
            other = list(bits)
            other[j] = 1
            b = VERTICES.index(tuple(other))
            weight = np.ones(27)
            for k in range(3):
                if k != j:
                    weight *= _POINTS[:, k] if bits[k] else 1-_POINTS[:, k]
            pairs[j, e] = (a, b)
            weights[j, e] = weight
            e += 1
    return pairs, weights


_PAIRS, _WEIGHTS = _edge_tables()


@njit(cache=False, fastmath=False, nogil=True, error_model='numpy')
def _fallback_kernel(v, spacing, delta, pairs, weights, plo, phi, clo, chi, samples):
    """Returns True iff every interval/sample of every cell is finite and ordered."""
    n = v.shape[0]
    ok = True
    flo = np.empty((27, 3, 3))
    fhi = np.empty((27, 3, 3))
    elo = np.empty((4, 3))
    ehi = np.empty((4, 3))
    blo = np.empty(27)
    bhi = np.empty(27)
    fs = np.empty((27, 3, 3))
    for cell in range(n):
        # ---- jacobian_intervals: F[p, r, j]
        for j in range(3):
            h = spacing[j]
            for e in range(4):
                a, b = pairs[j, e, 0], pairs[j, e, 1]
                for r in range(3):
                    d = v[cell, b, r]-v[cell, a, r]
                    slo, shi, g1 = _nr_outward(d, d)                      # point - point
                    x1, x2 = slo/h, shi/h                                  # interval / point h
                    qlo, qhi, g2 = _nr_outward(np.minimum(np.minimum(np.minimum(x1, x1), x2), x2),
                                               np.maximum(np.maximum(np.maximum(x1, x1), x2), x2))
                    elo[e, r], ehi[e, r] = qlo, qhi
                    ok = ok and g1 and g2
            for p in range(27):
                for r in range(3):
                    dlo, dhi = 0., 0.
                    for e in range(4):
                        w = weights[j, e, p]
                        wlo, whi, g3 = _nr_outward(elo[e, r]*w, ehi[e, r]*w)
                        dlo, dhi, g4 = _nr_add(dlo, dhi, wlo, whi)
                        ok = ok and g3 and g4
                    c = 1. if r == j else 0.
                    flo[p, r, j], fhi[p, r, j], g5 = _nr_add(dlo, dhi, c, c)
                    ok = ok and g5
        for p in range(27):
            p1l, p1h, g1 = _nr_mul(flo[p, 1, 1], fhi[p, 1, 1], flo[p, 2, 2], fhi[p, 2, 2])
            p2l, p2h, g2 = _nr_mul(flo[p, 1, 2], fhi[p, 1, 2], flo[p, 2, 1], fhi[p, 2, 1])
            al, ah, g3 = _nr_sub(p1l, p1h, p2l, p2h)
            p1l, p1h, g4 = _nr_mul(flo[p, 1, 2], fhi[p, 1, 2], flo[p, 2, 0], fhi[p, 2, 0])
            p2l, p2h, g5 = _nr_mul(flo[p, 1, 0], fhi[p, 1, 0], flo[p, 2, 2], fhi[p, 2, 2])
            bl, bh, g6 = _nr_sub(p1l, p1h, p2l, p2h)
            p1l, p1h, g7 = _nr_mul(flo[p, 1, 0], fhi[p, 1, 0], flo[p, 2, 1], fhi[p, 2, 1])
            p2l, p2h, g8 = _nr_mul(flo[p, 1, 1], fhi[p, 1, 1], flo[p, 2, 0], fhi[p, 2, 0])
            ccl, cch, g9 = _nr_sub(p1l, p1h, p2l, p2h)
            t1l, t1h, g10 = _nr_mul(flo[p, 0, 0], fhi[p, 0, 0], al, ah)
            t2l, t2h, g11 = _nr_mul(flo[p, 0, 1], fhi[p, 0, 1], bl, bh)
            sl, sh, g12 = _nr_add(t1l, t1h, t2l, t2h)
            t3l, t3h, g13 = _nr_mul(flo[p, 0, 2], fhi[p, 0, 2], ccl, cch)
            jl, jh, g14 = _nr_add(sl, sh, t3l, t3h)
            ml, mh, g15 = _nr_sub(jl, jh, delta, delta)                      # "- delta"
            plo[cell, p], phi[cell, p] = ml, mh
            blo[p], bhi[p] = ml, mh
            ok = (ok and g1 and g2 and g3 and g4 and g5 and g6 and g7 and g8 and g9 and g10
                  and g11 and g12 and g13 and g14 and g15)
        # ---- lagrange_to_bernstein, storage index 9*iz+3*iy+ix; axes x, then y, then z
        for stride, outer1, s1, outer2, s2 in ((1, 3, 9, 3, 3), (3, 3, 9, 3, 1), (9, 3, 3, 3, 1)):
            for i1 in range(outer1):
                for i2 in range(outer2):
                    i0 = i1*s1+i2*s2
                    a0, a1, a2 = i0, i0+stride, i0+2*stride
                    ml1, mh1, g1 = _nr_mul(blo[a1], bhi[a1], 2., 2.)
                    sl1, sh1, g2 = _nr_add(blo[a0], bhi[a0], blo[a2], bhi[a2])
                    hl1, hh1, g3 = _nr_mul(sl1, sh1, .5, .5)
                    nl, nh, g4 = _nr_sub(ml1, mh1, hl1, hh1)
                    blo[a1], bhi[a1] = nl, nh
                    ok = ok and g1 and g2 and g3 and g4
        for p in range(27):
            clo[cell, p], chi[cell, p] = blo[p], bhi[p]
        # ---- _sample_jacobians (float64 diagnostic)
        for p in range(27):
            for r in range(3):
                for j in range(3):
                    fs[p, r, j] = 1. if r == j else 0.
        for j in range(3):
            h = spacing[j]
            for e in range(4):
                a, b = pairs[j, e, 0], pairs[j, e, 1]
                for p in range(27):
                    w = weights[j, e, p]
                    for r in range(3):
                        fs[p, r, j] = fs[p, r, j]+((v[cell, b, r]-v[cell, a, r])/h)*w
        for p in range(27):
            finite = True
            for r in range(3):
                for j in range(3):
                    finite = finite and np.isfinite(fs[p, r, j])
            # all nine cofactor entries (the NumPy path checks every one)
            a00 = fs[p, 1, 1]*fs[p, 2, 2]-fs[p, 1, 2]*fs[p, 2, 1]
            a01 = fs[p, 1, 2]*fs[p, 2, 0]-fs[p, 1, 0]*fs[p, 2, 2]
            a02 = fs[p, 1, 0]*fs[p, 2, 1]-fs[p, 1, 1]*fs[p, 2, 0]
            a10 = fs[p, 2, 1]*fs[p, 0, 2]-fs[p, 2, 2]*fs[p, 0, 1]
            a11 = fs[p, 2, 2]*fs[p, 0, 0]-fs[p, 2, 0]*fs[p, 0, 2]
            a12 = fs[p, 2, 0]*fs[p, 0, 1]-fs[p, 2, 1]*fs[p, 0, 0]
            a20 = fs[p, 0, 1]*fs[p, 1, 2]-fs[p, 0, 2]*fs[p, 1, 1]
            a21 = fs[p, 0, 2]*fs[p, 1, 0]-fs[p, 0, 0]*fs[p, 1, 2]
            a22 = fs[p, 0, 0]*fs[p, 1, 1]-fs[p, 0, 1]*fs[p, 1, 0]
            for x in (a00, a01, a02, a10, a11, a12, a20, a21, a22):
                finite = finite and np.isfinite(x)
            t0 = fs[p, 0, 0]*a00
            t1 = fs[p, 0, 1]*a01
            t2 = fs[p, 0, 2]*a02
            s = 0.
            s += t0
            s += t1
            s += t2
            finite = finite and np.isfinite(t0) and np.isfinite(t1) and np.isfinite(t2) and np.isfinite(s)
            samples[cell, p] = s
            ok = ok and finite
    return ok


def fallback_arrays(vertices, spacing, delta, errors=None):
    """Drop-in for feasibility._fallback_arrays: (p, coefficients, samples) or None.

    The NumPy sample path honours the caller's underflow setting (errors['under']) and
    computes in the input dtype; outside the production case (float64 input, underflow
    ignored) this function simply delegates to the NumPy implementation.
    """
    from .feasibility import _spacing, _fallback_arrays
    raw = np.asarray(vertices)
    if (errors is not None and errors.get("under", "ignore") != "ignore") or raw.dtype != np.float64:
        return _fallback_arrays(vertices, spacing, delta, np.geterr() if errors is None else errors)
    v = as_float64_array(vertices, "vertices")
    if v.ndim != 3 or v.shape[1:] != (8, 3):
        raise InputValidationError("vertices: expected (cells,8,3)")
    h = np.asarray(_spacing(spacing), dtype=np.float64)
    n = len(v)
    plo, phi, clo, chi, samples = (np.empty((n, 27)) for _ in range(5))
    ok = _fallback_kernel(np.ascontiguousarray(v), h, float(delta), _PAIRS, _WEIGHTS,
                          plo, phi, clo, chi, samples)
    if not ok:
        return None
    shape = (n, 3, 3, 3)
    return (Interval(plo.reshape(shape), phi.reshape(shape)), Interval(clo.reshape(shape), chi.reshape(shape)),
            samples.reshape(shape))
