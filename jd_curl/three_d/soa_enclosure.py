"""E98 (G7: D3 + D5): the hoisted edge-hull margins kernel, row by row in structure-of-arrays form.

compiled_enclosure._chunk_hoisted_margins evaluates, per cell, 9 interval entries F[r, j] (edge-difference hull,
outward, /h, outward, + identity, outward) and the left-associated cofactor expansion of det F (outward after
every operation), then outward(J - delta); each outward step calls libm nextafter (84 calls per cell). This module
evaluates the same IEEE operation sequence per cell, with
  - nextafter replaced by an exact integer bit step (_up/_dn): identical bits to glibc nextafter(x, +-inf) for every
    input, NaN included (quiet bit set, as libm's x+y), with integer class tests (independent of DAZ/FTZ);
  - the cells of one (z, y) row processed together from a component-major copy of u, so that LLVM vectorizes the
    per-cell arithmetic (no fastmath, no FMA contraction: fastmath=False and separate IEEE operations).
For finite u every margin and flag is bit-identical to the libm kernel (the callers pass finite u; with several
NaN payloads in u, the payload of a NaN margin in an already flagged cell may differ, since the compiler may order two
NaN operands differently -- flags never differ). parallel_margins in compiled_enclosure compares both kernels until
a grid is verified. Cells are independent, so the row chunking cannot change any value.
"""
import numpy as np
from numba import njit, types
from numba.extending import intrinsic

from .thread_chunks import run_chunks

SOA_KERNEL = "numba-row-soa-bitstep-hoisted-edge-hull-v1"

_ABS = 0x7FFFFFFFFFFFFFFF            # magnitude bits
_SIGN = -0x7FFFFFFFFFFFFFFF-1         # 0x8000000000000000 (sign bit) as int64
_EXP = 0x7FF0000000000000             # +inf bits; magnitude > _EXP is NaN
_QUIET = 0x0008000000000000           # quiet-NaN bit (x86 sets it on x+y with a NaN operand)
_NEG_INF = _SIGN | _EXP
_NEG_TINY = _SIGN | 1                 # -min subnormal


@intrinsic
def _as_bits(typingctx, x):
    def codegen(context, builder, sig, args):
        return builder.bitcast(args[0], context.get_value_type(types.int64))
    return types.int64(types.float64), codegen


@intrinsic
def _from_bits(typingctx, b):
    def codegen(context, builder, sig, args):
        return builder.bitcast(args[0], context.get_value_type(types.float64))
    return types.float64(types.int64), codegen


@njit(cache=False, fastmath=False, inline='always', error_model='numpy')
def _up(x):
    """nextafter(x, +inf), bit for bit (glibc): NaN -> x with the quiet bit; +inf -> +inf; +-0 -> +min subnormal;
    otherwise one step of the magnitude away from (x >= 0) or towards (x < 0) zero. No int64 overflow: the
    magnitude a <= _ABS, a+1 only matters for a <= _EXP-1, a-1 only for a >= 1 (the rest is replaced)."""
    b = _as_bits(x)
    a = b & _ABS
    s = b & _SIGN
    m = a+1 if s == 0 else a-1
    r = s | (m & _ABS)
    r = 1 if a == 0 else r
    r = b if b == _EXP else r
    r = (b | _QUIET) if a > _EXP else r
    return _from_bits(r)


@njit(cache=False, fastmath=False, inline='always', error_model='numpy')
def _dn(x):
    """nextafter(x, -inf), bit for bit (glibc); the mirror image of _up."""
    b = _as_bits(x)
    a = b & _ABS
    s = b & _SIGN
    m = a-1 if s == 0 else a+1
    r = s | (m & _ABS)
    r = _NEG_TINY if a == 0 else r
    r = b if b == _NEG_INF else r
    r = (b | _QUIET) if a > _EXP else r
    return _from_bits(r)


@njit(cache=False, fastmath=False, inline='always', error_model='numpy')
def _out(lo, hi):
    """compiled_enclosure._nr_outward with the bit step."""
    lo, hi = _dn(lo), _up(hi)
    return lo, hi, np.isfinite(lo) & np.isfinite(hi) & (lo <= hi)


@njit(cache=False, fastmath=False, inline='always', error_model='numpy')
def _mul(alo, ahi, blo, bhi):
    """compiled_enclosure._nr_mul (the same four products and min/max chains)."""
    x, y, z, w = alo*blo, alo*bhi, ahi*blo, ahi*bhi
    return _out(np.minimum(np.minimum(np.minimum(x, y), z), w),
                np.maximum(np.maximum(np.maximum(x, y), z), w))


@njit(cache=False, fastmath=False, inline='always', error_model='numpy')
def _hull4(d0, d1, d2, d3, h, ident):
    """compiled_enclosure._nr_entry after its edge loop: hull from (+inf, -inf) in edge order, outward, /h,
    outward, + identity, outward; valid iff every difference and every step is."""
    good = np.isfinite(d0) & np.isfinite(d1) & np.isfinite(d2) & np.isfinite(d3)
    lower = np.minimum(np.minimum(np.minimum(np.minimum(np.inf, d0), d1), d2), d3)
    upper = np.maximum(np.maximum(np.maximum(np.maximum(-np.inf, d0), d1), d2), d3)
    lo, hi, g1 = _out(lower, upper)
    lo, hi, g2 = _out(lo/h, hi/h)
    lo, hi, g3 = _out(lo+ident, hi+ident)
    return lo, hi, good & g1 & g2 & g3


@njit(cache=False, fastmath=False, nogil=True, error_model='numpy', boundscheck=False)
def _row_entries(c00, c01, c10, c11, n, hx, hy, hz, r, FL, FH, OK):
    """Entries F[r, 0..2] of the n cells of one row. c{zy} is component r of node row (z0+z, y0+y). Edge order as
    _nr_entry's loop over the vertices a = 4z+2y+x with bit j clear, in increasing a."""
    i0 = 1. if r == 0 else 0.
    i1 = 1. if r == 1 else 0.
    i2 = 1. if r == 2 else 0.
    for x in range(n):
        # j = 0: a = 0, 2, 4, 6
        lo, hi, g = _hull4(c00[x+1]-c00[x], c01[x+1]-c01[x], c10[x+1]-c10[x], c11[x+1]-c11[x], hx, i0)
        FL[3*r, x] = lo; FH[3*r, x] = hi; OK[3*r, x] = g
        # j = 1: a = 0, 1, 4, 5
        lo, hi, g = _hull4(c01[x]-c00[x], c01[x+1]-c00[x+1], c11[x]-c10[x], c11[x+1]-c10[x+1], hy, i1)
        FL[3*r+1, x] = lo; FH[3*r+1, x] = hi; OK[3*r+1, x] = g
        # j = 2: a = 0, 1, 2, 3
        lo, hi, g = _hull4(c10[x]-c00[x], c10[x+1]-c00[x+1], c11[x]-c01[x], c11[x+1]-c01[x+1], hz, i2)
        FL[3*r+2, x] = lo; FH[3*r+2, x] = hi; OK[3*r+2, x] = g


@njit(cache=False, fastmath=False, nogil=True, error_model='numpy', boundscheck=False)
def _row_det(FL, FH, OK, n, delta, margin, flag, k0):
    """_chunk_hoisted_margins' cofactor expansion and - delta for the n cells of one row (FL[3*r+j] = F[r, j])."""
    for x in range(n):
        f00l = FL[0, x]; f00h = FH[0, x]; f01l = FL[1, x]; f01h = FH[1, x]; f02l = FL[2, x]; f02h = FH[2, x]
        f10l = FL[3, x]; f10h = FH[3, x]; f11l = FL[4, x]; f11h = FH[4, x]; f12l = FL[5, x]; f12h = FH[5, x]
        f20l = FL[6, x]; f20h = FH[6, x]; f21l = FL[7, x]; f21h = FH[7, x]; f22l = FL[8, x]; f22h = FH[8, x]
        e = OK[0, x] & OK[1, x] & OK[2, x] & OK[3, x] & OK[4, x] & OK[5, x] & OK[6, x] & OK[7, x] & OK[8, x]
        p1l, p1h, g1 = _mul(f11l, f11h, f22l, f22h)
        p2l, p2h, g2 = _mul(f12l, f12h, f21l, f21h)
        al, ah, g3 = _out(p1l-p2h, p1h-p2l)
        p1l, p1h, g4 = _mul(f12l, f12h, f20l, f20h)
        p2l, p2h, g5 = _mul(f10l, f10h, f22l, f22h)
        bl, bh, g6 = _out(p1l-p2h, p1h-p2l)
        p1l, p1h, g7 = _mul(f10l, f10h, f21l, f21h)
        p2l, p2h, g8 = _mul(f11l, f11h, f20l, f20h)
        ccl, cch, g9 = _out(p1l-p2h, p1h-p2l)
        t1l, t1h, g10 = _mul(f00l, f00h, al, ah)
        t2l, t2h, g11 = _mul(f01l, f01h, bl, bh)
        sl, sh, g12 = _out(t1l+t2l, t1h+t2h)
        t3l, t3h, g13 = _mul(f02l, f02h, ccl, cch)
        jl, jh, g14 = _out(sl+t3l, sh+t3h)
        ml, mh, g15 = _out(jl-delta, jh-delta)
        ok = e & g1 & g2 & g3 & g4 & g5 & g6 & g7 & g8 & g9 & g10 & g11 & g12 & g13 & g14 & g15
        margin[k0+x] = ml
        flag[k0+x] = np.uint8(0) if ok else np.uint8(1)


@njit(cache=False, fastmath=False, nogil=True, error_model='numpy', boundscheck=False)
def _soa_rows(U, cy, cx, hx, hy, hz, delta, row0, row1, margin, flag):
    """Rows [row0, row1) of cells (row = z0*cy + y0); U is the (3, nz, ny, nx) component-major copy of u."""
    FL = np.empty((9, cx)); FH = np.empty((9, cx)); OK = np.empty((9, cx), np.bool_)
    for row in range(row0, row1):
        y0 = row % cy
        z0 = row//cy
        for r in range(3):
            _row_entries(U[r, z0, y0], U[r, z0, y0+1], U[r, z0+1, y0], U[r, z0+1, y0+1], cx, hx, hy, hz, r, FL, FH, OK)
        _row_det(FL, FH, OK, cx, delta, margin, flag, row*cx)


@njit(cache=False, fastmath=False, nogil=True, error_model='numpy', boundscheck=False)
def _transpose(flat, Uf, n0, n1):
    """Uf[c, i] = flat[i, c] (pure copies: payloads and signed zeros are kept)."""
    for i in range(n0, n1):
        Uf[0, i] = flat[i, 0]
        Uf[1, i] = flat[i, 1]
        Uf[2, i] = flat[i, 2]


def soa_margins(u, grid, delta):
    """(margin, flag as bool) of every cell, bit for bit compiled_enclosure's libm kernel for finite u (the caller
    has run arithmetic_supported, as parallel_margins does)."""
    nz, ny, nx = grid.shape
    flat = np.ascontiguousarray(u, dtype=np.float64).reshape(-1, 3)
    nodes = nz*ny*nx
    U = np.empty((3, nz, ny, nx))
    Uf = U.reshape(3, nodes)
    run_chunks(lambda lo, hi: _transpose(flat, Uf, lo, hi), nodes)
    cz, cy, cx = nz-1, ny-1, nx-1
    margin = np.empty(grid.n_cells); flag = np.empty(grid.n_cells, dtype=np.uint8)
    hx, hy, hz = (float(h) for h in grid.spacing)
    run_chunks(lambda lo, hi: _soa_rows(U, cy, cx, hx, hy, hz, float(delta), lo, hi, margin, flag), cz*cy)
    return margin, flag.astype(bool)


def prewarm():
    """Compile the kernels for writeable and read-only inputs (outside a task clock)."""
    from .grid import make_grid_3d
    g = make_grid_3d(nx=5, ny=5, nz=5, hx=1., hy=1., hz=1.)
    for frozen in (False, True):
        u = np.zeros(g.shape+(3,))
        u.flags.writeable = not frozen
        soa_margins(u, g, 0.)
