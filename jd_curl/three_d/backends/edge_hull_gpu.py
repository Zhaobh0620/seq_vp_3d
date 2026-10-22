"""T84 (E101, throughput-v19): the hoisted edge-hull margins of the JD certificate on the GPU, bit for bit.

compiled_enclosure._chunk_hoisted_margins computes, per cell, the margin lo(J - delta) and an invalid-step flag from
the 9 interval entries F[r, j] (edge-difference hull, outward, /h_j, outward, + identity, outward) and the
left-associated cofactor expansion (outward after every operation). The kernel below performs the same IEEE
operations in the same order for one cell per lane:
  - outward steps are nextafter(lo, -inf) / nextafter(hi, +inf) as integer bit steps (soa_enclosure._up/_dn: glibc's
    results for every input, NaN given the quiet bit);
  - min/max are Numba's scalar np.minimum/np.maximum: the first operand when a <= b (a >= b) or a is NaN, else the
    second -- so equal values (+-0 included) give the first operand;
  - no FP contraction (enable_fp_fusion=False; the PTX is checked for fma), IEEE division (div.rn.f64);
  - h_x, h_y, h_z and delta are read from a float64 device tensor (Triton 3.1 would make Python floats fp32).
For finite u every flag and every margin of an unflagged cell is bit-identical to the CPU kernels (a flagged cell's
margin may differ only in a NaN payload; such tiles are never pre-proved). Only the margins move: the certificate's
prescreen, fallback, decisions and record stay compiled_enclosure/feasibility on the CPU.

Contract: docs/04 T84. Execution guards (GpuEdgeHull.margins): the input is a fresh device copy of the frozen host u
whose bytes the certificate hashes; buffers are pre-filled with margin = NaN and flag = 1 (an unwritten cell is
merely unproven); the first use per (grid, delta) is checked bit for bit against the CPU kernel on a deterministic
synthetic folding field and on the run's own informative input (the CPU result is returned until then); any
mismatch or failure switches the GPU part off for the run (the CPU kernel then runs, giving the same result).
throughput-v38 (gpu_hull_first_use_seed): the synthetic-field half of that first use may come from the F2' dry pass of the
same source, device and options (first_use_registry, cached by FC1) and is then seeded into _synthetic; the comparison on
the run's own input is never seeded and still happens in every run.
"""
import threading

import numpy as np
import torch

POLICY = "triton-edge-hull-margins-v1"
BLOCK, WARPS, MAXNREG = 128, 4, 255
_ABS, _EXPO, _QUIET = 0x7FFFFFFFFFFFFFFF, 0x7FF0000000000000, 0x0008000000000000

try:
    import triton
    import triton.language as tl
except ImportError:          # pragma: no cover
    triton = tl = None


def supported(device):
    return triton is not None and torch.device(device).type == "cuda"


if triton is not None:
    @triton.jit
    def _bits(x):
        return x.to(tl.int64, bitcast=True)

    @triton.jit
    def _real(b):
        return b.to(tl.float64, bitcast=True)

    @triton.jit
    def _finite(x):
        return (_bits(x) & 0x7FF0000000000000) != 0x7FF0000000000000

    @triton.jit
    def _up(x):
        b = _bits(x)
        a = b & 0x7FFFFFFFFFFFFFFF
        s = b & (-0x7FFFFFFFFFFFFFFF-1)
        r = s | (tl.where(s == 0, a+1, a-1) & 0x7FFFFFFFFFFFFFFF)
        r = tl.where(a == 0, 1, r)
        r = tl.where(b == 0x7FF0000000000000, b, r)
        r = tl.where(a > 0x7FF0000000000000, b | 0x0008000000000000, r)
        return _real(r)

    @triton.jit
    def _dn(x):
        b = _bits(x)
        a = b & 0x7FFFFFFFFFFFFFFF
        s = b & (-0x7FFFFFFFFFFFFFFF-1)
        r = s | (tl.where(s == 0, a-1, a+1) & 0x7FFFFFFFFFFFFFFF)
        r = tl.where(a == 0, (-0x7FFFFFFFFFFFFFFF-1) | 1, r)
        r = tl.where(b == ((-0x7FFFFFFFFFFFFFFF-1) | 0x7FF0000000000000), b, r)
        r = tl.where(a > 0x7FF0000000000000, b | 0x0008000000000000, r)
        return _real(r)

    @triton.jit
    def _min(a, b):
        return tl.where((a <= b) | (a != a), a, b)

    @triton.jit
    def _max(a, b):
        return tl.where((a >= b) | (a != a), a, b)

    @triton.jit
    def _out(lo, hi):
        lo2 = _dn(lo)
        hi2 = _up(hi)
        return lo2, hi2, _finite(lo2) & _finite(hi2) & (lo2 <= hi2)

    @triton.jit
    def _mul(alo, ahi, blo, bhi):
        x = alo*blo
        y = alo*bhi
        z = ahi*blo
        w = ahi*bhi
        return _out(_min(_min(_min(x, y), z), w), _max(_max(_max(x, y), z), w))

    @triton.jit
    def _sub(alo, ahi, blo, bhi):
        return _out(alo-bhi, ahi-blo)

    @triton.jit
    def _add(alo, ahi, blo, bhi):
        return _out(alo+blo, ahi+bhi)

    @triton.jit
    def _entry(d0, d1, d2, d3, h, ident):
        """_nr_entry after its edge loop: hull from (+inf, -inf) in edge order, outward, /h, outward, + identity."""
        ok = _finite(d0) & _finite(d1) & _finite(d2) & _finite(d3)
        inf = _real(tl.full(d0.shape, 0x7FF0000000000000, tl.int64))
        lower = _min(_min(_min(_min(inf, d0), d1), d2), d3)
        upper = _max(_max(_max(_max(-inf, d0), d1), d2), d3)
        lo, hi, g1 = _out(lower, upper)
        lo, hi, g2 = _out(lo/h, hi/h)
        lo, hi, g3 = _out(lo+ident, hi+ident)
        return lo, hi, ok & g1 & g2 & g3

    @triton.jit
    def _edges(U, n000, n001, n010, n011, n100, n101, n110, n111, r, m, j: tl.constexpr):
        """The four edge differences of component r along axis j (x: bit 0, y: bit 1, z: bit 2), vertices with bit j
        clear in increasing order: j=0 -> a = 0, 2, 4, 6; j=1 -> 0, 1, 4, 5; j=2 -> 0, 1, 2, 3."""
        if j == 0:
            d0 = tl.load(U+3*n001+r, mask=m)-tl.load(U+3*n000+r, mask=m)
            d1 = tl.load(U+3*n011+r, mask=m)-tl.load(U+3*n010+r, mask=m)
            d2 = tl.load(U+3*n101+r, mask=m)-tl.load(U+3*n100+r, mask=m)
            d3 = tl.load(U+3*n111+r, mask=m)-tl.load(U+3*n110+r, mask=m)
        elif j == 1:
            d0 = tl.load(U+3*n010+r, mask=m)-tl.load(U+3*n000+r, mask=m)
            d1 = tl.load(U+3*n011+r, mask=m)-tl.load(U+3*n001+r, mask=m)
            d2 = tl.load(U+3*n110+r, mask=m)-tl.load(U+3*n100+r, mask=m)
            d3 = tl.load(U+3*n111+r, mask=m)-tl.load(U+3*n101+r, mask=m)
        else:
            d0 = tl.load(U+3*n100+r, mask=m)-tl.load(U+3*n000+r, mask=m)
            d1 = tl.load(U+3*n101+r, mask=m)-tl.load(U+3*n001+r, mask=m)
            d2 = tl.load(U+3*n110+r, mask=m)-tl.load(U+3*n010+r, mask=m)
            d3 = tl.load(U+3*n111+r, mask=m)-tl.load(U+3*n011+r, mask=m)
        return d0, d1, d2, d3

    @triton.jit(do_not_specialize=["ncell", "ny", "nx", "cy", "cx"])     # one binary for every level grid
    def edge_hull_margins_kernel(U, K, MARGIN, FLAG, ncell, ny, nx, cy, cx, BLOCK: tl.constexpr):
        c = tl.program_id(0).to(tl.int64)*BLOCK+tl.arange(0, BLOCK).to(tl.int64)
        m = c < ncell
        x0 = c % cx
        y0 = (c//cx) % cy
        z0 = c//(cx*cy)
        n000 = (z0*ny+y0)*nx+x0
        n001 = n000+1
        n010 = n000+nx
        n011 = n010+1
        n100 = n000+ny*nx
        n101 = n100+1
        n110 = n100+nx
        n111 = n110+1
        hx = tl.load(K+0)
        hy = tl.load(K+1)
        hz = tl.load(K+2)
        delta = tl.load(K+3)
        one = tl.load(K+4)
        zero = tl.load(K+5)
        # entries in _chunk_hoisted_margins' order: j = 0 (r = 0, 1, 2), j = 1 (...), j = 2 (...)
        a0, a1, a2, a3 = _edges(U, n000, n001, n010, n011, n100, n101, n110, n111, 0, m, 0)
        f00l, f00h, e1 = _entry(a0, a1, a2, a3, hx, one)
        a0, a1, a2, a3 = _edges(U, n000, n001, n010, n011, n100, n101, n110, n111, 1, m, 0)
        f10l, f10h, e2 = _entry(a0, a1, a2, a3, hx, zero)
        a0, a1, a2, a3 = _edges(U, n000, n001, n010, n011, n100, n101, n110, n111, 2, m, 0)
        f20l, f20h, e3 = _entry(a0, a1, a2, a3, hx, zero)
        a0, a1, a2, a3 = _edges(U, n000, n001, n010, n011, n100, n101, n110, n111, 0, m, 1)
        f01l, f01h, e4 = _entry(a0, a1, a2, a3, hy, zero)
        a0, a1, a2, a3 = _edges(U, n000, n001, n010, n011, n100, n101, n110, n111, 1, m, 1)
        f11l, f11h, e5 = _entry(a0, a1, a2, a3, hy, one)
        a0, a1, a2, a3 = _edges(U, n000, n001, n010, n011, n100, n101, n110, n111, 2, m, 1)
        f21l, f21h, e6 = _entry(a0, a1, a2, a3, hy, zero)
        a0, a1, a2, a3 = _edges(U, n000, n001, n010, n011, n100, n101, n110, n111, 0, m, 2)
        f02l, f02h, e7 = _entry(a0, a1, a2, a3, hz, zero)
        a0, a1, a2, a3 = _edges(U, n000, n001, n010, n011, n100, n101, n110, n111, 1, m, 2)
        f12l, f12h, e8 = _entry(a0, a1, a2, a3, hz, zero)
        a0, a1, a2, a3 = _edges(U, n000, n001, n010, n011, n100, n101, n110, n111, 2, m, 2)
        f22l, f22h, e9 = _entry(a0, a1, a2, a3, hz, one)
        p1l, p1h, g1 = _mul(f11l, f11h, f22l, f22h)
        p2l, p2h, g2 = _mul(f12l, f12h, f21l, f21h)
        al, ah, g3 = _sub(p1l, p1h, p2l, p2h)
        p1l, p1h, g4 = _mul(f12l, f12h, f20l, f20h)
        p2l, p2h, g5 = _mul(f10l, f10h, f22l, f22h)
        bl, bh, g6 = _sub(p1l, p1h, p2l, p2h)
        p1l, p1h, g7 = _mul(f10l, f10h, f21l, f21h)
        p2l, p2h, g8 = _mul(f11l, f11h, f20l, f20h)
        ccl, cch, g9 = _sub(p1l, p1h, p2l, p2h)
        t1l, t1h, g10 = _mul(f00l, f00h, al, ah)
        t2l, t2h, g11 = _mul(f01l, f01h, bl, bh)
        sl, sh, g12 = _add(t1l, t1h, t2l, t2h)
        t3l, t3h, g13 = _mul(f02l, f02h, ccl, cch)
        jl, jh, g14 = _add(sl, sh, t3l, t3h)
        ml, mh, g15 = _out(jl-delta, jh-delta)
        ok = (e1 & e2 & e3 & e4 & e5 & e6 & e7 & e8 & e9 & g1 & g2 & g3 & g4 & g5 & g6 & g7 & g8 & g9 & g10 & g11
              & g12 & g13 & g14 & g15)
        tl.store(MARGIN+c, ml, mask=m)
        tl.store(FLAG+c, tl.where(ok, 0, 1).to(tl.int8), mask=m)


def new_record():
    """T84 record (one per backend): margins calls on the GPU path, CPU fallbacks, first-use checks, plan D rechecks."""
    return dict(policy=POLICY, calls=0, gpu_margins=0, cpu_fallbacks=0, compared=0, verified_grids=[],
                mismatches=0, mismatch_details=[], errors=0, error_types=[], unsupported=0, disabled=False,
                commit_rechecks=0, recheck_prestarts=0, recheck_prestarts_used=0, recheck_prestarts_dropped=0,
                recheck_dropped_errors=0, recheck_mismatches=0, recheck_errors=0, recheck_mismatch_details=[],
                rechecked_commits=0, recheck_wait_seconds=0.)


def _counterexample(d):
    """The T05 cell (tests/three_d/test_interval_certificate.counterexample): positive corner JD, negative inside."""
    v = np.array([(x, y, z) for z in (0., 1.) for y in (0., 1.) for x in (0., 1.)])
    x, y, z = v.T
    return np.stack((x, y+2*(x-.5)*z, -2*(x-.5)*y+d*z), axis=-1)-v


def synthetic_folding_field(grid):
    """T84 first use: a deterministic finite field on this grid exercising every branch of the per-cell arithmetic
    (review E101): zero boundary faces (as every certifiable state) and a constant block (edge differences exactly
    +-0: the outward steps' zero branches), a smooth part, rough patches, T05 counterexample cells (J < 0 inside with
    positive corners) and a collapsed cell (J = 0), cells at subnormal scale (the outward steps' subnormal branches),
    and one node with a huge displacement (overflowing products: flagged cells)."""
    from ..cell_q1 import cell_node_indices
    rng = np.random.default_rng(84)
    h = np.array(grid.spacing, dtype=np.float64)
    nz, ny, nx = grid.shape
    zz, yy, xx = np.meshgrid(*(np.linspace(0., 1., n) for n in grid.shape), indexing='ij')
    u = (.08*np.sin(np.pi*xx)*np.sin(2*np.pi*yy)*np.sin(np.pi*zz))[..., None]*h
    u = u+rng.normal(size=grid.shape+(3,))*(.02*h)
    count = max(8, (nz*ny*nx)//20000)
    corners = rng.integers(1, max(2, min(nz, ny, nx)-3), size=(count, 3))
    for k, (z, y, x) in enumerate(corners):
        z, y, x = min(z, nz-3), min(y, ny-3), min(x, nx-3)
        if k % 4 == 0:                                       # rough patch
            u[z:z+2, y:y+2, x:x+2] += rng.normal(size=(2, 2, 2, 3))*(.6*h)
        elif k % 4 == 1:                                     # constant block: exact zero differences
            u[z:z+2, y:y+2, x:x+2] = u[z, y, x]
        elif k % 4 == 2:                                     # subnormal scale
            u[z:z+2, y:y+2, x:x+2] = rng.normal(size=(2, 2, 2, 3))*5e-320
    flat = u.reshape(-1, 3)
    cells = rng.integers(0, grid.n_cells, size=max(4, count//2))
    for k, c in enumerate(cells):
        nodes = cell_node_indices(grid, slice(int(c), int(c)+1))[0]
        flat[nodes] = _counterexample(-.05 if k % 2 == 0 else 0.)*h + (0. if k % 3 else flat[nodes[0]])
    u[nz//2, ny//2, nx//2] = (1e300, -1e300, 1e300)
    u[[0, -1]] = 0.; u[:, [0, -1]] = 0.; u[:, :, [0, -1]] = 0.
    return np.ascontiguousarray(u)


def _same(gpu, cpu):
    """Every flag equal; every margin of an unflagged cell equal bit for bit (uint64)."""
    return _difference(gpu, cpu) is None


def _difference(gpu, cpu):
    """None, or the failure evidence: differing flags and unflagged margins (counts, first cell, both values)."""
    gm, gf = gpu
    cm, cf = cpu
    if gm.shape != cm.shape or gf.shape != cf.shape:
        return dict(shapes=[list(gm.shape), list(gf.shape), list(cm.shape), list(cf.shape)])
    flags = np.flatnonzero(gf != cf)
    keep = ~cf & (gf == cf)
    margins = np.flatnonzero(keep & (gm.view(np.uint64) != cm.view(np.uint64)))
    if flags.size == 0 and margins.size == 0:
        return None
    first = int(margins[0]) if margins.size else None
    return dict(flag_cells=int(flags.size), first_flag_cell=int(flags[0]) if flags.size else None,
                margin_cells=int(margins.size), first_margin_cell=first,
                first_margins=None if first is None else [float(gm[first]).hex(), float(cm[first]).hex()])


class GpuEdgeHull:
    """compiled_enclosure.parallel_margins(u, grid, delta), with the per-cell arithmetic on the GPU (T84).
    margins() runs on the certificate's kernel-pool thread, on a high-priority side stream of its own."""

    def __init__(self, device, record):
        self.device, self.record = torch.device(device), record
        self._lock = threading.Lock()
        self._verified, self._synthetic = set(), {}
        self._stream = None

    def _off(self, error):
        with self._lock:
            record = self.record
            record["errors"] += 1
            record["error_types"] = sorted(set(record["error_types"]) | {type(error).__name__})
            record["disabled"] = True

    def _gpu(self, u, grid, delta):
        nz, ny, nx = grid.shape
        if np.asarray(u).size != 3*nz*ny*nx:
            raise ValueError("T84 margins: u does not match the grid")      # never an out-of-bounds device read
        if self._stream is None:
            self._stream = torch.cuda.Stream(self.device, priority=-1)
        n = grid.n_cells
        with torch.cuda.stream(self._stream):
            # a fresh device copy of these very bytes (the ones the certificate hashes)
            du = torch.tensor(np.ascontiguousarray(u, dtype=np.float64).reshape(-1), device=self.device,
                              dtype=torch.float64)
            k = torch.tensor([float(h) for h in grid.spacing]+[float(delta), 1., 0.], device=self.device,
                             dtype=torch.float64)
            margin = torch.full((n,), float("nan"), dtype=torch.float64, device=self.device)
            flag = torch.ones(n, dtype=torch.int8, device=self.device)
            edge_hull_margins_kernel[(triton.cdiv(n, BLOCK),)](
                du, k, margin, flag, n, ny, nx, ny-1, nx-1, BLOCK=BLOCK, num_warps=WARPS, maxnreg=MAXNREG,
                enable_fp_fusion=False)
            out = margin.cpu().numpy(), flag.cpu().numpy()
        m, f = out
        if m.shape != (n,) or f.shape != (n,) or m.dtype != np.float64:
            raise RuntimeError("T84 margins of unexpected shape or type")
        return m, f.astype(bool)

    def margins(self, u, grid, delta):
        from .. import compiled_enclosure as ce
        from ..work_control import WorkBudgetExceeded
        record = self.record
        if record["disabled"]:
            record["cpu_fallbacks"] += 1
            return ce.parallel_margins(u, grid, delta)
        if not ce.arithmetic_supported(np.float64(0.)):
            return ce.parallel_margins(u, grid, delta)          # raises exactly as the CPU path does
        key = (grid.grid_id, float(delta).hex())
        try:
            result = self._gpu(u, grid, delta)
        except WorkBudgetExceeded:
            raise
        except Exception as error:
            self._off(error)
            record["cpu_fallbacks"] += 1
            return ce.parallel_margins(u, grid, delta)
        record["calls"] += 1
        if key in self._verified:
            record["gpu_margins"] += 1
            return result
        # first use on this (grid, delta): the synthetic folding field once, then this informative input
        try:
            if key not in self._synthetic:
                field = synthetic_folding_field(grid)
                self._synthetic[key] = _difference(self._gpu(field, grid, delta), ce.parallel_margins(field, grid, delta))
            reference = ce.parallel_margins(u, grid, delta)
        except WorkBudgetExceeded:
            raise
        except Exception as error:
            self._off(error)
            record["cpu_fallbacks"] += 1
            return ce.parallel_margins(u, grid, delta)
        record["compared"] += 1
        difference = self._synthetic[key]
        source = "synthetic"
        if difference is None:
            difference, source = _difference(result, reference), "input"
        if difference is not None:
            record["mismatches"] += 1
            record["mismatch_details"].append(dict(difference, grid_id=grid.grid_id, delta=float(delta).hex(),
                                                   field=source))           # failure evidence (hard constraint 5)
            record["disabled"] = True
            return reference
        if ce._informative(u, reference[0]):
            self._verified.add(key)
            record["verified_grids"].append([grid.grid_id, float(delta).hex()])
        return reference


def prewarm():
    """Compile the kernel outside any task clock."""
    from ..grid import make_grid_3d
    g = make_grid_3d(nx=5, ny=6, nz=7, hx=1., hy=1., hz=1., x0=0., y0=0., z0=0., unit='mm')
    GpuEdgeHull("cuda", new_record())._gpu(np.zeros(g.shape+(3,)), g, 0.)
    torch.cuda.synchronize()
