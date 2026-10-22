"""E51/S3 opt-in fused float64 Triton kernels for Q1 Gauss/closed27 geometry.

Same discrete objects as torch_ops (T02/T03/T08/T28/T39): Q1 physical
derivatives at the 27 Gauss points (JD/curl change penalties) and at the
closed27 uniform barrier points, the localized C2 barrier, and the Euclidean
transpose of the geometry fluxes. Only the evaluation order changes: Q1
derivatives use the exact edge-difference/bilinear tensor form instead of a
dense einsum, per-cell vertex contributions go to a buffer and are gathered
per node in the fixed vertex order 0..7 after the base term, soft terms before
barrier terms (the single-tile legacy order), and energies are block partial
sums added in a fixed order. Results agree with the torch path to roundoff,
NOT bitwise; GPU energies stay guarded comparisons with CPU fallback (T41) and
no GPU value is ever a feasibility certificate. Only the localized closed27
barrier used by the MRI route is supported; other problems use torch_ops.
"""
import numpy as np
import torch
import triton

from ...validation import InputValidationError
from . import triton_q1_kernels as K
from . import triton_q1_idx_kernels as KI       # T89 (S30-G4)
from . import triton_q1_f32_kernels as KF       # T90 (S30-G5 draft)

BLOCK_NODES = 256
POLICY = "triton-fused-q1-v1"
# E51 launch settings measured at native size (one cell per thread is fastest;
# BLOCK = 32*num_warps avoids duplicated lanes). Launch shape never changes a
# per-cell value; block partial sums are added in the same fixed block order.
LAUNCH = dict(soft_grad=(64, 2), barrier_grad=(64, 2), soft_energy=(64, 2), soft_energy_base=(128, 8),
              barrier_energy=(64, 2), ggn_diag=(32, 1), ggn_hvp=(32, 1))
# ggn_diag: E65 measured 0.38 s vs 0.87 s at native (same bits); ggn_hvp: same shape as ggn_diag, not tuned.
# E99 (throughput-v17): soft_energy_base at (128, 8) gives each cell to two threads (128 elements over 256 threads:
# every cell computed twice) with 128 registers; (128, 4) with maxnreg=255 computes it once (same BLOCK, so the same
# block partial sums; the PTX differs only in .maxntid/.maxnreg). ggn_diag gets maxnreg=255 at its shape. Measured
# on a native state: 237.7 -> 49.4 ms and 375 -> 261 ms, bit for bit. Without an explicit maxnreg ptxas may choose
# 40 registers for (128, 4): the launch records the registers and keeps the E51 shape below 128.
TUNED_LAUNCH = dict(soft_energy_base=(128, 4, 255), ggn_diag=(32, 1, 255))
TUNED_MIN_REGISTERS = 128


def supported(problem):
    return (problem.barrier_location == "closed27_uniform" and problem.barrier_activation is not None
            and getattr(problem, "barrier_basis", "samples") in ("samples", "bernstein")
            and problem._gauss.location == "gauss27")


def skip_preconditions(problem):
    """E98: the host conditions under which every barrier helper is an exact signed zero at J >= act and the
    guarded products stay finite: finite w >= 0, finite delta, act > max(delta, 0) with act - delta finite and
    (act-delta)^2 > 0 (scale = act - delta, t = 1), and every 1/h <= 1e30 (the gradient kernel's guard bounds the
    vertex values by 1e100, so every edge (u_b - u_a)/h is at most 2e130 and every cofactor is finite).
    make_problem already enforces 0 <= delta < act <= 1."""
    w, delta, act = float(problem.barrier_weight), float(problem.certificate_config.delta), float(problem.barrier_activation)
    ih = [1/float(h) for h in problem.grid.spacing]
    return (np.isfinite(w) and w >= 0 and np.isfinite(delta) and np.isfinite(act) and act > delta and act > 0
            and np.isfinite(act-delta) and (act-delta)*(act-delta) > 0
            and all(np.isfinite(x) and abs(x) <= 1e30 for x in ih))


class _TunedOff(Exception):
    """E99: the tuned launch compiled with too few registers; the reference shape is used (not an error)."""


ACTIVE_SET_CHECKS = 3        # T89: skipped calls per (kernel, grid) also computed in full and compared


def new_active_set_record():
    return dict(policy="native-active-set-skip-v1", models=0, unsupported=0, disabled=False, mismatches=0, errors=0,
                frozen_mismatch=0, fractions={},
                energy=dict(calls=0, skipped=0, compared=0, verified=[], misses=0),
                gradient=dict(calls=0, skipped=0, compared=0, verified=[], misses=0),
                sampling=dict(calls=0, skipped=0, compared=0, verified=[], misses=0))


def _same_values(a, b):
    """== with +0 == -0 and NaN at the same places (torch.equal is False for any NaN)."""
    if a.shape != b.shape:
        return False
    an, bn = torch.isnan(a), torch.isnan(b)
    return bool(torch.equal(an, bn) and torch.equal(torch.where(an, torch.zeros_like(a), a), torch.where(bn, torch.zeros_like(b), b)))


class _ActiveSets:
    """T89 (S30-G4): the device index sets of one problem's active weights (frozen nodes never move) and the caches
    of the contributions of the frozen part: block partial sums (energy), the last full gradient of the same
    reference (gradient), the last sampling (warped image and its gradient). `record` is the backend's shared dict.
    Every use is guarded: the frozen entries of the new u must equal the cached state's, as values."""

    def __init__(self, problem, device, record):
        from ..active_set import index_sets, block_mask
        self.record, self.device = record, device
        g = problem.grid
        self.grid_id = g.grid_id
        sets = index_sets(problem.active_weights, g, gradient_rings=2 if float(problem.curvature_weight) else 1)
        nz, ny, nx = g.shape
        i32 = lambda a: torch.tensor(np.ascontiguousarray(a, dtype=np.int32), device=device)
        self.cells_c1 = i32(np.flatnonzero(sets["c1_cells"].reshape(-1)))
        n1 = np.flatnonzero(sets["n1_nodes"].reshape(-1)).astype(np.int64)
        self.dofs_n1 = i32(np.concatenate([3*n1, 3*n1+1, 3*n1+2]))
        self.nodes_active = i32(np.flatnonzero(sets["active_nodes"].reshape(-1)))
        self.frozen3 = torch.tensor(np.repeat(sets["frozen_nodes"][..., None], 3, axis=-1), device=device)
        self.blocks = {b: i32(np.flatnonzero(block_mask(sets["active_cells"], b))) for b in (64, 128)}
        self.fractions = dict(nodes_active=float(sets["active_nodes"].mean()), nodes_n1=float(sets["n1_nodes"].mean()),
                              cells_c1=float(sets["c1_cells"].mean()), cells_active=float(sets["active_cells"].mean()),
                              blocks64=float(block_mask(sets["active_cells"], 64).mean()),
                              blocks128=float(block_mask(sets["active_cells"], 128).mean()))
        record["models"] += 1
        record["fractions"] = self.fractions
        self.soft_cache, self.barrier_cache, self.grad_cache, self.sample_cache = [], [], None, None
        self.templates = {}                  # G4b: (kind, with_base, blocks) -> (u, block outputs) of an all-frozen-equal call

    def enabled(self):
        return not self.record["disabled"]

    def frozen_equal(self, u, other, count=True):
        """The frozen entries of u and other are equal as values (one device synchronization)."""
        if u is other:
            return True
        if u.shape != other.shape:
            return False
        same = not bool(((u != other) & self.frozen3).any())
        if not same and count:
            self.record["frozen_mismatch"] += 1
        return same

    def store_template(self, key, u, ref, base, out):
        """G4b: keep the block outputs of a full call whose ref and base (when given) have u's frozen entries."""
        if (ref is None or self.frozen_equal(ref, u, count=False)) and (base is None or self.frozen_equal(base, u, count=False)):
            self.templates[key] = (u, out)

    def template(self, key, u, ref, base):
        """G4b: the stored block outputs for key when u, ref and base (when given) all have the template's frozen entries."""
        entry = self.templates.get(key)
        if entry is None:
            return None
        t = entry[0]
        if not self.frozen_equal(u, t):
            return None
        if ref is not None and not self.frozen_equal(ref, t):
            return None
        if base is not None and not self.frozen_equal(base, t):
            return None
        return entry[1]

    @staticmethod
    def _keep(cache, entry, n=2):
        cache.insert(0, entry)
        del cache[n:]

    def check(self, kind, name, out, reference_launch):
        """First-use comparison of a skipped result with the full computation; True when `out` may be used."""
        part = self.record[kind]
        key = [name, self.grid_id]
        if key in part["verified"]:
            return True
        try:
            reference = reference_launch()
            same = _same_values(out, reference)
        except Exception:
            self.record["errors"] += 1
            self.record["disabled"] = True
            return False
        part["compared"] += 1
        if not same:
            self.record["mismatches"] += 1
            self.record["disabled"] = True
            return False
        if sum(1 for k in part.get("verified_counts", []) if k == key) >= ACTIVE_SET_CHECKS-1:
            part["verified"].append(key)
        part.setdefault("verified_counts", []).append(key)
        return True


def prewarm_fp32():
    """T90: compile the float32 screening kernels (both WITH_BASE values) on a tiny grid; no record."""
    from ..grid import make_grid_3d
    g = make_grid_3d(nx=7, ny=6, nz=5, hx=1., hy=1., hz=1., x0=0., y0=0., z0=0., unit='mm')
    ncell, (nz, ny, nx) = g.n_cells, g.shape
    u = torch.zeros(g.shape+(3,), dtype=torch.float64, device='cuda')
    k = torch.zeros(39, dtype=torch.float64, device='cuda'); k[37] = .05; k[38] = .05
    block, warps = FusedQ1.F32_LAUNCH['soft_energy']
    blocks = triton.cdiv(ncell, block)
    for with_base in (False, True):
        out = torch.empty((blocks, 3), dtype=torch.float64, device='cuda')
        KF.soft_energy_f32_kernel[(blocks,)](u, u, u, k, out, ncell, ny, nx, ny-1, nx-1, WITH_BASE=with_base, BLOCK=block, num_warps=warps)
        out = torch.empty((blocks, 4), dtype=torch.float64, device='cuda')
        for kernel in (KF.barrier_energy_f32_kernel, KF.barrier_energy_skip_f32_kernel):
            kernel[(blocks,)](u, u, k, out, ncell, ny, nx, ny-1, nx-1, WITH_BASE=with_base, BLOCK=block, num_warps=warps)
    torch.cuda.synchronize()


def prewarm_active_set():
    """T89: compile the index-list kernels (both WITH_BASE values, the launch shapes used) on a tiny grid; no record."""
    from ..grid import make_grid_3d
    g = make_grid_3d(nx=7, ny=6, nz=5, hx=1., hy=1., hz=1., x0=0., y0=0., z0=0., unit='mm')
    ncell, (nz, ny, nx) = g.n_cells, g.shape
    u = torch.zeros(g.shape+(3,), dtype=torch.float64, device='cuda')
    k = torch.zeros(39, dtype=torch.float64, device='cuda'); k[37] = .05; k[38] = .05
    idx = torch.arange(ncell, dtype=torch.int32, device='cuda')
    loc = torch.empty((ncell, 24), dtype=torch.float64, device='cuda')
    for kernel, name in ((KI.soft_grad_idx_kernel, 'soft_grad'), (KI.barrier_grad_idx_kernel, 'barrier_grad'),
                         (KI.barrier_grad_skip_idx_kernel, 'barrier_grad')):
        block, warps = LAUNCH[name]
        args = (u, u, k) if kernel is KI.soft_grad_idx_kernel else (u, k)
        kernel[(triton.cdiv(ncell, block),)](*args, loc, idx, ncell, ncell, ny, nx, ny-1, nx-1, BLOCK=block, num_warps=warps)
    for with_base, shape in ((False, LAUNCH['soft_energy']), (True, LAUNCH['soft_energy_base']), (True, TUNED_LAUNCH['soft_energy_base'])):
        block, warps, *regs = shape
        blocks = triton.cdiv(ncell, block)
        blk = torch.arange(blocks, dtype=torch.int32, device='cuda')
        out = torch.empty((blocks, 3), dtype=torch.float64, device='cuda')
        KI.soft_energy_idx_kernel[(blocks,)](u, u, u, k, out, blk, ncell, ny, nx, ny-1, nx-1, WITH_BASE=with_base, BLOCK=block,
                                             num_warps=warps, **(dict(maxnreg=regs[0]) if regs else {}))
    block, warps = LAUNCH['barrier_energy']
    blocks = triton.cdiv(ncell, block)
    blk = torch.arange(blocks, dtype=torch.int32, device='cuda')
    out = torch.empty((blocks, 4), dtype=torch.float64, device='cuda')
    for kernel in (KI.barrier_energy_idx_kernel, KI.barrier_energy_skip_idx_kernel):
        for with_base in (False, True):
            kernel[(blocks,)](u, u, k, out, blk, ncell, ny, nx, ny-1, nx-1, WITH_BASE=with_base, BLOCK=block, num_warps=warps)
    nnode = nz*ny*nx
    dofs = torch.arange(nnode*3, dtype=torch.int32, device='cuda')
    base = torch.zeros((nnode*3,), dtype=torch.float64, device='cuda')
    KI.gather_idx_kernel[(triton.cdiv(nnode*3, BLOCK_NODES),)](base, loc, torch.empty_like(base), dofs, nnode*3, nnode, nz, ny, nx,
                                                              ny-1, nx-1, BLOCK=BLOCK_NODES)
    torch.cuda.synchronize()


def prewarm_ggn_reuse():
    """GG2: compile the indexed GGN diagonal kernel (the E51 launch shape) on a tiny grid; no record."""
    from ..grid import make_grid_3d
    g = make_grid_3d(nx=7, ny=6, nz=5, hx=1., hy=1., hz=1., x0=0., y0=0., z0=0., unit='mm')
    ncell, (nz, ny, nx) = g.n_cells, g.shape
    u = torch.zeros(g.shape+(3,), dtype=torch.float64, device='cuda')
    k = torch.zeros(39, dtype=torch.float64, device='cuda'); k[37] = .05; k[38] = .05
    idx = torch.arange(ncell, dtype=torch.int32, device='cuda')
    loc = torch.empty((ncell, 24), dtype=torch.float64, device='cuda')
    block, warps = LAUNCH['ggn_diag']
    KI.ggn_diag_idx_kernel[(triton.cdiv(ncell, block),)](u, k, loc, idx, ncell, ncell, ny, nx, ny-1, nx-1, BLOCK=block,
                                                         num_warps=warps)
    torch.cuda.synchronize()


class FusedQ1:
    """Constants and launches for one problem (grid, weights, barrier)."""

    def __init__(self, problem, device, barrier_skip=None, tuned_launch=None, reference_first=False, active_set=None):
        if not supported(problem):
            raise InputValidationError("fused Q1 kernels need gauss27 soft terms and the closed27 localized barrier")
        g = problem.grid
        nodes = (.5-np.sqrt(15.)/10, .5, .5+np.sqrt(15.)/10)
        qw = np.asarray(problem._gauss.weights, dtype=np.float64)
        if qw.shape != (27,):
            raise InputValidationError("unexpected Gauss weight layout")
        delta, act = problem.certificate_config.delta, problem.barrier_activation
        consts = np.concatenate([[1/h for h in g.spacing], nodes, qw,
                                 [problem.weights.mu, problem.weights.nu, problem.barrier_weight, delta, act, act-delta]])
        self.k = torch.tensor(consts, dtype=torch.float64, device=device)
        self.shape = g.shape
        nz, ny, nx = g.shape
        self.nz, self.ny, self.nx, self.cy, self.cx = nz, ny, nx, ny-1, nx-1
        self.ncell, self.nnode = g.n_cells, g.n_nodes if hasattr(g, 'n_nodes') else nz*ny*nx
        self.device = device
        self.barrier_weight = problem.barrier_weight
        # E58/T68: barrier kernels on the 27 Bernstein coefficients when selected.
        self.bernstein = getattr(problem, "barrier_basis", "samples") == "bernstein"
        # E98 (throughput-v16): barrier kernels that skip inactive closed27 points, bit for bit the kernels above;
        # `barrier_skip` is the backend's shared record (calls, first-use checks per kernel and grid, mismatches).
        self.skip = None
        self._grid_id = g.grid_id
        self._act = float(act)
        if barrier_skip is not None:
            if self.bernstein or not skip_preconditions(problem):
                barrier_skip["unsupported"] += 1
            else:
                self.skip = barrier_skip
        # E99 (throughput-v17): the tuned launch shapes (`tuned_launch` is the backend's shared record) and the
        # reference-first check of the skip kernels; both verified per kernel and grid against the E51/v16 launch.
        self.tuned = tuned_launch
        self.reference_first = bool(reference_first)
        # T89 (S30-G4): the active-set index sets and caches (an _ActiveSets of this problem), or None
        self.active = None
        if active_set is not None:
            if self.bernstein or not skip_preconditions(problem):
                active_set.record["unsupported"] += 1
            else:
                self.active = active_set

    def _grid(self, n, block):
        return (triton.cdiv(n, block),)

    def _cell_locals(self, kernel, name, *arrays):
        block, warps = LAUNCH[name]
        out = torch.empty((self.ncell, 24), dtype=torch.float64, device=self.device)
        kernel[self._grid(self.ncell, block)](*arrays, self.k, out, self.ncell, self.ny, self.nx, self.cy, self.cx,
                                              BLOCK=block, num_warps=warps)
        return out

    def _gather(self, base, local):
        out = torch.empty_like(base)
        K.gather_kernel[self._grid(self.nnode*3, BLOCK_NODES)](base, local, out, self.nnode, self.nz, self.ny, self.nx,
                                                               self.cy, self.cx, BLOCK=BLOCK_NODES)
        return out

    def _check(self, *arrays):
        for a in arrays:
            if a.dtype != torch.float64 or not a.is_contiguous() or tuple(a.shape) != self.shape+(3,):
                raise InputValidationError("fused kernels need contiguous float64 node arrays")

    def soft_locals(self, u, ref):
        self._check(u, ref)
        return self._cell_locals(K.soft_grad_kernel, 'soft_grad', u, ref)

    def _skipping(self):
        return self.skip is not None and not self.skip["disabled"]

    def _checked(self, name, launch, reference_launch, active):
        """E98: the skip kernel's output. Until (kernel, grid) is verified, the v15 kernel also runs and both outputs
        are compared bit for bit. A match verifies the pair only if the reference shows that the active branch ran
        (`active`), so a call with every point skipped (u = 0, say) is compared but proves nothing. A mismatch
        returns the reference and switches the skip kernels off for the run. A failure while comparing (e.g. out of
        memory for the second buffer or the comparison's temporaries) also switches them off and returns the v15
        output -- the reference if it exists, else the v15 kernel run alone after the failed call's buffers are
        released (outside the except block, whose traceback keeps them alive)."""
        out = launch()
        if [name, self._grid_id] in self.skip["verified"]:
            return out
        reference = verdict = None
        try:
            reference = reference_launch()
            same = out.shape == reference.shape and torch.equal(out.view(torch.int64), reference.view(torch.int64))
            verdict = (same, same and active(reference))
        except Exception:
            pass
        if verdict is None:
            out = None
            self.skip["errors"] += 1
            self.skip["disabled"] = True
            return reference if reference is not None else reference_launch()
        self.skip["compared"] += 1
        if not verdict[0]:
            self.skip["mismatches"] += 1
            self.skip["disabled"] = True
            return reference
        if verdict[1]:
            self.skip["verified"].append([name, self._grid_id])
        return out

    def _reference_first_skip(self, name, launch, reference_launch, active):
        """E99 (throughput-v17): the skip kernels' check with the reference first (_reference_first on the skip
        record): an unverified call whose reference shows no active point runs the v15 kernel only."""
        return self._reference_first(self.skip, name, launch, reference_launch, active)

    def barrier_locals(self, u):
        self._check(u)
        if self._skipping():
            self.skip["calls"] += 1
            # active: some local is nonzero or NaN (with w = 0 never: every call is then compared under v16; under
            # v17's reference-first check every such call runs the v15 kernel only)
            check = self._reference_first_skip if self.reference_first else self._checked       # E99: v17
            return check('barrier_grad', lambda: self._cell_locals(K.barrier_grad_skip_kernel, 'barrier_grad', u),
                         lambda: self._cell_locals(K.barrier_grad_kernel, 'barrier_grad', u),
                         lambda r: bool((r != 0).any()))
        kernel = K.barrier_grad_bern_kernel if self.bernstein else K.barrier_grad_kernel
        return self._cell_locals(kernel, 'barrier_grad', u)

    def _tuned_shape(self, name):
        return TUNED_LAUNCH[name] if (self.tuned is not None and not self.tuned["disabled"]) else None

    def _launch_tuned(self, name, launch):
        """E99: a launch at the tuned shape; its compiled registers are recorded (first launch per name), and fewer
        than TUNED_MIN_REGISTERS (maxnreg not applied) switch the tuned shapes off: the reference is used instead."""
        out, compiled = launch()
        regs = self.tuned["registers"]
        if name not in regs:
            regs[name] = int(getattr(compiled, "n_regs", 0) or 0)
            if regs[name] < TUNED_MIN_REGISTERS:
                self.tuned["low_registers"] += 1
                self.tuned["disabled"] = True
                raise _TunedOff()
        return out

    def _reference_first(self, record, name, launch, reference_launch, active):
        """E99 (throughput-v17): until (kernel, grid) is verified, the reference (the v16 kernel or launch shape) runs
        first; only when it shows that the compared path was exercised (`active`) does the new one run too, compared
        bit for bit; otherwise the reference is returned (counted as reference_only). A match verifies the pair; a
        mismatch or any failure switches `record` off for the run and returns the reference. Verified: the new one
        alone."""
        if not record["disabled"] and [name, self._grid_id] in record["verified"]:
            return launch()
        reference = reference_launch()
        if record["disabled"]:
            return reference
        out = verdict = None
        try:
            if active(reference):
                out = launch()
                verdict = bool(out.shape == reference.shape
                               and torch.equal(out.view(torch.int64), reference.view(torch.int64)))
            else:
                verdict = "inactive"
        except _TunedOff:
            verdict = "off"
        except Exception:
            pass
        out = None if verdict is not True else out
        if verdict is None:
            record["errors"] += 1
            record["disabled"] = True
            return reference
        if verdict in ("inactive", "off"):
            record["reference_only"] += verdict == "inactive"
            return reference
        record["compared"] += 1
        if not verdict:
            record["mismatches"] += 1
            record["disabled"] = True
            return reference
        record["verified"].append([name, self._grid_id])
        return out

    def _barrier_active(self, u):
        """E99: whether some closed27 point of u has J < act (or J is NaN), so that the barrier part of the GGN
        diagonal contributes; from the v16 barrier energy kernel (block minima), run only while ggn_diag's tuned
        shape is unverified."""
        block, warps = LAUNCH['barrier_energy']
        blocks = triton.cdiv(self.ncell, block)
        out = torch.empty((blocks, 4), dtype=torch.float64, device=self.device)
        K.barrier_energy_kernel[(blocks,)](u, u, self.k, out, self.ncell, self.ny, self.nx, self.cy, self.cx,
                                           WITH_BASE=False, BLOCK=block, num_warps=warps)
        return bool(torch.isnan(out[:, :2]).any() or (out[:, 1] < self._act).any())

    def _cell_locals_at(self, kernel, shape, *arrays):
        """E99: _cell_locals at an explicit (BLOCK, num_warps[, maxnreg]) shape; returns (out, compiled kernel)."""
        block, warps, *regs = shape
        out = torch.empty((self.ncell, 24), dtype=torch.float64, device=self.device)
        compiled = kernel[self._grid(self.ncell, block)](*arrays, self.k, out, self.ncell, self.ny, self.nx, self.cy,
                                                         self.cx, BLOCK=block, num_warps=warps,
                                                         **(dict(maxnreg=regs[0]) if regs else {}))
        return out, compiled

    def diagonal_nodes(self, base, u):
        """E65: base + per-node sum of the GGN diagonal geometry/barrier contributions (samples basis)."""
        if self.bernstein:
            raise InputValidationError("fused GGN diagonal supports the closed27 samples barrier only")
        self._check(u)
        tuned = self._tuned_shape('ggn_diag')
        if tuned is not None:                                          # E99
            self.tuned["calls"] += 1
            local = self._reference_first(
                self.tuned, 'ggn_diag',
                lambda: self._launch_tuned('ggn_diag', lambda: self._cell_locals_at(K.ggn_diag_kernel, tuned, u)),
                lambda: self._cell_locals(K.ggn_diag_kernel, 'ggn_diag', u),
                # active (review E99): nonzero locals AND an active barrier at u -- the soft-geometry part alone is
                # nonzero for every u, so a first call with no closed27 J < act would leave the barrier part unchecked
                lambda r: bool((r != 0).any()) and self._barrier_active(u))
            return self._gather(base.contiguous(), local)
        return self._gather(base.contiguous(), self._cell_locals(K.ggn_diag_kernel, 'ggn_diag', u))

    def diagonal_locals(self, u):
        """GG2: the per-cell locals diagonal_nodes(base, u) gathers -- the same launches, shapes and checks in the same
        order (E99 tuned shape with its reference-first check, else the E51 shape); diagonal_nodes itself is unchanged."""
        if self.bernstein:
            raise InputValidationError("fused GGN diagonal supports the closed27 samples barrier only")
        self._check(u)
        tuned = self._tuned_shape('ggn_diag')
        if tuned is not None:
            self.tuned["calls"] += 1
            return self._reference_first(
                self.tuned, 'ggn_diag',
                lambda: self._launch_tuned('ggn_diag', lambda: self._cell_locals_at(K.ggn_diag_kernel, tuned, u)),
                lambda: self._cell_locals(K.ggn_diag_kernel, 'ggn_diag', u),
                lambda r: bool((r != 0).any()) and self._barrier_active(u))
        return self._cell_locals(K.ggn_diag_kernel, 'ggn_diag', u)

    def diagonal_locals_indexed(self, u, cells, out):
        """GG2: the ggn_diag locals of the listed cells (int32 device tensor) written into their rows of `out`
        ((ncell, 24) float64); the other rows are left as they are. The kernel body is ggn_diag_kernel's (generated
        from the same text, scripts/gen_triton_q1.py); the backend compares it bit for bit on first use per grid."""
        if self.bernstein:
            raise InputValidationError("fused GGN diagonal supports the closed27 samples barrier only")
        self._check(u)
        if (out.dtype != torch.float64 or not out.is_contiguous() or tuple(out.shape) != (self.ncell, 24)
                or cells.dtype != torch.int32 or cells.dim() != 1):
            raise InputValidationError("GG2: indexed GGN diagonal needs an (ncell, 24) float64 store and int32 cells")
        n = int(cells.numel())
        if n:
            block, warps = LAUNCH['ggn_diag']
            KI.ggn_diag_idx_kernel[self._grid(n, block)](u, self.k, out, cells, n, self.ncell, self.ny, self.nx, self.cy,
                                                        self.cx, BLOCK=block, num_warps=warps)
        return out

    def gather_nodes(self, base, local):
        """GG2: base + the per-node sum of (ncell, 24) locals (the gather diagonal_nodes uses)."""
        return self._gather(base.contiguous(), local)

    def hvp_nodes(self, base, u, v):
        """E72: base + per-node sum of the frozen GGN action on v (geometry + closed27 barrier, samples basis)."""
        if self.bernstein:
            raise InputValidationError("fused GGN HVP supports the closed27 samples barrier only")
        self._check(u, v)
        return self._gather(base.contiguous(), self._cell_locals(K.ggn_hvp_kernel, 'ggn_hvp', u, v))

    def gradient_nodes(self, base, u, ref, *, soft=True, barrier=True):
        """base + sum soft contributions + sum barrier contributions (node array)."""
        act = self.active
        if act is not None and soft and barrier and act.enabled():
            got = self._gradient_nodes_active(base, u, ref)          # T89 (S30-G4)
            if got is not None:
                return got
        out = base.contiguous()
        if soft:
            out = self._gather(out, self.soft_locals(u, ref))
        if barrier:
            out = self._gather(out, self.barrier_locals(u))
        if act is not None and soft and barrier:
            act.grad_cache = (u, ref, out)
        return out

    def _active_locals(self, kernel, name, *arrays):
        """T89: the per-cell locals of the C1 cells only (index-list kernel), in a full-size buffer."""
        act = self.active
        block, warps = LAUNCH[name]
        out = torch.empty((self.ncell, 24), dtype=torch.float64, device=self.device)
        n = int(act.cells_c1.numel())
        kernel[self._grid(n, block)](*arrays, self.k, out, act.cells_c1, n, self.ncell, self.ny, self.nx, self.cy, self.cx,
                                     BLOCK=block, num_warps=warps)
        return out

    def _active_gather(self, out, base, local):
        """T89: out[i] = base[i] + sum of the 8 cell contributions for the N1 degrees of freedom i (fixed vertex order)."""
        act = self.active
        n = int(act.dofs_n1.numel())
        KI.gather_idx_kernel[self._grid(n, BLOCK_NODES)](base, local, out, act.dofs_n1, n, self.nnode, self.nz, self.ny, self.nx,
                                                        self.cy, self.cx, BLOCK=BLOCK_NODES)
        return out

    def _gradient_nodes_active(self, base, u, ref):
        """T89: the gradient with the frozen degrees of freedom copied from the last full gradient of the same
        reference (their 1-/2-ring is frozen, so every term of theirs is the same computation on the same values),
        or None (no usable cache: the caller computes in full and caches)."""
        act = self.active
        part = act.record["gradient"]
        part["calls"] += 1
        entry = act.grad_cache
        if entry is None or entry[1] is not ref or not act.frozen_equal(u, entry[0]):
            part["misses"] += 1
            return None
        skip_grad = self._skipping() and ['barrier_grad', self._grid_id] in self.skip["verified"]
        out = entry[2].clone()
        base = base.contiguous()
        soft = self._active_locals(KI.soft_grad_idx_kernel, 'soft_grad', u, ref)
        bar = self._active_locals(KI.barrier_grad_skip_idx_kernel if skip_grad else KI.barrier_grad_idx_kernel, 'barrier_grad', u)
        out = self._active_gather(out, base, soft)
        out = self._active_gather(out, out, bar)
        part["skipped"] += 1
        if not act.check('gradient', 'gradient_nodes', out,
                         lambda: self._gather(self._gather(base, self.soft_locals(u, ref)), self.barrier_locals(u))):
            return None
        act.grad_cache = (u, ref, out)
        return out

    def joint(self, image, total_base, u, ref):
        """(outer=image+barrier, total=image+prox+soft+barrier) sharing barrier locals."""
        bar = self.barrier_locals(u)
        outer = self._gather(image.contiguous(), bar)
        total = self._gather(self._gather(total_base.contiguous(), self.soft_locals(u, ref)), bar)
        if self.active is not None:
            self.active.grad_cache = (u, ref, total)                  # T89: the outer start's full gradient of this reference
        return outer, total

    def soft_energy(self, u, ref, base=None):
        self._check(u, ref) if base is None else self._check(u, ref, base)
        if self.active is not None and self.active.enabled():
            got = self._soft_energy_active(u, ref, base)              # T89 (S30-G4)
            if got is not None:
                return got
        out = self._soft_energy_out(u, ref, base)
        if self.active is not None:
            self.active.store_template(('soft', base is not None, out.shape[0]), u, ref, base, out)
        return out.sum(dim=0)   # [jd, curl, stable soft difference]

    def _soft_energy_out(self, u, ref, base):
        """The block partial sums [jd, curl, stable soft difference] of the E51/E99 launches (the body of soft_energy)."""
        tuned = None if base is None else self._tuned_shape('soft_energy_base')
        if tuned is not None:                                          # E99
            self.tuned["calls"] += 1

            def launch(shape):
                block, warps, *regs = shape
                out = torch.empty((triton.cdiv(self.ncell, block), 3), dtype=torch.float64, device=self.device)
                compiled = K.soft_energy_kernel[(triton.cdiv(self.ncell, block),)](
                    u, ref, base, self.k, out, self.ncell, self.ny, self.nx, self.cy, self.cx, WITH_BASE=True,
                    BLOCK=block, num_warps=warps, **(dict(maxnreg=regs[0]) if regs else {}))
                return out, compiled
            out = self._reference_first(self.tuned, 'soft_energy_base',
                                        lambda: self._launch_tuned('soft_energy_base', lambda: launch(tuned)),
                                        lambda: launch(LAUNCH['soft_energy_base'])[0],
                                        # active (review E99): every column has a nonzero partial (all three
                                        # are compared; e.g. u = ref = base gives all zeros: not informative)
                                        lambda r: bool((r[:, 0] != 0).any() and (r[:, 1] != 0).any()
                                                       and (r[:, 2] != 0).any()))
            return out
        block, warps = LAUNCH['soft_energy' if base is None else 'soft_energy_base']
        blocks = triton.cdiv(self.ncell, block)
        out = torch.empty((blocks, 3), dtype=torch.float64, device=self.device)
        K.soft_energy_kernel[(blocks,)](u, ref, u if base is None else base, self.k, out, self.ncell, self.ny, self.nx,
                                        self.cy, self.cx, WITH_BASE=base is not None, BLOCK=block, num_warps=warps)
        return out

    def _soft_shape(self, base):
        """T89: the launch shape the full path uses now (the tuned shape once verified on this grid), or None while
        the tuned shape is still being verified (the active path waits for the reference to settle)."""
        if base is None:
            return LAUNCH['soft_energy']
        tuned = self._tuned_shape('soft_energy_base')
        if tuned is None:
            return LAUNCH['soft_energy_base']
        return tuned if ['soft_energy_base', self._grid_id] in self.tuned["verified"] else None

    def _soft_energy_active(self, u, ref, base):
        """T89: the block partial sums with the frozen blocks copied from a cached evaluation of the same reference
        whose frozen entries equal u's (zero when the reference is u's own frozen state), the active blocks by the
        index-list kernel; or None (no usable cache / shape not settled: the caller computes in full and caches)."""
        act = self.active
        part = act.record["energy"]
        part["calls"] += 1
        shape = self._soft_shape(base)
        if shape is None:
            part["misses"] += 1
            return None
        block, warps, *regs = shape
        blocks = triton.cdiv(self.ncell, block)
        cached = act.template(('soft', base is not None, blocks), u, ref, base)     # G4b: exact frozen-block outputs
        if cached is None:
            part["misses"] += 1
            return None
        out = cached.clone()
        blk = act.blocks[block]
        KI.soft_energy_idx_kernel[(int(blk.numel()),)](u, ref, u if base is None else base, self.k, out, blk, self.ncell, self.ny, self.nx,
                                                     self.cy, self.cx, WITH_BASE=base is not None, BLOCK=block, num_warps=warps,
                                                     **(dict(maxnreg=regs[0]) if regs else {}))
        part["skipped"] += 1
        if not act.check('energy', 'soft_energy_base' if base is not None else 'soft_energy', out,
                         lambda: self._soft_energy_out(u, ref, base)):
            return None
        return out.sum(dim=0)

    # ---- T90 (S30-G5 draft): float32 screening evaluations (full, no caches; same column layout as the float64 ones)
    F32_LAUNCH = dict(soft_energy=(128, 4), barrier_energy=(128, 4))

    def soft_energy_f32(self, u, ref, base=None):
        """[jd, curl, stable soft difference] from the float32 kernel: every cell term and the block reduction in
        float32 (block partials cast to float64, summed in float64). A screening value only."""
        self._check(u, ref) if base is None else self._check(u, ref, base)
        block, warps = self.F32_LAUNCH['soft_energy']
        blocks = triton.cdiv(self.ncell, block)
        out = torch.empty((blocks, 3), dtype=torch.float64, device=self.device)
        KF.soft_energy_f32_kernel[(blocks,)](u, ref, u if base is None else base, self.k, out, self.ncell, self.ny, self.nx,
                                            self.cy, self.cx, WITH_BASE=base is not None, BLOCK=block, num_warps=warps)
        return out.sum(dim=0)

    def barrier_energy_f32(self, u, base=None):
        """[barrier value, min J, unweighted barrier change, min base J] from the float32 kernel (the E98 skip variant
        when the skip record is on and verified for this grid; same values either way). A screening value only."""
        self._check(u) if base is None else self._check(u, base)
        if self.bernstein:
            raise InputValidationError("the float32 barrier kernels support the closed27 samples barrier only")
        block, warps = self.F32_LAUNCH['barrier_energy']
        blocks = triton.cdiv(self.ncell, block)
        out = torch.empty((blocks, 4), dtype=torch.float64, device=self.device)
        name = 'barrier_energy_base' if base is not None else 'barrier_energy'
        kernel = (KF.barrier_energy_skip_f32_kernel if (self._skipping() and [name, self._grid_id] in self.skip["verified"])
                  else KF.barrier_energy_f32_kernel)
        kernel[(blocks,)](u, u if base is None else base, self.k, out, self.ncell, self.ny, self.nx, self.cy, self.cx,
                          WITH_BASE=base is not None, BLOCK=block, num_warps=warps)
        sums, mins = out[:, (0, 2)].sum(dim=0), out[:, (1, 3)].min(dim=0).values
        return torch.stack((sums[0], mins[0], sums[1], mins[1]))

    def barrier_energy(self, u, base=None):
        self._check(u) if base is None else self._check(u, base)
        if self.active is not None and self.active.enabled():
            got = self._barrier_energy_active(u, base)                # T89 (S30-G4)
            if got is not None:
                return got
        out = self._barrier_energy_out(u, base)
        if self.active is not None:
            self.active.store_template(('barrier', base is not None, out.shape[0]), u, None, base, out)
        sums, mins = out[:, (0, 2)].sum(dim=0), out[:, (1, 3)].min(dim=0).values
        # [barrier value, min J (min beta under T68), unweighted barrier change, min base J (base beta)]
        return torch.stack((sums[0], mins[0], sums[1], mins[1]))

    def _barrier_energy_out(self, u, base):
        """The block partials [value, min J, change, min base J] of the E51/E98 launches (the body of barrier_energy)."""
        block, warps = LAUNCH['barrier_energy']
        blocks = triton.cdiv(self.ncell, block)
        out = torch.empty((blocks, 4), dtype=torch.float64, device=self.device)
        kernel = K.barrier_energy_bern_kernel if self.bernstein else K.barrier_energy_kernel
        if self._skipping():
            self.skip["calls"] += 1

            def launch(k, o=None):
                o = torch.empty((blocks, 4), dtype=torch.float64, device=self.device) if o is None else o
                k[(blocks,)](u, u if base is None else base, self.k, o, self.ncell, self.ny, self.nx, self.cy, self.cx,
                             WITH_BASE=base is not None, BLOCK=block, num_warps=warps)
                return o
            act = self._act
            # active: the value branch ran -- some block's minimum trial J is below act, or its value sum or minimum
            # trial J is NaN (the value sum only receives contributions inside that branch); then, with a base, the
            # change branch ran too. A base J below act alone (incl. a NaN change) runs only the change branch: not
            # counted
            check = self._reference_first_skip if self.reference_first else self._checked       # E99: v17
            out = check('barrier_energy_base' if base is not None else 'barrier_energy',
                                lambda: launch(K.barrier_energy_skip_kernel, out), lambda: launch(kernel),
                                lambda r: bool(torch.isnan(r[:, :2]).any() or (r[:, 1] < act).any()))
        else:
            kernel[(blocks,)](u, u if base is None else base, self.k, out, self.ncell, self.ny, self.nx,
                              self.cy, self.cx, WITH_BASE=base is not None, BLOCK=block, num_warps=warps)
        return out

    def _barrier_energy_active(self, u, base):
        """T89: the block partials with the frozen blocks copied from a cached evaluation whose frozen entries equal
        u's (value and minimum J; the change is 0 and the base minimum is the same minimum), the active blocks by the
        index-list kernel (the E98 skip variant once that is verified on this grid); or None."""
        act = self.active
        part = act.record["energy"]
        part["calls"] += 1
        name = 'barrier_energy_base' if base is not None else 'barrier_energy'
        skipping = self._skipping()
        if skipping and [name, self._grid_id] not in self.skip["verified"]:
            part["misses"] += 1
            return None                          # the reference's own first-use check has not settled
        block, warps = LAUNCH['barrier_energy']
        blocks = triton.cdiv(self.ncell, block)
        cached = act.template(('barrier', base is not None, blocks), u, None, base)  # G4b: exact frozen-block outputs
        if cached is None:
            part["misses"] += 1
            return None
        out = cached.clone()
        blk = act.blocks[block]
        kernel = KI.barrier_energy_skip_idx_kernel if skipping else KI.barrier_energy_idx_kernel
        kernel[(int(blk.numel()),)](u, u if base is None else base, self.k, out, blk, self.ncell, self.ny, self.nx, self.cy, self.cx,
                                    WITH_BASE=base is not None, BLOCK=block, num_warps=warps)
        part["skipped"] += 1
        if not act.check('energy', name, out, lambda: self._barrier_energy_out(u, base)):
            return None
        sums, mins = out[:, (0, 2)].sum(dim=0), out[:, (1, 3)].min(dim=0).values
        return torch.stack((sums[0], mins[0], sums[1], mins[1]))
