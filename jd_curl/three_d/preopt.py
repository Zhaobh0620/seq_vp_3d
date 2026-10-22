"""T88 (S30-G3, explicit option): a float32 GPU pre-optimizer, an UNCERTIFIED initializer for the native level.

The pre-optimizer never produces a formal state: its output is a
node displacement field that the native transfer certifies (prepare_transfer_problem: whole-cell JD > 0 with the
alpha back-off) before the certified native solver starts. Everything reported as a result still comes from the
certified FP64 path.

Pre-objective (float32, device, sums without the volume factor; the weights are the level's mu = nu and alpha):
    P(u) = 1/2 sum_x (M(x + u(x)) - F(x))^2
         + mu/2 sum_cells [(J_c - 1)^2 + |curl_c|^2]           (Q1 cell-centre Jacobian/curl, as the soft geometry terms at identity)
         + alpha/2 sum_x |L u(x)|^2                            (seven-point Laplacian, physical spacing, T76's R)
         + kappa_pre sum_cells relu(j0 - J_c)^2                (soft positivity floor; the certificate decides later)
M is sampled trilinearly (torch grid_sample, align_corners) at x + u. The displacement is parametrised on the interior
nodes in voxel units (the six faces are zero). Steps: Adam moments of the Gaussian-smoothed gradient (sigma_grad
voxels), the update scaled so that the largest node move per iteration is at most `cap` voxels (FireANTs' half-voxel
rule). Levels 8/4/2/1 with the pyramid's own Gaussian+linear downsampling (nifti.downsample_array); the field is
carried to the next level by the pyramid's Q1 prolongation (multiresolution.prolong_displacement). The output is a
float64 field (exact conversion) with zero faces; the caller applies the P0c boundary-band taper and certifies.
"""
from dataclasses import dataclass, asdict
from time import perf_counter

import numpy as np

from ..validation import InputValidationError, finite_real, readonly_copy
from .validation import integer_at_least

PREOPT_POLICY = "fp32-preopt-v1"


@dataclass(frozen=True)
class PreoptConfig:
    iterations: tuple = (80, 80, 60, 40)          # per level 8/4/2/1
    lr: float = .25                               # Adam step in voxels (before the cap)
    cap: float = .5                               # max node move per iteration, voxels
    sigma_grad: float = 1.                        # Gaussian smoothing of the gradient, voxels (0 = off)
    sigma_field: float = 0.                       # S30-G3 r1: Gaussian smoothing of the field after every step, voxels (0 = off)
    mu: tuple = (1., 1., .1, .03)                 # geometry weights per level (the recommended NM hierarchy)
    alpha: tuple = (1e-3, 1e-3, 1e-3, 0.)         # curvature weights per level (NM)
    kappa_pre: float = 10.                        # soft positivity floor weight
    j0: float = .2                                # positivity floor on the cell-centre Jacobian
    seconds: float = 8.                           # wall cap for the whole pre-optimization
    beta1: float = .9
    beta2: float = .99
    eps: float = 1e-8

    def __post_init__(self):
        if len(self.iterations) != 4 or any(isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 0 for n in self.iterations):
            raise InputValidationError("preopt iterations: four nonnegative integers")
        object.__setattr__(self, "iterations", tuple(int(n) for n in self.iterations))
        for name in ("lr", "cap", "kappa_pre", "j0", "seconds", "beta1", "beta2", "eps"):
            object.__setattr__(self, name, finite_real(getattr(self, name), "preopt "+name, positive=True))
        object.__setattr__(self, "sigma_grad", finite_real(self.sigma_grad, "preopt sigma_grad"))
        object.__setattr__(self, "sigma_field", finite_real(self.sigma_field, "preopt sigma_field"))
        if self.sigma_field < 0:
            raise InputValidationError("preopt: sigma_field >= 0")
        if self.sigma_grad < 0 or not (self.beta1 < 1 and self.beta2 < 1) or self.j0 >= 1:
            raise InputValidationError("preopt: sigma_grad >= 0, betas < 1, j0 < 1")
        for name in ("mu", "alpha"):
            values = getattr(self, name)
            if len(values) != 4 or any(isinstance(v, bool) or not isinstance(v, (int, float, np.floating)) or not np.isfinite(v) or v < 0 for v in values):
                raise InputValidationError(f"preopt {name}: four finite nonnegative numbers")
            object.__setattr__(self, name, tuple(float(v) for v in values))
        if any(m <= 0 for m in self.mu):
            raise InputValidationError("preopt mu: positive")

# S30 PRE speed (2026-10-15): explicit execution switches (PreoptExecution; the
# default None runs the original loop verbatim, see _run_level_default): lean_sync (P-1), static_grid (P-2),
# cuda_graph (P-3). Each keeps every tensor operation, its order and its operands, so the field is bit for bit the
# one of the default path; the CUDA-graph path is checked bit for bit against an eager iteration at its first replay
# on every level and falls back on a mismatch (recorded).

PREOPT_EXECUTION_POLICY = "preopt-exec-v1"       # S30 PRE speed (P-1/P-2/P-3)
GRAPH_WARMUP_ITERATIONS = 3                        # eager iterations on the side stream before a level's capture


@dataclass(frozen=True)
class PreoptExecution:
    """S30 PRE speed, implementation only (explicit; None = the original loop verbatim).

    lean_sync (P-1): the per-iteration term floats (data, geometry, floor, curvature, min det, total) are read from the
      device only at the first and the last iteration of a level; the nonfinite check runs once per level on the last
      total and the field (same FloatingPointError, raised at the level's end instead of mid-level). Needs no observer.
    static_grid (P-2): a level's identity voxel grid (the three meshgrid tensors) is built once per level instead of
      once per iteration; the ops that use it are unchanged.
    cuda_graph (P-3): after GRAPH_WARMUP_ITERATIONS eager iterations on a side stream, a level's iteration body is
      captured into two CUDA graphs around the host-side cap (A: energy, gradient, smoothing, Adam step and the cap
      reduction; B: the scaled update and the field smoothing); the per-iteration host scalars (the two Adam bias
      corrections as float32 reciprocals, the cap scale) enter through pinned buffers so the captured kernels compute
      exactly what the eager ops compute (PyTorch divides a float32 tensor by a CPU scalar as a multiplication by the
      float32 reciprocal). The first replay of every level is compared bit for bit (u, m, v) with an eager iteration
      on a copy of the state; a mismatch, a capture error or a level with fewer than GRAPH_WARMUP_ITERATIONS + 2
      iterations keeps that level eager, and a mismatch disables the graphs for the rest of the run (recorded).
      Requires lean_sync, static_grid and a CUDA device.
    parallel_setup (S30 P-12/P-6: P-12, bit for bit): the three coarse levels' downsampling of the fixed and moving images
      (nifti.downsample_array, the same calls) and the native images' float32 conversion run in a thread pool started
      with the pre-optimization; each level takes its results when it starts (the waits are recorded).
    analytic_grad (S30 P-12/P-6: P-6, ROUNDING LEVEL): the energy and the gradient of every iteration come from
      preopt_grad (Triton on CUDA, the torch reference elsewhere) instead of torch.autograd; the first iteration of
      every level is checked against autograd (contract §3.5) and a failed check returns the run to autograd. Not with
      lean_sync, static_grid or cuda_graph.
    """
    lean_sync: bool = False
    static_grid: bool = False
    cuda_graph: bool = False
    parallel_setup: bool = False
    analytic_grad: bool = False

    def __post_init__(self):
        for name in ("lean_sync", "static_grid", "cuda_graph", "parallel_setup", "analytic_grad"):
            if not isinstance(getattr(self, name), bool):
                raise InputValidationError(f"preopt execution {name}: boolean")
        if self.cuda_graph and not (self.lean_sync and self.static_grid):
            raise InputValidationError("preopt execution cuda_graph requires lean_sync and static_grid")
        if self.analytic_grad and (self.lean_sync or self.static_grid or self.cuda_graph):
            raise InputValidationError("preopt execution analytic_grad is not combined with lean_sync, static_grid or cuda_graph")

    @property
    def any(self):
        return self.lean_sync or self.static_grid or self.cuda_graph or self.parallel_setup or self.analytic_grad

    @property
    def per_iteration(self):
        """The P-1/P-2/P-3 loop (_run_level_exec) is needed."""
        return self.lean_sync or self.static_grid or self.cuda_graph


def _gaussian_1d(sigma, device, dtype):
    import torch
    r = int(np.ceil(3*sigma)); x = torch.arange(-r, r+1, device=device, dtype=dtype)
    k = torch.exp(-.5*(x/sigma)**2); return k/k.sum()


def _smooth(field, sigma):
    """Separable Gaussian smoothing of a (C, D, H, W) tensor along the three spatial axes (replicate padding)."""
    import torch
    import torch.nn.functional as Fn
    if sigma <= 0:
        return field
    k = _gaussian_1d(sigma, field.device, field.dtype); r = (k.numel()-1)//2
    c = field.shape[0]; out = field[None]
    for axis in range(3):
        shape = [1, 1, 1, 1, 1]; shape[2+axis] = k.numel()
        w = k.view(shape).expand(c, 1, *shape[2:]).contiguous()
        pad = [0, 0, 0, 0, 0, 0]; pad[2*(2-axis)] = pad[2*(2-axis)+1] = r
        out = Fn.conv3d(Fn.pad(out, pad, mode="replicate"), w, groups=c)
    return out[0]


def _identity_mesh(D, H, W, device, dtype):
    """The three meshgrid tensors (z, y, x) of a level, as the per-iteration code built them."""
    import torch
    return torch.meshgrid(torch.arange(D, device=device, dtype=dtype), torch.arange(H, device=device, dtype=dtype),
                          torch.arange(W, device=device, dtype=dtype), indexing="ij")


def _pre_energy_tensors(u_int, fixed, moving, spacing, mu, alpha, kappa_pre, j0, mesh=None, hvec=None):
    """Pre-objective of the interior displacement u_int (3, D-2, H-2, W-2) in voxel units: the tensors
    (total, det, data, geometry, floor, curvature). mesh = the level's identity meshgrid (P-2) or None (built here,
    the original per-iteration path); hvec = the level's spacing vector tensor (P-3: built once per level, a host
    copy is not capturable) or None (built here as before); the same operations either way."""
    import torch
    import torch.nn.functional as Fn
    u = Fn.pad(u_int, (1, 1, 1, 1, 1, 1))                           # zero faces -> (3, D, H, W)
    D, H, W = fixed.shape
    hz, hy, hx = spacing[2], spacing[1], spacing[0]
    # data term: sample M at x + u, grid_sample wants normalised (x, y, z) in [-1, 1] with align_corners
    zz, yy, xx = _identity_mesh(D, H, W, u.device, u.dtype) if mesh is None else mesh
    gx = 2*(xx+u[0])/max(W-1, 1)-1; gy = 2*(yy+u[1])/max(H-1, 1)-1; gz = 2*(zz+u[2])/max(D-1, 1)-1
    grid = torch.stack((gx, gy, gz), dim=-1)[None]
    warped = Fn.grid_sample(moving[None, None], grid, mode="bilinear", padding_mode="border", align_corners=True)[0, 0]
    data = .5*((warped-fixed)**2).sum()
    # cell-centre Q1 Jacobian: d u_c / d x_d (voxel units, times h_c/h_d for physical units) averaged over the cell's edges
    def d(axis):
        g = torch.diff(u, dim=1+axis)
        for ax in range(3):
            if ax != axis:
                g = .5*(g.narrow(1+ax, 0, g.shape[1+ax]-1)+g.narrow(1+ax, 1, g.shape[1+ax]-1))
        return g                                                       # (3, D-1, H-1, W-1): components x, y, z
    gz_, gy_, gx_ = d(0), d(1), d(2)                                   # derivatives along z, y, x
    h = (hx, hy, hz)
    def J(c, dd):                                                      # d u_c / d x_dd, physical
        g = (gx_, gy_, gz_)[dd][c]
        return g*(h[c]/h[dd])
    F00 = 1+J(0, 0); F01 = J(0, 1); F02 = J(0, 2)
    F10 = J(1, 0); F11 = 1+J(1, 1); F12 = J(1, 2)
    F20 = J(2, 0); F21 = J(2, 1); F22 = 1+J(2, 2)
    det = F00*(F11*F22-F12*F21)-F01*(F10*F22-F12*F20)+F02*(F10*F21-F11*F20)
    curl = torch.stack((F21-F12, F02-F20, F10-F01))
    geometry = .5*mu*(((det-1)**2).sum()+(curl**2).sum())
    floor = kappa_pre*(torch.relu(j0-det)**2).sum()
    curvature = torch.zeros((), device=u.device, dtype=u.dtype)
    if alpha > 0:
        # seven-point Laplacian of u (mm) on the interior nodes: u_mm,c = h_c u_vox,c; neighbours over the zero faces
        ui = u[:, 1:-1, 1:-1, 1:-1]
        lap = ((u[:, 2:, 1:-1, 1:-1]-2*ui+u[:, :-2, 1:-1, 1:-1])/(hz*hz)
               + (u[:, 1:-1, 2:, 1:-1]-2*ui+u[:, 1:-1, :-2, 1:-1])/(hy*hy)
               + (u[:, 1:-1, 1:-1, 2:]-2*ui+u[:, 1:-1, 1:-1, :-2])/(hx*hx))
        lap = lap*(torch.tensor([hx, hy, hz], device=u.device, dtype=u.dtype) if hvec is None else hvec)[:, None, None, None]
        curvature = .5*alpha*(lap**2).sum()
    total = data+geometry+floor+curvature
    return total, det, data, geometry, floor, curvature


def _terms(det, data, geometry, floor, curvature):
    """The per-iteration record floats of the original path (one device read each)."""
    return dict(data=float(data), geometry=float(geometry), floor=float(floor), curvature=float(curvature), min_det=float(det.min()))


def _pre_energy(u_int, fixed, moving, spacing, mu, alpha, kappa_pre, j0):
    """Pre-objective terms for the interior displacement u_int (3, D-2, H-2, W-2) in voxel units (the original
    interface: the total tensor and the record floats)."""
    total, det, data, geometry, floor, curvature = _pre_energy_tensors(u_int, fixed, moving, spacing, mu, alpha, kappa_pre, j0)
    return total, _terms(det, data, geometry, floor, curvature)


def _run_level_default(u_int, m, v, F, M, h, level, config, started, observer, counters):
    """The original iteration loop of a level (verbatim; the default path). Returns (first, last, done, stopped)."""
    import torch
    first = last = None; done = 0; stopped = None
    for it in range(config.iterations[level]):
        if perf_counter()-started >= config.seconds:
            stopped = dict(level=level, iteration=it); break
        done = it+1
        total, terms = _pre_energy(u_int, F, M, h, config.mu[level], config.alpha[level], config.kappa_pre, config.j0)
        if first is None:
            first = terms
        last = dict(terms, total=float(total))
        if not np.isfinite(float(total)):
            raise FloatingPointError("preopt: nonfinite pre-objective")
        grad, = torch.autograd.grad(total, u_int)
        with torch.no_grad():
            g = _smooth(grad, config.sigma_grad)
            m.mul_(config.beta1).add_(g, alpha=1-config.beta1); v.mul_(config.beta2).addcmul_(g, g, value=1-config.beta2)
            t = it+1
            step = config.lr*(m/(1-config.beta1**t))/((v/(1-config.beta2**t)).sqrt()+config.eps)
            largest = float(step.abs().amax())
            scale = 1. if largest <= config.cap else config.cap/largest
            counters["cap_hits"] += scale < 1.
            u_int.sub_(step*scale)
            if config.sigma_field > 0:
                # S30-G3 r1: elastic smoothing of the field (FireANTs' smooth_warp_sigma); it removes the node-level
                # (hourglass) modes that the cell-centre terms cannot see
                u_int.copy_(_smooth(u_int, config.sigma_field))
        if observer is not None:
            observer(dict(level=level, iteration=it, **last))
    return first, last, done, stopped


def _f32_reciprocal(x):
    """float32(1) / float32(x): the reciprocal PyTorch's CUDA division by a CPU scalar multiplies with."""
    return np.float32(1.)/np.float32(x)


class _LevelGraphs:
    """P-3: a level's iteration body as two CUDA graphs around the host-side cap (see PreoptExecution)."""

    def __init__(self, u_int, m, v, F, M, h, mu, alpha, config, mesh, hvec):
        import torch
        self.torch = torch
        self.u_int, self.m, self.v = u_int, m, v
        dev = u_int.device
        self.inv1 = torch.zeros((), device=dev, dtype=torch.float32); self.inv2 = torch.zeros((), device=dev, dtype=torch.float32)
        self.scale = torch.zeros((), device=dev, dtype=torch.float32)
        self.inv1_h = torch.zeros((), dtype=torch.float32).pin_memory(); self.inv2_h = torch.zeros((), dtype=torch.float32).pin_memory()
        self.scale_h = torch.zeros((), dtype=torch.float32).pin_memory()
        pool = torch.cuda.graph_pool_handle()
        self.graph_a = torch.cuda.CUDAGraph(); self.graph_b = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph_a, pool=pool):
            self.inv1.copy_(self.inv1_h, non_blocking=True); self.inv2.copy_(self.inv2_h, non_blocking=True)
            total, det, data, geometry, floor, curvature = _pre_energy_tensors(u_int, F, M, h, mu, alpha, config.kappa_pre, config.j0, mesh, hvec)
            grad, = torch.autograd.grad(total, u_int)
            with torch.no_grad():
                g = _smooth(grad, config.sigma_grad)
                m.mul_(config.beta1).add_(g, alpha=1-config.beta1); v.mul_(config.beta2).addcmul_(g, g, value=1-config.beta2)
                step = config.lr*(m*self.inv1)/((v*self.inv2).sqrt()+config.eps)
                largest = step.abs().amax()
                det_min = det.min()
        self.step, self.largest = step, largest
        self.outputs = (total, det_min, data, geometry, floor, curvature)
        with torch.cuda.graph(self.graph_b, pool=pool):
            self.scale.copy_(self.scale_h, non_blocking=True)
            with torch.no_grad():
                u_int.sub_(self.step*self.scale)
                if config.sigma_field > 0:
                    u_int.copy_(_smooth(u_int, config.sigma_field))

    def run_a(self, t, beta1, beta2):
        self.inv1_h.fill_(float(_f32_reciprocal(1-beta1**t))); self.inv2_h.fill_(float(_f32_reciprocal(1-beta2**t)))
        self.graph_a.replay()
        return float(self.largest)                       # the one host read of the iteration (the cap)

    def run_b(self, scale):
        self.scale_h.fill_(float(np.float32(scale)))
        self.graph_b.replay()

    def terms(self):
        total, det_min, data, geometry, floor, curvature = self.outputs
        return float(total), dict(data=float(data), geometry=float(geometry), floor=float(floor), curvature=float(curvature), min_det=float(det_min))


def _eager_iteration(u_int, m, v, F, M, h, mu, alpha, config, mesh, hvec, t):
    """One iteration of the lean/static path (same ops as the default loop; the term floats are not read).
    Returns (outputs, scale) with outputs = (total, det, data, geometry, floor, curvature) tensors."""
    import torch
    total, det, data, geometry, floor, curvature = _pre_energy_tensors(u_int, F, M, h, mu, alpha, config.kappa_pre, config.j0, mesh, hvec)
    grad, = torch.autograd.grad(total, u_int)
    with torch.no_grad():
        g = _smooth(grad, config.sigma_grad)
        m.mul_(config.beta1).add_(g, alpha=1-config.beta1); v.mul_(config.beta2).addcmul_(g, g, value=1-config.beta2)
        step = config.lr*(m/(1-config.beta1**t))/((v/(1-config.beta2**t)).sqrt()+config.eps)
        largest = float(step.abs().amax())
        scale = 1. if largest <= config.cap else config.cap/largest
        u_int.sub_(step*scale)
        if config.sigma_field > 0:
            u_int.copy_(_smooth(u_int, config.sigma_field))
    return (total, det, data, geometry, floor, curvature), scale


def _run_level_exec(u_int, m, v, F, M, h, level, config, started, execution, counters, record):
    """The lean/static/graph loop of a level (P-1/P-2/P-3). Returns (first, last, done, stopped)."""
    import torch
    D, H, W = F.shape
    mesh = _identity_mesh(D, H, W, u_int.device, u_int.dtype) if execution.static_grid else None
    hvec = torch.tensor([h[0], h[1], h[2]], device=u_int.device, dtype=u_int.dtype) if execution.static_grid else None
    mu, alpha = config.mu[level], config.alpha[level]
    first = last = None; done = 0; stopped = None
    outputs = None; graphs = None
    use_graph = execution.cuda_graph and not record["graphs"]["disabled"] and u_int.is_cuda \
        and config.iterations[level] >= GRAPH_WARMUP_ITERATIONS+2
    side = torch.cuda.Stream() if use_graph else None
    for it in range(config.iterations[level]):
        if perf_counter()-started >= config.seconds:
            stopped = dict(level=level, iteration=it); break
        done = it+1; t = it+1
        if use_graph and it < GRAPH_WARMUP_ITERATIONS:
            # warm-up on a side stream (the same eager ops; a stream does not change a kernel's result)
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                outputs, scale = _eager_iteration(u_int, m, v, F, M, h, mu, alpha, config, mesh, hvec, t)
            torch.cuda.current_stream().wait_stream(side)              # the terms/record reads below use the current stream
            counters["cap_hits"] += scale < 1.
            if it == GRAPH_WARMUP_ITERATIONS-1:
                try:
                    graphs = _LevelGraphs(u_int, m, v, F, M, h, mu, alpha, config, mesh, hvec)
                    record["graphs"]["levels_captured"].append(level)
                except Exception as exc:                                     # capture failed: the level stays eager
                    torch.cuda.synchronize()
                    graphs = None; use_graph = False
                    record["graphs"]["capture_errors"].append(dict(level=level, error=f"{type(exc).__name__}: {exc}"[:300]))
        elif graphs is not None:
            if it == GRAPH_WARMUP_ITERATIONS:
                # first use: the first replay against an eager iteration on a copy of the state, bit for bit
                u_c, m_c, v_c = u_int.detach().clone().requires_grad_(True), m.clone(), v.clone()
                outs_c, scale_c = _eager_iteration(u_c, m_c, v_c, F, M, h, mu, alpha, config, mesh, hvec, t)
                largest = graphs.run_a(t, config.beta1, config.beta2)
                scale = 1. if largest <= config.cap else config.cap/largest
                graphs.run_b(scale)
                equal = bool(torch.equal(u_int.detach(), u_c.detach()) and torch.equal(m, m_c) and torch.equal(v, v_c)) and scale == scale_c
                record["graphs"]["first_use"].append(dict(level=level, iteration=it, equal=equal))
                if not equal:
                    # the eager result is the path of record: restore it and run the rest eagerly
                    with torch.no_grad():
                        u_int.copy_(u_c); m.copy_(m_c); v.copy_(v_c)
                    record["graphs"]["disabled"] = dict(level=level, iteration=it, reason="first_use_mismatch")
                    graphs = None; use_graph = False
                    scale = scale_c
                    outputs = outs_c                                  # the eager terms of this iteration
                else:
                    outputs = graphs
            else:
                largest = graphs.run_a(t, config.beta1, config.beta2)
                scale = 1. if largest <= config.cap else config.cap/largest
                graphs.run_b(scale)
                outputs = graphs
            counters["cap_hits"] += scale < 1.
        else:
            outputs, scale = _eager_iteration(u_int, m, v, F, M, h, mu, alpha, config, mesh, hvec, t)
            counters["cap_hits"] += scale < 1.
        if it == 0 or not execution.lean_sync:
            total_f, terms = (outputs.terms() if isinstance(outputs, _LevelGraphs) else (float(outputs[0]), _terms(*outputs[1:])))
            if first is None:
                first = terms
            last = dict(terms, total=total_f)
            if not np.isfinite(total_f):
                raise FloatingPointError("preopt: nonfinite pre-objective")
    if done and execution.lean_sync:
        total_f, terms = (outputs.terms() if isinstance(outputs, _LevelGraphs) else (float(outputs[0]), _terms(*outputs[1:])))
        last = dict(terms, total=total_f)
        if first is None:
            first = terms
        if not np.isfinite(total_f) or not bool(torch.isfinite(u_int).all()):
            raise FloatingPointError("preopt: nonfinite pre-objective")
    return first, last, done, stopped


PARALLEL_SETUP_WORKERS = 7      # S30 P-12/P-6: six coarse downsamplings and the native float32 conversion


def _native_float32(fixed_image, moving_image):
    """P-12: the native images as float32 arrays (IEEE round-to-nearest, as torch.tensor(float64, dtype=float32))."""
    with np.errstate(over="ignore"):                                 # review M5: |x| > FLT_MAX -> inf, as torch (silent)
        return (np.asarray(fixed_image, dtype=np.float64).astype(np.float32),
                np.asarray(moving_image, dtype=np.float64).astype(np.float32))


def _run_level_analytic(u_int, m, v, F, M, h, level, config, started, observer, counters, analytic):
    """S30 P-12/P-6 (P-6): the default loop with the energy and the gradient from the analytic path (the first iteration
    of every level checked against autograd by `analytic`); the smoothing, Adam, cap and update are unchanged."""
    import torch
    first = last = None; done = 0; stopped = None
    mu, alpha = config.mu[level], config.alpha[level]

    def autograd_step():
        total, det, data, geometry, floor, curvature = _pre_energy_tensors(u_int, F, M, h, mu, alpha, config.kappa_pre, config.j0)
        grad, = torch.autograd.grad(total, u_int)
        return total, det.min(), data, geometry, floor, curvature, grad

    for it in range(config.iterations[level]):
        if perf_counter()-started >= config.seconds:
            stopped = dict(level=level, iteration=it); break
        done = it+1
        total, det_min, data, geometry, floor, curvature, grad = analytic(level, u_int, F, M, h, mu, alpha, config.kappa_pre,
                                                                          config.j0, autograd_step)
        terms = dict(data=float(data), geometry=float(geometry), floor=float(floor), curvature=float(curvature), min_det=float(det_min))
        if first is None:
            first = terms
        last = dict(terms, total=float(total))
        if not np.isfinite(float(total)):
            raise FloatingPointError("preopt: nonfinite pre-objective")
        with torch.no_grad():
            g = _smooth(grad, config.sigma_grad)
            m.mul_(config.beta1).add_(g, alpha=1-config.beta1); v.mul_(config.beta2).addcmul_(g, g, value=1-config.beta2)
            t = it+1
            step = config.lr*(m/(1-config.beta1**t))/((v/(1-config.beta2**t)).sqrt()+config.eps)
            largest = float(step.abs().amax())
            scale = 1. if largest <= config.cap else config.cap/largest
            counters["cap_hits"] += scale < 1.
            u_int.sub_(step*scale)
            if config.sigma_field > 0:
                u_int.copy_(_smooth(u_int, config.sigma_field))
        if observer is not None:
            observer(dict(level=level, iteration=it, **last))
    return first, last, done, stopped


def _conv3d_backends(torch, dev, sigma):
    """The ConvBackend PyTorch selects for _smooth's three conv3d calls (one per axis) at this sigma: C = 3 channels,
    groups = 3 (depthwise), float32, contiguous 5-D input. torch._C._select_conv_backend reads metadata only (no conv
    is run); small tensors of the same layout stand in for the field (the CUDA test checks native-size shapes too)."""
    k = _gaussian_1d(sigma, dev, torch.float32)
    n = k.numel()
    names = []
    for axis in range(3):
        shape = [1, 1, 1, 1, 1]
        shape[2+axis] = n
        w = k.view(shape).expand(3, 1, *shape[2:]).contiguous()
        x = torch.empty((1, 3, n+2, n+2, n+2), device=dev, dtype=torch.float32)      # metadata only (no launch)
        names.append(torch._C._select_conv_backend(x, w, None, [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 3).name)
    return names


def float32_precision_record(torch, dev, config=None):
    """TF32 (record only -- behaviour unchanged, the actual values
    recorded, the paper describes them as they are). The float32 pre-optimizer's only TF32-eligible operation is the
    Gaussian-smoothing conv3d in _smooth (no matmul, einsum or tl.dot). TF32 is possible there only if PyTorch dispatches
    that conv to cuDNN AND the cuDNN TF32 switch is on AND the device has TF32 tensor cores (sm_80+). Recorded: the
    process-wide switches (read when the pre-optimizer finishes; nothing is set here), the device capability, and the
    backend PyTorch selects for each smoothing sigma in use (review 10-18: a contiguous 5-D depthwise float32 conv may
    go to the native CudaDepthwise3d kernel, plain FP32, whatever the switches say). conv3d_may_use_tf32: True only for
    a cuDNN backend with the switches on and a capable device; None if a backend could not be determined. The output is
    an uncertified initializer; the native transfer certifies it in float64."""
    cudnn_ok = bool(torch.backends.cudnn.is_available())
    try:
        version = int(torch.backends.cudnn.version()) if cudnn_ok else None
    except Exception as error:                                          # review 10-18: record, never fail the run
        version = "unknown: "+type(error).__name__
    rec = dict(cudnn_available=cudnn_ok, cudnn_version=version,
               cudnn_enabled=bool(torch.backends.cudnn.enabled), cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32),
               cuda_matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
               float32_matmul_precision=str(torch.get_float32_matmul_precision()), torch_version=str(torch.__version__))
    capable = False
    if dev.type == "cuda":
        major, minor = torch.cuda.get_device_capability(dev)
        capable = major >= 8                                            # TF32 tensor cores: Ampere (sm_80) and later
        rec.update(device_capability=[int(major), int(minor)])
    rec["tf32_capable_device"] = capable
    backends = {}
    for name in ("sigma_grad", "sigma_field"):
        sigma = None if config is None else float(getattr(config, name))
        if sigma is not None and sigma > 0:
            try:
                backends[name] = _conv3d_backends(torch, dev, sigma)
            except Exception as error:                                  # record, never fail the pre-optimizer
                backends[name] = "unknown: "+type(error).__name__
    rec["conv3d_backends"] = backends
    if config is None or any(isinstance(v, str) for v in backends.values()):
        may = None                                                      # not determined (no config, or a lookup failed)
    else:
        cudnn_conv = any(b.startswith("Cudnn") for v in backends.values() for b in v)
        may = bool(cudnn_conv and rec["cudnn_enabled"] and rec["cudnn_allow_tf32"] and capable)
    rec["conv3d_may_use_tf32"] = may
    return rec


def preoptimize(fixed_image, moving_image, grid, config=None, device="cuda", observer=None, execution=None):
    """Return (u float64 (D, H, W, 3) in mm with zero faces, record). Uncertified; the caller certifies.
    execution: None (the original loop) or a PreoptExecution (S30 PRE speed; bit for bit the same field)."""
    import torch
    from .multiresolution import four_level_grids, prolong_displacement
    from .nifti import downsample_array
    config = PreoptConfig() if config is None else config
    if not isinstance(config, PreoptConfig):
        raise InputValidationError("preopt config: PreoptConfig required")
    if execution is not None:
        if not isinstance(execution, PreoptExecution):
            raise InputValidationError("preopt execution: PreoptExecution required")
        if not execution.any:
            execution = None
        elif execution.lean_sync and observer is not None:
            raise InputValidationError("preopt execution lean_sync cannot serve an observer (per-iteration terms)")
    started = perf_counter()
    grids = four_level_grids(grid)
    dev = torch.device(device); dtype = torch.float32
    if execution is not None and execution.cuda_graph and dev.type != "cuda":
        raise InputValidationError("preopt execution cuda_graph requires a CUDA device")
    u_mm = None; levels = []; counters = dict(cap_hits=0); stopped = None
    # S30 P-12/P-6: the record keeps the v31/v32 keys (and order); the new switches appear only when set
    exec_record = None if execution is None else dict(policy=PREOPT_EXECUTION_POLICY, lean_sync=execution.lean_sync,
                                                      static_grid=execution.static_grid, cuda_graph=execution.cuda_graph,
                                                      graphs=dict(levels_captured=[], first_use=[], capture_errors=[], disabled=False))
    analytic = None
    if execution is not None and execution.analytic_grad:
        from .preopt_grad import AnalyticPreGradient
        analytic = AnalyticPreGradient(dev)
        exec_record["analytic_grad"] = analytic.record
    pool = setup = None
    if execution is not None and execution.parallel_setup:
        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(max_workers=PARALLEL_SETUP_WORKERS, thread_name_prefix="preopt-setup")
        setup = {("native",): pool.submit(_native_float32, fixed_image, moving_image)}
        for lv in range(3):
            setup[(lv, "fixed")] = pool.submit(downsample_array, fixed_image, grid, grids[lv].shape)
            setup[(lv, "moving")] = pool.submit(downsample_array, moving_image, grid, grids[lv].shape)
        exec_record["parallel_setup"] = dict(workers=PARALLEL_SETUP_WORKERS, wait_seconds=[])
    try:
        return _preoptimize_levels(fixed_image, moving_image, grid, grids, config, dev, dtype, observer, execution,
                                   exec_record, analytic, setup, started, u_mm, levels, counters, stopped)
    finally:
        if pool is not None:
            # review M3: wait for running downsamplings (after an error or a coarse-level stop) so that no CPU work
            # outlives the pre-optimization; on the normal path every future has already been consumed
            pool.shutdown(wait=True, cancel_futures=True)


def _preoptimize_levels(fixed_image, moving_image, grid, grids, config, dev, dtype, observer, execution, exec_record,
                        analytic, setup, started, u_mm, levels, counters, stopped):
    """The level loop of preoptimize (moved here unchanged by S30 P-12/P-6 so that the setup pool is shut down)."""
    import torch
    from .multiresolution import prolong_displacement
    from .nifti import downsample_array
    for level, level_grid in enumerate(grids):
        tick = perf_counter()
        native32 = None
        if setup is not None:
            tick_wait = perf_counter()
            if level == 3:
                native32 = setup[("native",)].result()
            else:
                fixed, gf, _ = setup[(level, "fixed")].result()
                moving, gm, _ = setup[(level, "moving")].result()
                if gf.grid_id != level_grid.grid_id or gm.grid_id != level_grid.grid_id:
                    raise InputValidationError("preopt: pyramid grids differ")
            exec_record["parallel_setup"]["wait_seconds"].append(perf_counter()-tick_wait)
        elif level == 3:
            fixed, moving = np.asarray(fixed_image, dtype=np.float64), np.asarray(moving_image, dtype=np.float64)
        else:
            fixed, gf, _ = downsample_array(fixed_image, grid, level_grid.shape)
            moving, gm, _ = downsample_array(moving_image, grid, level_grid.shape)
            if gf.grid_id != level_grid.grid_id or gm.grid_id != level_grid.grid_id:
                raise InputValidationError("preopt: pyramid grids differ")
        h = tuple(float(s) for s in level_grid.spacing)                 # (hx, hy, hz)
        seed = np.zeros(level_grid.shape+(3,)) if u_mm is None else prolong_displacement(u_mm, grids[level-1], level_grid)
        # voxel units, interior, (3, D-2, H-2, W-2)
        seed_vox = np.moveaxis(seed[1:-1, 1:-1, 1:-1], -1, 0)/np.array(h)[:, None, None, None]
        u_int = torch.tensor(seed_vox, device=dev, dtype=dtype, requires_grad=True)
        if native32 is not None:                                     # P-12: the same float32 values, converted off-thread
            F = torch.from_numpy(native32[0]).to(dev); M = torch.from_numpy(native32[1]).to(dev)
        else:
            F = torch.tensor(fixed, device=dev, dtype=dtype); M = torch.tensor(moving, device=dev, dtype=dtype)
        m = torch.zeros_like(u_int); v = torch.zeros_like(u_int)
        if execution is None or not (execution.per_iteration or execution.analytic_grad):
            first, last, done, stopped = _run_level_default(u_int, m, v, F, M, h, level, config, started, observer, counters)
        elif execution.analytic_grad:
            first, last, done, stopped = _run_level_analytic(u_int, m, v, F, M, h, level, config, started, observer, counters, analytic)
        else:
            first, last, done, stopped = _run_level_exec(u_int, m, v, F, M, h, level, config, started, execution, counters, exec_record)
        with torch.no_grad():
            out = np.zeros(level_grid.shape+(3,))
            out[1:-1, 1:-1, 1:-1] = np.moveaxis(u_int.detach().double().cpu().numpy(), 0, -1)*np.array(h)
        u_mm = out
        levels.append(dict(level=level, shrink_factor=(8, 4, 2, 1)[level], shape_zyx=list(level_grid.shape), iterations=done,
                           seconds=perf_counter()-tick, first=first, last=last))
        if stopped is not None:
            break
    if u_mm.shape != grid.shape+(3,):                                  # time cap on a coarse level: carry to native
        for level in range(levels[-1]["level"]+1, 4):
            u_mm = prolong_displacement(u_mm, grids[level-1], grids[level])
    record = dict(policy=PREOPT_POLICY, config=asdict(config), device=str(dev), dtype="float32", levels=levels,
                  cap_hits=int(counters["cap_hits"]), stopped=stopped, seconds=perf_counter()-started,
                  role="uncertified_preoptimizer_initializer_NOT_outer_commit",
                  **({} if exec_record is None else dict(execution=exec_record)),
                  float32_precision=float32_precision_record(torch, dev, config))   # TF32 record (10-18; record only)
    return readonly_copy(np.ascontiguousarray(u_mm)), record


def boundary_taper(u, grid, L=4):
    """P0c (I2, 2026-10-13): nodes within L of a face scale linearly to zero on the face (order preserving); nodes at
    least L from every face keep u. Returns (tapered field float64, weight field)."""
    rs = []
    for n in grid.shape:
        k = np.arange(n); rs.append(np.minimum(np.minimum(k, n-1-k)/float(L), 1.))
    w = rs[0][:, None, None]*rs[1][None, :, None]*rs[2][None, None, :]
    return readonly_copy(w[..., None]*np.asarray(u, dtype=np.float64)), w
