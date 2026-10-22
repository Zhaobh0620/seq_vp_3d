"""Explicit float64 tiled Q1/Spline/GGN/DST-I device arithmetic for 3D-D.

The CPU remains authoritative for certification and nonlinear acceptance. This
module does NOT evaluate uncertified trial maps. No whole-volume Gauss cache.
"""
from dataclasses import dataclass

import numpy as np
import torch

from ...validation import InputValidationError, StateMismatchError
from ..linearization import build_linearization
from ..poisson import build_poisson_inverse
from ..residuals import QUADRATURE_ID
from ..work_control import checkpoint
from ..validation import integer_at_least
from .spline import TorchSpline
from .selective_kernels import geometry_selected, barrier_selected


def laplacian_nodes(field, inv):
    """T76 device mirror of curvature.laplacian_nodes: seven-point Laplacian of a zero-boundary
    node field with inv = (1/hz^2, 1/hy^2, 1/hx^2); zero on the boundary nodes; same axis order."""
    iz, iy, ix = inv
    out = torch.zeros_like(field)
    c = field[1:-1, 1:-1, 1:-1]
    out[1:-1, 1:-1, 1:-1] = ((field[1:-1, 1:-1, 2:]-2.*c+field[1:-1, 1:-1, :-2])*ix
                             + (field[1:-1, 2:, 1:-1]-2.*c+field[1:-1, :-2, 1:-1])*iy
                             + (field[2:, 1:-1, 1:-1]-2.*c+field[:-2, 1:-1, 1:-1])*iz)
    return out


@dataclass
class Transfers:
    upload_calls: int = 0
    upload_bytes: int = 0
    download_calls: int = 0
    download_bytes: int = 0
    scalar_reads: int = 0


class Device:
    def __init__(self, name):
        requested = torch.device(name)
        if requested.type not in ("cpu", "cuda"):
            raise InputValidationError("only explicit cpu/cuda devices supported")
        # Allocation must fail if CUDA is unavailable: never silently fall back.
        self.device = torch.empty(0, device=requested, dtype=torch.float64).device
        self.transfers = Transfers()

    lean_host = False   # E76 (explicit): skip host-side copies that never change a value

    def upload(self, array):
        if (self.lean_host and self.device.type == "cuda" and isinstance(array, np.ndarray)
                and array.dtype == np.float64 and array.ndim > 0 and array.flags.c_contiguous):
            pass                                       # torch.tensor copies it to the device anyway (same strides)
        else:
            array = np.array(array, copy=True)
        self.transfers.upload_calls += 1
        self.transfers.upload_bytes += array.nbytes
        return torch.tensor(array, device=self.device, dtype=torch.float64)

    def check(self, x, shape=None):
        if (not isinstance(x, torch.Tensor) or x.dtype != torch.float64 or x.device != self.device
                or x.requires_grad or (shape is not None and tuple(x.shape) != tuple(shape))):
            raise InputValidationError("device/float64/shape/autograd mismatch")

    def download(self, tensor):
        self.check(tensor)
        self.transfers.download_calls += 1
        self.transfers.download_bytes += tensor.numel()*tensor.element_size()
        if self.lean_host and tensor.is_cuda:
            host = tensor.detach().cpu().numpy()       # .cpu() of a CUDA tensor is already a fresh host copy
            return host if host.flags.c_contiguous else host.copy()
        return tensor.detach().cpu().numpy().copy()

    def download_owned(self, tensor):
        """E89: readonly_copy(download(tensor)) without the second copy: the bytes go straight into a
        fresh NumPy-owned C-contiguous float64 array, which is then frozen. The temporary torch view
        of that buffer is dropped before returning, so no writeable alias survives."""
        self.check(tensor)
        self.transfers.download_calls += 1
        self.transfers.download_bytes += tensor.numel()*tensor.element_size()
        host = np.empty(tuple(tensor.shape), dtype=np.float64)
        torch.from_numpy(host).copy_(tensor.detach())   # synchronous (non_blocking=False)
        host.flags.writeable = False
        return host

    def scalar(self, tensor):
        self.transfers.scalar_reads += 1
        return float(tensor.item())

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)


def dst1(x, axis):
    n = x.shape[axis]
    shape = list(x.shape)
    shape[axis] = 1
    zero = torch.zeros(shape, dtype=x.dtype, device=x.device)
    extension = torch.cat((zero, x, zero, -x.flip((axis,))), dim=axis)
    return -torch.fft.fft(extension, dim=axis).imag.narrow(axis, 1, n)/(2*(n+1))**.5


class TorchPoisson:
    def __init__(self, device, inverse):
        self.device, self.grid = device, inverse.grid
        self._denominator = device.upload(inverse.denominator)
        self._denominator32 = None

    def apply(self, rhs):
        self.device.check(rhs, (3*self.grid.n_free_scalar,))
        x = rhs.reshape((3,)+self.grid.interior_shape)
        for axis in (1, 2, 3):
            x = dst1(x, axis)
        x = x/self._denominator
        for axis in (1, 2, 3):
            x = dst1(x, axis)
        return x.reshape(-1)

    def apply_fp32(self, rhs):
        """E106 (C14 phase A, explicit option fp32_poisson): the same spectral solve in float32 -- rhs rounded to
        float32, the six DST-I and the division by the denominator (rounded to float32 once per grid) in float32 --
        and the result converted back to float64 exactly. Callers keep everything else in float64."""
        self.device.check(rhs, (3*self.grid.n_free_scalar,))
        if self._denominator32 is None:
            self._denominator32 = self._denominator.to(torch.float32)
        x = rhs.reshape((3,)+self.grid.interior_shape).to(torch.float32)
        for axis in (1, 2, 3):
            x = dst1(x, axis)
        x = x/self._denominator32
        for axis in (1, 2, 3):
            x = dst1(x, axis)
        return x.reshape(-1).to(torch.float64)


def curl(V):
    return torch.stack((V[..., 2, 1]-V[..., 1, 2], V[..., 0, 2]-V[..., 2, 0],
                        V[..., 1, 0]-V[..., 0, 1]), dim=-1)


def torch_first_roots(b0, b1, b2, b3, bisections=50):
    """T66 device mirror of solvers.global_jd_step.first_roots (b0 > 0 points only)."""
    a, b, c = 3*b3, 2*b2, b1
    one = torch.ones_like(b1)
    nan = torch.full_like(b1, float("nan"))
    linear = a == 0
    r_lin = torch.where(linear & (b != 0), -c/torch.where(b != 0, b, one), nan)
    disc = b*b-4*a*c
    sq = torch.sqrt(torch.where(disc >= 0, disc, torch.zeros_like(disc)))
    q = -(b+torch.where(b >= 0, sq, -sq))/2
    r1 = torch.where(~linear & (disc >= 0) & (a != 0), q/torch.where(a != 0, a, one), nan)
    r2 = torch.where(~linear & (disc >= 0) & (q != 0), c/torch.where(q != 0, q, one), nan)
    r1 = torch.where(linear, r_lin, r1)
    r2 = torch.where(linear, nan, r2)
    r1 = torch.where((r1 > 0) & (r1 < 1), r1, one)
    r2 = torch.where((r2 > 0) & (r2 < 1), r2, one)
    t1, t2 = torch.minimum(r1, r2), torch.maximum(r1, r2)

    def qv(s):
        return b0+s*(b1+s*(b2+s*b3))

    def touch(s):
        return 4*2.220446049250313e-16*(b0.abs()+s*(b1.abs()+s*(b2.abs()+s*b3.abs())))
    q1, q2, q3 = qv(t1) <= touch(t1), qv(t2) <= touch(t2), qv(one) <= 0
    zero = torch.zeros_like(b1)
    hi = torch.where(q1, t1, torch.where(q2, t2, torch.where(q3, one, nan)))
    lo = torch.where(q1, zero, torch.where(q2, t1, torch.where(q3, t2, nan)))
    has = torch.isfinite(hi)
    lo, hi = lo[has], hi[has]
    d0, d1, d2, d3 = b0[has], b1[has], b2[has], b3[has]
    for _ in range(bisections):
        mid = .5*(lo+hi)
        neg = d0+mid*(d1+mid*(d2+mid*d3)) <= 0
        hi = torch.where(neg, mid, hi)
        lo = torch.where(neg, lo, mid)
    out = torch.full_like(b0, float("inf"))
    out[has] = lo
    return out


def geometry(local, D):
    V = torch.einsum("jpa,tar->tprj", D, local)
    F = V+torch.eye(3, dtype=V.dtype, device=V.device)
    # Cross products of cyclic rows are cofactor rows, NOT the adjugate.
    A = torch.stack((torch.linalg.cross(F[..., 1, :], F[..., 2, :]),
                     torch.linalg.cross(F[..., 2, :], F[..., 0, :]),
                     torch.linalg.cross(F[..., 0, :], F[..., 1, :])), dim=-2)
    J = (F[..., 0, :]*A[..., 0, :]).sum(dim=-1)
    return F, A, J, curl(V)


def barrier(J, delta, weight, activation=None):
    if activation is not None:
        scale = activation-delta
        t = (torch.minimum(J, torch.full_like(J, activation))-delta)/scale
        r = torch.clamp(activation-J, min=0)/scale
        logt = torch.where(r < .5, torch.log1p(-r), torch.log(t))
        return (-weight*r*r*logt, weight*(2*r*logt-r*r/t)/scale,
                weight*(-2*logt+4*r/t+(r/t)**2)/scale**2)
    gap, scale = J-delta, 1-delta
    x = (J-1)/scale
    h = torch.full_like(x, -1/11)
    for k in range(8, -1, -1):
        h = h*x+(-1.)**k/(k+2)
    value = weight*torch.where(x.abs() < .01, x*x*h, x-torch.log(gap/scale))
    return value, weight*x/gap, (weight/gap)/gap


@dataclass(frozen=True)
class TensorBlock:
    name: str
    tile: tuple | None
    values: torch.Tensor
    problem_id: str
    eval_id: str
    quadrature_id: str = QUADRATURE_ID


class TorchModel:
    """Owned device constants for exactly one CPU problem/reference identity."""
    def __init__(self, device, problem, *, tile_size=None, selective_kernels=False, fused_kernels=False,
                 fused_jd_root=False, fused_diagonal=False, fused_hvp=False, barrier_skip=None, laplacian_memo=None,
                 fused_spline=None, tuned_launch=None, skip_reference_first=False, laplacian_cache=False,
                 fused_laplacian=None, active_set_skip=None):
        if not isinstance(selective_kernels, bool):
            raise InputValidationError("selective_kernels must be boolean")
        if not isinstance(fused_kernels, bool):
            raise InputValidationError("fused_kernels must be boolean")
        if not isinstance(fused_jd_root, bool):
            raise InputValidationError("fused_jd_root must be boolean")
        if not isinstance(fused_diagonal, bool) or (fused_diagonal and not fused_kernels):
            raise InputValidationError("fused_diagonal must be boolean and requires fused_kernels")
        if not isinstance(fused_hvp, bool) or (fused_hvp and not fused_kernels):
            raise InputValidationError("fused_hvp must be boolean and requires fused_kernels")
        # E55b: opt-in fused Triton global JD root (T66 trial-step helper only); else torch.
        self._jd_root = None
        if fused_jd_root and device.device.type == "cuda":
            from .triton_jd_root import FusedJDRoot, supported as root_supported
            if root_supported(problem):
                self._jd_root = FusedJDRoot(problem, device.device)
        self._selective_kernels = selective_kernels
        # E51: opt-in Triton Q1 kernels for the supported MRI objective only; otherwise torch.
        self._fused = None
        # T89 (S30-G4): the active-set index sets and caches when the problem carries active weights and the backend
        # passes its record (`active_set_skip`); nothing otherwise
        self._active = None
        if active_set_skip is not None and getattr(problem, "active_weights", None) is not None and device.device.type == "cuda":
            from .triton_q1 import _ActiveSets
            self._active = _ActiveSets(problem, device.device, active_set_skip)
        if fused_kernels and device.device.type == "cuda":
            from .triton_q1 import FusedQ1, supported
            if supported(problem):
                extra = dict(**({} if barrier_skip is None else dict(barrier_skip=barrier_skip)),          # E98
                             **({} if tuned_launch is None else dict(tuned_launch=tuned_launch)),         # E99
                             **(dict(reference_first=True) if skip_reference_first else {}),             # E99
                             **({} if self._active is None else dict(active_set=self._active)))        # T89
                self._fused = FusedQ1(problem, device.device, **extra)
        # E65: opt-in fused GGN diagonal (samples barrier basis); otherwise the torch loop below.
        self._fused_diagonal = bool(fused_diagonal and self._fused is not None and not self._fused.bernstein)
        # E72: opt-in fused GGN action (samples barrier basis); otherwise the torch loop in hvp().
        self._fused_hvp = bool(fused_hvp and self._fused is not None and not self._fused.bernstein)
        self.fused_counts = {}     # E72: calls that actually ran a fused kernel (the backend may share its dict)
        # E98 (throughput-v16): the last L u kept for its device tensor (a weak reference and the tensor version);
        # `laplacian_memo` is the backend's shared count record. None keeps every Laplacian computed afresh.
        self._lap_record, self._lap_memo = laplacian_memo, None
        # E101 (C2): with laplacian_cache the memo keeps a slot for the reference tensor (self._ref, the outer start
        # since the last refresh) and the two most recently used other tensors (this makes C1's reordering
        # unnecessary); (C3) fused_laplacian is the backend's record of the one-pass kernel (laplacian_fused). Neither
        # changes a value.
        self._lap_slots = [] if laplacian_cache else None
        self._lap_fused = fused_laplacian
        self.device, self.problem = device, problem
        self.grid, self.w = problem.grid, problem.weights
        self._tile_size = problem.memory.cell_tile_size if tile_size is None else integer_at_least(tile_size, 'device tile_size')
        up = device.upload
        self._q = up(self.grid.node_weights())
        # T80 (E80): q m on the image term only (same tensor as q when the problem has no data weight).
        self._qd = self._q if problem.data_weight is None else up(self.grid.node_weights()*problem.data_weight)
        self._fixed = up(problem.fixed_image)
        self._ref = up(problem.controls.u_ref)
        self._identity = up(self.grid.identity())
        self._D = up(problem._gauss.coefficients)
        self._Dc = up(problem.barrier_ops.coefficients)
        # T68 (E58): Bernstein barrier basis (27x27 map); None keeps the sample basis exactly.
        self._TB = None
        if problem.barrier_basis == "bernstein":
            from ..bernstein_barrier import torch_matrix
            self._TB = torch_matrix(device.device)
        self._qw = up(problem._gauss.weights)
        # T76 (E78): curvature regularizer constants; computed in torch outside the fused kernels.
        self._alpha = float(problem.curvature_weight)
        if self._alpha:
            from ..curvature import curvature_diagonal_nodes, inverse_square_spacing
            self._inv = tuple(float(x) for x in inverse_square_spacing(self.grid))
            self._curv_weight = up(np.sqrt(self._alpha*self.grid.node_weights())[..., None])
            self._curv_alpha_q = up((self._alpha*self.grid.node_weights())[..., None])
            self._curv_diag = up(curvature_diagonal_nodes(problem))
            self._interior_mask = self._strict_mask_float()
        self._offsets = torch.tensor([z*self.grid.shape[1]*self.grid.shape[2]+y*self.grid.shape[2]+x
                                     for z in (0, 1) for y in (0, 1) for x in (0, 1)], device=device.device)
        self.spline = (TorchSpline(problem.sampler, up) if fused_spline is None else
                       TorchSpline(problem.sampler, up, fused=fused_spline))      # E99
        self._strict = torch.zeros(self.grid.shape, dtype=torch.bool, device=device.device)
        self._strict[1:-1, 1:-1, 1:-1] = True

    def sample_u(self, u):
        """The moving image and its gradient at identity + u on every node (spline.sample). T89 (S30-G4): with an
        active set, only the active nodes are sampled afresh; the frozen nodes' values come from the last sampling
        whose frozen entries equal u's (the same points, the same per-point arithmetic). First uses are compared
        with the full sampling; a mismatch switches the active-set skipping off for the run."""
        act = self._active
        chunk = self.problem.memory.spline_query_chunk
        if act is None or act.record["disabled"]:
            return self.spline.sample(self._identity+u, chunk_size=chunk, strict=self._strict)
        part = act.record["sampling"]
        part["calls"] += 1
        entry = act.sample_cache
        if entry is not None and act.frozen_equal(u, entry[0]):
            idx = act.nodes_active
            points = self._identity.reshape(-1, 3)[idx]+u.reshape(-1, 3)[idx]
            value, gradient = self.spline.sample(points, chunk_size=chunk, strict=self._strict.reshape(-1)[idx])
            warped, grad = entry[1].clone(), entry[2].clone()
            warped.reshape(-1)[idx] = value
            grad.reshape(-1, 3)[idx] = gradient
            part["skipped"] += 1
            if act.check('sampling', 'sample', torch.cat((warped.reshape(-1), grad.reshape(-1))),
                         lambda: torch.cat([t.reshape(-1) for t in self.spline.sample(self._identity+u, chunk_size=chunk, strict=self._strict)])):
                act.sample_cache = (u, warped, grad)
                return warped, grad
        else:
            part["misses"] += 1
        warped, grad = self.spline.sample(self._identity+u, chunk_size=chunk, strict=self._strict)
        act.sample_cache = (u, warped, grad)
        return warped, grad

    def _strict_mask_float(self):
        mask = torch.zeros(self.grid.shape, dtype=torch.float64, device=self.device.device)
        mask[1:-1, 1:-1, 1:-1] = 1.
        return mask

    def _laplacian(self, field):
        """laplacian_nodes(field, self._inv); E101 (C3): by the fused kernel once it is verified on this grid."""
        if self._lap_fused is None:
            return laplacian_nodes(field, self._inv)
        if not hasattr(self, "_lap_kernel"):
            from .laplacian_fused import FusedLaplacian, supported
            self._lap_kernel = (FusedLaplacian(self._lap_fused, self._inv, field.device)
                                if supported(field.device) else None)
            if self._lap_kernel is None:
                self._lap_fused["unsupported"] += 1
        if self._lap_kernel is None:
            return laplacian_nodes(field, self._inv)
        return self._lap_kernel(field, lambda f: laplacian_nodes(f, self._inv))

    def _laplacian_of(self, u):
        """laplacian_nodes(u, self._inv); E98: the same tensor values reused while u is the same, unmodified tensor
        (the same deterministic computation would give the same bits; consumers never write into it)."""
        if self._lap_record is None:
            return self._laplacian(u)
        if self._lap_slots is not None:
            return self._laplacian_cached(u)
        memo = self._lap_memo
        if memo is not None and memo[0]() is u and memo[1] == u._version:
            self._lap_record["hits"] += 1
            return memo[2]
        import weakref
        out = self._laplacian(u)
        self._lap_memo = (weakref.ref(u), u._version, out)
        self._lap_record["misses"] += 1
        return out

    def _laplacian_cached(self, u):
        """E101 (C2): the E98 memo with a reference slot and two recently used slots (same keys: tensor identity and
        _version; the same deterministic values). The trial energy's base (the last accepted state), the outer start
        (the reference) and the trial itself all hit after their first computation. The slot list is rebuilt, never
        changed in place (review E101: a copied model shares the list object)."""
        import weakref
        slots = self._lap_slots
        for i, (ref, version, out) in enumerate(slots):
            if ref() is u and version == u._version:
                self._lap_record["hits"] += 1
                self._lap_slots = [slots[i]]+slots[:i]+slots[i+1:]
                return out
        out = self._laplacian(u)
        self._lap_record["misses"] += 1
        slots = [(weakref.ref(u), u._version, out)]+slots
        pinned = [e for e in slots if e[0]() is self._ref]
        recent = [e for e in slots if e[0]() is not self._ref and e[0]() is not None][:2]
        self._lap_slots = pinned[:1]+recent
        return out

    def curvature_residual(self, u):
        """T76: sqrt(alpha q) L u on the nodes."""
        return self._curv_weight*self._laplacian_of(u)

    def curvature_gradient(self, u):
        """T76: L^T (alpha q L u) = L (alpha q L u) on the nodes (zero boundary)."""
        return self._laplacian(self._curv_alpha_q*self._laplacian_of(u))

    def geometry(self, local, D, *, cofactor=True, with_curl=True):
        if self._selective_kernels:
            return geometry_selected(local, D, cofactor=cofactor, with_curl=with_curl)
        return geometry(local, D)

    def barrier_vals(self, J):
        """(..., 27) point values -> the values the barrier acts on (T68)."""
        return J if self._TB is None else J @ self._TB.T

    def barrier_back(self, g):
        """Transpose map back to point space (T68)."""
        return g if self._TB is None else g @ self._TB

    def barrier_vals_points_axis1(self, x):
        """Apply the T68 map along axis 1 (points) of (t, p, ...) arrays."""
        return x if self._TB is None else torch.einsum("kp,tp...->tk...", self._TB, x)

    def barrier_term(self, J, *, order):
        p = self.problem
        if self._selective_kernels:
            return barrier_selected(J, p.certificate_config.delta, p.barrier_weight,
                                    p.barrier_activation, order=order)
        return barrier(J, p.certificate_config.delta, p.barrier_weight, p.barrier_activation)[order]

    def tiles(self, select=None):
        """Cell tiles in order. E93: with ``select`` (one bool per tile) an unselected tile yields
        ((start, end), None) without building its indices; every tile passes the same checkpoint."""
        nz, ny, nx = self.grid.shape
        size = self._tile_size
        for number, start in enumerate(range(0, self.grid.n_cells, size)):
            checkpoint("device_cell_tiles", completed=start, total=self.grid.n_cells)
            end = min(start+size, self.grid.n_cells)
            if select is not None and not select[number]:
                yield (start, end), None
                continue
            yield (start, end), self.tile_indices(start, end)

    def tile_indices(self, start, end):
        """Node indices (cells start..end-1, 8 vertices) on the device; no checkpoint (E93)."""
        nz, ny, nx = self.grid.shape
        c = torch.arange(start, end, device=self.device.device)
        x = c % (nx-1)
        y = (c//(nx-1)) % (ny-1)
        z = c//((nx-1)*(ny-1))
        return (z*ny*nx+y*nx+x)[:, None]+self._offsets

    def inject(self, p):
        self.device.check(p, (3*self.grid.n_free_scalar,))
        v = torch.zeros(self.grid.shape+(3,), dtype=p.dtype, device=p.device)
        v[1:-1, 1:-1, 1:-1] = p.reshape((3,)+self.grid.interior_shape).movedim(0, -1)
        return v

    @staticmethod
    def restrict(out):
        return out[1:-1, 1:-1, 1:-1].movedim(-1, 0).contiguous().reshape(-1)

    @staticmethod
    def scatter(out, indices, D, A, jd, c=None, *, also_into=None):
        flux = A*jd[..., None, None]
        if c is not None:
            flux[..., 2, 1] += c[..., 0]
            flux[..., 1, 2] -= c[..., 0]
            flux[..., 0, 2] += c[..., 1]
            flux[..., 2, 0] -= c[..., 1]
            flux[..., 1, 0] += c[..., 2]
            flux[..., 0, 1] -= c[..., 2]
        local = torch.einsum("jpa,tprj->tar", D, flux)
        flat = out.view(-1, 3)
        # Each vertex slice has unique indices. No nondeterministic CUDA atomics.
        for a in range(8):
            flat[indices[:, a]] += local[:, a]
            if also_into is not None:
                also_into.view(-1, 3)[indices[:, a]] += local[:, a]

    def linearize(self, state):
        build_linearization(self.problem, state)  # CPU ownership/certificate check
        return TorchLinearization(self, state)

    def poisson(self, *, a, b, metric):
        return TorchPoisson(self.device, build_poisson_inverse(self.grid, a=a, b=b, metric=metric))


TRACE_ROWS = 256      # E94 first-use check: cells whose torch values are kept to test cell_j27 across kernel blocks


class TorchLinearization:
    def __init__(self, model, state):
        self.model, self.state = model, state
        self._u = model.device.upload(state.u)
        self._warped, self._b = model.sample_u(self._u)                  # T89: the active-node sampling when enabled

    def _local(self, indices, corners=False, ref=False, *, cofactor=True, with_curl=True):
        m = self.model
        return m.geometry((m._ref if ref else self._u).reshape(-1, 3)[indices], m._Dc if corners else m._D,
                          cofactor=cofactor, with_curl=with_curl)

    def _block(self, name, values, tile=None):
        return TensorBlock(name, tile, values, self.model.problem.problem_id, self.state.eval_id)

    def residuals(self, p=None):
        """Stream R0 or K0 p in the exact B block order and quadrature."""
        m, w = self.model, self.model.w
        v = None if p is None else m.inject(p)
        yield self._block("image", m._qd.sqrt()*((self._warped-m._fixed) if v is None else (self._b*v).sum(dim=-1)))
        yield self._block("proximity", (w.lam*m._q).sqrt()[..., None]*((self._u-m._ref) if v is None else v))
        if m._alpha:
            yield self._block("curvature", m.curvature_residual(self._u if v is None else v))   # T76
        for tile, indices in m.tiles():
            _, A, J, C = self._local(indices, cofactor=v is not None, with_curl=v is None)
            if v is None:
                _, _, Jr, Cr = self._local(indices, ref=True, cofactor=False)
                jd, c = J-Jr, C-Cr
            else:
                V = torch.einsum("jpa,tar->tprj", m._D, v.reshape(-1, 3)[indices])
                jd, c = (A*V).sum(dim=(-2, -1)), curl(V)
            yield self._block("jd", (w.mu*m._qw).sqrt()*jd, tile)
            yield self._block("curl", (w.nu*m._qw).sqrt()[None, :, None]*c, tile)

    def vjp(self, blocks):
        m, w, iterator = self.model, self.model.w, iter(blocks)
        def read(name, shape, tile=None):
            item = next(iterator, None)
            if (not isinstance(item, TensorBlock) or item.name != name or item.tile != tile
                    or item.problem_id != m.problem.problem_id or item.eval_id != self.state.eval_id
                    or item.quadrature_id != QUADRATURE_ID):
                raise StateMismatchError("device residual block identity/order mismatch")
            m.device.check(item.values, shape)
            return item.values
        out = (self._b*(m._qd.sqrt()*read("image", m.grid.shape))[..., None]
               +(w.lam*m._q).sqrt()[..., None]*read("proximity", m.grid.shape+(3,)))
        if m._alpha:
            z = m._curv_weight*read("curvature", m.grid.shape+(3,))
            z = z*m._interior_mask[..., None]                     # T76: L^T has interior rows only
            out = out+laplacian_nodes(z, m._inv)
        for tile, indices in m.tiles():
            shape = (tile[1]-tile[0], 27)
            jd = (w.mu*m._qw).sqrt()*read("jd", shape, tile)
            c = (w.nu*m._qw).sqrt()[None, :, None]*read("curl", shape+(3,), tile)
            _, A, _, _ = self._local(indices, with_curl=False)
            m.scatter(out, indices, m._D, A, jd, c)
        if next(iterator, None) is not None:
            raise StateMismatchError("extra device residual block")
        return m.restrict(out)

    def energy(self):
        """Float64 diagnostic energies only, NOT authoritative acceptance sums."""
        m = self.model
        values = {name: torch.zeros((), device=m.device.device, dtype=torch.float64)
                  for name in ("image", "proximity", "jd", "curl", "barrier", "curvature")}
        for block in self.residuals():
            values[block.name] += .5*(block.values**2).sum()
        for _, indices in m.tiles():
            _, _, J, _ = self._local(indices, corners=True, cofactor=False, with_curl=False)
            values["barrier"] += m.barrier_term(m.barrier_vals(J), order=0).sum()
        return values

    def gradient(self, term="total"):
        if term not in ("total", "outer", "image", "proximity", "jd", "curl", "barrier", "curvature"):
            raise InputValidationError("unknown device gradient term")
        m, w = self.model, self.model.w
        if m._fused is not None and term in ("total", "outer"):
            base = torch.zeros_like(self._u)
            base += m._qd[..., None]*(self._warped-m._fixed)[..., None]*self._b
            if term == "total":
                base += w.lam*m._q[..., None]*(self._u-m._ref)
            if m._alpha:
                base += m.curvature_gradient(self._u)   # T76: node term, before the fused kernels
            return m.restrict(m._fused.gradient_nodes(base, self._u, m._ref, soft=term == "total", barrier=True))
        out = torch.zeros_like(self._u)
        if term in ("total", "outer", "image"):
            out += m._qd[..., None]*(self._warped-m._fixed)[..., None]*self._b
        if term in ("total", "proximity"):
            out += w.lam*m._q[..., None]*(self._u-m._ref)
        if term in ("total", "outer", "curvature") and m._alpha:
            out += m.curvature_gradient(self._u)   # T76
        for _, indices in m.tiles():
            if term in ("total", "jd", "curl"):
                _, A, J, C = self._local(indices)
                _, _, Jr, Cr = self._local(indices, ref=True, cofactor=False)
                jd = w.mu*m._qw*(J-Jr) if term != "curl" else torch.zeros_like(J)
                c = w.nu*m._qw[None, :, None]*(C-Cr) if term != "jd" else None
                m.scatter(out, indices, m._D, A, jd, c)
            if term in ("total", "outer", "barrier"):
                _, A, J, _ = self._local(indices, corners=True, with_curl=False)
                first = m.barrier_back(m.barrier_term(m.barrier_vals(J), order=1))
                m.scatter(out, indices, m._Dc, A, first)
        return m.restrict(out)

    def joint_outer_gradients(self):
        """GF/GD/GE, sharing local geometry/flux but not their sum orders."""
        m, w = self.model, self.model.w
        image = m._qd[..., None]*(self._warped-m._fixed)[..., None]*self._b
        outer, total = torch.zeros_like(self._u), torch.zeros_like(self._u)
        outer += image
        total += image
        total += w.lam*m._q[..., None]*(self._u-m._ref)
        if m._alpha:
            curvature = m.curvature_gradient(self._u)   # T76: in F and E, not in D
            outer += curvature
            total += curvature
        if m._fused is not None:
            outer, total = m._fused.joint(outer, total, self._u, m._ref)
            return tuple(m.restrict(v) for v in (outer, image, total))
        for _, indices in m.tiles():
            _, A, J, C = self._local(indices)
            _, _, Jr, Cr = self._local(indices, ref=True, cofactor=False)
            m.scatter(total, indices, m._D, A, w.mu*m._qw*(J-Jr),
                      w.nu*m._qw[None, :, None]*(C-Cr))
            _, A, J, _ = self._local(indices, corners=True, with_curl=False)
            first = m.barrier_back(m.barrier_term(m.barrier_vals(J), order=1))
            m.scatter(total, indices, m._Dc, A, first, also_into=outer)
        return tuple(m.restrict(v) for v in (outer, image, total))

    def hvp(self, p):
        m, w = self.model, self.model.w
        v = m.inject(p)
        out = m._qd[..., None]*(self._b*(self._b*v).sum(dim=-1)[..., None])+w.lam*m._q[..., None]*v   # T80
        if m._alpha:
            out = out+m.curvature_gradient(v)   # T76: exact Hessian of the quadratic R
        if m._fused_hvp:
            m.fused_counts["fused_hvp_calls"] = m.fused_counts.get("fused_hvp_calls", 0)+1
            return m.restrict(m._fused.hvp_nodes(out, self._u, v.contiguous()))
        for _, indices in m.tiles():
            _, A, _, _ = self._local(indices, with_curl=False)
            V = torch.einsum("jpa,tar->tprj", m._D, v.reshape(-1, 3)[indices])
            m.scatter(out, indices, m._D, A, w.mu*m._qw*(A*V).sum(dim=(-2, -1)),
                      w.nu*m._qw[None, :, None]*curl(V))
            _, A, J, _ = self._local(indices, corners=True, with_curl=False)
            second = m.barrier_term(m.barrier_vals(J), order=2)
            V = torch.einsum("jpa,tar->tprj", m._Dc, v.reshape(-1, 3)[indices])
            m.scatter(out, indices, m._Dc, A, m.barrier_back(second*m.barrier_vals((A*V).sum(dim=(-2, -1)))))
        return m.restrict(out)

    def global_jd_root(self, p, limit):
        """T66: min over all cells' closed27 points of the first root of det(F+tau V)-delta in (0, limit].

        Same coefficients and piecewise-monotone bisection as solvers.global_jd_step (CPU);
        returns (root_s or inf, invalid_base_points, points) as host Python scalars.
        """
        m = self.model
        if m.problem.barrier_location != "closed27_uniform":
            raise InputValidationError("T66 uses the closed27 barrier points")
        delta = m.problem.certificate_config.delta
        v = m.inject(m.device.upload(p))
        if m._jd_root is not None:
            return m._jd_root(self._u.contiguous(), v.contiguous(), limit)
        best = torch.full((), float("inf"), dtype=torch.float64, device=self._u.device)
        invalid = torch.zeros((), dtype=torch.int64, device=self._u.device)
        finite = torch.ones((), dtype=torch.bool, device=self._u.device)
        points = 0
        for _, indices in m.tiles():
            F, AF, _, _ = self._local(indices, corners=True, with_curl=False)
            V = torch.einsum("jpa,tar->tprj", m._Dc, v.reshape(-1, 3)[indices])
            AV = torch.stack((torch.linalg.cross(V[..., 1, :], V[..., 2, :]),
                              torch.linalg.cross(V[..., 2, :], V[..., 0, :]),
                              torch.linalg.cross(V[..., 0, :], V[..., 1, :])), dim=-2)
            c0 = (F[..., 0, :]*AF[..., 0, :]).sum(dim=-1)-delta
            b1 = (AF*V).sum(dim=(-2, -1))*limit
            b2 = (F*AV).sum(dim=(-2, -1))*limit**2
            b3 = (V[..., 0, :]*AV[..., 0, :]).sum(dim=-1)*limit**3
            finite &= torch.isfinite(c0).all() & torch.isfinite(b1).all() & torch.isfinite(b2).all() & torch.isfinite(b3).all()
            valid = c0 > 0
            invalid += (~valid).sum()
            points += c0.numel()
            maybe = valid & ~(c0 > b1.abs()+b2.abs()+b3.abs())
            if bool(maybe.any()):
                roots = torch_first_roots(c0[maybe], b1[maybe], b2[maybe], b3[maybe])
                best = torch.minimum(best, roots.min())
        if not bool(finite):
            raise FloatingPointError("T66 nonfinite cubic coefficients")
        return float(best), int(invalid), points

    @staticmethod
    def node_estimator(compiled, top4=None):
        """The T69 face/interior estimator node_min_closed27_jd calls on a tile's candidate rows: the NumPy one, the
        E81 compiled one, or (E100) the compiled one with the top4 record (same values, bit for bit)."""
        if compiled:
            from ..solvers.jd_affine_compiled import face_interior_minima_parallel as estimator
            return estimator if top4 is None else (lambda J27: estimator(J27, top4=top4))
        if top4 is not None:
            raise InputValidationError("E100 top4 selection needs the compiled estimator")
        from ..solvers.jd_affine import face_interior_minima_parallel as estimator
        return estimator

    def node_min_closed27_jd(self, extension_activation=None, stats=None, compiled=False, fused_values=None,
                             checkpoints=True, trace=None, top4=None, estimates=None):
        """T62 J_a on the full grid: closed27 values and exact edge-quadratic minima.

        Node minima use scatter_reduce(amin), which is order independent.
        E94: fused_values = (minima, candidate mask or None, J27 of listed cells) from backends.node_jd_fused (the
        same values this loop computes, bit for bit); then only the estimator, the stats and the scatter run here,
        with the same rows and calls per tile. checkpoints=False walks the same tiles without the work
        checkpoint (the first-use reference run). trace (a list, first-use check only) receives per tile the
        cell values before the extension, the candidate cells and their 27 values, the estimator's values (E100), and
        the 27 values of the first TRACE_ROWS cells (torch path only); nothing else changes.
        E100: top4 (a jd_affine_compiled.new_top4_record(), compiled estimator only) selects the estimator's Newton
        starts without the argsort (the same indices); estimates (with fused_values; no trace) gives each tile's
        candidate cells and estimator values computed earlier by the backend's node_min_prepare (NodeEstimates) from the same rows by the same
        calls -- an exception the earlier pass met is raised at its tile, before that tile's stats.
        """
        from ..solvers.jd_affine import EDGE_INDICES
        m = self.model
        if m.problem.barrier_location != "closed27_uniform" or m.problem.barrier_activation is None:
            raise InputValidationError("T62 requires the closed27 local barrier route")
        if estimates is not None and (fused_values is None or trace is not None or extension_activation is None):
            raise InputValidationError("E100 estimates need fused values and the extension, without a trace")
        edges = torch.as_tensor(EDGE_INDICES, device=self._u.device)
        TB = None
        if extension_activation is not None:
            from ..solvers.jd_affine import LOWERED_RELATIVE, _count
            face_interior_minima = self.node_estimator(compiled, top4)
            if compiled:      # E81: bit-identical Numba estimator
                from ..solvers.jd_affine_compiled import COMPILED_ESTIMATOR
                if stats is not None:
                    stats["estimator"] = COMPILED_ESTIMATOR
        node = torch.full(m.grid.shape, float("inf"), dtype=torch.float64, device=self._u.device).view(-1)
        if checkpoints:
            tiles = m.tiles()
        else:
            size, n = m._tile_size, m.grid.n_cells
            tiles = (((s, min(s+size, n)), m.tile_indices(s, min(s+size, n))) for s in range(0, n, size))
        for number, ((start, end), indices) in enumerate(tiles):
            if fused_values is not None:
                minima = fused_values[0][start:end].clone()
            else:
                _, _, J, _ = self._local(indices, corners=True, with_curl=False)
                f = J[:, edges]
                f0, f1, f2 = f[..., 0], f[..., 1], f[..., 2]
                a = 2*(f0-2*f1+f2)
                b = -3*f0+4*f1-f2
                inside = (a > 0) & (b < 0) & (-b < 2*a)
                vertex = torch.where(inside, f0-b*b/(4*torch.where(inside, a, torch.ones_like(a))),
                                     torch.full_like(a, float("inf")))
                minima = torch.minimum(J.min(dim=1).values, vertex.min(dim=1).values)
            entry = None
            if trace is not None:
                entry = dict(start=start, minima=minima.clone(), cand=None, rows=None)
                if fused_values is None and start < TRACE_ROWS:
                    entry["j"] = J[:TRACE_ROWS-start].clone()
                trace.append(entry)
            if extension_activation is not None:
                # T69: candidates on the device; their 27 values go to the shared CPU estimator.
                if estimates is not None:
                    cand, host_mu = estimates.at(number, start, end)      # E100 (raises a stored exception here)
                elif fused_values is not None:
                    cand = torch.nonzero(fused_values[1][start:end]).flatten()
                    if cand.numel():
                        cand_values = fused_values[2](cand+start)       # the tile loop's rows J[cand], bit for bit
                else:
                    if TB is None:
                        from ..bernstein_barrier import torch_matrix
                        TB = torch_matrix(J.device).T
                    beta_min = (J @ TB).min(dim=1).values
                    cand = torch.nonzero(beta_min < torch.clamp(minima, max=float(extension_activation))).flatten()
                    if cand.numel():
                        cand_values = J[cand]
                if entry is not None:
                    entry["cand"] = cand.clone()
                    entry["rows"] = cand_values.clone() if cand.numel() else None
                if cand.numel():
                    mu = torch.as_tensor(host_mu if estimates is not None else
                                         face_interior_minima(cand_values.cpu().numpy()), dtype=torch.float64,
                                         device=minima.device)
                    if entry is not None:
                        entry["mu"] = mu.clone()          # E100: the estimator's values (first-use checks only)
                    _count(stats, int(cand.numel()), int((mu < minima[cand]*(1-LOWERED_RELATIVE)).sum()),
                           int((mu < minima[cand]).sum()), float(mu.min()))
                    minima[cand] = torch.minimum(minima[cand], mu)
            for k in range(8):
                node.scatter_reduce_(0, indices[:, k], minima, reduce="amin")
        return node.view(m.grid.shape)

    def diagonal_base(self):
        """GG2: the node part of diagonal() before the geometry/barrier locals are added (image GN + proximal, + T76),
        the same expressions in the same order; diagonal() itself is unchanged."""
        m, w = self.model, self.model.w
        out = m._qd[..., None]*self._b**2+w.lam*m._q[..., None]
        if m._alpha:
            out = out+m._curv_diag
        return out

    def diagonal(self):
        m, w = self.model, self.model.w
        out = m._qd[..., None]*self._b**2+w.lam*m._q[..., None]   # T80
        if m._alpha:
            out = out+m._curv_diag   # T76
        if m._fused_diagonal:
            return m.restrict(m._fused.diagonal_nodes(out, self._u))
        for _, indices in m.tiles():
            _, A, _, _ = self._local(indices, with_curl=False)
            _, Ac, Jc, _ = self._local(indices, corners=True, with_curl=False)
            curvature = m.barrier_term(m.barrier_vals(Jc), order=2)
            lj = torch.einsum("tprj,jpa->tpar", A, m._D)
            lc = m.barrier_vals_points_axis1(torch.einsum("tprj,jpa->tpar", Ac, m._Dc))
            cs = torch.stack([sum(m._D[j]**2 for j in range(3) if j != r) for r in range(3)], dim=-1)
            local = (m._qw[None, :, None, None]*(w.mu*lj**2+w.nu*cs[None])).sum(dim=1)
            local += (curvature[..., None, None]*lc**2).sum(dim=1)
            for a in range(8):
                out.view(-1, 3)[indices[:, a]] += local[:, a]
        return m.restrict(out)
