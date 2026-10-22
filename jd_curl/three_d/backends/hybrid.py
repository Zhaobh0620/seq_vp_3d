"""CPU-certified solver controllers with optional resident Torch directions.

No independent acceptance loop lives here. CPU E/F differences, reference
refresh, feasibility and outer stopping remain the shared C implementation.
"""
from functools import wraps
from time import perf_counter

import numpy as np
import torch

from ...validation import as_float64_array, finite_real, InputValidationError, require_finite_result
from ..linearization import energy_gradient
from ..solvers.common import gradient_norm_mass
from ..solvers.lm import LMConfig3D, LMDirection3D
from ..work_control import operation
from .torch_ops import Device, TorchModel, TorchPoisson


FP32_RANGE = 2.**100        # E106: float32 spectral solve only when max|s*g| <= 2^100 (a screen; the finite check decides)


def timed(name):
    def decorate(fn):
        @wraps(fn)
        def run(self, *args, **kwargs):
            # E105 (direction_prefetch): an object may defer one of the two full synchronizations while its own
            # prefetched device work is in flight (only then; otherwise both run exactly as before)
            deferred = getattr(self, "_sync_deferred", None)
            if deferred is None or not deferred(name, "start"):
                self.device.synchronize()
            started = perf_counter()
            try:
                with operation(name):
                    return fn(self, *args, **kwargs)
            finally:
                if deferred is None or not deferred(name, "end"):
                    self.device.synchronize()
                self.timings[name] = self.timings.get(name, 0.)+perf_counter()-started
        return run
    return decorate


def tensor_norm(x):
    # Match the scaled CPU norm contract without under/overflow in sum(x*x).
    scale = x.abs().max()
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    return safe*torch.linalg.vector_norm(x/safe)


class HybridBackend:
    def __init__(self, device="cuda"):
        self.device = Device(device)
        self.backend_id = "torch-float64-"+str(self.device.device)+"-cpu-certified-hybrid"
        self._model = self._lin = None
        self.cpu_gradient_checks = 0
        self.timings = {}

    def linearize(self, lin):
        # CPU Linearization validates ownership before any device cache reuse.
        if self._model is None or self._model.problem is not lin.problem:
            self._model = TorchModel(self.device, lin.problem)
            self._lin = None
        if self._lin is None or self._lin.state is not lin.state:
            self._lin = self._model.linearize(lin.state)
        return self._lin

    @timed("gradient")
    def gradient(self, lin, target):
        gradient = self.device.download(self.linearize(lin).gradient())
        require_finite_result(gradient, "device gradient")
        if gradient_norm_mass(gradient, lin.problem.grid.omega) <= 1.01*target:
            self.cpu_gradient_checks += 1
            gradient = energy_gradient(lin)
        return gradient

    def poisson(self, inverse):
        return _HostPoisson(self.device, inverse, self.timings)

    @timed("diagonal")
    def diagonal(self, lin):
        result = self.linearize(lin).diagonal()
        if not bool(torch.isfinite(result).all()) or not bool((result > 0).all()):
            raise FloatingPointError("invalid device GGN diagonal")
        return result

    def diagonal_launch(self, lin):
        """E89 (P07): diagonal(lin) in two parts. The device work is queued now (no host synchronization);
        result() performs the same checks and the exact download, timed as 'diagonal'."""
        return _PendingDiagonal(self, self.linearize(lin).diagonal())

    @timed("global_jd_root")
    def global_jd_root(self, lin, direction, limit):
        """T66 device evaluation; returns (root in s-units or inf, invalid_base_points, points)."""
        return self.linearize(lin).global_jd_root(direction, limit)

    @timed("node_min_jd")
    def node_min_jd(self, lin, *, extension_activation=None, stats=None, compiled=False):
        """T62 metric-only node JD measure (host array, full grid shape); T69 extension optional.
        compiled (E81): the bit-identical Numba face/interior estimator."""
        return self.device.download(self._node_min_device(lin, extension_activation, stats, compiled))

    def _node_min_device(self, lin, extension_activation, stats, compiled):
        """Device node array of node_min_jd (no timer of its own; E94 overrides it in the accelerated backend)."""
        return self.linearize(lin).node_min_closed27_jd(
            extension_activation=extension_activation, stats=stats, **({"compiled": True} if compiled else {}))

    @timed("lm_direction")
    def lm_direction(self, lin, gradient, *, gamma, config, diagonal=None):
        """Zero-start PCG; device vectors stay resident until p/Hp extraction.

        Scalar dot/norm reads synchronize CG control; they are counted separately
        from whole-field transfers. A fresh true undamped Hp is always computed.
        """
        n = 3*lin.problem.grid.n_free_scalar
        gradient = as_float64_array(gradient, "gradient", (n,))
        gamma = finite_real(gamma, "gamma", positive=True)
        if not isinstance(config, LMConfig3D) or not config.gamma_min <= gamma <= config.gamma_max:
            raise InputValidationError("invalid LM configuration/damping")
        dl = self.linearize(lin)
        m, dev = dl.model, self.device
        p = Hp = info = relative = slope = None
        iterations = n_hvp = n_preconditioner = 0
        true_checked = False
        def finish(valid, reason):
            return LMDirection3D(valid, reason, p, Hp, info, iterations, relative, slope,
                                 n_hvp, n_preconditioner, true_checked)
        def scalar(x):
            value = dev.scalar(x)
            if not np.isfinite(value):
                raise FloatingPointError("nonfinite device linear solve scalar")
            return value
        try:
            g = dev.upload(gradient)
            gnorm = scalar(tensor_norm(g))
            if gnorm == 0:
                return finish(False, "ZERO_GRADIENT")
            inverse = m.poisson(a=m.w.lam+gamma, b=(m.w.mu+m.w.nu)/2, metric="seven_point")
            scale = None
            if config.preconditioner == "ggn_diagonal_poisson":
                diagonal = dl.diagonal() if diagonal is None else diagonal
                dev.check(diagonal, (n,))
                if not bool(torch.isfinite(diagonal).all()) or not bool((diagonal > 0).all()):
                    raise FloatingPointError("invalid device GGN diagonal")
                mass_diagonal = m.grid.omega*(m.w.lam+gamma+(m.w.mu+m.w.nu)*sum(1/h**2 for h in m.grid.spacing))
                scale = mass_diagonal**.5/torch.sqrt(diagonal+m.grid.omega*gamma)
            x, r, direction, rho_old = torch.zeros_like(g), -g.clone(), None, None
            info = config.max_cg_iterations
            for _ in range(config.max_cg_iterations):
                if scalar(tensor_norm(r)) <= config.cg_rtol*gnorm:
                    info = 0
                    break
                n_preconditioner += 1
                z = inverse.apply(r) if scale is None else scale*inverse.apply(scale*r)
                rho = scalar(torch.dot(r, z))
                if rho <= 0:
                    return finish(False, "NON_DESCENT_DIRECTION")
                direction = z.clone() if direction is None else z+(rho/rho_old)*direction
                n_hvp += 1
                q = dl.hvp(direction)+m.grid.omega*gamma*direction
                curvature = scalar(torch.dot(direction, q))
                if curvature <= 0:
                    return finish(False, "NON_DESCENT_DIRECTION")
                alpha = rho/curvature
                x += alpha*direction
                r -= alpha*q
                rho_old = rho
                iterations += 1
            n_hvp += 1
            hx = dl.hvp(x)
            true_checked = True
            residual_norm = scalar(tensor_norm(hx+m.grid.omega*gamma*x+g))
            relative = residual_norm/gnorm
            # Exactly two whole-vector downloads per completed direction, none per HVP.
            p, Hp = dev.download(x), dev.download(hx)
            require_finite_result(p, "device CG direction")
            require_finite_result(Hp, "device undamped Hp")
            if info != 0:
                return finish(False, "CG_NOT_CONVERGED")
            if residual_norm > config.cg_rtol*gnorm:
                return finish(False, "TRUE_RESIDUAL_FAILED")
            slope = float(gradient@p)
            require_finite_result(slope, "device LM slope")
            if slope >= 0:
                return finish(False, "NON_DESCENT_DIRECTION")
            return finish(True, "LINEAR_SOLVE_ACCEPTED")
        except FloatingPointError:
            return finish(False, "NONFINITE_LINEAR_SOLVE")


class _PendingDiagonal:
    def __init__(self, backend, tensor):
        self._backend, self._tensor = backend, tensor

    def result(self):
        backend, result = self._backend, self._tensor
        self._tensor = None
        started = perf_counter()
        try:
            with operation("diagonal"):
                if not bool(torch.isfinite(result).all()) or not bool((result > 0).all()):
                    raise FloatingPointError("invalid device GGN diagonal")
                return backend.device.download(result)
        finally:
            backend.device.synchronize()
            backend.timings["diagonal"] = backend.timings.get("diagonal", 0.)+perf_counter()-started


class _HostPoisson:
    def __init__(self, device, inverse, timings, device_gradient=None, keep_direction=None, prefetch=None, fp32=None):
        self.device, self.timings = device, timings
        self.inverse = TorchPoisson(device, inverse)
        self._device_gradient = device_gradient     # E95: host gradient -> its device tensor (or None)
        self._scale = None
        self._keep_direction = keep_direction       # E97: (frozen host direction, its device tensor) -> backend
        self._prefetch = prefetch                   # E105: the backend holding this Poisson's prefetched direction
        self._fp32 = fp32                           # E106: the backend holding the fp32_poisson policy and record

    def _sync_deferred(self, name, when):
        """E105: the 'poisson' operation of a prefetched direction does not wait for that direction's device work
        at its start (the host work between() runs meanwhile, as in v22 after it queues the same kernels)."""
        return self._prefetch is not None and name == "poisson" and when == "start" and self._prefetch.prefetch_taken(self)

    @timed("poisson")
    def apply(self, rhs):
        if self._prefetch is not None:
            self._prefetch.prefetch_drop("plain_apply")
        return self.device.download(self.inverse.apply(self.device.upload(rhs)))

    def prepare_equilibrated(self, scale, gradient):
        """E95: the device pre-scale s*g (single IEEE multiplications, as the compiled host pass) for
        apply_equilibrated, or None -- no device gradient path, or s*g not finite (then the host path runs
        and behaves, operations and checkpoints included, exactly as v13). No operation scope is entered."""
        if self._device_gradient is None:
            return None
        if self._prefetch is not None:
            # E105: this gradient's direction may already be queued (same kernels, same inputs, same order); taking it
            # hands out the device gradient memo exactly as below. Any mismatch drops it and the v22 path runs.
            taken = self._prefetch.prefetch_take(self, scale, gradient)
            if isinstance(taken, str):               # s*g not finite: v22 returns None here (memo handed out)
                return None
            if taken is not None:
                return taken
        if self._scale is None or self._scale[0] is not scale:
            self._scale = (scale, self.device.upload(scale))
        device_scale = self._scale[1]
        device_gradient = self._device_gradient(gradient)
        if device_gradient is None:
            device_gradient = self.device.upload(gradient)
        pre = device_scale*device_gradient
        if self._fp32 is not None and self._fp32.fp32_active():
            # E106: the finite flag (as v23) and the float32 range flag max|pre| <= 2^100, in one synchronization
            finite, in_range = torch.stack((torch.isfinite(pre).all(), pre.abs().amax() <= FP32_RANGE)).tolist()
            if not finite:
                return None
            return dict(scale=device_scale, pre=pre, gradient=device_gradient, use32=bool(in_range))
        if not bool(torch.isfinite(pre).all()):
            return None
        return device_scale, pre

    @timed("poisson")
    def apply_equilibrated(self, prepared, between=None):
        """E95: one 'poisson' operation, as v13's inner.apply: x = M^-1(pre) and -(s*x) on the device (a
        multiplication and a negation in separate kernels, as the compiled host pass). Returns (direction,
        True) when every value is finite, else ('x', x) with x downloaded, and the caller finishes exactly as
        v13 does after its own inner.apply (compiled negated product, then the NumPy expression).
        between: host work (reading none of these values) run once while the device computes."""
        if isinstance(prepared, dict):               # E106: the fp32_poisson path
            return self._apply_fp32(prepared, between)
        if len(prepared) == 3:                       # E105: (x, out, out_finite) queued by the prefetch
            x, out, out_finite = prepared
            self._prefetch.prefetch_used(self)
        else:
            device_scale, pre = prepared
            x = self.inverse.apply(pre)
            out = -(device_scale*x)
            out_finite = torch.isfinite(out).all()
        if between is not None:
            between()                                # the device work above is queued, not awaited
        if not bool(out_finite):
            return "x", self._download_numpy(x)
        host = self._download_numpy(out)
        if self._keep_direction is not None:
            host.flags.writeable = False                # E97: the twin of `out` (device_candidate)
            self._keep_direction(host, out)
        return host, True

    def _apply_fp32(self, d, between):
        """E106 (C14 phase A): x = M^-1 pre in float32 (TorchPoisson.apply_fp32) when pre is within range, else in
        float64; out = -(s*x) in float64. A nonfinite float32 out, an error, or a first-use check above the limit
        gives the float64 direction of this step (v23's values) instead. The first FP32_CHECKS uses per level grid
        also compute the float64 direction and compare: relative l2 and max differences of out and of g.out.
        The norms |g| and |out| are kept with the host direction for the backend's descent guard."""
        from ..work_control import WorkBudgetExceeded
        fp = self._fp32
        scale, pre, gradient = d["scale"], d["pre"], d["gradient"]
        use32 = d["use32"]
        if d.get("from_prefetch"):
            self._prefetch.prefetch_used(self)
        if not use32:
            fp.fp32_poisson_record["range_fallbacks"] += 1
        checking, values = False, None
        try:
            if d.get("queued") is not None:
                x, out, out_finite = d["queued"]
            else:
                x = self.inverse.apply_fp32(pre) if use32 else self.inverse.apply(pre)
                out = -(scale*x)
                out_finite = torch.isfinite(out).all()
            if use32:
                checking = fp.fp32_checking(self.inverse.grid)
                parts = [out_finite.to(torch.float64), torch.linalg.vector_norm(gradient), torch.linalg.vector_norm(out)]
                if checking:
                    x64 = self.inverse.apply(pre)
                    out64 = -(scale*x64)
                    difference = out-out64
                    parts += [torch.linalg.vector_norm(difference), torch.linalg.vector_norm(out64),
                              difference.abs().amax(), out64.abs().amax(), torch.dot(gradient, out),
                              torch.dot(gradient, out64)]
                stacked = torch.stack(parts)
        except WorkBudgetExceeded:
            raise
        except Exception as error:      # noqa: BLE001 - recorded; v23's float64 direction from here on
            if not use32:
                raise                       # review E106 (round 2): the plain float64 solve fails as in v23
            fp.fp32_error(error)
            use32, checking = False, False
            x = self.inverse.apply(pre)
            out = -(scale*x)
            out_finite = torch.isfinite(out).all()
        if between is not None:
            between()                                # once, outside the guards (its own errors propagate as in v23)
        if use32:
            try:
                values = stacked.tolist()
            except WorkBudgetExceeded:
                raise
            except Exception as error:  # noqa: BLE001
                fp.fp32_error(error)
                use32, checking = False, False
                x = self.inverse.apply(pre)
                out = -(scale*x)
                out_finite = torch.isfinite(out).all()
        if use32 and values[0] != 1.:
            fp.fp32_poisson_record["nonfinite_fallbacks"] += 1          # float32 out not finite: this step in float64
            use32, checking = False, False
            x = self.inverse.apply(pre)
            out = -(scale*x)
            out_finite = torch.isfinite(out).all()
        if checking:
            ratio = lambda a, b: a/b if b > 0 else (0. if a == 0 else float("inf"))
            measured = (ratio(values[3], values[4]), ratio(values[5], values[6]), ratio(abs(values[7]-values[8]), abs(values[8])))
            if not fp.fp32_check_result(self.inverse.grid, *measured):
                use32 = False                                   # above the limit: this step's float64 direction
                x, out, out_finite = x64, out64, torch.isfinite(out64).all()
        if not bool(out_finite):
            return "x", self._download_numpy(x)
        host = self._download_numpy(out)
        if self._keep_direction is not None:
            host.flags.writeable = False                # E97: the twin of `out` (device_candidate)
            self._keep_direction(host, out)
        if use32:
            fp.fp32_poisson_record["fp32_used"] += 1

            def rerun(pre=pre, scale=scale, replaced=host):
                # the float64 direction of this gradient (v23's values); returned even if not finite, so the solver
                # then raises exactly where v23's direction check does
                x64 = self.inverse.apply(pre)
                out64 = -(scale*x64)
                finite = bool(torch.isfinite(out64).all())
                again = self._download_numpy(out64)
                fp.forget_twin(replaced)            # the float32 direction is never used: its device copy goes
                if finite and self._keep_direction is not None:
                    again.flags.writeable = False
                    self._keep_direction(again, out64)
                return again, finite
            fp.fp32_note(host, values[1], values[2], rerun)
        return host, True

    def release(self):
        """E96: drop the device copy of S (called when the inner solve that built this Poisson ends)."""
        self._scale = None
        if self._prefetch is not None:
            self._prefetch.prefetch_drop("release", owner=self)

    def _download_numpy(self, tensor):
        """device.download into a NumPy-allocated array (as v13's host passes allocate their outputs)."""
        self.device.check(tensor)
        self.device.transfers.download_calls += 1
        self.device.transfers.download_bytes += tensor.numel()*tensor.element_size()
        host = np.empty(tuple(tensor.shape), dtype=np.float64)
        torch.from_numpy(host).copy_(tensor.detach())
        return host
