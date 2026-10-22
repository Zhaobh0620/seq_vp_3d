"""Tensor-product not-a-knot cubic evaluation using the frozen SciPy coefficients.

No grid_sample, image prefilter, padding, extrapolation or coordinate clipping.
The span is restricted only at the closed upper endpoint, not the query itself.
"""
import torch
from ..work_control import checkpoint

from ...validation import InputValidationError


def cubic_basis(t, x, n):
    span = (torch.searchsorted(t, x.contiguous(), right=True)-1).clamp(3, n-1)
    basis = torch.ones((x.numel(), 1), dtype=x.dtype, device=x.device)
    quadratic = None
    for degree in range(1, 4):
        values, saved = [], torch.zeros_like(x)
        for r in range(degree):
            right = t[span+r+1]-x
            left = x-t[span+1-degree+r]
            temp = basis[:, r]/(right+left)
            values.append(saved+right*temp)
            saved = left*temp
        values.append(saved)
        basis = torch.stack(values, dim=-1)
        if degree == 2:
            quadratic = basis
    indices = span[:, None]+torch.arange(-3, 1, device=x.device)
    zero = torch.zeros_like(x[:, None])
    a, b = t[indices+3]-t[indices], t[indices+4]-t[indices+1]
    # Repeated endpoint knots give zero coefficients, not divisions by zero.
    derivative = (3*torch.cat((zero, quadratic), dim=-1)/torch.where(a != 0, a, 1)
                  -3*torch.cat((quadratic, zero), dim=-1)/torch.where(b != 0, b, 1))
    return indices, basis, derivative


class TorchSpline:
    def __init__(self, sampler, upload, fused=None):
        self._knots = tuple(upload(t) for t in sampler._spline.t)
        self._c = upload(sampler._spline.c)
        self._bounds = tuple((float(a[0]), float(a[-1])) for a in sampler.grid.axes)
        # E99 (C5, throughput-v17): the fused kernel, verified per grid against this torch path (`fused` is the
        # backend's shared record). None (every profile up to v16) keeps the torch path only.
        self._fused = None
        if fused is not None:
            from .spline_fused import FusedSpline
            self._fused = FusedSpline(self._knots, self._c, sampler.grid.grid_id, fused)

    def sample_f32(self, points, *, chunk_size):
        """T90 (S30-G5 draft): the moving image and its gradient at `points` (device float64 or float32 xyz) from the
        float32 fused kernel on float32 copies of the knots and coefficients; returns float32 (value, gradient). A
        screening sampler: no bound checks beyond finiteness, never bit-identical to sample()."""
        if (not isinstance(points, torch.Tensor) or points.dtype not in (torch.float64, torch.float32)
                or points.device != self._c.device or points.shape[-1:] != (3,) or points.requires_grad):
            raise InputValidationError("spline requires explicit device float xyz points, without autograd")
        if not bool(torch.isfinite(points).all()):
            raise InputValidationError("nonfinite spline query")
        if not isinstance(chunk_size, int) or chunk_size < 1:
            raise InputValidationError("positive integer spline chunk required")
        if not hasattr(self, "_f32"):
            self._f32 = (tuple(t.to(torch.float32).contiguous() for t in self._knots), self._c.to(torch.float32).contiguous(),
                         max(9, max(int(t.numel()).bit_length() for t in self._knots)))
        from . import spline_fused_f32_kernels as KF
        from .spline_fused import BLOCK, WARPS
        import triton
        (tz, ty, tx), c, nstep = self._f32
        flat = points.reshape(-1, 3).contiguous()
        n = flat.shape[0]
        if 3*n >= 2**31 or c.numel() >= 2**31:
            raise InputValidationError("float32 spline sampler: int32 offsets exceeded")
        value = torch.empty(n, dtype=torch.float32, device=flat.device)
        gradient = torch.empty((n, 3), dtype=torch.float32, device=flat.device)
        nz, ny, nx = c.shape
        for start in range(0, n, chunk_size):
            end = min(start+chunk_size, n)
            KF.spline_fused_f32_kernel[(triton.cdiv(end-start, BLOCK),)](
                flat, c, tz, ty, tx, value, gradient, start, end, ny, nx, tz.numel(), ty.numel(), tx.numel(),
                nz-1, ny-1, nx-1, NSTEP=nstep, BLOCK=BLOCK, num_warps=WARPS, enable_fp_fusion=False)
        return value.reshape(points.shape[:-1]), gradient.reshape(points.shape)

    def sample(self, points, *, chunk_size, strict=None):
        if (not isinstance(points, torch.Tensor) or points.dtype != torch.float64
                or points.device != self._c.device or points.shape[-1:] != (3,) or points.requires_grad):
            raise InputValidationError("spline requires explicit device float64 xyz points, without autograd")
        if not bool(torch.isfinite(points).all()):
            raise InputValidationError("nonfinite spline query")
        if not isinstance(chunk_size, int) or chunk_size < 1:
            raise InputValidationError("positive integer spline chunk required")
        valid = torch.ones(points.shape[:-1], dtype=torch.bool, device=points.device)
        if strict is not None and (strict.shape != valid.shape or strict.dtype != torch.bool or strict.device != points.device):
            raise InputValidationError("strict query mask mismatch")
        for r, (lo, hi) in enumerate(self._bounds):
            valid &= (points[..., r] >= lo) & (points[..., r] <= hi)
            if strict is not None:
                valid &= ~strict | ((points[..., r] > lo) & (points[..., r] < hi))
        if not bool(valid.all()):
            raise InputValidationError("spline query outside permitted support")
        flat = points.reshape(-1, 3)
        value = torch.empty(flat.shape[0], device=points.device, dtype=points.dtype)
        gradient = torch.empty_like(flat)
        fused = self._fused if (self._fused is not None and self._fused.usable(flat)) else None
        if fused is not None:
            fused.record["calls"] += 1
            if fused.verified():
                return self._sample_fused(fused, flat, value, gradient, chunk_size, points.shape)
        inside = 0 if fused is not None else None
        for start in range(0, len(flat), chunk_size):
            checkpoint("device_spline_sample", completed=start, total=len(flat))
            end = min(start+chunk_size, len(flat))
            bases = [cubic_basis(t, flat[start:end, 2-axis], self._c.shape[axis])
                     for axis, t in enumerate(self._knots)]
            iz, bz, dz = bases[0]
            iy, by, dy = bases[1]
            ix, bx, dx = bases[2]
            coeff = self._c[iz[:, :, None, None], iy[:, None, :, None], ix[:, None, None, :]]
            def contract(z, y, x):
                return (coeff*z[:, :, None, None]*y[:, None, :, None]*x[:, None, None, :]).sum(dim=(1, 2, 3))
            value[start:end] = contract(bz, by, bx)
            gradient[start:end] = torch.stack((contract(bz, by, dx), contract(bz, dy, bx), contract(dz, by, bx)), dim=-1)
            if inside is not None:            # E99: informative points for the fused kernel's verification
                from .spline_fused import inside_count
                inside = inside+(inside_count(flat[start:end, 2], self._knots[0], iz)
                                 & inside_count(flat[start:end, 1], self._knots[1], iy)
                                 & inside_count(flat[start:end, 0], self._knots[2], ix)).sum()
        if inside is not None:
            fused.compare(flat, value, gradient, inside, chunk_size)
        return value.reshape(points.shape[:-1]), gradient.reshape(points.shape)

    def _sample_fused(self, fused, flat, value, gradient, chunk_size, shape):
        """E99: the verified fused kernel per chunk, with the same checkpoints; a chunk whose launch fails is
        computed by the torch path (_torch_chunk) and switches the kernel off for the run."""
        for start in range(0, len(flat), chunk_size):
            checkpoint("device_spline_sample", completed=start, total=len(flat))
            end = min(start+chunk_size, len(flat))
            done = False
            if fused is not None:
                try:
                    fused.chunk(flat, start, end, value, gradient)
                    done = True
                except Exception:
                    pass
                if not done:
                    fused.record["errors"] += 1
                    fused.record["disabled"] = True
                    fused = None
            if not done:
                self._torch_chunk(flat, start, end, value, gradient)
        return value.reshape(shape[:-1]), gradient.reshape(shape)

    def _torch_chunk(self, flat, start, end, value, gradient):
        """E99: one chunk of the torch path above, operation for operation (the fallback of _sample_fused)."""
        bases = [cubic_basis(t, flat[start:end, 2-axis], self._c.shape[axis])
                 for axis, t in enumerate(self._knots)]
        iz, bz, dz = bases[0]
        iy, by, dy = bases[1]
        ix, bx, dx = bases[2]
        coeff = self._c[iz[:, :, None, None], iy[:, None, :, None], ix[:, None, None, :]]
        def contract(z, y, x):
            return (coeff*z[:, :, None, None]*y[:, None, :, None]*x[:, None, None, :]).sum(dim=(1, 2, 3))
        value[start:end] = contract(bz, by, bx)
        gradient[start:end] = torch.stack((contract(bz, by, dx), contract(bz, dy, bx), contract(dz, by, bx)), dim=-1)
