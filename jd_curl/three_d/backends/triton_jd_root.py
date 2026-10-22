"""E55b (T66, implementation only): fused Triton evaluation of the global closed27 JD root.

Same quantities as TorchLinearization.global_jd_root / solvers.global_jd_step: per closed27
point the cubic det(F + tau V) - delta (F = I + grad u from exact Q1 edge differences,
V = grad v), its first root in (0, limit] by the same piecewise-monotone bisection (with the
near-tangent tolerance), minimum over all cells. One program handles BLOCK cells; each writes
its own minimum/count/flag and the host reduction is torch.min/sum (no atomics, deterministic).
Agreement with the torch path is to roundoff, not bitwise: Triton contracts multiply-adds
into FMA by default (the Horner error stays inside the 4*eps touch band); this is a
trial-step heuristic, never a certificate.
"""
import torch
import triton

from ...validation import InputValidationError
from . import triton_jd_root_kernels as K

POLICY = "triton-fused-global-jd-root-v1"
BLOCK, WARPS = 64, 2


def supported(problem):
    return problem.barrier_location == "closed27_uniform"


class FusedJDRoot:
    def __init__(self, problem, device):
        if not supported(problem):
            raise InputValidationError("fused JD root needs the closed27 barrier points")
        g = problem.grid
        self.inv_h = [1/h for h in g.spacing]
        self.delta = float(problem.certificate_config.delta)
        nz, ny, nx = g.shape
        self.shape, self.ny, self.nx, self.cy, self.cx = g.shape, ny, nx, ny-1, nx-1
        self.ncell = g.n_cells
        self.device = device

    def __call__(self, u, v, limit):
        """(root in s-units or inf, number of c0 <= 0 points, number of points)."""
        for a in (u, v):
            if a.dtype != torch.float64 or not a.is_contiguous() or tuple(a.shape) != self.shape+(3,):
                raise InputValidationError("fused JD root needs contiguous float64 node arrays")
        k = torch.tensor(self.inv_h+[self.delta, limit, limit**2, limit**3], dtype=torch.float64, device=self.device)
        programs = triton.cdiv(self.ncell, BLOCK)
        mins = torch.empty(programs, dtype=torch.float64, device=self.device)
        inv = torch.empty(programs, dtype=torch.int32, device=self.device)
        bad = torch.empty(programs, dtype=torch.int32, device=self.device)
        K.global_jd_root_kernel[(programs,)](u, v, k, mins, inv, bad, self.ncell, self.ny, self.nx, self.cy, self.cx,
                                             BLOCK=BLOCK, num_warps=WARPS)
        if bool(bad.any()):
            raise FloatingPointError("T66 nonfinite cubic coefficients (fused)")
        return float(mins.min()), int(inv.to(torch.int64).sum()), 27*self.ncell
