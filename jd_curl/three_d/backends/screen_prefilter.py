"""E93 (implementation only): rigorous device prefilter for the closed27 witness screen.

The witness screen (witness_screen.screen) evaluates J at the 27 closed points of every cell
with the torch einsum path, tile by tile, and only acts on tiles whose minimum is <= delta.
This prefilter evaluates the same 27 values per cell once, in a fused Triton kernel, together
with an a-priori bound e (docs/04 §E93) such that |J_legacy - J_fast| <= e at every point, for
any summation order and with or without FMA in the legacy path. A tile is skipped only when
fl(jmin - e) > delta holds for all of its cells, which proves the legacy minimum of that tile is
> delta (nonfinite legacy values count as +inf there). All other tiles run the unchanged legacy
code, so the screen's result, witness and observer calls are bit-identical. A nonfinite value
anywhere only keeps the tile. The prefilter is never a feasibility statement.
"""
from fractions import Fraction
import math

import numpy as np
import torch

from ...validation import InputValidationError

POLICY = "triton-screen-prefilter-v1"
BLOCK, WARPS = 64, 2
U = Fraction(1, 2**53)


def gamma(n):
    """gamma_n = n u / (1 - n u) as an exact fraction."""
    return n*U/(1-n*U)


def up(x):
    """A double >= the exact fraction x."""
    y = float(x)
    return y if Fraction(y) >= x else math.nextafter(y, math.inf)


def supported(device):
    if device.type != "cuda":
        return False
    try:
        import triton  # noqa: F401
    except Exception:
        return False
    return True


def bound_constants(D, spacing):
    """Kernel constants of the E93 bound for one closed27 coefficient table D (3, 27, 8).

    Returns (k, info). ih_j must equal the doubles 1/h_j the operator used (checked: every nonzero
    D entry is +-w * fl(1/h_j) with w in {1, 1/2, 1/4} exactly, and the support is the Q1 edge
    pattern); otherwise the prefilter is not used (None).
    """
    D = np.asarray(D, dtype=np.float64)
    if D.shape != (3, 27, 8) or not np.all(np.isfinite(D)):
        return None, "unexpected coefficient table"
    ih = [1/float(h) for h in spacing]
    pts = [(x, y, z) for z in (0., .5, 1.) for y in (0., .5, 1.) for x in (0., .5, 1.)]
    for j in range(3):
        for p, point in enumerate(pts):
            for a in range(8):
                bits = ((a >> 0) & 1, (a >> 1) & 1, (a >> 2) & 1)
                w = 1.
                for k in range(3):
                    if k != j:
                        w *= point[k] if bits[k] else 1-point[k]
                expected = (1. if bits[j] else -1.)*w*ih[j]     # exact: w is 0, 1/4, 1/2 or 1
                if D[j, p, a] != expected:
                    return None, "coefficients are not the exact Q1 edge form"
    dsum = max(sum((Fraction(abs(float(x))) for x in D[j, p]), Fraction(0)) for j in range(3) for p in range(27))
    cA = up((gamma(8)+gamma(5))*dsum)
    cAo = up(gamma(8)*dsum)
    g3p = up(1+gamma(3))
    inv1mu = up(1/(1-U))
    c12 = up(12*gamma(5))
    two_u = up(2*U)
    safety = up(1+Fraction(1, 2**20))
    tiny = 2.**-1000
    k = ih+[cA, cAo, g3p, inv1mu, c12, two_u, safety, tiny]
    return k, dict(dsum=float(dsum), cA=cA, cAo=cAo)


class ScreenPrefilter:
    """Per-grid constants and buffers; flagged_tiles() keeps every tile that may hold J <= delta."""

    def __init__(self, grid, D_host, delta, device):
        from . import triton_screen_prefilter_kernels as K
        self._K = K
        k, info = bound_constants(D_host, grid.spacing)
        if k is None:
            raise InputValidationError("E93 prefilter unavailable: "+info)
        self.info = info
        self.D_host = D_host
        self.spacing = tuple(float(h) for h in grid.spacing)
        self.delta = float(delta)
        self.device = device
        self.k = torch.tensor(k, dtype=torch.float64, device=device)
        nz, ny, nx = grid.shape
        self.shape, self.ny, self.nx, self.cy, self.cx = grid.shape, ny, nx, ny-1, nx-1
        self.ncell = grid.n_cells
        self._buffers = {}

    def _buffers_for(self, tile_size):
        if tile_size not in self._buffers:
            n_tiles = -(-self.ncell//tile_size)
            jmin = torch.full((n_tiles*tile_size,), math.inf, dtype=torch.float64, device=self.device)
            e = torch.zeros(n_tiles*tile_size, dtype=torch.float64, device=self.device)
            self._buffers[tile_size] = (n_tiles, jmin, e, torch.empty_like(jmin))
        return self._buffers[tile_size]

    def cell_values(self, device_u, tile_size):
        """(jmin, e) per cell (views of length n_cells) for a contiguous float64 (nz, ny, nx, 3) array."""
        if (device_u.dtype != torch.float64 or not device_u.is_contiguous()
                or tuple(device_u.shape) != self.shape+(3,)):
            raise InputValidationError("E93 prefilter needs a contiguous float64 node array")
        n_tiles, jmin, e, _ = self._buffers_for(tile_size)
        import triton
        programs = triton.cdiv(self.ncell, BLOCK)
        self._K.screen_prefilter_kernel[(programs,)](device_u, self.k, jmin, e, self.ncell,
                                                     self.ny, self.nx, self.cy, self.cx,
                                                     BLOCK=BLOCK, num_warps=WARPS)
        return jmin[:self.ncell], e[:self.ncell]

    def flagged_tiles(self, device_u, tile_size):
        """Host bool array: True = the tile may hold a legacy value <= delta (run the legacy code)."""
        n_tiles, jmin, e, margin = self._buffers_for(tile_size)
        self.cell_values(device_u, tile_size)
        torch.sub(jmin, e, out=margin)          # padding: inf - 0 = inf
        keep = ~(margin.view(n_tiles, tile_size).amin(dim=1) > self.delta)    # NaN keeps the tile
        return keep.cpu().numpy()


def prewarm():
    """Compile the kernel outside the task clock (integer arguments are not specialized)."""
    from ..cell_q1 import CellQ1Operators3D
    from ..grid import make_grid_3d
    grid = make_grid_3d(nx=7, ny=6, nz=5, hx=1., hy=1., hz=1., unit='mm')
    device = torch.device('cuda')
    u = torch.zeros(grid.shape+(3,), dtype=torch.float64, device=device)
    ScreenPrefilter(grid, CellQ1Operators3D(grid, location="closed27_uniform").coefficients, 0., device).flagged_tiles(u, 64)
    torch.cuda.synchronize()
