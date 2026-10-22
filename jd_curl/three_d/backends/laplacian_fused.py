"""E101 (C3, throughput-v19): the T76 seven-point Laplacian of a node field in one Triton pass, bit for bit
torch_ops.laplacian_nodes.

torch evaluates, on the interior nodes,
    ((f[x+1] - 2.*c + f[x-1])*ix + (f[y+1] - 2.*c + f[y-1])*iy) + (f[z+1] - 2.*c + f[z-1])*iz
with one IEEE rounding per operation (2.*c is exact; the subtraction is a + (-1)*b, exact negation then one rounding),
left to right, and +0 on the boundary nodes (zeros_like). The kernel performs the same operations in the same order,
FP contraction off; ix, iy, iz come from a float64 device tensor (never Python floats: Triton 3.1 makes those fp32).
The first use on every level grid is compared with the torch path on a deterministic rough synthetic field (not
u = 0, which would make every value 0); a mismatch or a failure switches the kernel off for the run.
"""
import numpy as np
import torch

POLICY = "triton-fused-laplacian-v1"
BLOCK, WARPS = 256, 8

try:
    import triton
    import triton.language as tl
except ImportError:          # pragma: no cover
    triton = tl = None


if triton is not None:
    @triton.jit(do_not_specialize=["n", "nz", "ny", "nx"])          # one binary for every level grid
    def laplacian_kernel(F, INV, OUT, n, nz, ny, nx, BLOCK: tl.constexpr):
        e = tl.program_id(0).to(tl.int64)*BLOCK+tl.arange(0, BLOCK).to(tl.int64)
        m = e < n
        node = e//3
        x = node % nx
        y = (node//nx) % ny
        z = node//(nx*ny)
        inner = m & (x > 0) & (x < nx-1) & (y > 0) & (y < ny-1) & (z > 0) & (z < nz-1)
        ix = tl.load(INV+2)
        iy = tl.load(INV+1)
        iz = tl.load(INV+0)
        c = tl.load(F+e, mask=inner, other=0.)
        c2 = c+c                                   # == 2.*c exactly (a doubling is exact, overflow alike)
        sx = 3
        sy = 3*nx
        sz = 3*nx*ny
        tx = (tl.load(F+e+sx, mask=inner, other=0.)-c2)+tl.load(F+e-sx, mask=inner, other=0.)
        ty = (tl.load(F+e+sy, mask=inner, other=0.)-c2)+tl.load(F+e-sy, mask=inner, other=0.)
        tz = (tl.load(F+e+sz, mask=inner, other=0.)-c2)+tl.load(F+e-sz, mask=inner, other=0.)
        out = (tx*ix+ty*iy)+tz*iz
        zero = tl.zeros_like(out)
        tl.store(OUT+e, tl.where(inner, out, zero), mask=m)


def new_record():
    return dict(policy=POLICY, calls=0, verified_grids=[], compared=0, mismatches=0, errors=0, unsupported=0,
                disabled=False)


def supported(device):
    return triton is not None and device.type == "cuda"


def _synthetic(shape, device):
    generator = torch.Generator(device=device)
    generator.manual_seed(101)
    field = torch.randn(shape, generator=generator, dtype=torch.float64, device=device)
    field = field*torch.exp(3*torch.randn(shape, generator=generator, dtype=torch.float64, device=device))
    return field.contiguous()


class FusedLaplacian:
    """laplacian_nodes(field, inv) on one level grid's node fields (nz, ny, nx, 3), float64, contiguous."""

    def __init__(self, record, inv, device):
        self.record = record
        self.inv = torch.tensor([float(v) for v in inv], dtype=torch.float64, device=device)   # (iz, iy, ix)
        self._inv_host = tuple(float(v) for v in inv)

    def _launch(self, field):
        nz, ny, nx = field.shape[:3]
        out = torch.empty_like(field)
        n = field.numel()
        laplacian_kernel[(triton.cdiv(n, BLOCK),)](field, self.inv, out, n, nz, ny, nx, BLOCK=BLOCK,
                                                   num_warps=WARPS, enable_fp_fusion=False)
        return out

    def __call__(self, field, reference):
        """reference(field) is the torch laplacian_nodes; it decides until this grid is verified."""
        record = self.record
        if (record["disabled"] or field.dtype != torch.float64 or not field.is_contiguous()
                or field.dim() != 4 or field.shape[3] != 3):
            return reference(field)
        key = list(field.shape)
        if key not in record["verified_grids"]:          # per level grid, shared by the backend's models (record)
            try:
                probe = _synthetic(field.shape, field.device)
                same = torch.equal(self._launch(probe).view(torch.int64), reference(probe).view(torch.int64))
            except Exception:
                record["errors"] += 1
                record["disabled"] = True
                return reference(field)
            record["compared"] += 1
            if not same:
                record["mismatches"] += 1
                record["disabled"] = True
                return reference(field)
            record["verified_grids"].append(key)
        try:
            out = self._launch(field)
        except Exception:
            record["errors"] += 1
            record["disabled"] = True
            return reference(field)
        record["calls"] += 1
        return out


def prewarm():
    """Compile the kernel outside any task clock."""
    from ..grid import make_grid_3d
    from ..curvature import inverse_square_spacing
    g = make_grid_3d(nx=5, ny=6, nz=7, hx=1., hy=1., hz=1., x0=0., y0=0., z0=0., unit='mm')
    kernel = FusedLaplacian(new_record(), inverse_square_spacing(g), torch.device('cuda'))
    kernel._launch(torch.zeros(g.shape+(3,), dtype=torch.float64, device='cuda'))
    torch.cuda.synchronize()
