"""E94 (G2/C4, implementation only): fused Triton per-cell T62/T69 measure, bit for bit the torch path.

TorchLinearization.node_min_closed27_jd computes, tile by tile, J at the 27 closed points (einsum with the closed27
table, cross product, three-term sum), the exact edge-quadratic minima, and with the T69 extension the Bernstein
minimum that selects the candidate cells. The generated kernel (scripts/gen_triton_node_jd.py) reproduces those
per-cell values bit for bit in one pass, without materializing F: the measured cuBLAS/torch evaluation orders are
written out explicitly and the kernel is compiled with enable_fp_fusion=False. Candidate cells, the CPU estimator,
the node scatter and the stats stay the torch code. Equality rests on measured library behaviour, so the backend
checks every level grid's first use bit for bit against the torch path and switches the kernel off on a mismatch.
Metric only: never a bound, a certificate or an acceptance input.
"""
import numpy as np
import torch

from ...validation import InputValidationError

POLICY = "triton-fused-node-jd-v1"
BLOCK, WARPS = 64, 2


def _check_trace_rows():
    """The first-use check compares cell_j27 on 3*BLOCK + 5 cells: the torch trace must keep that many rows."""
    from .torch_ops import TRACE_ROWS
    if TRACE_ROWS < 3*BLOCK+5:
        raise InputValidationError("E94: TRACE_ROWS must cover three kernel blocks")


_check_trace_rows()


def supported(device):
    if device.type != "cuda":
        return False
    try:
        import triton  # noqa: F401
    except Exception:
        return False
    return True


def exact_edge_form(D, spacing):
    """True iff every entry of the closed27 table is +-w*fl(1/h_j), w the Q1 bilinear weight (0, 1/4, 1/2, 1)."""
    D = np.asarray(D, dtype=np.float64)
    if D.shape != (3, 27, 8) or not np.all(np.isfinite(D)):
        return False
    ih = [1/float(h) for h in spacing]
    pts = [(x, y, z) for z in (0., .5, 1.) for y in (0., .5, 1.) for x in (0., .5, 1.)]
    for j in range(3):
        for p, point in enumerate(pts):
            for a in range(8):
                bits = (a & 1, (a >> 1) & 1, (a >> 2) & 1)
                w = 1.
                for k in range(3):
                    if k != j:
                        w *= point[k] if bits[k] else 1-point[k]
                if D[j, p, a] != (1. if bits[j] else -1.)*w*ih[j]:
                    return False
    return True


def tables_match():
    """The edge list and Bernstein map the kernel was generated from equal the live ones."""
    from . import triton_node_jd_kernels as K
    from ..bernstein_barrier import TB
    from ..solvers.jd_affine import EDGE_INDICES
    M = np.asarray(TB, dtype=np.float64).T
    live = [[(p, float(M[p, k])) for p in range(27) if M[p, k] != 0.] for k in range(27)]
    return (np.array_equal(np.asarray(K.EDGES), np.asarray(EDGE_INDICES)) and
            [[(int(p), float(v)) for p, v in col] for col in K.M_COLUMNS] == live)


class FusedNodeJD:
    """Per-grid constants and output buffers; cell_values() returns the torch path's per-cell values."""

    def __init__(self, grid, D_host, device):
        from . import triton_node_jd_kernels as K
        if not exact_edge_form(D_host, grid.spacing):
            raise InputValidationError("E94 fused node JD: closed27 table is not the exact Q1 edge form")
        if not tables_match():
            raise InputValidationError("E94 fused node JD: kernel tables differ from the live edge/Bernstein tables")
        self._K = K
        self.D_host = D_host
        self.device = device
        self.ih = [1/float(h) for h in grid.spacing]
        nz, ny, nx = grid.shape
        self.shape, self.ny, self.nx, self.cy, self.cx = grid.shape, ny, nx, ny-1, nx-1
        self.ncell = grid.n_cells
        self.minima = torch.empty(self.ncell, dtype=torch.float64, device=device)
        self.cand = torch.zeros(self.ncell, dtype=torch.int8, device=device)
        self.bad = torch.empty(self.ncell, dtype=torch.int8, device=device)

    def cell_values(self, device_u, activation=None):
        """(minima, candidate mask or None, any nonfinite J) for a contiguous float64 (nz, ny, nx, 3) array.
        The returned tensors are this object's buffers (overwritten by the next call)."""
        if (device_u.dtype != torch.float64 or not device_u.is_contiguous()
                or tuple(device_u.shape) != self.shape+(3,)):
            raise InputValidationError("E94 fused node JD needs a contiguous float64 node array")
        import triton
        ext = activation is not None
        k = torch.tensor(self.ih+[float(activation) if ext else 0.], dtype=torch.float64, device=self.device)
        self._K.node_jd_kernel[(triton.cdiv(self.ncell, BLOCK),)](
            device_u, k, self.minima, self.cand, self.bad, self.ncell, self.ny, self.nx, self.cy, self.cx,
            EXT=ext, BLOCK=BLOCK, num_warps=WARPS, enable_fp_fusion=False)
        return self.minima, (self.cand.bool() if ext else None), bool(self.bad.any())

    def cell_j27(self, device_u, cells):
        """(len(cells), 27) closed27 J of the listed cells (int64 device tensor), the same values as cell_values."""
        import triton
        cells = cells.to(torch.int64).contiguous()
        out = torch.empty((cells.numel(), 27), dtype=torch.float64, device=self.device)
        if cells.numel():
            k = torch.tensor(self.ih+[0.], dtype=torch.float64, device=self.device)
            self._K.cell_j27_kernel[(triton.cdiv(cells.numel(), BLOCK),)](
                device_u, k, cells, out, cells.numel(), self.ny, self.nx, self.cy, self.cx,
                BLOCK=BLOCK, num_warps=WARPS, enable_fp_fusion=False)
        return out


def prewarm():
    """Compile both kernel variants outside the task clock (integer arguments are not specialized)."""
    from ..cell_q1 import CellQ1Operators3D
    from ..grid import make_grid_3d
    grid = make_grid_3d(nx=7, ny=6, nz=5, hx=1., hy=1., hz=1., unit='mm')
    device = torch.device('cuda')
    fused = FusedNodeJD(grid, CellQ1Operators3D(grid, location="closed27_uniform").coefficients, device)
    u = torch.zeros(grid.shape+(3,), dtype=torch.float64, device=device)
    fused.cell_values(u)
    fused.cell_values(u, .25)
    fused.cell_j27(u, torch.arange(3, device=device))
    torch.cuda.synchronize()
