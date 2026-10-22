"""T68 (E58): the localized barrier on the 27 Bernstein coefficients of the cell's triquadratic J.

J restricted to a Q1 cell has degree <= 2 in each reference coordinate, so its values at the
closed27 points (0, 1/2, 1)^3 determine it exactly, and beta = (T x T x T) J are its Bernstein
coefficients (T = [[1,0,0], [-1/2,2,-1/2], [0,0,1]] per axis; the certificate's
lagrange_to_bernstein). min beta <= min over the cell of J (convex hull property), so a finite
barrier on beta implies J > delta on the whole cell. Point index p = 9*iz+3*iy+ix (closed27
operator order); coefficient index k uses the same layout.
    B = w * sum_cells sum_k psi(beta_k),  dB/dJ_p = w * sum_k T_kp psi'(beta_k),
    GGN (dJ/du)^T T^T diag(w psi''(beta)) T (dJ/du).
"""
import numpy as np


T1 = np.array([[1., 0., 0.], [-.5, 2., -.5], [0., 0., 1.]])
TB = np.kron(T1, np.kron(T1, T1))          # (27, 27): row k (z,y,x), column p (z,y,x)
BASES = ("samples", "bernstein")
UNIT_ROUNDOFF = 2.**-53
BOUND_FACTOR = 3e4       # per-cell a-priori bound B_c = BOUND_FACTOR*u*phi^2*(sigma+phi) (E58b)


class BarrierDomainError(FloatingPointError):
    """A Bernstein coefficient is <= delta: the trial lies outside the T68 barrier domain."""


def to_bernstein(J):
    """(..., 27) closed27 values -> (..., 27) Bernstein coefficients (= J @ TB.T).

    Separable and elementwise (x, then y, then z; middle entry 2*b - .5*(a+c), the operations of
    the certificate and of the Triton kernels), so a cell's coefficients depend only on its own
    values, never on how many cells share the batch (a BLAS matmul does not guarantee that).
    """
    b = np.array(J, dtype=np.float64, order="C")   # C order: the reshape below is a view of b
    v = b.reshape(b.shape[:-1]+(3, 3, 3))
    v[..., 1] = 2.*v[..., 1]-.5*(v[..., 0]+v[..., 2])
    v[..., 1, :] = 2.*v[..., 1, :]-.5*(v[..., 0, :]+v[..., 2, :])
    v[..., 1, :, :] = 2.*v[..., 1, :, :]-.5*(v[..., 0, :, :]+v[..., 2, :, :])
    return b


def to_points(g):
    """Transpose map (= g @ TB): per axis T^T = [[1,-.5,0],[0,2,0],[0,-.5,1]], x then y then z."""
    b = np.array(g, dtype=np.float64, order="C")
    v = b.reshape(b.shape[:-1]+(3, 3, 3))
    for axis in (-1, -2, -3):
        w = np.moveaxis(v, axis, -1)                 # a view: writes go to b
        d1 = w[..., 1].copy()
        w[..., 0] -= .5*d1
        w[..., 2] -= .5*d1
        w[..., 1] = 2.*d1
    return b


def torch_matrix(device):
    import torch
    return torch.as_tensor(TB, dtype=torch.float64, device=device)


def barrier_values(problem, J):
    """The quantity the barrier acts on: J itself (samples) or its Bernstein coefficients."""
    return to_bernstein(J) if problem.barrier_basis == "bernstein" else J


def point_first(problem, first):
    """d B / d J at the points from d B / d (barrier values)."""
    return to_points(first) if problem.barrier_basis == "bernstein" else first


def min_coefficient(problem, u, *, workers=None, fast=False):
    """CPU min over all cells of the barrier values (for transfer/domain checks).

    Exact path: the tiles are exactly the energy path's (problem.memory.cell_tile_size), so
    every cell's values, and hence the min (order independent), are bitwise those of the serial
    loop; tiles run on the E54 thread pool. The work-budget checkpoint is a context variable:
    pool threads do not inherit it, so with several workers the budget is checked only here,
    before and after the scan (with one worker the per-tile checkpoints also run, inline).

    fast=True (Bernstein basis only): a compiled single-pass kernel returns, per cell, its min
    coefficient beta_c and an a-priori bound B_c on |compiled - exact| (both paths: 8-term dot
    products for F, the determinant, the separable transform; see _min_beta_kernel). If
    min_c(beta_c - B_c) > delta every exact value is > delta: accept, return the compiled min.
    If some beta_c + B_c <= delta that cell's exact value is <= delta: reject, return the
    compiled min. Otherwise (or on any non-finite value) the exact path decides and its value
    is returned. The decision is therefore always the exact path's.
    """
    from .work_control import checkpoint
    checkpoint("bernstein_domain", completed=0, total=problem.grid.n_cells)
    if fast and problem.barrier_basis == "bernstein":
        value, lower, upper, finite = _compiled_min_beta(problem, u, workers)
        delta = problem.certificate_config.delta
        if finite and (lower > delta or upper <= delta):
            checkpoint("bernstein_domain", completed=problem.grid.n_cells, total=problem.grid.n_cells)
            return value
    best = _exact_min(problem, u, workers)
    checkpoint("bernstein_domain", completed=problem.grid.n_cells, total=problem.grid.n_cells)
    return best


def _exact_min(problem, u, workers):
    from .cell_q1 import cell_tiles
    from .geometry import _compute_cell_geometry
    from .thread_chunks import run_chunks
    tiles = list(cell_tiles(problem.grid, problem.memory.cell_tile_size))

    def kernel(lo, hi):
        best = np.inf
        for tile in tiles[lo:hi]:
            J = _compute_cell_geometry(u, problem.barrier_ops, tile=tile).J
            best = min(best, float(barrier_values(problem, J).min()))
        return best
    # NumPy tiles hold the GIL between large temporaries; more than 4 threads measured slower.
    from .thread_chunks import scan_workers
    workers = min(4, scan_workers()) if workers is None else workers
    return min(run_chunks(kernel, len(tiles), workers=workers, chunks_per_worker=1))


_KERNEL = None


def _min_beta_kernel():
    global _KERNEL
    if _KERNEL is None:
        from numba import njit

        @njit(cache=False, fastmath=False, nogil=True, error_model='numpy')
        def kernel(u, D, offsets, cy, cx, ny, nx, lo, hi, unit, factor):
            """Over cells [lo, hi): min beta, min(beta_c - B_c), min(beta_c + B_c), all finite.

            beta: the 27 Bernstein coefficients of det(I + grad u) at the closed27 points.
            B_c = factor*unit*phi^2*(sigma+phi), sigma = max_{r,j,p} sum_a |u_ar D_jpa|,
            phi = max |F_rj| over the cell's points. Per path, each F entry is within
            gamma_9*(sigma+phi); the determinant then within 18 phi^2 of that plus 6 gamma_5 phi^3;
            the transform multiplies by <= 27 (row sums of T x T x T) and adds 6*unit*27*6 phi^3.
            Two paths: about unit*(8.7e3 phi^2 (sigma+phi) + 3.6e3 phi^3) <= 1.3e4 unit phi^2 (sigma+phi);
            factor = 3e4 leaves room for second-order terms.
            """
            best = np.inf
            lower = np.inf
            upper = np.inf
            finite = True
            b = np.empty(27)
            F = np.empty((3, 3))
            for c in range(lo, hi):
                x0 = c % cx
                y0 = (c//cx) % cy
                z0 = c//(cx*cy)
                base = (z0*ny+y0)*nx+x0
                sigma = 0.
                phi = 0.
                for p in range(27):
                    for r in range(3):
                        for j in range(3):
                            acc = 0.
                            mag = 0.
                            for a in range(8):
                                term = u[base+offsets[a], r]*D[j, p, a]
                                acc += term
                                mag += abs(term)
                            F[r, j] = acc
                            if mag > sigma:
                                sigma = mag
                        F[r, r] += 1.
                    for r in range(3):
                        for j in range(3):
                            if abs(F[r, j]) > phi:
                                phi = abs(F[r, j])
                    b[p] = (F[0, 0]*(F[1, 1]*F[2, 2]-F[1, 2]*F[2, 1])
                            + F[0, 1]*(F[1, 2]*F[2, 0]-F[1, 0]*F[2, 2])
                            + F[0, 2]*(F[1, 0]*F[2, 1]-F[1, 1]*F[2, 0]))
                for stride in (1, 3, 9):       # separable T along x, y, z (index 9*iz+3*iy+ix)
                    for p in range(27):
                        if (p//stride) % 3 == 1:
                            b[p] = 2.*b[p]-.5*(b[p-stride]+b[p+stride])
                bound = factor*unit*phi*phi*(sigma+phi)
                cell = np.inf
                for p in range(27):
                    v = b[p]
                    if not (v == v) or v == np.inf or v == -np.inf:
                        finite = False
                    elif v < cell:
                        cell = v
                if not (bound == bound) or bound == np.inf:
                    finite = False
                if cell < best:
                    best = cell
                if cell-bound < lower:
                    lower = cell-bound
                if cell+bound < upper:
                    upper = cell+bound
            return best, lower, upper, finite
        _KERNEL = kernel
    return _KERNEL


def _compiled_min_beta(problem, u, workers):
    """(min beta, min(beta_c - B_c), min(beta_c + B_c), all finite) over all cells."""
    from ..validation import InputValidationError
    from .cell_q1 import VERTICES
    from .thread_chunks import run_chunks
    grid = problem.grid
    nz, ny, nx = grid.shape
    cz, cy, cx = grid.cell_shape
    u = np.asarray(u, dtype=np.float64)
    if u.shape != grid.shape+(3,):
        raise InputValidationError("Bernstein domain check: u must have the grid shape + (3,)")
    flat = np.ascontiguousarray(u.reshape(-1, 3))
    D = np.ascontiguousarray(problem.barrier_ops.coefficients)
    offsets = np.array([(z*ny+y)*nx+x for x, y, z in VERTICES], dtype=np.int64)
    kernel = _min_beta_kernel()
    parts = run_chunks(lambda lo, hi: kernel(flat, D, offsets, cy, cx, ny, nx, lo, hi, UNIT_ROUNDOFF, BOUND_FACTOR),
                       grid.n_cells, workers=workers)
    return (min(v[0] for v in parts), min(v[1] for v in parts), min(v[2] for v in parts), all(v[3] for v in parts))
