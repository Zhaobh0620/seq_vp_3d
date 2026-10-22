"""T75 (E71): rigorous floating-point Bernstein prescreen for the full-cell JD certificate.

For a Q1 cell, J = det(I + grad u) restricted to the cell is triquadratic; its 27 values at the
closed27 points (0, 1/2, 1)^3 determine it, and its Bernstein coefficients beta = (T x T x T) J
satisfy min beta <= min over the cell of J (convex hull). The interval fallback of the
certificate computes beta with outward rounding. This kernel computes beta in ordinary
float64 arithmetic and a per-cell a-priori bound B on |beta_fl - beta_exact|; if
min beta_fl - B > delta, then J > delta on the whole cell, rigorously. Cells that do not pass
go to the unchanged interval fallback, so the prescreen can only add proofs, never remove or
weaken one, and it never declares a cell infeasible.

Bound (u = 2^-53, gamma_n = n u / (1 - n u), eta = 1e-290 covers gradual underflow). Full proof: docs/04 T75 and the
paper's Appendix C (manuscript outside this repository); the constants below are the ones the kernel uses, each at
least the one the proof needs:
  sigma = max_{p,r,j} sum_a |u_ar D_jpa|,  phi = max_{p,r,j} |F_rj|  (computed),
  e     = gamma_12 (sigma + phi) + eta         error of every F entry vs the exact F. The proof needs
          gamma_10 sigma + gamma_1 phi: each product u*D carries two roundings (D = fl(1/h) times an exact factor,
          and the product), the sequential 8-term sum seven more (gamma_9 against the exact magnitudes, gamma_10
          against the computed sigma), and the diagonal adds the identity with one rounding (gamma_1 phi);
          gamma_12 (sigma + phi) is used, with slack,
  Phi   = phi + e                              upper bound of the exact entries,
  E_J   = 3 sqrt 3 e (3 Phi^2 + 3 Phi e + e^2) + 6 gamma_6 Phi^3
          (det is multilinear in the rows; Hadamard |det| <= prod row norms, rows <= sqrt3 Phi,
           error rows <= sqrt3 e; plus the evaluation error of the fixed cofactor expression,
           at most five roundings on any path: gamma_5 times the six monomial magnitudes, gamma_6 used),
  M     = max_p |J_fl(p)|,
  B     = 1.01 (27 E_J + 135 gamma_3 M) + eta
          (T x T x T has absolute row sums at most 27, which propagates the J error. The transform's own rounding:
           each pass replaces every middle entry x_m of a line by 2 x_m - (x_l + x_r)/2 with two roundings (the
           halving and the doubling are exact), committing at most gamma_2 (2|x_m| + |x_l|/2 + |x_r|/2) <= 3 gamma_2 X
           for inputs bounded by X; the inputs of the three passes are bounded by M, 3M, 9M (times (1 + 3 gamma_2)
           powers) and a later pass amplifies an earlier error by at most 3, so the proof needs
           (9*3 + 3*9 + 27) gamma_2 M (1 + 3 gamma_2)^2 = 81 gamma_2 M (1 + 3 gamma_2)^2; 135 gamma_3 M is used, with
           slack; the factor 1.01 covers the roundings in evaluating E_J and B themselves).
The derivation assumes IEEE float64 with every +, -, x rounded separately in the stated order: no fused multiply-add
and no reassociation. The kernel is compiled with fastmath=False, which gives LLVM no permission to contract or
reassociate; a one-off inspection of the generated code (2026-10-18, paper review) found no FMA. There is no
regression test of the generated code for this kernel (the interval kernels E32/E33/E98 have one).
The coefficient table must be finite with no subnormal nonzero entry (spacing above ~2^1020 makes
f/h subnormal, so 'D carries u' fails); otherwise no cell is proven here (all NaN).
"""
import numpy as np

from ..validation import InputValidationError

UNIT = 2.**-53
ETA = 1e-290
SQRT27_UP = 5.196152422706633   # >= 3 sqrt 3


def _gamma(n):
    return n*UNIT/(1-n*UNIT)


G12, G6, G3 = _gamma(12), _gamma(6), _gamma(3)
PRESCREEN_KERNEL = "numba-float-bernstein-apriori-v1"
_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        from numba import njit

        @njit(cache=False, fastmath=False, nogil=True, error_model='numpy')
        def kernel(u, D, offsets, cells, cy, cx, ny, nx, lo, hi, g12, g6, g3, sq27, eta, out):
            """out[i] = min beta_fl - B for cells[i] (NaN if any value is not finite)."""
            b = np.empty(27)
            F = np.empty((3, 3))
            for i in range(lo, hi):
                c = cells[i]
                x0 = c % cx
                y0 = (c//cx) % cy
                z0 = c//(cx*cy)
                base = (z0*ny+y0)*nx+x0
                sigma = 0.
                phi = 0.
                big = 0.
                finite = True
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
                    v = (F[0, 0]*(F[1, 1]*F[2, 2]-F[1, 2]*F[2, 1])
                         + F[0, 1]*(F[1, 2]*F[2, 0]-F[1, 0]*F[2, 2])
                         + F[0, 2]*(F[1, 0]*F[2, 1]-F[1, 1]*F[2, 0]))
                    b[p] = v
                    if abs(v) > big:
                        big = abs(v)
                for stride in (1, 3, 9):       # separable T along x, y, z (index 9*iz+3*iy+ix)
                    for p in range(27):
                        if (p//stride) % 3 == 1:
                            b[p] = 2.*b[p]-.5*(b[p-stride]+b[p+stride])
                e = g12*(sigma+phi)+eta
                P = phi+e
                ej = sq27*e*(3.*P*P+3.*P*e+e*e)+6.*g6*P*P*P
                bound = 1.01*(27.*ej+135.*g3*big)+eta
                low = np.inf
                for p in range(27):
                    w = b[p]
                    if not (w == w) or w == np.inf or w == -np.inf:
                        finite = False
                    elif w < low:
                        low = w
                if not (bound == bound) or bound == np.inf or not (sigma == sigma) or sigma == np.inf:
                    finite = False
                out[i] = low-bound if finite else np.nan
        _KERNEL = kernel
    return _KERNEL


def prescreen_lower_bounds(u, grid, coefficients, cells, *, workers=None):
    """Rigorous lower bounds of J over each listed cell (NaN where not finite).

    u: (nz, ny, nx, 3) float64 nodes; coefficients: the closed27 Q1 derivative table
    (3, 27, 8) of the same grid; cells: increasing int64 cell ids."""
    from .cell_q1 import VERTICES
    from .thread_chunks import run_chunks
    u = np.asarray(u, dtype=np.float64)
    if u.shape != grid.shape+(3,):
        raise InputValidationError("prescreen: u must have the grid shape + (3,)")
    D = np.ascontiguousarray(np.asarray(coefficients, dtype=np.float64))
    if D.shape != (3, 27, 8):
        raise InputValidationError("prescreen: closed27 coefficient table (3, 27, 8) required")
    cells = np.ascontiguousarray(np.asarray(cells, dtype=np.int64))
    out = np.full(len(cells), np.nan)
    nonzero = np.abs(D[D != 0])
    if not np.all(np.isfinite(D)) or (nonzero.size and nonzero.min() < np.finfo(np.float64).tiny):
        # Review E71: the bound assumes every table entry is fl(1/h) times an exact factor, within u.
        # A subnormal (or non-finite) entry breaks that, so such grids get no prescreen proofs.
        return out
    nz, ny, nx = grid.shape
    cz, cy, cx = grid.cell_shape
    flat = np.ascontiguousarray(u.reshape(-1, 3))
    offsets = np.array([(z*ny+y)*nx+x for x, y, z in VERTICES], dtype=np.int64)
    if len(cells):
        kernel = _kernel()
        run_chunks(lambda lo, hi: kernel(flat, D, offsets, cells, cy, cx, ny, nx, lo, hi,
                                         G12, G6, G3, SQRT27_UP, ETA, out), len(cells), workers=workers)
    return out
