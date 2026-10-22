"""T62 (E53): node JD measure and affine factors of the conservative A1 metric.

Only the SPD PG metric uses these values; they are not a certificate, never
enter F/E, and never decide acceptance. Closed27 point order is the
CellQ1Operators3D one: index 9*iz+3*iy+ix for reference nodes (0, 1/2, 1).
"""
import numpy as np

from ...validation import InputValidationError, as_float64_array, finite_real, require_finite_result
from ..cell_q1 import cell_node_indices, cell_tiles
from ..geometry import _compute_cell_geometry

JD_AFFINE_FLOOR = 1e-12
JD_AFFINE_ACTIVATION = .25   # pre-registered E53 value
EXTENSION_POLICY = "bernstein-candidates-9cubed-4start-bounded-newton-30-v1"   # T69 (E59)

# Three closed27 points (t=0, 1/2, 1) on each of the 12 cell edges.
EDGE_INDICES = np.array(
    [[9*iz+3*iy+ix for ix in range(3)] for iz in (0, 2) for iy in (0, 2)]
    + [[9*iz+3*iy+ix for iy in range(3)] for iz in (0, 2) for ix in (0, 2)]
    + [[9*iz+3*iy+ix for iz in range(3)] for iy in (0, 2) for ix in (0, 2)])


def edge_quadratic_minima(f0, f1, f2):
    """Exact interior minimum of q(t)=f0+B t+A t^2 through t=0,1/2,1; inf if none.

    A=2(f0-2f1+f2), B=-3f0+4f1-f2; an interior minimum exists iff A>0 and
    0<-B<2A (t*=-B/(2A) in (0,1)). Endpoints are closed27 values already.
    """
    with np.errstate(over="raise", invalid="raise"):
        a = 2*(f0-2*f1+f2)
        b = -3*f0+4*f1-f2
        inside = (a > 0) & (b < 0) & (-b < 2*a)
        out = np.full(np.shape(f0), np.inf)
        out[inside] = f0[inside]-b[inside]**2/(4*a[inside])
    return out


def closed27_cell_minima(J):
    """m_c = min(closed27 values, exact edge-quadratic minima) per cell; J is (cells, 27)."""
    J = np.asarray(J, dtype=np.float64)
    if J.ndim != 2 or J.shape[1] != 27:
        raise InputValidationError("closed27 JD values must have shape (cells, 27)")
    require_finite_result(J, "T62 closed27 JD")
    f = J[:, EDGE_INDICES]
    return np.minimum(J.min(axis=1), edge_quadratic_minima(f[..., 0], f[..., 1], f[..., 2]).min(axis=1))


# ---- T69 (E59): face/interior extension of m_c on candidate cells -------------------------
# J on a cell is triquadratic: J = sum C[k,j,i] x^i y^j z^k with C from the 27 closed27 values
# through L_a(t) = sum_i M[a,i] t^i (Lagrange on 0, 1/2, 1). Every operation below acts on one cell's
# row; only the BLAS product J27 @ _V9.T could in principle round differently with the batch, which
# affects the metric at rounding level only.
_M = np.array([[1., -3., 2.], [0., 4., -4.], [0., -1., 2.]])
_T9 = np.linspace(0., 1., 9)


def _lagrange(t):
    t = np.asarray(t, dtype=np.float64)
    return np.stack([2*(t-.5)*(t-1), -4*t*(t-1), 2*t*(t-.5)], axis=-1)


_V9 = np.kron(_lagrange(_T9), np.kron(_lagrange(_T9), _lagrange(_T9)))      # (729, 27), z-major rows
_G9 = np.array([(x, y, z) for z in _T9 for y in _T9 for x in _T9])            # matching (x, y, z)


def _monomial(J27):
    return np.einsum("ncba,ck,bj,ai->nkji", J27.reshape(-1, 3, 3, 3), _M, _M, _M).reshape(-1, 27)


def _powers(t):
    one, zero = np.ones_like(t), np.zeros_like(t)
    return np.stack([one, t, t*t], -1), np.stack([zero, one, 2*t], -1), np.stack([zero, zero, 2*one], -1)


def _row(Z, Y, X):
    return (Z[:, :, None, None]*Y[:, None, :, None]*X[:, None, None, :]).reshape(len(Z), 27)


def _value(C, x):
    return np.sum(C*_row(_powers(x[:, 2])[0], _powers(x[:, 1])[0], _powers(x[:, 0])[0]), axis=1)


def _derivatives(C, x):
    (px, dx, hx), (py, dy, hy), (pz, dz, hz) = _powers(x[:, 0]), _powers(x[:, 1]), _powers(x[:, 2])
    e = lambda Z, Y, X: np.sum(C*_row(Z, Y, X), axis=1)
    g = np.stack([e(pz, py, dx), e(pz, dy, px), e(dz, py, px)], -1)
    H = np.empty((len(x), 3, 3))
    H[:, 0, 0], H[:, 1, 1], H[:, 2, 2] = e(pz, py, hx), e(pz, hy, px), e(hz, py, px)
    H[:, 0, 1] = H[:, 1, 0] = e(pz, dy, dx)
    H[:, 0, 2] = H[:, 2, 0] = e(dz, py, dx)
    H[:, 1, 2] = H[:, 2, 1] = e(dz, dy, px)
    return g, H


def _bounded_newton(C, x, best, iterations):
    """Monotone projected steps on [0,1]^3: Newton on the free coordinates when their Hessian is
    positive definite, else a scaled gradient step; halving backtracking, strict decrease only.
    A coordinate is fixed while it sits on a face with the gradient pointing outward."""
    x, best = x.copy(), best.copy()
    live = np.ones(len(x), dtype=bool)
    for _ in range(iterations):
        idx = np.flatnonzero(live)
        if idx.size == 0:
            break
        xi, Ci = x[idx], C[idx]
        g, H = _derivatives(Ci, xi)
        free = ~(((xi <= 0.) & (g > 0)) | ((xi >= 1.) & (g < 0)))
        gf = np.where(free, g, 0.)
        Hf = H*free[:, :, None]*free[:, None, :]+np.eye(3)*(~free)[:, :, None]
        w, Q = np.linalg.eigh(Hf)
        scale = np.maximum(1., np.abs(w).max(axis=1))
        ok = np.all(w > 1e-12*scale[:, None], axis=1)
        newton = -np.einsum("nij,nj,nkj,nk->ni", Q, 1/np.where(ok[:, None], w, 1.), Q, gf)
        d = np.where(ok[:, None], newton, -gf/scale[:, None])
        step, done = np.ones(idx.size), np.zeros(idx.size, dtype=bool)
        for _ in range(40):
            xt = np.clip(xi+step[:, None]*d, 0., 1.)
            ft = _value(Ci, xt)
            take = (~done) & (ft < best[idx])
            x[idx[take]], best[idx[take]] = xt[take], ft[take]
            done |= take
            step = np.where(done, step, .5*step)
            if done.all():
                break
        live[idx[~done]] = False          # no strict decrease: converged to rounding (or stuck)
    return best


def face_interior_minima(J27, *, starts=4, iterations=30):
    """mu_c per cell (T69): min of the 9^3 grid values and of bounded Newton from the `starts`
    best grid points. mu_c is a value the cell's J attains, so mu_c >= its true minimum; it is an
    estimate for the SPD metric, never a lower bound or a certificate."""
    J27 = np.asarray(J27, dtype=np.float64)
    if J27.ndim != 2 or J27.shape[1] != 27:
        raise InputValidationError("closed27 JD values must have shape (cells, 27)")
    require_finite_result(J27, "T69 closed27 JD")
    n = len(J27)
    if n == 0:
        return np.empty(0)
    with np.errstate(over="raise", invalid="raise"):
        C = _monomial(J27)
        grid_values = J27 @ _V9.T
        order = np.argsort(grid_values, axis=1, kind="stable")[:, :starts]
        best = grid_values.min(axis=1)
        x0 = _G9[order.T.ravel()]                                  # start-major (starts*n, 3)
        Cs = np.tile(C, (order.shape[1], 1))
        found = _bounded_newton(Cs, x0, _value(Cs, x0), iterations).reshape(order.shape[1], n)
        result = np.minimum(best, found.min(axis=0))
    require_finite_result(result, "T69 face/interior minima")
    return result


def extension_candidates(J, minima, activation):
    """Cells whose Bernstein hull allows a value below min(m_c, J_act); only these can change r."""
    from ..bernstein_barrier import to_bernstein
    return np.flatnonzero(to_bernstein(J).min(axis=1) < np.minimum(minima, activation))


PARALLEL_ROWS = 4096   # E66: below this many candidate cells the estimator runs inline


def face_interior_minima_parallel(J27):
    """face_interior_minima over row chunks on the E54 thread pool (NumPy releases the GIL).

    Every row is estimated independently, so a row's value can differ from the serial call only
    through the rounding of the per-chunk BLAS product J27 @ _V9.T (metric only)."""
    from ..thread_chunks import run_chunks
    J27 = np.asarray(J27, dtype=np.float64)
    if len(J27) < PARALLEL_ROWS:
        return face_interior_minima(J27)
    parts = run_chunks(lambda lo, hi: face_interior_minima(J27[lo:hi]), len(J27), chunks_per_worker=1)
    return np.concatenate(parts)


def extend_minima(J, minima, activation, stats, compiled=False):
    """In place: m_c <- min(m_c, mu_c) on candidate cells; stats accumulate.
    compiled (E81): the bit-identical Numba estimator (jd_affine_compiled)."""
    candidates = extension_candidates(J, minima, activation)
    if candidates.size:
        if compiled:
            from .jd_affine_compiled import COMPILED_ESTIMATOR, face_interior_minima_parallel as estimator
            if stats is not None:
                stats["estimator"] = COMPILED_ESTIMATOR          # provenance; values are the NumPy estimator's
        else:
            estimator = face_interior_minima_parallel
        mu = estimator(J[candidates])
        _count(stats, candidates.size, int(np.sum(mu < minima[candidates]*(1-LOWERED_RELATIVE))),
               int(np.sum(mu < minima[candidates])), float(mu.min()))
        minima[candidates] = np.minimum(minima[candidates], mu)
    return minima


LOWERED_RELATIVE = 1e-3    # T69 record: a cell counts as lowered if mu_c < (1 - 1e-3) m_c


def _count(stats, candidates, lowered, lowered_any, mu_min):
    if stats is not None:
        stats["candidates"] = stats.get("candidates", 0)+int(candidates)
        stats["lowered_cells"] = stats.get("lowered_cells", 0)+int(lowered)          # by more than 0.1%
        stats["lowered_cells_any"] = stats.get("lowered_cells_any", 0)+int(lowered_any)  # including rounding
        stats["mu_min"] = min(stats.get("mu_min", float("inf")), float(mu_min))


def node_min_jd(problem, state, *, extension_activation=None, stats=None, compiled=False):
    """CPU J_a = min over cells containing node a of m_c (T62), full grid shape.

    extension_activation (T69, triggered): m_c also includes the face/interior estimate mu_c
    on candidate cells (min beta_c < min(m_c, J_act)); None keeps T62 exactly."""
    if problem.barrier_location != "closed27_uniform" or problem.barrier_activation is None:
        raise InputValidationError("T62 requires the closed27 local barrier route")
    grid = problem.grid
    ops = problem.barrier_ops
    u = as_float64_array(state.u, "T62 u", grid.shape+(3,))   # validated once, not per tile
    node = np.full(grid.shape, np.inf)
    flat = node.ravel()
    for tile in cell_tiles(grid, problem.memory.cell_tile_size):
        J = _compute_cell_geometry(u, ops, tile=tile).J
        minima = closed27_cell_minima(J)
        if extension_activation is not None:
            minima = extend_minima(J, minima, extension_activation, stats, compiled=compiled)
        indices = cell_node_indices(grid, tile)
        for a in range(8):
            np.minimum.at(flat, indices[:, a], minima)
    return node


def affine_factors(node_jd, grid, activation):
    """r = min(1, max(J_a, floor)/J_act) on free nodes, in the (3, interior) gradient layout."""
    activation = finite_real(activation, "JD affine activation", positive=True)
    if not JD_AFFINE_FLOOR < activation <= 1:
        raise InputValidationError("JD affine activation must be in (floor, 1]")
    node = np.asarray(node_jd, dtype=np.float64)
    if node.shape != grid.shape:
        raise InputValidationError("node JD must have the grid shape")
    inner = node[1:-1, 1:-1, 1:-1]
    if not np.all(np.isfinite(inner)):
        # Every free node belongs to a cell (+inf = unvisited); nan/-inf = invalid geometry.
        raise FloatingPointError("T62 node JD not finite")
    r = np.minimum(1., np.maximum(inner, JD_AFFINE_FLOOR)/activation)
    require_finite_result(r, "T62 affine factors")
    return np.broadcast_to(r, (3,)+r.shape).ravel(order="C")
