"""T66 (E55): global closed27 cubic JD step prediction. A trial-step heuristic, NEVER a certificate.

For every cell and closed27 point, q(tau) = det(A + tau V) - delta with A = I + grad u (base) and
V = grad p (direction). The first positive root in (0, tau0] is found per point on normalized
s = tau/tau0: split (0, 1] at the roots of q' (monotone pieces), take the first piece whose end
has q <= 0, bisect it 50 times and keep the lower end (q > 0 there). The global minimum over all
points is returned; the solver uses 0.9 * that (same safety as the historical predictor).
"""
from time import perf_counter

import numpy as np

from ...validation import InputValidationError
from ..boundary import inject_free
from ..cell_q1 import cell_node_indices, cell_tiles
from ..geometry import _compute_cell_geometry, cofactor

POLICY = "global-closed27-cubic-v1"
BISECTIONS = 50


def _critical_points(b1, b2, b3):
    """Roots of 3 b3 s^2 + 2 b2 s + b1 in (0, 1), sorted; invalid entries are 1."""
    a, b, c = 3*b3, 2*b2, b1
    one = np.ones_like(b1)
    with np.errstate(all='ignore'):
        linear = a == 0
        r_lin = np.where(linear & (b != 0), -c/np.where(b != 0, b, 1.), np.nan)
        disc = b*b-4*a*c
        sq = np.sqrt(np.where(disc >= 0, disc, 0.))
        q = -(b+np.where(b >= 0, sq, -sq))/2
        r1 = np.where(~linear & (disc >= 0) & (a != 0), q/np.where(a != 0, a, 1.), np.nan)
        r2 = np.where(~linear & (disc >= 0) & (q != 0), c/np.where(q != 0, q, 1.), np.nan)
        r1 = np.where(linear, r_lin, r1)
        r2 = np.where(linear, np.nan, r2)
    ok1 = (r1 > 0) & (r1 < 1)
    ok2 = (r2 > 0) & (r2 < 1)
    r1 = np.where(ok1, r1, one)
    r2 = np.where(ok2, r2, one)
    return np.minimum(r1, r2), np.maximum(r1, r2)


def _q(b0, b1, b2, b3, s):
    return b0+s*(b1+s*(b2+s*b3))


def _touch(b0, b1, b2, b3, s):
    """Rounding scale of q(s): an end value q <= 4 eps * sum |b_k| s^k counts as touching zero."""
    return 4*np.finfo(float).eps*(np.abs(b0)+s*(np.abs(b1)+s*(np.abs(b2)+s*np.abs(b3))))


def first_roots(b0, b1, b2, b3):
    """First root in (0, 1] of b0 + b1 s + b2 s^2 + b3 s^3 with b0 > 0 (lower bracket end); inf if none.

    q > 0 at the returned value unless it equals a near-tangent critical point (|q| within rounding),
    which is then returned as the root estimate. Outside the rounding band of q the estimate is never
    later than the true first root; inside it (near-double roots) it can be late by about sqrt(eps)
    relative, which the 0.9 safety factor absorbs. When (0,1) has no valid critical point, the
    placeholder end s=1 also gets the rounding tolerance (a root 'touching' at s=1 is reported).
    """
    b0, b1, b2, b3 = (np.asarray(x, dtype=np.float64) for x in (b0, b1, b2, b3))
    out = np.full(b0.shape, np.inf)
    with np.errstate(all='ignore'):
        maybe = ~(b0 > np.abs(b1)+np.abs(b2)+np.abs(b3))
    if not np.any(maybe):
        return out
    b0, b1, b2, b3 = b0[maybe], b1[maybe], b2[maybe], b3[maybe]
    t1, t2 = _critical_points(b1, b2, b3)
    one = np.ones_like(b0)
    q1 = _q(b0, b1, b2, b3, t1) <= _touch(b0, b1, b2, b3, t1)
    q2 = _q(b0, b1, b2, b3, t2) <= _touch(b0, b1, b2, b3, t2)
    q3 = _q(b0, b1, b2, b3, one) <= 0
    # A near-tangent end (|q| within rounding) counts as a root: conservative (earlier) estimate.
    hi = np.where(q1, t1, np.where(q2, t2, np.where(q3, 1., np.nan)))
    lo = np.where(q1, 0., np.where(q2, t1, np.where(q3, t2, np.nan)))
    has = np.isfinite(hi)
    lo, hi = lo[has], hi[has]
    c = [x[has] for x in (b0, b1, b2, b3)]
    for _ in range(BISECTIONS):
        mid = .5*(lo+hi)
        neg = _q(*c, mid) <= 0
        hi = np.where(neg, mid, hi)
        lo = np.where(neg, lo, mid)
    roots = np.full(b0.shape, np.inf)
    roots[has] = lo
    out[maybe] = roots
    return out


def point_coefficients(F, V, delta):
    """(c0, c1, c2, c3) of det(F + tau V) - delta; F includes the identity, V does not."""
    AF, AV = cofactor(F), cofactor(V)
    c0 = np.sum(F[..., 0, :]*AF[..., 0, :], axis=-1)-delta
    c1 = np.sum(AF*V, axis=(-2, -1))
    c2 = np.sum(F*AV, axis=(-2, -1))
    c3 = np.sum(V[..., 0, :]*AV[..., 0, :], axis=-1)
    return c0, c1, c2, c3


def summarize(root_s, limit, invalid_base, points, started):
    s = float(np.min(root_s)) if np.size(root_s) else np.inf
    root = None if not np.isfinite(s) else s*limit
    return dict(policy=POLICY, limit=float(limit), root=root, points=int(points),
                invalid_base_points=int(invalid_base), seconds=perf_counter()-started)


def global_root_cpu(problem, base, direction, limit):
    """CPU reference implementation over all cells (tests/small grids)."""
    started = perf_counter()
    if problem.barrier_location != "closed27_uniform":
        raise InputValidationError("T66 uses the closed27 barrier points")
    grid, ops, delta = problem.grid, problem.barrier_ops, problem.certificate_config.delta
    field = inject_free(direction, grid)   # (nz, ny, nx, 3)
    best, invalid, points = np.inf, 0, 0
    with np.errstate(over='raise', invalid='raise'):
        for tile in cell_tiles(grid, problem.memory.cell_tile_size):
            F = _compute_cell_geometry(base.u, ops, tile=tile).F
            local = field.reshape(-1, 3)[cell_node_indices(grid, tile)]            # (cells, 8, 3)
            V = np.einsum("jpa,tar->tprj", ops.coefficients, local)              # grad p, no identity
            c0, c1, c2, c3 = point_coefficients(F, V, delta)
            coefficients = np.stack((c0, c1*limit, c2*limit**2, c3*limit**3))
            if not np.all(np.isfinite(coefficients)):
                raise FloatingPointError("T66 nonfinite cubic coefficients")
            valid = c0 > 0
            invalid += int(np.count_nonzero(~valid))
            points += c0.size
            if np.any(valid):
                r = first_roots(*(x[valid] for x in coefficients))
                best = min(best, float(r.min()))
    return summarize(np.array([best]), limit, invalid, points, started)
