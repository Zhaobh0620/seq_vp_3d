"""T87 (S30-G2, explicit option, path only): the pulled-back direction p~(x) = p(x + u(x)).

The candidate stays u + tau p~ on the same Q1 nodes: the step proposal, the domain bound, the JD step prediction,
the witness screen, the certificate, Armijo and the E/F acceptance are unchanged. Only the direction vector differs:
p is the Poisson direction, p~ samples it (trilinear, nearest-boundary clamp; the six faces of p are zero) at the
displaced node x + u(x), so that the Jacobian of the candidate changes multiplicatively, (I + tau grad p)(phi), instead
of additively (CPU proxy probe of 2026-10-14: 2-4x more JD step room in the first native commits). p~ replaces p only
when it descends clearly: g.p~ < 0 and |g.p~| >= r |g.p| (r = pullback_min_slope_ratio); otherwise the Poisson
direction is kept and the attempt row records the fallback. Numba kernel over z-chunks (order-independent: every
node is written once), with a NumPy/SciPy reference for the tests.
"""
from time import perf_counter

import numpy as np

from ..boundary import inject_free, restrict_free
from ..thread_chunks import run_chunks

PULLBACK_POLICY = "direction-pullback-trilinear-v1"
PULLBACK_MIN_SLOPE_RATIO = .5

try:
    from numba import njit, config as _numba_config
    _NUMBA = not _numba_config.DISABLE_JIT
except ImportError:  # pragma: no cover
    _NUMBA = False


if _NUMBA:
    @njit(cache=False, fastmath=False, nogil=True, error_model="numpy")
    def _pullback_chunk(p, u, ihx, ihy, ihz, out, z0, z1):
        """out[z, y, x, :] = trilinear p at (x + u_x/hx, y + u_y/hy, z + u_z/hz), clamped to the node box; interior
        nodes only (the faces of out stay zero). Returns the count of nonfinite coordinates."""
        nz, ny, nx = u.shape[0], u.shape[1], u.shape[2]
        bad = 0
        for z in range(z0, z1):
            for y in range(1, ny-1):
                for x in range(1, nx-1):
                    px = x+u[z, y, x, 0]*ihx
                    py = y+u[z, y, x, 1]*ihy
                    pz = z+u[z, y, x, 2]*ihz
                    if not (px == px and py == py and pz == pz and abs(px) < np.inf and abs(py) < np.inf and abs(pz) < np.inf):
                        bad += 1
                        continue
                    if px < 0.:
                        px = 0.
                    elif px > nx-1:
                        px = nx-1.
                    if py < 0.:
                        py = 0.
                    elif py > ny-1:
                        py = ny-1.
                    if pz < 0.:
                        pz = 0.
                    elif pz > nz-1:
                        pz = nz-1.
                    x0 = int(np.floor(px)); y0 = int(np.floor(py)); z0f = int(np.floor(pz))
                    if x0 >= nx-1:
                        x0 = nx-2
                    if y0 >= ny-1:
                        y0 = ny-2
                    if z0f >= nz-1:
                        z0f = nz-2
                    fx = px-x0; fy = py-y0; fz = pz-z0f
                    x1 = x0+1; y1 = y0+1; z1f = z0f+1
                    w000 = (1.-fz)*(1.-fy)*(1.-fx); w001 = (1.-fz)*(1.-fy)*fx
                    w010 = (1.-fz)*fy*(1.-fx); w011 = (1.-fz)*fy*fx
                    w100 = fz*(1.-fy)*(1.-fx); w101 = fz*(1.-fy)*fx
                    w110 = fz*fy*(1.-fx); w111 = fz*fy*fx
                    for c in range(3):
                        out[z, y, x, c] = (w000*p[z0f, y0, x0, c]+w001*p[z0f, y0, x1, c]
                                           + w010*p[z0f, y1, x0, c]+w011*p[z0f, y1, x1, c]
                                           + w100*p[z1f, y0, x0, c]+w101*p[z1f, y0, x1, c]
                                           + w110*p[z1f, y1, x0, c]+w111*p[z1f, y1, x1, c])
        return bad


def pullback_reference(direction, u, grid):
    """SciPy reference (order 1, nearest clamp) of the same map, for tests; returns the free vector."""
    from scipy.ndimage import map_coordinates
    p = inject_free(direction, grid)
    nz, ny, nx = grid.shape
    iz, iy, ix = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    hx, hy, hz = grid.spacing
    coords = np.stack([iz+u[..., 2]/hz, iy+u[..., 1]/hy, ix+u[..., 0]/hx])
    out = np.zeros_like(p)
    for c in range(3):
        out[..., c] = map_coordinates(p[..., c], coords, order=1, mode="nearest")
    out[0] = out[-1] = 0.; out[:, 0] = out[:, -1] = 0.; out[:, :, 0] = out[:, :, -1] = 0.
    return restrict_free(out, grid)


def pullback_direction(direction, u, grid):
    """(p~ as a read-only free vector, finite) for a free direction vector and the base displacement u (zyx, 3)."""
    direction = np.asarray(direction, dtype=np.float64)
    if direction.shape != (3*grid.n_free_scalar,) or u.shape != grid.shape+(3,):
        raise ValueError("pullback_direction: direction must be the free vector of the grid and u its displacement field")
    p = inject_free(direction, grid)
    out = np.zeros_like(p)
    hx, hy, hz = grid.spacing
    if not _NUMBA:
        tilde = pullback_reference(direction, u, grid)
        finite = bool(np.all(np.isfinite(u)) and np.all(np.isfinite(tilde)))
        tilde = np.array(tilde, copy=True); tilde.flags.writeable = False
        return tilde, finite
    nz = grid.shape[0]
    bad = sum(run_chunks(lambda lo, hi: _pullback_chunk(p, u, 1./hx, 1./hy, 1./hz, out, max(lo, 1), min(hi, nz-1)),
                         nz))
    tilde = restrict_free(out, grid)
    finite = bool(bad == 0 and np.all(np.isfinite(tilde)))
    tilde.flags.writeable = False
    return tilde, finite


def pullback_counts(rows):
    """Per level (rows carry 'level'; None when absent): attempts with a pull-back record, used, fallbacks
    (not clearly descending or nonfinite), accepted pulled-back steps, JD-refused pulled-back trials, mean seconds."""
    out = {}
    for r in rows:
        m = r.get("pullback")
        if m is None:
            continue
        c = out.setdefault(str(r.get("level")), dict(attempts=0, used=0, fallbacks=0, accepted=0, jd_rejected=0, seconds=0.))
        c["attempts"] += 1
        c["seconds"] += float(m.get("seconds") or 0.)
        if m["used"]:
            c["used"] += 1
            if r.get("accepted"):
                c["accepted"] += 1
            if r.get("reason") == "JACOBIAN_VIOLATION":
                c["jd_rejected"] += 1
        else:
            c["fallbacks"] += 1
    return out
