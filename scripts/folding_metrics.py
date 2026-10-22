"""WP-0.3 (engineering plan 2026-10-04): auxiliary folding indicators, reported NEXT TO the certificate, never
instead of it (read-only; no solver call).

- Tetrahedral volume ratios of every Q1 cell of the node map y = x + u: 24 tetrahedra (FAIR/Heldmann: edge
  endpoints, face centre, cell centre), 6 tetrahedra (Kuhn, main diagonal corner 1 -> 8) and 5 tetrahedra.
  Each ratio is det(deformed tetrahedron edges) / det(identity edges), so the identity gives exactly 1 and no
  per-face sign bookkeeping is needed. Neither direction implies the certificate's whole-cell J > 0 (the plan
  gives counterexamples both ways), so these are proxies (P1 Heldmann-24 etc.).
- Consistency identity: for a trilinear map the mean of the 24 ratios equals the cell average of J, which is the
  mean of the 27 Bernstein coefficients of J (checked in the tests).
- Learn2Reg-style indicators (as described in the plan, eval_version 2.0; verify against the official script before
  quoting): central differences with zero padding, J = I + grad u (u in voxels), 2-voxel crop, std of
  log(clip(det + 3, 1e-9, 1e9)) over the whole volume; and the fraction of voxels with central-difference det <= 0.
Corner order: 1..8 = offsets (x, y, z) 000, 100, 010, 110, 001, 101, 011, 111 (x fastest), arrays are (z, y, x, 3).
Usage: python folding_metrics.py out.json run_dir...   (final_native.npz of each run)
"""
import json
import sys
from pathlib import Path

import numpy as np

OFFSETS = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)])   # (x, y, z)
FACES = {'z0': (0, 1, 3, 2), 'x1': (1, 3, 7, 5), 'y0': (0, 1, 5, 4), 'x0': (2, 0, 4, 6), 'y1': (3, 2, 6, 7), 'z1': (4, 5, 7, 6)}
KUHN = [(0, 1, 3, 7), (0, 1, 5, 7), (0, 2, 3, 7), (0, 2, 6, 7), (0, 4, 5, 7), (0, 4, 6, 7)]
FIVE = [(0, 3, 5, 6), (1, 0, 3, 5), (2, 0, 3, 6), (4, 0, 5, 6), (7, 3, 5, 6)]


def _det(a, b, c):
    return (a[..., 0]*(b[..., 1]*c[..., 2]-b[..., 2]*c[..., 1]) - a[..., 1]*(b[..., 0]*c[..., 2]-b[..., 2]*c[..., 0])
            + a[..., 2]*(b[..., 0]*c[..., 1]-b[..., 1]*c[..., 0]))


def _tets24(P):
    """P: (..., 8, 3) corner positions -> list of 24 (A, B, C, D) point arrays."""
    M = P.mean(axis=-2)
    out = []
    for ring in FACES.values():
        F = sum(P[..., i, :] for i in ring)/4.
        for k in range(4):
            out.append((P[..., ring[k], :], P[..., ring[(k+1) % 4], :], F, M))
    return out


def _tets(P, scheme):
    if scheme == 24:
        return _tets24(P)
    table = KUHN if scheme == 6 else FIVE
    return [(P[..., a, :], P[..., b, :], P[..., c, :], P[..., d, :]) for a, b, c, d in table]


def _ratios(P, P0, scheme):
    out = []
    for (a, b, c, d), (a0, b0, c0, d0) in zip(_tets(P, scheme), _tets(P0, scheme)):
        out.append(_det(b-a, c-a, d-a)/_det(b0-a0, c0-a0, d0-a0))
    return np.stack(out, axis=-1)


def _corner_positions(u, spacing, zs):
    """Local corner coordinates (corner 1 subtracted) of the cells in z-slab zs; returns (P, P0) as (..., 8, 3) xyz."""
    hx, hy, hz = spacing
    ny, nx = u.shape[1]-1, u.shape[2]-1
    P = np.empty((len(range(*zs.indices(u.shape[0]-1))), ny, nx, 8, 3))
    base = u[zs, :-1, :-1]
    for k, (ox, oy, oz) in enumerate(OFFSETS):
        z = slice(zs.start+oz, zs.stop+oz)
        P[..., k, :] = u[z, oy:oy+ny, ox:ox+nx]-base+np.array([ox*hx, oy*hy, oz*hz])
    P0 = np.array([[ox*hx, oy*hy, oz*hz] for ox, oy, oz in OFFSETS], dtype=np.float64)
    return P, np.broadcast_to(P0, P.shape)


def tet_ratio_summary(u, spacing, schemes=(24, 6, 5), slab=8):
    """Per scheme: min ratio, fraction of cells with a ratio <= 0, fraction of tetrahedra <= 0; and the per-cell
    mean of the 24 ratios (cell average J) for the consistency check."""
    u = np.asarray(u, dtype=np.float64)
    nzc = u.shape[0]-1
    stats = {s: dict(min=np.inf, cells_nonpositive=0, tets_nonpositive=0, tets=0) for s in schemes}
    cells = 0
    for z0 in range(0, nzc, slab):
        P, P0 = _corner_positions(u, spacing, slice(z0, min(z0+slab, nzc)))
        cells += P.shape[0]*P.shape[1]*P.shape[2]
        for s in schemes:
            r = _ratios(P, P0, s)
            st = stats[s]
            st['min'] = min(st['min'], float(r.min()))
            st['cells_nonpositive'] += int(np.sum(np.any(r <= 0, axis=-1)))
            st['tets_nonpositive'] += int(np.sum(r <= 0))
            st['tets'] += int(r.size)
    return {f"tet{s}": dict(min_ratio=st['min'], cells_nonpositive=st['cells_nonpositive'], cells=cells,
                            tets_nonpositive=st['tets_nonpositive'], tets=st['tets']) for s, st in stats.items()}


def cell_mean_ratio24(u, spacing):
    P, P0 = _corner_positions(np.asarray(u, dtype=np.float64), spacing, slice(0, u.shape[0]-1))
    return _ratios(P, P0, 24).mean(axis=-1)


def l2r_indicators(u, spacing):
    """Learn2Reg-style (per the plan's description): u in mm on a (z, y, x) grid -> voxel units, central differences
    with zero padding, J = I + grad u, 2-voxel crop, SDlogJ = std log clip(det + 3, 1e-9, 1e9); plus the fraction of
    voxels with det <= 0 (our own report, same differences)."""
    hx, hy, hz = spacing
    v = np.asarray(u, dtype=np.float64)/np.array([hx, hy, hz])            # components x, y, z in voxels
    p = np.pad(v, ((1, 1), (1, 1), (1, 1), (0, 0)))
    d = [(p[2:, 1:-1, 1:-1]-p[:-2, 1:-1, 1:-1])/2., (p[1:-1, 2:, 1:-1]-p[1:-1, :-2, 1:-1])/2., (p[1:-1, 1:-1, 2:]-p[1:-1, 1:-1, :-2])/2.]
    dz, dy, dx = d                                                        # derivatives along z, y, x of all components
    J = [[(1. if r == c else 0.)+(dx, dy, dz)[c][..., r] for c in range(3)] for r in range(3)]
    det = (J[0][0]*(J[1][1]*J[2][2]-J[1][2]*J[2][1]) - J[0][1]*(J[1][0]*J[2][2]-J[1][2]*J[2][0])
           + J[0][2]*(J[1][0]*J[2][1]-J[1][1]*J[2][0]))
    crop = det[2:-2, 2:-2, 2:-2]
    return dict(l2r_sdlogj=float(np.log(np.clip(crop+3., 1e-9, 1e9)).std()),
                central_det_nonpositive_fraction=float(np.mean(crop <= 0)), central_det_min=float(crop.min()))


def run_summary(run):
    run = Path(run)
    proto = json.loads((run/'protocol.json').read_text())
    with np.load(run/proto['task']['name']/'final_native.npz') as z:
        u = np.array(z['u'])
    man = json.loads((run/'inputs'/proto['task']['pair']/'manifest.json').read_text())
    spacing = tuple(man['spacing_xyz'])
    summary = json.loads((run/proto['task']['name']/'summary.json').read_text())
    out = dict(run=run.name, certificate_min_jd_lower_bound=summary['certificate']['min_jd_lower_bound'],
               **tet_ratio_summary(u, spacing), **l2r_indicators(u, spacing))
    return out


if __name__ == '__main__':
    rows = []
    for r in sys.argv[2:]:
        s = run_summary(r); rows.append(s)
        print(f"{s['run'][:44]:44s} cert minJD {s['certificate_min_jd_lower_bound']:.2e} | tet24 min {s['tet24']['min_ratio']:.3g} "
              f"nonpos cells {s['tet24']['cells_nonpositive']} | tet6 min {s['tet6']['min_ratio']:.3g} nonpos {s['tet6']['cells_nonpositive']} | "
              f"tet5 min {s['tet5']['min_ratio']:.3g} nonpos {s['tet5']['cells_nonpositive']} | L2R SDlogJ {s['l2r_sdlogj']:.4f} "
              f"central det<=0 {s['central_det_nonpositive_fraction']:.2e} (min {s['central_det_min']:.3g})", flush=True)
    Path(sys.argv[1]).write_text(json.dumps(rows, indent=1))
