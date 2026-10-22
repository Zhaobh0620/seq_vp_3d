"""WP-0.2: Learn2Reg-style evaluation of saved final native maps (read-only).

Re-implements the official definitions of MDL-UzL/L2R evaluation (eval_version 2.0, evaluation.py + utils.py) — the
repository has no licence file, so the code is not copied; scripts/../results cross-checks it against the official
functions (see docs). Arrays are ours ((z, y, x), u in mm with components x, y, z); all metrics are invariant to the
axis permutation/flip between our arrays and the release:
- labels warped by scipy map_coordinates(moving_seg, identity + disp_vox, order=0) (outside -> 0);
- Dice per label 1..35; a label missing in the fixed or in the UNWARPED moving labels is NaN; case value = nanmean;
- HD95 = surface_distance.compute_robust_hausdorff(compute_surface_distances(fixed==l, warped==l, ones(3)), 95),
  same NaN rule; case value = nanmean;
- SDlogJ: central differences (correlate with [-.5, 0, .5], zero padding) of disp_vox, J = grad + I, crop 2 voxels,
  std of log(clip(det + 3, 1e-9, 1e9)) over the cropped volume (the "+3" is in the official code);
  num_foldings = count(det + 3 <= 0) (as official);
- aggregate: mean, std (ddof 0) and the .3 quantile of the case means for Dice/HD95 (.7 for SDlogJ); the ranking-style
  DSC30 (mean Dice over the 30% (case, label) entries with the lowest initial Dice) is reported too.
Usage: python evaluate_l2r_style.py out.json run_dir...
"""
import json
import sys
from pathlib import Path

import numpy as np
import scipy.ndimage
from scipy.ndimage import map_coordinates

LABELS = list(range(1, 36))


def disp_voxels(u, spacing_xyz):
    """(z, y, x, 3) mm displacement with xyz components -> (3, z, y, x) voxel displacement in array-axis order."""
    hx, hy, hz = spacing_xyz
    return np.stack([u[..., 2]/hz, u[..., 1]/hy, u[..., 0]/hx])


def warp_labels_l2r(moving_seg, d):
    ident = np.meshgrid(*[np.arange(n) for n in moving_seg.shape], indexing='ij')
    return map_coordinates(moving_seg, ident+d, order=0)


def dice_l2r(fixed, moving, warped, labels=LABELS):
    out = []
    for l in labels:
        if (fixed == l).sum() == 0 or (moving == l).sum() == 0:
            out.append(np.nan)
        else:
            a, b = fixed == l, warped == l
            s = a.sum()+b.sum()
            out.append(2.*np.logical_and(a, b).sum()/s if s else np.nan)
    return float(np.nanmean(out)), out


def hd95_l2r(fixed, moving, warped, labels=LABELS):
    from surface_distance import compute_robust_hausdorff, compute_surface_distances
    out = []
    for l in labels:
        if (fixed == l).sum() == 0 or (moving == l).sum() == 0:
            out.append(np.nan)
        else:
            out.append(float(compute_robust_hausdorff(compute_surface_distances(fixed == l, warped == l, np.ones(3)), 95.)))
    return float(np.nanmean(out)), out


def sdlogj_l2r(d):
    k = np.array([-.5, 0., .5])
    grads = []
    for axis in range(3):
        w = k.reshape([3 if a == axis else 1 for a in range(3)])
        grads.append(np.stack([scipy.ndimage.correlate(d[c], w, mode='constant', cval=0.) for c in range(3)]))
    G = np.stack(grads)                          # G[a, c] = d disp_c / d axis_a
    J = G+np.eye(3).reshape(3, 3, 1, 1, 1)
    J = J[:, :, 2:-2, 2:-2, 2:-2]
    det = (J[0, 0]*(J[1, 1]*J[2, 2]-J[1, 2]*J[2, 1]) - J[1, 0]*(J[0, 1]*J[2, 2]-J[0, 2]*J[2, 1])
           + J[2, 0]*(J[0, 1]*J[1, 2]-J[0, 2]*J[1, 1]))
    jac = np.clip(det+3., 1e-9, 1e9)
    return float(np.log(jac).std()), float((jac <= 0).sum()), float((det <= 0).mean())


def evaluate_run(run):
    run = Path(run)
    proto = json.loads((run/'protocol.json').read_text())
    pair = proto['task']['pair']
    with np.load(run/'inputs'/pair/'pair.npz') as z:
        fixed, moving = np.array(z['fixed_seg35']), np.array(z['moving_seg35'])
    man = json.loads((run/'inputs'/pair/'manifest.json').read_text())
    with np.load(run/proto['task']['name']/'final_native.npz') as z:
        u = np.array(z['u'])
    d = disp_voxels(u, man['spacing_xyz'])
    warped = warp_labels_l2r(moving, d)
    zero = warp_labels_l2r(moving, np.zeros_like(d))
    dice, dice_l = dice_l2r(fixed, moving, warped)
    dice0, dice0_l = dice_l2r(fixed, moving, zero)
    hd, _ = hd95_l2r(fixed, moving, warped)
    sd, folds, neg = sdlogj_l2r(d)
    return dict(run=run.name, pair=pair, dice=dice, dice_labels=dice_l, initial_dice_labels=dice0_l, initial_dice=dice0,
                hd95=hd, sdlogj=sd, num_foldings=folds, central_det_nonpositive_fraction=neg)


def aggregate(rows):
    dice = np.array([r['dice'] for r in rows]); hd = np.array([r['hd95'] for r in rows]); sd = np.array([r['sdlogj'] for r in rows])
    entries = [(r0, r1) for r in rows for r0, r1 in zip(r['initial_dice_labels'], r['dice_labels']) if not (np.isnan(r0) or np.isnan(r1))]
    entries.sort(key=lambda e: e[0])
    k = max(1, int(round(.3*len(entries))))
    return dict(cases=len(rows), dice_mean=float(dice.mean()), dice_std=float(dice.std()), dice30_quantile=float(np.quantile(dice, .3)),
                dsc30_ranking=float(np.mean([e[1] for e in entries[:k]])), hd95_mean=float(hd.mean()), hd95_30=float(np.quantile(hd, .3)),
                sdlogj_mean=float(sd.mean()), sdlogj_70=float(np.quantile(sd, .7)))


if __name__ == '__main__':
    rows = []
    for r in sys.argv[2:]:
        x = evaluate_run(r); rows.append(x)
        print(f"{x['run'][:44]:44s} L2R Dice {x['dice']:.4f} (initial {x['initial_dice']:.4f})  HD95 {x['hd95']:.3f}  SDlogJ {x['sdlogj']:.4f}  "
              f"foldings {x['num_foldings']:.0f}  central det<=0 {x['central_det_nonpositive_fraction']:.2e}", flush=True)
    agg = aggregate(rows) if len(rows) > 1 else None
    if agg:
        print(json.dumps(agg))
    Path(sys.argv[1]).write_text(json.dumps(dict(cases=rows, aggregate=agg), indent=1, default=float))
