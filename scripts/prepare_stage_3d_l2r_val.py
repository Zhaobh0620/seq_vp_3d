"""WP-0.2: freeze the 19 Learn2Reg OASIS validation pairs (fixed OASIS_n <- moving OASIS_{n+1}, n = 0395..0413) in the
E11 input format, with NO spatial resampling (same reader and checks as prepare_stage_3d_e11_native.py).

Learn2Reg ids are positions in the neurite-OASIS subject list (/mnt/d/data/subjects.txt): OASIS_0395 is line 395
(OAS1_0438), ..., OASIS_0414 is line 414 (OAS1_0457) — the DIO mapping int(id) - 395 + 438 agrees for this range.
Orientation differs from the Learn2Reg release (axis permutation/flip); Dice, HD95 and SDlogJ are invariant to it, so
these arrays are evaluated directly (scripts/evaluate_l2r_style.py); only a submission needs the release orientation.
Usage: python prepare_stage_3d_l2r_val.py   (writes data/three_d/stage_L2R_val_v1; refuses to overwrite)"""
from pathlib import Path, PureWindowsPath
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.run_stage_3d_c import dump                                                    # noqa: E402
from jd_curl.three_d.nifti import load_scalar, require_common_grid, file_sha256, dice_labels  # noqa: E402

DEST = ROOT/'data/three_d/stage_L2R_val_v1'
L2R_IDS = list(range(395, 414))          # fixed ids; moving = id + 1


def subject_dirs():
    lines = [l.strip() for l in Path('/mnt/d/data/subjects.txt').read_text().splitlines() if l.strip()]
    def subject(line_no):                 # 1-based line number = Learn2Reg id
        name = PureWindowsPath(lines[line_no-1]).parent.name
        return Path('/mnt/d/data')/name
    return {i: subject(i) for i in range(395, 415)}


def main():
    DEST.mkdir(parents=True, exist_ok=False); started = perf_counter()
    subj = subject_dirs()
    assert subj[395].name == 'OASIS_OAS1_0438_MR1' and subj[414].name == 'OASIS_OAS1_0457_MR1', (subj[395], subj[414])
    schedule = [dict(pair=f'{subj[i].name}__{subj[i+1].name}', fixed=str(subj[i]), moving=str(subj[i+1]),
                     l2r_fixed=f'OASIS_{i:04d}', l2r_moving=f'OASIS_{i+1:04d}') for i in L2R_IDS]
    dump(DEST/'selection.json', dict(rule='Learn2Reg OASIS validation pairs (evaluation_configs/OASIS_VAL_evaluation_config.json): '
                                     'fixed OASIS_n <- moving OASIS_{n+1}, n = 0395..0413; ids are subjects.txt line numbers',
                                     pairs=schedule, no_metric_based_selection=True))
    for pair in schedule:
        out = DEST/pair['pair']; out.mkdir(); arrays = {}; sources = {}; reference = None
        for side in ('fixed', 'moving'):
            for kind in ('norm', 'seg4', 'seg35'):
                v = load_scalar(Path(pair[side])/f'aligned_{kind}.nii.gz', labels=kind != 'norm')
                if reference is None:
                    reference = v
                grid = require_common_grid(reference, v); assert grid.shape == (192, 224, 160) and grid.spacing == (1., 1., 1.)
                key = side if kind == 'norm' else side+'_'+kind; arrays[key] = v.values; sources[key] = v.metadata
        arrays['initial_u'] = np.zeros(grid.shape+(3,)); labels = {}
        for kind, n in [('seg4', 4), ('seg35', 35)]:
            labels[kind] = dice_labels(arrays['fixed_'+kind], arrays['moving_'+kind], labels=list(range(1, n+1)))
        q = grid.node_weights()
        initial_mse = float(np.sum(q*(arrays['moving']-arrays['fixed'])**2, dtype=np.longdouble)/np.sum(q, dtype=np.longdouble))
        L = np.prod([a[-1]-a[0] for a in grid.axes])**(1/3)
        np.savez_compressed(out/'pair.npz', **arrays)
        record = dict(dataset='OASIS native aligned full-volume, Learn2Reg OASIS validation pair', pair=pair['pair'],
                      l2r_fixed=pair['l2r_fixed'], l2r_moving=pair['l2r_moving'],
                      fixed_subject=Path(pair['fixed']).name, moving_subject=Path(pair['moving']).name, mapping='fixed -> moving pullback',
                      shape_zyx=grid.shape, spacing_xyz=grid.spacing, origin_xyz=[float(a[0]) for a in grid.axes], unit='mm',
                      preprocessing='pure RAS storage reorientation only; no resampling/smoothing/intensity adjustment', sources=sources,
                      pair_sha256=file_sha256(out/'pair.npz'), baseline_label_metrics=labels, baseline_native_mse=initial_mse,
                      weights=dict(lam=.1/L**2, kappa=.002, strong_mu_nu=1., weak_mu_nu=.001),
                      labels_policy='seg4 primary; seg35 audit-only; no labels in loss')
        dump(out/'manifest.json', record); assert all(file_sha256(s['source']) == s['sha256'] for s in sources.values())
        print(pair['pair'], pair['l2r_fixed'], 'baseline seg4', round(labels['seg4']['macro_dice'], 4), 'seg35', round(labels['seg35']['macro_dice'], 4), flush=True)
    dump(DEST/'completion.json', dict(complete=True, pairs=len(schedule), elapsed_seconds=perf_counter()-started, no_optimizer_runs=True))


if __name__ == '__main__':
    main()
