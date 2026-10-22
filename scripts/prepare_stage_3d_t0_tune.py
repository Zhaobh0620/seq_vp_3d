"""E88 (T0): freeze a 12-pair tuning set from the Learn2Reg OASIS TRAINING range, disjoint from the 19 validation pairs and
from the development/E11 pairs, in the E11 input format (same reader and checks as prepare_stage_3d_l2r_val.py; no
spatial resampling).

Pre-registered selection rule (written before any registration on these subjects):
- Subjects are the lines of /mnt/d/data/subjects.txt (Learn2Reg id = line number). Training range = lines 1..394.
  Lines 1..10 (OAS1_0001..0011, the development pair and the four E11 pairs) are excluded; validation lines 395..414 are
  outside the range anyway.
- Candidate pairs follow the Learn2Reg convention: fixed = line n, moving = line n+1, for n = 11, 12, ..., 393.
- Each candidate gets r = ln(LV_moving / LV_fixed), LV = voxel count of seg35 labels 3 and 22 (lateral ventricles) in
  aligned_seg35. Stratum: 'moving larger' r > .5, 'moving smaller' r < -.5, 'balanced' |r| <= .5. (E87 and the 2026-10-05
  audit found that ventricle-volume mismatch explains 54% of the across-pair seg35 variance; the 19 validation pairs are
  6 larger / 6 smaller / 7 balanced.)
- Walk the candidates in ascending n and accept a pair if its stratum has fewer than 4 pairs and neither subject is in an
  accepted pair; stop at 12 pairs (4 per stratum). Labels are used only for this stratification; no registration outcome
  is looked at.
- The 24 selected subjects are reserved for tuning: they must never be used to train a segmentation network or any other
  learned component (P26 in docs/performance_analysis_levers_20261005.md).
Usage: python prepare_stage_3d_t0_tune.py   (writes data/three_d/stage_T0_tune_v1; refuses to overwrite)"""
from pathlib import Path, PureWindowsPath
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.run_stage_3d_c import dump                                                    # noqa: E402
from jd_curl.three_d.nifti import load_scalar, require_common_grid, file_sha256, dice_labels  # noqa: E402

DEST = ROOT/'data/three_d/stage_T0_tune_v1'
FIRST_LINE, LAST_LINE = 11, 394            # training range without the development/E11 subjects (lines 1..10)
QUOTA, THRESHOLD, LV_LABELS = 4, .5, (3, 22)


def subject_lines():
    lines = [l.strip() for l in Path('/mnt/d/data/subjects.txt').read_text().splitlines() if l.strip()]
    return {i: Path('/mnt/d/data')/PureWindowsPath(lines[i-1]).parent.name for i in range(1, len(lines)+1)}


def lv_volume(subject, cache):
    if subject not in cache:
        v = load_scalar(subject/'aligned_seg35.nii.gz', labels=True)
        cache[subject] = int(np.isin(v.values, LV_LABELS).sum())
    return cache[subject]


def select(subj):
    e11 = {p for row in __import__('json').loads((ROOT/'data/three_d/stage_E11_native_v1/selection.json').read_text())['pairs']
           for p in row['pair'].split('__')}
    assert {subj[i].name for i in range(1, 11)} == e11, 'lines 1..10 must be exactly the development/E11 subjects'
    cache, used, chosen, walked = {}, set(), {'moving larger': [], 'moving smaller': [], 'balanced': []}, []
    for n in range(FIRST_LINE, LAST_LINE):
        f, m = subj[n], subj[n+1]
        r = float(np.log(lv_volume(m, cache)/lv_volume(f, cache)))
        stratum = 'moving larger' if r > THRESHOLD else 'moving smaller' if r < -THRESHOLD else 'balanced'
        accepted = len(chosen[stratum]) < QUOTA and f not in used and m not in used
        walked.append(dict(line=n, fixed=f.name, moving=m.name, log_lv_ratio=r, stratum=stratum, accepted=accepted))
        if accepted:
            chosen[stratum].append(dict(pair=f'{f.name}__{m.name}', fixed=str(f), moving=str(m), l2r_fixed=f'OASIS_{n:04d}',
                                        l2r_moving=f'OASIS_{n+1:04d}', log_lv_ratio=r, stratum=stratum))
            used |= {f, m}
        if all(len(v) == QUOTA for v in chosen.values()):
            break
    assert all(len(v) == QUOTA for v in chosen.values()), {k: len(v) for k, v in chosen.items()}
    schedule = sorted((p for v in chosen.values() for p in v), key=lambda p: int(p['l2r_fixed'][6:]))
    return schedule, walked, cache


def main():
    DEST.mkdir(parents=True, exist_ok=False); started = perf_counter()
    subj = subject_lines()
    assert subj[394].name == 'OASIS_OAS1_0437_MR1' and subj[395].name == 'OASIS_OAS1_0438_MR1', (subj[394], subj[395])
    schedule, walked, cache = select(subj)
    reserved = sorted({Path(p[k]).name for p in schedule for k in ('fixed', 'moving')})
    dump(DEST/'selection.json', dict(
        rule=__doc__.split('Pre-registered selection rule')[1].split('Usage:')[0].strip(),
        pairs=schedule, walked_candidates=walked, lv_voxels={k.name: v for k, v in cache.items()},
        reserved_subjects_never_for_training=reserved, no_metric_based_selection=True))
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
        record = dict(dataset='OASIS native aligned full-volume, E88 tuning pair (Learn2Reg training range)', pair=pair['pair'],
                      l2r_fixed=pair['l2r_fixed'], l2r_moving=pair['l2r_moving'], stratum=pair['stratum'],
                      log_lv_ratio=pair['log_lv_ratio'],
                      fixed_subject=Path(pair['fixed']).name, moving_subject=Path(pair['moving']).name, mapping='fixed -> moving pullback',
                      shape_zyx=grid.shape, spacing_xyz=grid.spacing, origin_xyz=[float(a[0]) for a in grid.axes], unit='mm',
                      preprocessing='pure RAS storage reorientation only; no resampling/smoothing/intensity adjustment', sources=sources,
                      pair_sha256=file_sha256(out/'pair.npz'), baseline_label_metrics=labels, baseline_native_mse=initial_mse,
                      weights=dict(lam=.1/L**2, kappa=.002, strong_mu_nu=1., weak_mu_nu=.001),
                      labels_policy='seg4 primary; seg35 audit-only; no labels in loss; labels used only to stratify the selection')
        dump(out/'manifest.json', record); assert all(file_sha256(s['source']) == s['sha256'] for s in sources.values())
        print(pair['pair'], pair['l2r_fixed'], pair['stratum'], 'ln LV ratio %+.2f' % pair['log_lv_ratio'],
              'baseline seg4', round(labels['seg4']['macro_dice'], 4), 'seg35', round(labels['seg35']['macro_dice'], 4), flush=True)
    dump(DEST/'completion.json', dict(complete=True, pairs=len(schedule), elapsed_seconds=perf_counter()-started, no_optimizer_runs=True))


if __name__ == '__main__':
    main()
