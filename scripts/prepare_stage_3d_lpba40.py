"""D1 (2026-10-14, docs/design_draft_D1_LPBA40.md): freeze the LPBA40 pair set in the
E11 input format (pair.npz + manifest.json per pair, selection.json), so that run_stage_3d_e48_pg.py --pair can use it.

- Images: nii/subjects/Sxx/delineation/image.nii.gz (skull-stripped, 1 mm, affinely aligned delineation space); labels: the
  same folder's label.nii.gz (56 structures, original ids). Read with jd_curl.three_d.nifti.load_scalar: pure RAS storage
  reorientation, no resampling (the four-level pyramid accepts any shape; no padding).
- Units: the LPBA40 delineation headers leave xyzt units 'unknown' (zooms 1.0; LPBA40 is 1 mm). load_scalar requires explicit
  mm, so each file is first copied to DEST/_nifti_mm/ with nibabel set_xyzt_units('mm') and nothing else changed (data, affine,
  qform/sform untouched); source and copy SHA-256 are recorded in the manifest.
- Intensity: each image divided by the 99.5th percentile of its nonzero voxels, then clipped to [0, 1] (OASIS inputs are in
  [0, ~.86]); recorded per image.
- Label sets (manifest label_sets, read by e11.label_evaluation): lpba56 (primary, the 56 ids) and brain (image > 0, id 1;
  brain_mask_set for the brain SDlogJ). Labels never enter the loss.
- Pairs (selection.json, fixed by rule before any run): role 'test' = all 12 ordered pairs of S37..S40 (splits/test.txt);
  role 'tune' = 12 disjoint consecutive pairs of splits/train.txt in file order (t0 <- t1, t2 <- t3, ...). No metric-based choice.
Usage: python prepare_stage_3d_lpba40.py   (writes data/three_d/stage_LPBA40_v1; refuses to overwrite)"""
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.run_stage_3d_c import dump                                                    # noqa: E402
from jd_curl.three_d.nifti import load_scalar, require_common_grid, file_sha256, dice_labels  # noqa: E402

SRC = Path('/mnt/d/data/LPBA40')
DEST = ROOT/'data/three_d/stage_LPBA40_v1'
PCT = 99.5


def split(name):
    return [l.strip() for l in (SRC/'splits'/f'{name}.txt').read_text().splitlines() if l.strip()]


def schedule():
    test = split('test'); train = split('train')
    assert test == ['S37', 'S38', 'S39', 'S40'], test
    assert len(train) >= 24 and not set(train) & set(test), train
    rows = [dict(pair=f'LPBA40_{f}__LPBA40_{m}', fixed=f, moving=m, role='test') for f in test for m in test if f != m]
    rows += [dict(pair=f'LPBA40_{train[2*k]}__LPBA40_{train[2*k+1]}', fixed=train[2*k], moving=train[2*k+1], role='tune')
             for k in range(12)]
    return rows


def main():
    DEST.mkdir(parents=True, exist_ok=False); started = perf_counter()
    rows = schedule()
    dump(DEST/'selection.json', dict(rule='LPBA40 delineation space; test = all 12 ordered pairs of splits/test.txt (S37..S40); '
                                     'tune = 12 disjoint consecutive pairs of splits/train.txt in file order; fixed <- moving',
                                     pairs=rows, no_metric_based_selection=True, design='docs/design_draft_D1_LPBA40.md'))
    cache = {}

    mm = DEST/'_nifti_mm'; mm.mkdir()

    def with_mm(s, kind):
        import nibabel as nib
        src = SRC/'nii/subjects'/s/'delineation'/f'{kind}.nii.gz'; dst = mm/f'{s}_{kind}.nii.gz'
        if not dst.exists():
            im = nib.load(src); assert all(abs(float(z)-1.) < 1e-6 for z in im.header.get_zooms()[:3])
            out = nib.Nifti1Image(np.asanyarray(im.dataobj), im.affine, im.header.copy()); out.header.set_xyzt_units('mm')
            nib.save(out, dst)
            assert np.array_equal(np.asanyarray(nib.load(dst).dataobj), np.asanyarray(im.dataobj))
            unit_copies[str(dst)] = dict(source=str(src), source_sha256=file_sha256(src), copy_sha256=file_sha256(dst))
        return dst
    unit_copies = {}

    def subject(s, reference):
        if s not in cache:
            img = load_scalar(with_mm(s, 'image'))
            lab = load_scalar(with_mm(s, 'label'), labels=True)
            grid = require_common_grid(img, lab)
            v = np.asarray(img.values, dtype=np.float64); scale = float(np.percentile(v[v > 0], PCT))
            cache[s] = dict(grid=grid, image=np.clip(v/scale, 0., 1.), label=np.asarray(lab.values, dtype=np.float64),
                            brain=(v > 0).astype(np.float64), scale=scale, image_meta=img.metadata, label_meta=lab.metadata, raw=img)
        if reference is not None:
            require_common_grid(reference, cache[s]['raw'])
        return cache[s]

    ids = None
    for row in rows:
        out = DEST/row['pair']; out.mkdir()
        f = subject(row['fixed'], None); m = subject(row['moving'], f['raw']); grid = f['grid']
        assert grid.spacing == (1., 1., 1.)
        if ids is None:
            ids = sorted(int(i) for i in np.unique(f['label']) if i != 0); assert len(ids) == 56, len(ids)
        assert sorted(int(i) for i in np.unique(m['label']) if i != 0) == ids
        arrays = dict(fixed=f['image'], moving=m['image'], fixed_lpba56=f['label'], moving_lpba56=m['label'],
                      fixed_brain=f['brain'], moving_brain=m['brain'], initial_u=np.zeros(grid.shape+(3,)))
        label_sets = dict(lpba56=ids, brain=[1])
        labels = {k: dice_labels(arrays['fixed_'+k], arrays['moving_'+k], labels=v) for k, v in label_sets.items()}
        q = grid.node_weights()
        initial_mse = float(np.sum(q*(arrays['moving']-arrays['fixed'])**2, dtype=np.longdouble)/np.sum(q, dtype=np.longdouble))
        L = np.prod([a[-1]-a[0] for a in grid.axes])**(1/3)
        np.savez_compressed(out/'pair.npz', **arrays)
        sources = {'fixed': f['image_meta'], 'fixed_lpba56': f['label_meta'], 'moving': m['image_meta'], 'moving_lpba56': m['label_meta']}
        record = dict(dataset='LPBA40 delineation space (skull-stripped, 56 labels)', pair=row['pair'], role=row['role'],
                      fixed_subject=row['fixed'], moving_subject=row['moving'], mapping='fixed -> moving pullback',
                      shape_zyx=grid.shape, spacing_xyz=grid.spacing, origin_xyz=[float(a[0]) for a in grid.axes], unit='mm',
                      preprocessing=f'pure RAS storage reorientation; intensity / p{PCT} of nonzero voxels, clipped to [0, 1]; no resampling',
                      intensity_scale=dict(fixed=f['scale'], moving=m['scale']), sources=sources,
                      unit_copies={k: v for k, v in unit_copies.items() if f"/{row['fixed']}_" in k or f"/{row['moving']}_" in k},
                      pair_sha256=file_sha256(out/'pair.npz'), baseline_label_metrics=labels, baseline_native_mse=initial_mse,
                      weights=dict(lam=.1/L**2, kappa=.002, strong_mu_nu=1., weak_mu_nu=.001),
                      label_sets=label_sets, label_primary='lpba56', label_audit_only='brain', brain_mask_set='brain',
                      labels_policy='lpba56 primary (56 structures); brain = image > 0; no labels in loss')
        dump(out/'manifest.json', record); assert all(file_sha256(s['source']) == s['sha256'] for s in sources.values())
        print(row['pair'], row['role'], 'baseline lpba56', round(labels['lpba56']['macro_dice'], 4), flush=True)
    dump(DEST/'completion.json', dict(complete=True, pairs=len(rows), elapsed_seconds=perf_counter()-started, no_optimizer_runs=True))


if __name__ == '__main__':
    main()
