"""E48+ from-zero PG four-level runs (E34 configuration) with explicit execution profiles.

The mathematical configuration is the E34 hour PG route: Sequential + absolute
closed27 C2 local barrier (activation .05, kappa .002, delta 0), JD-step PG with
5 inner steps, four levels, safe coarse transfer and alpha backtracking. Only
the execution profile (and, when explicitly requested, a later-stage option)
differs. Budget end is NOT convergence; independent audits stay pending.
"""
import argparse
import json
import numpy as np
import os
import shutil
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.profile_stage_3d_e31_fixed import dump, freeze_source, verify_source
from scripts.run_stage_3d_e19_mri import worker, PAIR, HIERARCHY_POLICIES, export_reserve_for
from jd_curl.three_d.nifti import file_sha256

SOURCE = 'results/mri_3d/agent_e34_weak_four_3600s_20260928T013155Z/inputs'
BUDGETS = (30, 45, 60, 90, 120, 180, 240, 330, 600, 3600)   # 330: the 2026-10-08 recommended schedule (15/45/105 s coarse, 300 s solve, 30 s export)
# 30/45/60 (S30-G1, 2026-10-14): short end-to-end budgets with a 3 s export reserve (export_reserve_for); the longer
# budgets keep the 30 s reserve bit for bit.  90/120 (S30 schedule probe, 2026-10-15): the pre-optimizer
# start (T88) at shorter walls, also with the 3 s reserve; with --pre-optimizer the coarse levels are skipped, so the
# coarse entries of --level-shares are inert and the native cutoff is wall - reserve (see test_budget_90_120_s30.py).
SERVE_POLICY = 'resident-service-v1'   # S30-G1: one warm process runs a queue of prepared directories (same frozen source)
# 240: the 2026-10-10 recommended schedule (TA2, user's 'safe' choice: 7.5/22.5/52.5 s nominal coarse, 210 s solve, 30 s export)
E53_OPTIONS = ('hierarchy_policy', 'resolution_exit', 'pg_jd_affine_activation', 'jd_step_global', 'level_shares',
               'barrier_basis', 'pg_native_metric_policy', 'bb2_ggn_guard', 'curvature_weight', 'geometry_weight',
               'geometry_weight_levels', 'pg_metric_policy_levels', 'data_weight', 'first_trial_cap_levels', 'first_trial_scope',
               'inner_steps_levels', 'native_start_scale', 'transfer_smoothing', 'curvature_weight_levels',
               'practical_stop', 'momentum_first_levels', 'pre_optimizer', 'active_set', 'native_transfer_alphas',
               'ggn_diagonal_reuse', 'native_geometry_weights', 'native_reference', 'native_metric_ablation',   # T92: these four
               'native_inner',                                                                          # T92: D2-e (LM)
               'native_penalty_scale')                                                                  # T92: D2-c' down-weighted arm
# contrast_mix is applied at prepare time (derived inputs); it is recorded in the protocol but never forwarded to the worker.


CONTRAST_MIX_POLICY = 'contrast-mix-v1'
# E66 / E87 / E88: pre-registered input pair sets (name, folder with selection.json and <pair>/{pair.npz, manifest.json}).
# The development pair is an E11 pair whose files come from the frozen E34 inputs (SOURCE), as before.
PAIR_DATASETS = (('stage_E11_native_v1', 'data/three_d/stage_E11_native_v1'),
                 ('stage_L2R_val_v1', 'data/three_d/stage_L2R_val_v1'),       # E87: the 19 Learn2Reg OASIS validation pairs
                 ('stage_T0_tune_v1', 'data/three_d/stage_T0_tune_v1'),       # E88: 12 tuning pairs, disjoint from the others
                 ('stage_LPBA40_v1', 'data/three_d/stage_LPBA40_v1'),         # D1 (2026-10-14): LPBA40 tuning + test pairs
                 # HOLDOUTREG (2026-10-22): held-out sets H1 (50 OASIS pairs), H3 (36 IBSR18 pairs), H4 (20 large-deformation
                 # OASIS pairs). Registered before the freeze; the folders are written by scripts/prepare_stage_3d_holdout.py
                 # only after it (a set whose selection.json is missing is skipped by resolve_pair).
                 ('stage_H1_v1', 'data/three_d/stage_H1_v1'),
                 ('stage_H3_IBSR18_v1', 'data/three_d/stage_H3_IBSR18_v1'),
                 ('stage_H4_v1', 'data/three_d/stage_H4_v1'))
DEVELOPMENT_PAIR_DATASET = 'development_pair_E34_inputs'
HELD_OUT_PAIR_SETS = ('stage_H1_v1', 'stage_H3_IBSR18_v1', 'stage_H4_v1')   # served only once completion.json says complete


def _complete_set(folder):
    """HOLDOUTREG: a held-out set folder whose writer finished (completion.json with complete = true)."""
    path = ROOT/folder/'completion.json'
    try:
        record = json.loads(path.read_text()) if path.is_file() else None
    except (OSError, ValueError):
        return False
    return isinstance(record, dict) and record.get('complete') is True


def resolve_pair(pair):
    """(source folder, dataset name) of a pre-registered input pair. A name must be a bare folder name, be listed
    in exactly one pair set's selection.json, and its manifest must name the same pair."""
    if not isinstance(pair, str) or not pair or pair in ('.', '..') or '/' in pair or '\\' in pair or pair != Path(pair).name:
        raise ValueError(f'unknown input pair {pair!r} (a pair must be a bare folder name)')   # E66 message kept
    found = []
    for name, folder in PAIR_DATASETS:
        selection = ROOT/folder/'selection.json'
        if name in HELD_OUT_PAIR_SETS and not _complete_set(folder):       # HOLDOUTREG: absent or partly written
            continue
        if selection.is_file() and pair in {row['pair'] for row in json.loads(selection.read_text())['pairs']}:
            found.append((name, folder))
    if not found:
        raise ValueError(f'unknown input pair {pair!r} (only the pre-registered pair sets {[n for n, _ in PAIR_DATASETS]})')
    if len(found) > 1:
        raise ValueError(f'input pair {pair!r} is registered in more than one pair set: {[n for n, _ in found]}')
    name, folder = found[0]
    if pair == PAIR:
        if name != 'stage_E11_native_v1':
            raise ValueError('the development pair must come from the E11 pair set')
        source, name = ROOT/SOURCE/PAIR, DEVELOPMENT_PAIR_DATASET
    else:
        source = ROOT/folder/pair
    if not all((source/f).is_file() for f in ('pair.npz', 'manifest.json')):
        raise ValueError(f'unknown input pair {pair!r}')
    if json.loads((source/'manifest.json').read_text()).get('pair') != pair:
        raise ValueError('input manifest names a different pair')
    return source, name


def _contrast_mix_image(image, a, sigma, eps_frac):
    """E80 N (T79, input preprocessing only): (1-a) I + a N(I) with the local-contrast normalisation
    N(I) = 1{I > 0} (I - G*I) / sqrt(G*(I - G*I)^2 + (eps_frac std(I))^2), G a Gaussian of width sigma voxels
    (nearest boundary), then rescaled so that the mixed image keeps the global std of I (review T79: otherwise the
    data term grows 1-2 orders of magnitude against the geometry terms). a = 0 returns the image bit-for-bit.
    Labels and the displacement are never touched."""
    from scipy.ndimage import gaussian_filter
    if image.dtype != np.float64:
        raise ValueError('contrast_mix: float64 images required')
    if a == 0.:
        return np.array(image, copy=True)
    std = float(image.std())
    if not std > 0.:
        raise ValueError('contrast_mix: a constant image has no contrast to normalise')
    mean = gaussian_filter(image, sigma, mode='nearest')
    centred = image-mean
    scale = np.sqrt(gaussian_filter(centred*centred, sigma, mode='nearest')+(eps_frac*std)**2)
    mixed = (1.-a)*image+a*(centred/scale)*(image > 0.)
    out = mixed*(std/float(mixed.std()))
    if not np.all(np.isfinite(out)):
        raise ValueError('contrast_mix produced non-finite values')
    return out


def parse_data_weight(spec):
    """'brain-shell-v1:beta=0.1,sigma_mm=2' (or a dict) -> validated dict, or None."""
    if spec is None:
        return None
    from jd_curl.three_d.data_weight import DATA_WEIGHT_POLICY
    if isinstance(spec, dict):
        policy, params = spec.get('policy'), {k: v for k, v in spec.items() if k != 'policy'}
    else:
        policy, _, rest = str(spec).partition(':')
        params = dict(item.split('=') for item in rest.split(',') if item)
    if policy != DATA_WEIGHT_POLICY or set(params) != {'beta', 'sigma_mm'} or any(isinstance(v, (bool,)) for v in params.values()):
        raise ValueError('data_weight: brain-shell-v1:beta=<0..1>,sigma_mm=<>0>')
    beta, sigma_mm = float(params['beta']), float(params['sigma_mm'])
    if not (np.isfinite(beta) and 0. <= beta <= 1. and np.isfinite(sigma_mm) and sigma_mm > 0.):
        raise ValueError('data_weight: 0 <= beta <= 1 and sigma_mm > 0')
    return dict(policy=DATA_WEIGHT_POLICY, beta=beta, sigma_mm=sigma_mm)


def parse_contrast_mix(spec):
    """'a,sigma,eps_frac' -> validated dict, or None."""
    if spec is None:
        return None
    if isinstance(spec, dict):
        a, sigma, eps_frac = spec['a'], spec['sigma'], spec['eps_frac']
    else:
        parts = [float(x) for x in (spec if isinstance(spec, (list, tuple)) else str(spec).split(','))]
        if len(parts) != 3:
            raise ValueError('contrast_mix: a,sigma,eps_frac')
        a, sigma, eps_frac = parts
    a, sigma, eps_frac = float(a), float(sigma), float(eps_frac)
    if not (np.isfinite(a) and 0. <= a <= 1. and np.isfinite(sigma) and sigma > 0. and np.isfinite(eps_frac) and eps_frac > 0.):
        raise ValueError('contrast_mix: 0 <= a <= 1, sigma > 0, eps_frac > 0')
    return dict(policy=CONTRAST_MIX_POLICY, a=a, sigma=sigma, eps_frac=eps_frac)


def prewarm_jit(throughput, barrier_basis=None, first_trial_cap=False, ggn_reuse=False):
    """Compile the explicit parallel proof kernel before the task clock (cost recorded)."""
    import numpy as np
    from jd_curl.three_d.grid import make_grid_3d
    tick = perf_counter()
    if first_trial_cap:
        # E84 T81: compile the max-node displacement scan outside the task clock (only when the option is set).
        from jd_curl.three_d import fast_checks
        if fast_checks._NUMBA:
            from jd_curl.validation import readonly_copy
            fast_checks.max_node_sq(np.zeros(3*8), (1., 1., 1.))
            fast_checks.max_node_sq(readonly_copy(np.zeros(3*8)), (1., 1., 1.))
    if barrier_basis == 'bernstein':
        # T68 compiled transfer domain check (E58b), any throughput profile.
        from jd_curl.three_d.bernstein_barrier import min_coefficient
        from jd_curl.three_d.problem import make_problem, Weights3D
        from jd_curl.three_d.feasibility import EnclosureCertificateConfig
        g = make_grid_3d(nx=5, ny=5, nz=5, hx=1., hy=1., hz=1., x0=0., y0=0., z0=0., unit='mm')
        warm = make_problem(grid=g, fixed_image=np.zeros(g.shape), moving_image=np.zeros(g.shape),
            u_ref=np.zeros(g.shape+(3,)), weights=Weights3D(.01, .001, .001, .002),
            certificate_config=EnclosureCertificateConfig(delta=0.), barrier_location='closed27_uniform',
            barrier_activation=.05, barrier_basis='bernstein')
        # both Numba specializations: production seeds are read-only, scaled seeds writable
        from jd_curl.validation import readonly_copy
        min_coefficient(warm, np.zeros(g.shape+(3,)), fast=True)
        min_coefficient(warm, readonly_copy(np.zeros(g.shape+(3,))), fast=True)
    if throughput not in ('v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v19', 'v20', 'v21', 'v22', 'v23', 'v24', 'v25', 'v26', 'v27', 'v28', 'v29', 'v30', 'v31', 'v32', 'v33', 'v34', 'v35', 'v36', 'v37', 'v38', 'v39', 'v41'):
        return perf_counter()-tick if (barrier_basis == 'bernstein' or first_trial_cap) else 0.
    from jd_curl.three_d.compiled_enclosure import parallel_margins
    grid = make_grid_3d(nx=5, ny=5, nz=5, hx=1., hy=1., hz=1., x0=0., y0=0., z0=0., unit='mm')
    parallel_margins(np.zeros(grid.shape+(3,)), grid, 0.)
    if throughput in ('v9', 'v10', 'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v19', 'v20', 'v21', 'v22', 'v23', 'v24', 'v25', 'v26', 'v27', 'v28', 'v29', 'v30', 'v31', 'v32', 'v33', 'v34', 'v35', 'v36', 'v37', 'v38', 'v39', 'v41'):
        # E76: compile the allocation-free step-bound and max-abs-difference kernels.
        from jd_curl.three_d import fast_checks
        if fast_checks._NUMBA:
            from jd_curl.validation import readonly_copy
            for ro in (False, True):          # production arrays are read-only: compile both signatures
                wrap = readonly_copy if ro else (lambda a: a)
                v = wrap(np.zeros((3, 3, 3, 3)))
                fast_checks.step_bound_xu(v, wrap(np.zeros(grid.shape+(3,))), grid.axes)
                fast_checks.max_abs_diff(wrap(np.zeros(8)), wrap(np.ones(8)))
                fast_checks.max_abs(wrap(np.zeros(8)))
    if throughput in ('v10', 'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v19', 'v20', 'v21', 'v22', 'v23', 'v24', 'v25', 'v26', 'v27', 'v28', 'v29', 'v30', 'v31', 'v32', 'v33', 'v34', 'v35', 'v36', 'v37', 'v38', 'v39', 'v41'):
        # E81: compile the face/interior estimator kernels outside the task clock.
        from jd_curl.three_d.solvers.jd_affine_compiled import prewarm as _prewarm_extension
        _prewarm_extension()
    if throughput in ('v18', 'v19', 'v20', 'v21', 'v22', 'v23', 'v24', 'v25', 'v26', 'v27', 'v28', 'v29', 'v30', 'v31', 'v32', 'v33', 'v34', 'v35', 'v36', 'v37', 'v38', 'v39', 'v41'):
        # E100: the compiled stable top-4 selection of the estimator's Newton starts.
        from jd_curl.three_d.solvers.jd_affine_compiled import prewarm_top4
        prewarm_top4()
    if throughput in ('v19', 'v20', 'v21', 'v22', 'v23', 'v24', 'v25', 'v26', 'v27', 'v28', 'v29', 'v30', 'v31', 'v32', 'v33', 'v34', 'v35', 'v36', 'v37', 'v38', 'v39', 'v41'):
        # E101: the T84 edge-hull margins kernel and the fused Laplacian kernel.
        from jd_curl.three_d.backends.edge_hull_gpu import prewarm as _prewarm_hull
        from jd_curl.three_d.backends.laplacian_fused import prewarm as _prewarm_laplacian
        _prewarm_hull(); _prewarm_laplacian()
    if throughput in ('v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v19', 'v20', 'v21', 'v22', 'v23', 'v24', 'v25', 'v26', 'v27', 'v28', 'v29', 'v30', 'v31', 'v32', 'v33', 'v34', 'v35', 'v36', 'v37', 'v38', 'v39', 'v41'):
        # E89: compile the throughput-v11 host kernels outside the task clock.
        from jd_curl.three_d.host_kernels import prewarm as _prewarm_host
        from scripts.run_stage_3d_e19_mri import host_prewarm_kwargs
        _prewarm_host(**host_prewarm_kwargs(throughput))      # E96: the E95 kernels only for lean_checks profiles
    if throughput in ('v8', 'v9', 'v10', 'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v19', 'v20', 'v21', 'v22', 'v23', 'v24', 'v25', 'v26', 'v27', 'v28', 'v29', 'v30', 'v31', 'v32', 'v33', 'v34', 'v35', 'v36', 'v37', 'v38', 'v39', 'v41'):
        # E71: compile the T75 prescreen kernel outside the task clock.
        from jd_curl.three_d.cell_q1 import CellQ1Operators3D
        from jd_curl.three_d.certificate_prescreen import prescreen_lower_bounds
        prescreen_lower_bounds(np.zeros(grid.shape+(3,)), grid, CellQ1Operators3D(grid, location='closed27_uniform').coefficients,
                               np.arange(grid.n_cells))
    if throughput in ('v28', 'v30'):
        # T89 (S30-G4): the index-list kernels of the active set (both WITH_BASE values, the shapes used).
        from jd_curl.three_d.backends.triton_q1 import prewarm_active_set
        prewarm_active_set()
    if throughput in ('v29', 'v30', 'v37', 'v38', 'v39', 'v41'):
        # T90 (S30-G5): the float32 screening kernels.
        from jd_curl.three_d.backends.triton_q1 import prewarm_fp32
        prewarm_fp32()
    if ggn_reuse:
        # GG2: the indexed GGN diagonal kernel (only when the option is set).
        from jd_curl.three_d.backends.triton_q1 import prewarm_ggn_reuse
        prewarm_ggn_reuse()
    if throughput in ('v16', 'v17', 'v18', 'v19', 'v20', 'v21', 'v22', 'v23', 'v24', 'v25', 'v26', 'v27', 'v28', 'v29', 'v30', 'v31', 'v32', 'v33', 'v34', 'v35', 'v36', 'v37', 'v38', 'v39', 'v41'):
        # E98: the row-SoA certificate kernel, and the read-only specializations of the libm margins (first-use
        # check) and of the prescreen: the owned certificate front end passes the frozen trial u itself.
        from jd_curl.three_d.soa_enclosure import prewarm as _prewarm_soa
        from jd_curl.three_d.compiled_enclosure import _libm_margins
        from jd_curl.three_d.cell_q1 import CellQ1Operators3D
        from jd_curl.three_d.certificate_prescreen import prescreen_lower_bounds
        from jd_curl.validation import readonly_copy
        _prewarm_soa()
        frozen = readonly_copy(np.zeros(grid.shape+(3,)))
        _libm_margins(frozen, grid, 0.)
        prescreen_lower_bounds(frozen, grid, CellQ1Operators3D(grid, location='closed27_uniform').coefficients,
                               np.arange(grid.n_cells))
    if throughput in ('v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v19', 'v20', 'v21', 'v22', 'v23', 'v24', 'v25', 'v26', 'v27', 'v28', 'v29', 'v30', 'v31', 'v32', 'v33', 'v34', 'v35', 'v36', 'v37', 'v38', 'v39', 'v41'):
        from jd_curl.three_d.compiled_fallback import fallback_arrays
        fallback_arrays(np.zeros((3, 8, 3)), (1., 1., 1.), 0.)
    if throughput in ('v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v19', 'v20', 'v21', 'v22', 'v23', 'v24', 'v25', 'v26', 'v27', 'v28', 'v29', 'v30', 'v31', 'v32', 'v33', 'v34', 'v35', 'v36', 'v37', 'v38', 'v39', 'v41'):
        # Compile every Triton variant used by the run (integer args are not specialized,
        # so these binaries are the ones every level grid uses).
        import torch
        from jd_curl.three_d.problem import make_problem, Weights3D
        from jd_curl.three_d.feasibility import EnclosureCertificateConfig
        from jd_curl.three_d.backends.triton_q1 import FusedQ1
        g = make_grid_3d(nx=7, ny=6, nz=5, hx=1., hy=1., hz=1., x0=0., y0=0., z0=0., unit='mm')
        image = np.sin(g.identity()[..., 0])
        problem = make_problem(grid=g, fixed_image=image, moving_image=image, u_ref=np.zeros(g.shape+(3,)),
            weights=Weights3D(.01, .001, .001, .002), certificate_config=EnclosureCertificateConfig(delta=0.),
            barrier_location='closed27_uniform', barrier_activation=.05,
            **(dict(barrier_basis=barrier_basis) if barrier_basis else {}))
        fused = FusedQ1(problem, torch.device('cuda'))
        u = torch.zeros(g.shape+(3,), dtype=torch.float64, device='cuda')
        fused.gradient_nodes(u.clone(), u, u); fused.joint(u.clone(), u.clone(), u, u)
        fused.soft_energy(u, u); fused.soft_energy(u, u, u); fused.barrier_energy(u); fused.barrier_energy(u, u)
        if throughput in ('v6', 'v7', 'v8', 'v9', 'v10', 'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v19', 'v20', 'v21', 'v22', 'v23', 'v24', 'v25', 'v26', 'v27', 'v28', 'v29', 'v30', 'v31', 'v32', 'v33', 'v34', 'v35', 'v36', 'v37', 'v38', 'v39', 'v41') and not fused.bernstein:
            fused.diagonal_nodes(u.clone(), u)
        if throughput in ('v8', 'v9', 'v10', 'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v19', 'v20', 'v21', 'v22', 'v23', 'v24', 'v25', 'v26', 'v27', 'v28', 'v29', 'v30', 'v31', 'v32', 'v33', 'v34', 'v35', 'v36', 'v37', 'v38', 'v39', 'v41') and not fused.bernstein:
            fused.hvp_nodes(u.clone(), u, u.clone())
        if throughput in ('v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v19', 'v20', 'v21', 'v22', 'v23', 'v24', 'v25', 'v26', 'v27', 'v28', 'v29', 'v30', 'v31', 'v32', 'v33', 'v34', 'v35', 'v36', 'v37', 'v38', 'v39', 'v41'):
            # E93: the witness-screen prefilter kernel.
            from jd_curl.three_d.backends.screen_prefilter import prewarm as _prewarm_prefilter
            _prewarm_prefilter()
        if throughput in ('v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v19', 'v20', 'v21', 'v22', 'v23', 'v24', 'v25', 'v26', 'v27', 'v28', 'v29', 'v30', 'v31', 'v32', 'v33', 'v34', 'v35', 'v36', 'v37', 'v38', 'v39', 'v41'):
            # E94: the fused node JD kernel (both variants).
            from jd_curl.three_d.backends.node_jd_fused import prewarm as _prewarm_node_jd
            _prewarm_node_jd()
        if throughput in ('v16', 'v17', 'v18', 'v19', 'v20', 'v21', 'v22', 'v23', 'v24', 'v25', 'v26', 'v27', 'v28', 'v29', 'v30', 'v31', 'v32', 'v33', 'v34', 'v35', 'v36', 'v37', 'v38', 'v39', 'v41'):
            # E98: the barrier energy/gradient kernels that skip inactive points (a throwaway check record).
            skip = FusedQ1(problem, torch.device('cuda'), barrier_skip=dict(calls=0, compared=0, verified=[], mismatches=0,
                                                                              errors=0, disabled=False, unsupported=0))
            skip.barrier_locals(u); skip.barrier_energy(u); skip.barrier_energy(u, u)
        if throughput in ('v17', 'v18', 'v19', 'v20', 'v21', 'v22', 'v23', 'v24', 'v25', 'v26', 'v27', 'v28', 'v29', 'v30', 'v31', 'v32', 'v33', 'v34', 'v35', 'v36', 'v37', 'v38', 'v39', 'v41'):
            # E99: the tuned launch shapes (soft-geometry energy with a base, GGN diagonal) and the fused spline kernel.
            import triton
            from jd_curl.three_d.backends.triton_q1 import TUNED_LAUNCH
            from jd_curl.three_d.backends import triton_q1_kernels as TK
            from jd_curl.three_d.backends.spline_fused import prewarm as _prewarm_spline
            block, warps, regs = TUNED_LAUNCH['soft_energy_base']
            out = torch.empty((triton.cdiv(fused.ncell, block), 3), dtype=torch.float64, device='cuda')
            TK.soft_energy_kernel[(triton.cdiv(fused.ncell, block),)](u, u, u, fused.k, out, fused.ncell, fused.ny,
                fused.nx, fused.cy, fused.cx, WITH_BASE=True, BLOCK=block, num_warps=warps, maxnreg=regs)
            fused._cell_locals_at(TK.ggn_diag_kernel, TUNED_LAUNCH['ggn_diag'], u)
            _prewarm_spline()
        if throughput == 'v5':
            from jd_curl.three_d.backends.triton_jd_root import FusedJDRoot
            FusedJDRoot(problem, torch.device('cuda'))(u, u.clone(), 1.)
        if throughput in ('v34', 'v35', 'v36', 'v37', 'v38', 'v39', 'v41'):        # FX1: v35 = v34 + ...
            # S30 P-12/P-6 (P-6): the pre-optimizer's three analytic-gradient kernels, outside the task clock.
            from jd_curl.three_d.preopt_grad import prewarm as _prewarm_preopt_grad
            _prewarm_preopt_grad()
        torch.cuda.synchronize()
    return perf_counter()-tick


def parse_transfer_smoothing(text):
    """A19: None, a dict, or 'kernel:sigma_mm[:scope]' -> the validated JSON-able dict (or None)."""
    if text is None:
        return None
    if isinstance(text, str):
        parts = text.split(':')
        if len(parts) not in (2, 3):
            raise ValueError('transfer_smoothing: kernel:sigma_mm[:scope]')
        try:
            sigma = float(parts[1])
        except ValueError:
            raise ValueError('transfer_smoothing: sigma_mm must be a number') from None
        text = dict(kernel=parts[0], sigma_mm=sigma, **({'scope': parts[2]} if len(parts) == 3 else {}))
    from jd_curl.three_d.transfer_smoothing import validate_transfer_smoothing
    from jd_curl.validation import InputValidationError as _IVE
    try:
        return validate_transfer_smoothing(dict(text))
    except (_IVE, TypeError) as exc:
        raise ValueError(f'transfer_smoothing: {exc}') from None


def parse_pre_optimizer(text):
    """T88 (S30-G3): None, a dict, or 'fp32-v1[:key=value,...]' with keys iterations (a/b/c/d), lr, cap, sigma_grad,
    mu (a/b/c/d), alpha (a/b/c/d), kappa_pre, j0, seconds, taper -> the validated JSON-able dict (or None)."""
    if text is None:
        return None
    from jd_curl.three_d.preopt import PreoptConfig, PREOPT_POLICY
    from jd_curl.validation import InputValidationError as _IVE
    if isinstance(text, str):
        head, _, rest = text.partition(':')
        if head != 'fp32-v1':
            raise ValueError('pre_optimizer: fp32-v1[:key=value,...]')
        spec = {}
        for item in [x for x in rest.split(',') if x]:
            key, sep, value = item.partition('=')
            if not sep or key in spec:
                raise ValueError('pre_optimizer: key=value items')
            try:
                if key in ('iterations', 'mu', 'alpha'):
                    spec[key] = [int(v) if key == 'iterations' else float(v) for v in value.split('/')]
                elif key == 'taper':
                    spec[key] = int(value)
                else:
                    spec[key] = float(value)
            except ValueError:
                raise ValueError(f'pre_optimizer: {key} must be numeric') from None
        text = dict(spec, policy=PREOPT_POLICY)
    text = dict(text)
    if text.get('policy') != PREOPT_POLICY:
        raise ValueError(f'pre_optimizer: policy {PREOPT_POLICY}')
    taper = text.get('taper', 4)
    if isinstance(taper, bool) or not isinstance(taper, int) or taper < 1:
        raise ValueError('pre_optimizer: taper must be a positive integer')
    try:
        config = PreoptConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in text.items() if k not in ('policy', 'taper')})
    except (_IVE, TypeError) as exc:
        raise ValueError(f'pre_optimizer: {exc}') from None
    from dataclasses import asdict as _asdict
    return dict(_asdict(config), policy=PREOPT_POLICY, taper=taper)


def parse_practical_stop(text):
    """H17: None, a dict, or 'epsilon=E[,window_seconds=W][,min_seconds=T][,min_commits=N]' -> the validated
    JSON-able dict (or None); epsilon is required."""
    if text is None:
        return None
    if isinstance(text, str):
        spec = {}
        for item in text.split(','):
            key, sep, value = item.partition('=')
            if not sep or key.strip() not in ('epsilon', 'window_seconds', 'min_seconds', 'min_commits') or key.strip() in spec:
                raise ValueError('practical_stop: epsilon=E[,window_seconds=W][,min_seconds=T][,min_commits=N]')
            try:
                spec[key.strip()] = int(value) if key.strip() == 'min_commits' else float(value)
            except ValueError:
                raise ValueError(f'practical_stop: {key.strip()} must be a number') from None
        text = spec
    from dataclasses import asdict
    from jd_curl.three_d.practical_stop import practical_stop_config
    from jd_curl.validation import InputValidationError as _IVE
    try:
        return asdict(practical_stop_config(dict(text)))
    except (_IVE, TypeError) as exc:
        raise ValueError(f'practical_stop: {exc}') from None


def prepare(output, strength, seconds, *, throughput, stage='E48', first_level=0, lean_commits=False,
            pg_metric_policy=None, pg_shift_factor=1., hierarchy_policy=None, resolution_exit=False,
            pg_jd_affine_activation=None, jd_step_global=False, level_shares=None, barrier_basis=None,
            pg_native_metric_policy=None, bb2_ggn_guard=False, pair=None, curvature_weight=0., geometry_weight=None,
            geometry_weight_levels=None, pg_metric_policy_levels=None, contrast_mix=None, data_weight=None,
            first_trial_cap_levels=None, first_trial_scope=None, inner_steps_levels=None, native_start_scale=None,
            transfer_smoothing=None, curvature_weight_levels=None, practical_stop=None, momentum_first_levels=None,
            pre_optimizer=None, active_set=None, export_reserve=None, native_transfer_alphas=None, ggn_diagonal_reuse=None,
            native_geometry_weights=None, native_reference=None, native_metric_ablation=None, native_penalty_scale=None,
            native_inner=None):
    if seconds not in BUDGETS:
        raise ValueError('only declared budgets')
    reserve = export_reserve_for(seconds)
    if export_reserve is not None:
        # FX1: an explicit export reserve; the default stays export_reserve_for(seconds)
        if isinstance(export_reserve, bool) or not isinstance(export_reserve, (int, float)) or not 0. < float(export_reserve) < float(seconds):
            raise ValueError('export_reserve: a positive number of seconds below the wall budget')
        reserve = float(export_reserve)
    contrast_mix = parse_contrast_mix(contrast_mix)      # validated before any directory exists
    data_weight = parse_data_weight(data_weight)          # T80, validated before any directory exists
    curvature_weight = float(curvature_weight)
    if not np.isfinite(curvature_weight) or curvature_weight < 0:
        raise ValueError('curvature_weight must be a finite nonnegative float')
    if geometry_weight is not None:
        if isinstance(geometry_weight, (bool, str)):
            raise ValueError('geometry_weight must be a real number')
        geometry_weight = float(geometry_weight)
        if not np.isfinite(geometry_weight) or geometry_weight <= 0:
            raise ValueError('geometry_weight must be a finite positive float')
    if geometry_weight_levels is not None:
        # T78 (E80): per-level mu = nu, exclusive with the single value.
        if geometry_weight is not None:
            raise ValueError('geometry_weight and geometry_weight_levels are exclusive')
        if isinstance(geometry_weight_levels, (str, bytes)) or len(geometry_weight_levels) != 4 \
                or any(isinstance(w, (bool, str)) for w in geometry_weight_levels):
            raise ValueError('geometry_weight_levels must be four real numbers')
        geometry_weight_levels = [float(w) for w in geometry_weight_levels]
        if not all(np.isfinite(w) and w > 0 for w in geometry_weight_levels):
            raise ValueError('geometry_weight_levels must be finite positive floats')
    if pg_metric_policy_levels is not None:
        # T78 (E80): per-level PG metric policies, exclusive with the coarse/native pair.
        if pg_metric_policy is not None or pg_native_metric_policy is not None:
            raise ValueError('pg_metric_policy_levels is exclusive with pg_metric_policy/pg_native_metric_policy')
        if isinstance(pg_metric_policy_levels, (str, bytes)) or len(pg_metric_policy_levels) != 4 \
                or not all(isinstance(p, str) and p for p in pg_metric_policy_levels):
            raise ValueError('pg_metric_policy_levels must be four policy names')
        pg_metric_policy_levels = list(pg_metric_policy_levels)
        from scripts.run_stage_3d_e19_mri import _e53_policy
        if not all(_e53_policy(p) for p in pg_metric_policy_levels):
            raise ValueError('pg_metric_policy_levels: every level must be an E53-type policy')
        from jd_curl.three_d.solvers.poisson_gradient import ConservativePGConfig3D
        for policy in pg_metric_policy_levels:
            ConservativePGConfig3D(metric_policy=policy, shift_factor=pg_shift_factor, resolution_exit=bool(resolution_exit),
                                   jd_affine_activation=pg_jd_affine_activation, jd_step_global=bool(jd_step_global))
    if (geometry_weight_levels is not None or pg_metric_policy_levels is not None) and \
            (first_level != 0 or (hierarchy_policy is not None and HIERARCHY_POLICIES[hierarchy_policy][strength] != 0)):
        # review E80: with a skipped level the level-0 entry would be silently unused.
        raise ValueError('per-level geometry weights/metric policies require every level to be solved (first_level 0)')
    if barrier_basis not in (None, 'samples', 'bernstein'):
        raise ValueError('barrier_basis: samples or bernstein')
    if strength not in ('weak', 'strong') or throughput not in (False, 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v19', 'v20', 'v21', 'v22', 'v23', 'v24', 'v25', 'v26', 'v27', 'v28', 'v29', 'v30', 'v31', 'v32', 'v33', 'v34', 'v35', 'v36', 'v37', 'v38', 'v39', 'v41'):
        raise ValueError('strength weak/strong and throughput False/v1/v2/v3/v4/v5/v6/v7/v8/v9/v10/v11/v12/v13/v14/v15/v16/v17/v18/v19/v20/v21/v22/v23/v24/v25/v26/v27 required')
    if hierarchy_policy is not None:
        if first_level != 0:
            raise ValueError('hierarchy_policy and an explicit first_level are exclusive')
        first_level = HIERARCHY_POLICIES[hierarchy_policy][strength]
    if level_shares is not None:
        level_shares = [float(x) for x in level_shares]
        if (len(level_shares) != 4 or level_shares[-1] != 1. or not level_shares[0] > 0.
                or not all(a < b for a, b in zip(level_shares, level_shares[1:]))):
            raise ValueError('level_shares: four strictly increasing cumulative fractions ending at 1')
    e53 = bool(hierarchy_policy is not None or resolution_exit or jd_step_global or bb2_ggn_guard or pg_jd_affine_activation is not None
               or pg_metric_policy_levels is not None or pg_metric_policy == 'legacy_q1' or (pg_metric_policy or '').endswith(('_jd_affine_v1', '_jd_affine_v2', '_jd_affine_v3', '_jd_affine_triggered_v1', '_jd_affine_always_v1')))
    if e53 and (resolution_exit or jd_step_global or pg_jd_affine_activation is not None or pg_metric_policy is not None):
        # Validate the inner configuration before any directory or source freeze exists.
        from jd_curl.three_d.solvers.poisson_gradient import ConservativePGConfig3D
        ConservativePGConfig3D(metric_policy=(pg_metric_policy_levels or [pg_metric_policy])[0] or 'legacy_q1', shift_factor=pg_shift_factor,
                               resolution_exit=bool(resolution_exit), jd_affine_activation=pg_jd_affine_activation,
                               jd_step_global=bool(jd_step_global))
    if pg_native_metric_policy is not None:
        # T71 (E62): native-level metric policy; the coarse levels keep pg_metric_policy.
        if not e53:
            raise ValueError('pg_native_metric_policy requires the E53-type PG route (set pg_metric_policy)')
        from jd_curl.three_d.solvers.poisson_gradient import ConservativePGConfig3D
        ConservativePGConfig3D(metric_policy=pg_native_metric_policy, shift_factor=pg_shift_factor,
                               resolution_exit=bool(resolution_exit), jd_step_global=bool(jd_step_global))
    if first_trial_cap_levels is not None:
        # E84 T81: per-level first-trial cap (level voxels; None = off), path only; validated before any directory
        # with the worker's own rules (review E84: the same route predicate, so prepare never accepts what run refuses).
        from scripts.run_stage_3d_e19_mri import conservative_pg_route, validate_first_trial_caps
        first_trial_cap_levels = list(validate_first_trial_caps(first_trial_cap_levels))
        if any(c is not None for c in first_trial_cap_levels[:first_level]):
            raise ValueError('first_trial_cap_levels: a skipped level cannot carry a cap')
        if not conservative_pg_route(resolution_exit=resolution_exit, jd_step_global=jd_step_global, bb2_ggn_guard=bb2_ggn_guard,
                                     pg_jd_affine_activation=pg_jd_affine_activation,
                                     pg_metric_policy=(pg_metric_policy_levels or [pg_metric_policy])[0]):
            raise ValueError('first_trial_cap_levels requires the E53-type PG route (an E53 metric policy or E53 option)')
        first_trial_scope = 'initial' if first_trial_scope is None else first_trial_scope
        from jd_curl.three_d.solvers.poisson_gradient import ConservativePGConfig3D
        for cap in first_trial_cap_levels:
            if cap is not None:
                ConservativePGConfig3D(first_trial_cap=cap, first_trial_scope=first_trial_scope)
    elif first_trial_scope is not None:
        raise ValueError('first_trial_scope requires first_trial_cap_levels')
    if momentum_first_levels is not None:
        # E111 T85: per-level momentum first direction (path only); the worker's own rules, before any directory.
        from scripts.run_stage_3d_e19_mri import conservative_pg_route, validate_momentum_levels
        momentum_first_levels = list(validate_momentum_levels(momentum_first_levels))
        if any(momentum_first_levels[:first_level]):
            raise ValueError('momentum_first_levels: a skipped level cannot be switched on')
        if not conservative_pg_route(resolution_exit=resolution_exit, jd_step_global=jd_step_global, bb2_ggn_guard=bb2_ggn_guard,
                                     pg_jd_affine_activation=pg_jd_affine_activation,
                                     pg_metric_policy=(pg_metric_policy_levels or [pg_metric_policy])[0]):
            raise ValueError('momentum_first_levels requires the E53-type PG route (an E53 metric policy or E53 option)')
    if native_start_scale is not None:
        # E90 T83: native start scale beta in [0.1, 1], validated before any directory (the launcher always uses
        # transfer backtracking, which T83 requires).
        from jd_curl.three_d.multiresolution import validate_start_scale
        from jd_curl.validation import InputValidationError as _IVE
        try:
            native_start_scale = validate_start_scale(native_start_scale)
        except _IVE as exc:
            raise ValueError(f'native_start_scale: {exc}') from None
    transfer_smoothing = parse_transfer_smoothing(transfer_smoothing)   # A19, validated before any directory
    pre_optimizer = parse_pre_optimizer(pre_optimizer)                   # T88, validated before any directory
    if active_set is not None:                                           # T89, validated before any directory
        from jd_curl.three_d.active_set import parse_active_set
        from jd_curl.validation import InputValidationError as _IVE
        try:
            active_set = parse_active_set(active_set).record()
        except _IVE as exc:
            raise ValueError(f'active_set: {exc}') from None
    if throughput in ('v28', 'v30') and active_set is None:
        raise ValueError(f'throughput {throughput} (active_set_skip) requires --active-set')
    # C2: T83's native start scale may be combined with the
    # pre-optimizer route (the native transfer then tries beta * .9^k times the tapered seed); it was refused here before.
    if pre_optimizer is not None and (first_level != 0 or hierarchy_policy is not None or transfer_smoothing is not None):
        raise ValueError('pre_optimizer requires first_level 0 (no hierarchy policy) and no transfer_smoothing')
    if native_transfer_alphas is not None:                               # AP2, validated before any directory
        from jd_curl.three_d.multiresolution import TRANSFER_ALPHA_LISTS
        if not isinstance(native_transfer_alphas, str) or native_transfer_alphas not in TRANSFER_ALPHA_LISTS:
            raise ValueError('native_transfer_alphas must be one of ' + ', '.join(sorted(TRANSFER_ALPHA_LISTS)))
        if pre_optimizer is None:
            raise ValueError('native_transfer_alphas requires --pre-optimizer')
        if native_start_scale not in (None, 1.):                         # C2: AP2's finer list and beta stay exclusive
            raise ValueError('native_transfer_alphas and native_start_scale are exclusive')
    if ggn_diagonal_reuse is not None:                                   # GG2, validated before any directory
        from jd_curl.three_d.backends.accelerated import parse_ggn_reuse
        from jd_curl.validation import InputValidationError as _IVE
        from scripts.run_stage_3d_e19_mri import THROUGHPUT_POLICIES
        try:
            parse_ggn_reuse(ggn_diagonal_reuse)
        except _IVE as exc:
            raise ValueError(f'ggn_diagonal_reuse: {exc}') from None
        if pre_optimizer is None:
            raise ValueError('ggn_diagonal_reuse requires --pre-optimizer (the native level is then the only PG level)')
        from jd_curl.three_d.solvers.poisson_gradient import T69_POLICY, T73_POLICY
        native_policy = (pg_native_metric_policy if pg_native_metric_policy is not None else
                         pg_metric_policy_levels[3] if pg_metric_policy_levels is not None else pg_metric_policy)
        if native_policy not in (T69_POLICY, T73_POLICY):
            raise ValueError('ggn_diagonal_reuse requires the T69 or T73 metric on the native level')   # review GG2 S1
        prof = THROUGHPUT_POLICIES.get(throughput) if isinstance(throughput, str) else None
        if prof is None or not all(prof.get(k) for k in ('fused_diagonal', 'fused_node_jd', 'estimator_overlap')):
            raise ValueError('ggn_diagonal_reuse requires a throughput profile with the fused diagonal, the E94 fused node JD '
                             'and the E100 estimator overlap (v18 or later)')
    practical_stop = parse_practical_stop(practical_stop)               # H17, validated before any directory
    from scripts.run_stage_3d_e19_mri import validate_native_ablation  # T92 (D2), validated before any directory
    native_ablation = validate_native_ablation(native_geometry_weights=native_geometry_weights, native_reference=native_reference,
        native_metric_ablation=native_metric_ablation, native_penalty_scale=native_penalty_scale,
        pre_optimizer=pre_optimizer, geometry_weight_levels=geometry_weight_levels,
        active_set=active_set, practical_stop=practical_stop, momentum_first_levels=momentum_first_levels,
        native_policy=(pg_native_metric_policy if pg_native_metric_policy is not None else
                       pg_metric_policy_levels[3] if pg_metric_policy_levels is not None else pg_metric_policy),
        first_trial_cap_levels=first_trial_cap_levels, native_inner=native_inner)
    if curvature_weight_levels is not None:                               # G2, validated before any directory
        if curvature_weight:
            raise ValueError('curvature_weight and curvature_weight_levels are exclusive')
        if isinstance(curvature_weight_levels, (str, bytes)) or len(curvature_weight_levels) != 4 \
                or any(isinstance(a, (bool, str, bytes)) for a in curvature_weight_levels):
            raise ValueError('curvature_weight_levels must be four real numbers')
        curvature_weight_levels = [float(a) for a in curvature_weight_levels]
        if not all(np.isfinite(a) and a >= 0 for a in curvature_weight_levels):
            raise ValueError('curvature_weight_levels must be finite nonnegative floats')
        if first_level != 0:
            raise ValueError('curvature_weight_levels requires first_level 0')
    if inner_steps_levels is not None:
        # E85 T82: per-level inner step limit (None = 5), validated before any directory with the worker's own rule.
        from scripts.run_stage_3d_e19_mri import validate_inner_steps_levels
        inner_steps_levels = list(validate_inner_steps_levels(inner_steps_levels, first_level))
    pair = PAIR if pair is None else pair
    # E66: the four further pre-registered pairs (first ten complete subjects, consecutive pairs; no
    # metric-based selection) come from the frozen E11 native set; the development pair keeps its source.
    # E87: the 19 Learn2Reg OASIS validation pairs (data/three_d/stage_L2R_val_v1) are a second pair set.
    source, pair_dataset = resolve_pair(pair)
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    data = output/'inputs'/pair; data.mkdir(parents=True)
    # Copy the two small input files (drvfs volumes such as /mnt/d do not keep symlinks reliably).
    for name in ('pair.npz', 'manifest.json'):
        shutil.copyfile(source/name, data/name)
    if contrast_mix is not None:
        # E80 N (T79): derived inputs; the original files' SHAs are kept in the manifest and the protocol.
        original = {name: file_sha256(data/name) for name in ('pair.npz', 'manifest.json')}
        with np.load(data/'pair.npz') as z:
            arrays = {k: np.array(z[k]) for k in z.files}
        for key in ('fixed', 'moving'):
            arrays[key] = _contrast_mix_image(arrays[key], contrast_mix['a'], contrast_mix['sigma'], contrast_mix['eps_frac'])
        np.savez(data/'pair.npz', **arrays)
        manifest = json.loads((data/'manifest.json').read_text())
        manifest['pair_sha256'] = file_sha256(data/'pair.npz')
        manifest['preprocessing_e80'] = dict(contrast_mix, source_pair_sha256=original['pair.npz'],
            source_manifest_sha256=original['manifest.json'], source_preprocessing=manifest.get('preprocessing'),
            source_baseline_native_mse=manifest.pop('baseline_native_mse', None),
            note='fixed/moving replaced by (1-a) I + a N(I) (background kept zero, rescaled to the source std); '
                 'labels and initial_u unchanged; recorded ratio is on these images')
        manifest['preprocessing'] = (f"E80 T79 contrast-mix-v1 a={contrast_mix['a']:g} sigma={contrast_mix['sigma']:g} "
                                     f"eps_frac={contrast_mix['eps_frac']:g} on top of: {manifest.get('preprocessing')}")
        (data/'manifest.json').write_text(json.dumps(manifest, indent=1))
    inputs = {name: file_sha256(data/name) for name in ('pair.npz', 'manifest.json')}
    task = dict(index=0, pair=pair, route='pg_'+strength, strength=strength, hierarchy='four',
                wall_seconds=seconds, name=f'pg_{strength}__four__{seconds}s')
    dump(output/'protocol.json', dict(stage=stage, task=task, input_sha256=inputs,
        pair_dataset=pair_dataset, pair_source=str(source.relative_to(ROOT)),
        source_sha256=freeze_source(output), throughput=throughput, first_level=first_level, lean_commits=lean_commits,
        pg_metric_policy=pg_metric_policy, pg_shift_factor=pg_shift_factor,
        hierarchy_policy=hierarchy_policy, resolution_exit=resolution_exit, pg_jd_affine_activation=pg_jd_affine_activation,
        jd_step_global=jd_step_global, level_shares=level_shares, barrier_basis=barrier_basis,
        pg_native_metric_policy=pg_native_metric_policy, bb2_ggn_guard=bb2_ggn_guard,
        curvature_weight=curvature_weight, geometry_weight=geometry_weight,
        geometry_weight_levels=geometry_weight_levels, pg_metric_policy_levels=pg_metric_policy_levels,
        contrast_mix=contrast_mix, data_weight=data_weight,
        first_trial_cap_levels=first_trial_cap_levels, first_trial_scope=first_trial_scope,
        inner_steps_levels=inner_steps_levels, native_start_scale=native_start_scale,
        transfer_smoothing=transfer_smoothing, curvature_weight_levels=curvature_weight_levels,
        practical_stop=practical_stop, momentum_first_levels=momentum_first_levels, pre_optimizer=pre_optimizer,
        active_set=active_set,
        **({} if native_transfer_alphas is None else dict(native_transfer_alphas=native_transfer_alphas)),   # AP2: key only when set
        **({} if ggn_diagonal_reuse is None else dict(ggn_diagonal_reuse=ggn_diagonal_reuse)),   # GG2: key only when set
        **({} if native_ablation is None else {native_ablation['switch']: native_ablation['value']}),   # T92: key only when set
        **({} if native_ablation is None or 'penalty_scale' not in native_ablation
           else dict(native_penalty_scale=native_ablation['penalty_scale'])),                         # T92: D2-c' arm
        initialization='original_zero_no_resume', export_reserve_seconds=reserve,
        **({} if export_reserve is None else dict(export_reserve_policy='explicit-v1 (FX1; user 2026-10-16 "降到 1 s")')),
        math='E34 PG route: Sequential + closed27 C2 local barrier (a=.05, kappa=.002, delta=0) + JD-step PG, '+
             ('5 inner' if inner_steps_levels is None else f'inner steps per level {inner_steps_levels} (None = 5; E85 T82)'),
        scope=('E53 explicit options (T62 PG metric / T63 backtracking exit / T64 hierarchy; user-approved 2026-09-29); '
               if e53 else 'execution-profile comparison; ')+
              ('E55 T66 global closed27 cubic JD step prediction (trial step only); ' if jd_step_global else '')+
              ('E56 T67 explicit level time shares; ' if level_shares is not None else '')+
              ('E58 T68 Bernstein-coefficient barrier (MODEL CHANGE: F differs); ' if barrier_basis == 'bernstein' else '')+
              ('E62 T71 native-level PG metric policy (metric only); ' if pg_native_metric_policy is not None else '')+
              ('E64 T72 GGN-model cap of the first BB2 step (step proposal only); ' if bb2_ggn_guard else '')+
              (f'E78 T76 curvature regularizer alpha={curvature_weight:g} (MODEL CHANGE, user-approved 2026-10-02: F = D + R + B); '
               if curvature_weight else '')+
              (f'G2 per-level curvature weights alpha={curvature_weight_levels} (MODEL PARAMETER CHANGE, user-approved '
               '2026-10-12: level L uses F = D + R_L + B with R_L = alpha_L/2 sum q |L u|^2); '
               if curvature_weight_levels is not None else '')+
              (f'E79 T77 explicit geometry weight mu=nu={geometry_weight:g} instead of the {strength} default '
               '(MODEL PARAMETER CHANGE, user-approved 2026-10-03); ' if geometry_weight is not None else '')+
              (f'E80 T78 per-level geometry weights mu=nu={geometry_weight_levels} instead of the {strength} default '
               '(MODEL PARAMETER CHANGE, user-approved 2026-10-03); ' if geometry_weight_levels is not None else '')+
              (f'E80 T78 per-level PG metric policies {pg_metric_policy_levels} (metric only); ' if pg_metric_policy_levels is not None else '')+
              (f'E80 T79 contrast-mix inputs a={contrast_mix["a"]:g} sigma={contrast_mix["sigma"]:g} eps_frac={contrast_mix["eps_frac"]:g} '
               '(INPUT PREPROCESSING, user-approved 2026-10-03; mixed image rescaled to the source std, background kept zero; '
               'the recorded MSE ratio is on the mixed images, NOT comparable with original-image ratios); ' if contrast_mix is not None else '')+
              (f'E80 T80 data-term node weights {data_weight} (MODEL CHANGE of D, user-approved 2026-10-03; recorded MSE ratios unweighted); '
               if data_weight is not None else '')+
              (f'E84 T81 first-trial max node displacement cap {first_trial_cap_levels} level voxels, scope {first_trial_scope} '
               '(trial step upper bound only, path; user-authorized 2026-10-04); ' if first_trial_cap_levels is not None else '')+
              (f'S30-G4 T89 native active set {active_set}: PG direction p = -(wS) M^-1 (wS) g with the tapered support-distance weights '
               '(path only; nodes with w = 0 never move; F/barrier/certificate/E-F/GF unchanged; user 2026-10-14); ' if active_set is not None else '')+
              (f'AP2 native transfer alpha list {native_transfer_alphas}: the pre-optimizer seed is certified at the first passing '
               'scale of (1, .99, .98, .96, .94, .92, then .9^k) instead of .9^k (CHANGES the T38g back-off list on that transfer; '
               'explicit option, user-approved 2026-10-17, relayed; path only); ' if native_transfer_alphas is not None else '')+
              (f'T92 (D2 ablation, user-confirmed 2026-10-19; docs/04 T92) {native_ablation["switch"]}={native_ablation["value"]}'
               +(f' with lam, mu, nu x {native_ablation["penalty_scale"]:g} (the metric follows)' if 'penalty_scale' in native_ablation else '')+': '
               'the native level only (the pre-optimizer and the transfer unchanged); an ablation of the model or of the path, '
               'never a recommended configuration; ' if native_ablation is not None else '')+
              (f'GG2 GGN diagonal reuse {ggn_diagonal_reuse}: the PG metric diagonal keeps the per-cell geometry locals of cells '
               'far from the barrier across outers (full build every period-th outer; metric only, F/barrier/certificate/E-F/GF '
               'unchanged; explicit option, user 2026-10-17 "4(a)可以做", relayed); ' if ggn_diagonal_reuse is not None else '')+
              (f'S30-G3 T88 float32 pre-optimizer {pre_optimizer} replaces the coarse levels as an UNCERTIFIED initializer; the native '
               'transfer certifies the tapered field (alpha back-off); F/E/certificate/E-F/GF unchanged (user 2026-10-14); ' if pre_optimizer is not None else '')+
              (f'E111 T85 momentum first direction on levels {momentum_first_levels} (first inner step along the last outer '
               'increment when it descends; path only; user-authorized 2026-10-13); ' if momentum_first_levels is not None else '')+
              (f'E85 T82 inner step limit per level {inner_steps_levels} (None = 5; MAINLINE PARAMETER as an explicit option, '
               'user-confirmed 2026-10-04; path only); ' if inner_steps_levels is not None else '')+
              ((f'C2 (T83 on the T88 route) native start scale beta={native_start_scale:g}: the native transfer tries beta*.9^k '
                'times the tapered pre-optimizer seed (CHANGES the T38g alpha = 1 first rule; explicit option, user-approved '
                '2026-10-18; path only); ' if pre_optimizer is not None else
                f'E90 T83 native start scale beta={native_start_scale:g}: the native transfer tries beta*.9^k*P u_L2 (CHANGES the '
                'T38g alpha = 1 first rule; explicit option, user-confirmed 2026-10-05; path only); ')
               if native_start_scale is not None else '')+
              (f'H17 practical stop {practical_stop}: the native level may end after a formal commit when the projected '
               'ratio gain over the remaining budget is <= epsilon (CHANGE OF THE STOPPING RULE as an explicit option, '
               'user-approved 2026-10-12; status native_practical_stop, success false, never convergence; GF threshold '
               'unchanged); ' if practical_stop is not None else '')+
              (f'A19 transfer low-pass {transfer_smoothing["kernel"]} sigma={transfer_smoothing["sigma_mm"]:g} mm on the '
               f'{transfer_smoothing["scope"]} transfer(s): alpha_k * S_sigma(P u) is certified (path only; user-approved '
               '2026-10-12); ' if transfer_smoothing is not None else '')+
              ('unchanged full-cell JD>0/E-F/forcing/GF; budget end is not convergence' if curvature_weight or curvature_weight_levels is not None or geometry_weight is not None or geometry_weight_levels is not None or data_weight is not None else
               'unchanged F/E/full-cell JD>0/E-F/forcing/GF; budget end is not convergence')))
    print(output, flush=True)


def enable_jit_cache(throughput):
    """E107: the content-addressed Numba cache (jd_curl.three_d.jit_cache, root JD_CURL_JIT_CACHE or
    ~/.cache/seq_vp_3d/numba_jit) for profiles with jit_cache, before any kernel compiles; else None (nothing read
    or written)."""
    from scripts.run_stage_3d_e19_mri import THROUGHPUT_POLICIES
    flag = isinstance(throughput, str) and bool(THROUGHPUT_POLICIES.get(throughput, {}).get('jit_cache'))
    if not flag:
        return None
    from jd_curl.three_d import jit_cache
    return jit_cache.enable_if(flag)


def parse_switches(text):
    """E111: '0,0,0,1' (or on/off, true/false) -> four booleans; anything else is refused before any directory."""
    values = {'0': False, '1': True, 'off': False, 'on': True, 'false': False, 'true': True}
    items = [x.strip().lower() for x in text.split(',')]
    if any(x not in values for x in items):
        raise ValueError('--momentum-first-levels: four entries of 0/1 (on/off, true/false)')
    return [values[x] for x in items]


def worker_options(cfg):
    """The worker keyword options of a prepared directory (FX1: shared by the run and its first-use dry pass)."""
    return dict(low_load=True, start_memory_gate=False, memory_limits=False,
                advance_stalled_levels=True, transfer_backtracking=True, local_barrier=True,
                efficient_cpu_budget=True, gpu_near_boundary=True, reuse_reference=True,
                device_batch_size=32768, selective_kernels=True, compiled_certificate=True,
                hoisted_certificate=True, certificate_tile_size=8192, throughput=cfg['throughput'],
                first_level=cfg.get('first_level', 0), lean_commits=cfg.get('lean_commits', False),
                pg_metric_policy=cfg.get('pg_metric_policy'), pg_shift_factor=cfg.get('pg_shift_factor', 1.),
                export_reserve=cfg.get('export_reserve_seconds', 30.),          # S30-G1: recorded at prepare time
                **{k: cfg[k] for k in E53_OPTIONS if cfg.get(k) is not None})


FIRST_USE_DIR = '/dev/shm'          # FX1 F2': the dry pass's synthetic inputs and records (no arrays of the run)
FIRST_USE_POLICY = 'first-use-prewarm-dry-pass-v1'


def first_use_prewarm(output, cfg):
    """FX1 F2' (throughput-v35): before the task clock, a dry pass of the worker
    with this directory's options on a synthetic pair of the same native grid geometry (no image data of the pair) and a
    synthetic native seed with strongly compressed regions, so that the E93/E94/E100/E99 first-use checks run once; it
    writes no arrays and ends after its first native formal commit. E100's overlap is not covered (on the synthetic
    field its second metric build has no candidate cells); the run checks it on its own first use, as v34 does. reset_process_counters() follows (the row-SoA
    certificate check is repeated on the run's data, as before). Returns (registry or None, record or None)."""
    from scripts.run_stage_3d_e19_mri import THROUGHPUT_POLICIES, reset_process_counters
    profile = THROUGHPUT_POLICIES.get(cfg['throughput']) if isinstance(cfg['throughput'], str) else None
    if not (profile and profile.get('first_use_prewarm')):
        return None, None
    import gc, tempfile
    from jd_curl.three_d.backends.first_use_registry import synthetic_inputs, synthetic_seed
    tick = perf_counter()
    record = dict(policy=FIRST_USE_POLICY, scope='E93 prefilter, E94 fused node JD with E100 overlap, E99 skip/tuned kernels'
                  + (", T84's synthetic-field half (v38; the input half stays in the run)"
                     if profile.get('gpu_hull_first_use_seed') else ''),
                  role='done before the task clock on a synthetic field of the native grid; the run seeds its backend with the keys')
    if cfg.get('pre_optimizer') is None:
        return None, dict(record, skipped='no pre-optimizer (the dry pass reaches the native level by the pre-optimizer route)',
                          seconds=perf_counter()-tick)
    cache_inputs = None
    if profile.get('first_use_cache'):
        # FC1 (v36): the registry of an earlier dry pass with exactly these key inputs
        # (source, device, native grid, profile, options) is used as is; otherwise the dry pass runs and its registry is stored.
        from jd_curl.three_d.backends import first_use_cache as fc
        manifest = json.loads((output/'inputs'/cfg['task']['pair']/'manifest.json').read_text())
        cache_inputs = fc.key_inputs(ROOT, manifest, profile, worker_options(cfg))
        cached, record['cache'] = fc.load(cache_inputs)
        if cached is not None:
            record.update(source='disk_cache', skipped=cached.get('skipped'), seconds=perf_counter()-tick,
                          families={k: len(v['verified'] if isinstance(v, dict) else v) for k, v in cached['families'].items()})
            return cached, record
    registry = {}
    root = Path(tempfile.mkdtemp(prefix='seq_vp_3d_first_use_', dir=FIRST_USE_DIR))
    try:
        manifest = json.loads((output/'inputs'/cfg['task']['pair']/'manifest.json').read_text())
        data = root/'inputs'; data.mkdir()
        arrays, grid = synthetic_inputs(manifest)
        np.savez(data/'pair.npz', **arrays)
        (data/'manifest.json').write_text(json.dumps(dict(
            shape_zyx=manifest['shape_zyx'], spacing_xyz=manifest['spacing_xyz'], origin_xyz=manifest['origin_xyz'],
            weights=manifest['weights'], pair_sha256=file_sha256(data/'pair.npz'), role='FX1 first-use dry pass (synthetic)')))
        dry = dict(seed=synthetic_seed(grid), registry=registry, stop_after_native_commits=1, max_seconds=45.)
        del arrays
        worker(dict(cfg['task'], name='first_use_dry'), root/'run', data, device='cuda', task_started=None,
               host_storage=None, first_use_dry=dry, **worker_options(cfg))
        record['dry_pass'] = dry.get('result')
    except Exception as exc:                 # the run then checks every first use itself, as v34 does (recorded)
        registry = {}
        record['error'] = repr(exc)
    finally:
        shutil.rmtree(root, ignore_errors=True)
        record['counters_reset'] = reset_process_counters()
        gc.collect()
    record.update(families={k: len(v['verified'] if isinstance(v, dict) else v) for k, v in registry.get('families', {}).items()},
                  skipped=registry.get('skipped'), seconds=perf_counter()-tick)
    if cache_inputs is not None and registry.get('families') and 'error' not in record:
        if registry.get('evidence'):
            # v38 (review): a dry pass that switched T84 off is not stored -- its evidence stays with this run only, and
            # the next run repeats the dry pass instead of carrying an old failure (or an old T84 skip) forward
            record['cache'].update(stored=False, store_skipped='the dry pass switched T84 off; its evidence stays with this run')
        else:
            try:                                              # FC1 (v36): only a clean dry pass is stored
                record['cache'].update(fc.store(cache_inputs, registry, record.get('dry_pass')))
            except OSError as exc:
                record['cache'].update(stored=False, store_error=repr(exc))
    return (registry if registry.get('families') else None), record


def _run_pair(output, cfg, *, jit, prewarm, resident=None, preverified=None, first_use=None):
    """One prepared directory in this process (run: the only one; serve: one of a queue). The task clock starts
    before the provenance checks, as before; their time is recorded as provenance_seconds."""
    started = perf_counter()
    with (output/'started.json').open('x') as stream:
        json.dump(dict(pid=os.getpid(), start=started, **({} if resident is None else dict(resident=resident))), stream)
    verify_source(output, cfg['source_sha256']); data = output/'inputs'/cfg['task']['pair']
    for name, digest in cfg['input_sha256'].items():
        if file_sha256(data/name) != digest:
            raise ValueError('input identity mismatch')
    provenance = perf_counter()-started
    # The storage guard watches the Windows volume that actually receives the writes.
    host = '/mnt/d' if str(output).startswith('/mnt/d/') else '/mnt/c'
    try:
        worker(cfg['task'], output/cfg['task']['name'], data, device='cuda', task_started=started,
            host_storage=host, **worker_options(cfg),                         # FX1: the same options as before, shared
            **({} if resident is None else dict(resident=resident)),
            **({} if preverified is None else dict(preverified=preverified)))   # FX1 F2'
    finally:
        verify_source(output, cfg['source_sha256'])
        for name, digest in cfg['input_sha256'].items():
            if file_sha256(data/name) != digest:
                raise ValueError('input changed during run')
    summary = json.loads((output/cfg['task']['name']/'summary.json').read_text())
    e2e = summary.get('e2e_seconds')
    dump(output/'completion.json', dict(seconds=perf_counter()-started, status=summary['status'],
        success=summary['success'], independent_audit_pending=True, jit_prewarm_seconds_outside_clock=prewarm,
        provenance_seconds=provenance, e2e_seconds=e2e,
        e2e_registration_seconds=None if e2e is None else e2e-provenance,
        cold_equivalent_seconds=None if e2e is None else e2e+prewarm+(0. if first_use is None else first_use['seconds']),
        **({} if first_use is None else dict(first_use_prewarm=first_use,                     # FX1 F2'
                                             first_use_prewarm_seconds_outside_clock=first_use['seconds'])),
        e2e_scope='S30-G1: e2e = task clock start to final_native.npz written (labels/smoothness/audit excluded); '
                  'registration = e2e minus the provenance checks; cold_equivalent adds the JIT prewarm',
        **({} if resident is None else dict(resident=resident)),
        **({} if 'practical_stop' not in summary else dict(practical_stop=summary['practical_stop'])),   # H17
        **({} if jit is None else dict(jit_cache=__import__('jd_curl.three_d.jit_cache', fromlist=['record']).record()))))


def run(output):
    output = Path(output).resolve(); cfg = json.loads((output/'protocol.json').read_text())
    tick = perf_counter()
    jit = enable_jit_cache(cfg['throughput'])
    prewarm = prewarm_jit(cfg['throughput'], cfg.get('barrier_basis'),  # outside the task clock, recorded below
                          first_trial_cap=cfg.get('first_trial_cap_levels') is not None,
                          ggn_reuse=cfg.get('ggn_diagonal_reuse') is not None)
    if jit is not None:
        prewarm = perf_counter()-tick                 # E107: the cache set-up is part of the start-up cost
    preverified, first_use = first_use_prewarm(output, cfg)     # FX1 F2' (v35 only): outside the task clock
    _run_pair(output, cfg, jit=jit, prewarm=prewarm, preverified=preverified, first_use=first_use)


def read_queue(queue):
    """S30-G1: one prepared directory per line (first token; the run_queue_disk.sh format with a watchdog column is
    accepted), blank lines and # comments ignored, no directory twice."""
    rows = [line.split()[0] for line in Path(queue).read_text().splitlines() if line.strip() and not line.lstrip().startswith('#')]
    if not rows:
        raise ValueError('serve: empty queue')
    outputs = [Path(r).resolve() for r in rows]
    if len(set(outputs)) != len(outputs):
        raise ValueError('serve: a directory is listed twice')
    return outputs


SERVE_SAME = ('throughput', 'source_sha256', 'barrier_basis')


def serve(queue, *, resident_checks='per_pair'):
    """S30-G1: one warm process for a queue of prepared directories. The JIT cache set-up and the prewarm run once;
    every directory must carry the same frozen source digests, throughput profile and barrier basis (the prewarm is
    specific to them), and must not have been started. Each pair is then exactly a `run` of that directory in this
    process: provenance checks, the worker with a new backend (first-use checks on real inputs repeated per pair),
    completion.json with the resident record. The first failure stops the queue; nothing is rerun."""
    if resident_checks != 'per_pair':
        raise ValueError("serve: only resident_checks 'per_pair' exists (first-use checks on real inputs repeat per pair)")
    outputs = read_queue(queue)
    cfgs = []
    for output in outputs:
        if not (output/'protocol.json').is_file():
            raise ValueError(f'serve: not a prepared directory: {output}')
        if (output/'started.json').exists() or (output/'completion.json').exists():
            raise ValueError(f'serve: already started: {output}')
        cfgs.append(json.loads((output/'protocol.json').read_text()))
    head = cfgs[0]
    for output, cfg in zip(outputs, cfgs):
        for key in SERVE_SAME:
            if cfg.get(key) != head.get(key):
                raise ValueError(f'serve: {output} differs from the first directory in {key}')
        if (cfg.get('first_trial_cap_levels') is not None) != (head.get('first_trial_cap_levels') is not None):
            raise ValueError(f'serve: {output} differs from the first directory in first_trial_cap_levels')
        if (cfg.get('ggn_diagonal_reuse') is not None) != (head.get('ggn_diagonal_reuse') is not None):
            raise ValueError(f'serve: {output} differs from the first directory in ggn_diagonal_reuse')   # GG2 prewarm
    tick = perf_counter()
    jit = enable_jit_cache(head['throughput'])
    prewarm = prewarm_jit(head['throughput'], head.get('barrier_basis'),
                          first_trial_cap=head.get('first_trial_cap_levels') is not None,
                          ggn_reuse=head.get('ggn_diagonal_reuse') is not None)
    if jit is not None:
        prewarm = perf_counter()-tick
    service = dict(policy=SERVE_POLICY, queue=str(Path(queue).resolve()), pid=os.getpid(), pairs=len(outputs),
                   prewarm_seconds_once=prewarm, first_use_checks=resident_checks)
    import gc
    for index, (output, cfg) in enumerate(zip(outputs, cfgs)):
        resident = dict(service, index=index, pairs_before=index)
        try:
            _run_pair(output, cfg, jit=jit, prewarm=prewarm, resident=resident)
        except BaseException as exc:
            dump(output/'serve_failure.json', dict(index=index, error=repr(exc)))
            raise
        gc.collect()
        print(json.dumps(dict(serve=index, output=str(output))), flush=True)
    return service


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'serve')); parser.add_argument('output', type=Path)
    parser.add_argument('--strength', choices=('weak', 'strong'), default='weak')
    parser.add_argument('--seconds', type=int, choices=BUDGETS, default=600)
    parser.add_argument('--throughput', choices=('none', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v19', 'v20', 'v21', 'v22', 'v23', 'v24', 'v25', 'v26', 'v27', 'v28', 'v29', 'v30', 'v31', 'v32', 'v33', 'v34', 'v35', 'v36', 'v37', 'v38', 'v39', 'v41'), default='none')
    parser.add_argument('--stage', default='E48')
    parser.add_argument('--first-level', type=int, choices=(0, 1, 2), default=0)
    parser.add_argument('--lean-commits', action='store_true')
    parser.add_argument('--pg-metric-policy', default=None)
    parser.add_argument('--pg-shift-factor', type=float, default=1.)
    parser.add_argument('--hierarchy-policy', choices=tuple(HIERARCHY_POLICIES), default=None)
    parser.add_argument('--resolution-exit', action='store_true', help='T63 backtracking exit (E53)')
    parser.add_argument('--pg-jd-affine-activation', type=float, default=None, help='T62 J_act (default .25)')
    parser.add_argument('--jd-step-global', action='store_true', help='T66 global closed27 cubic step prediction (E55)')
    parser.add_argument('--barrier-basis', choices=('samples', 'bernstein'), default=None,
                        help='T68 barrier on closed27 samples (default) or on Bernstein coefficients (E58, model change)')
    parser.add_argument('--pg-native-metric-policy', default=None,
                        help='T71 PG metric policy for the native level only (E62); coarse levels keep --pg-metric-policy')
    parser.add_argument('--bb2-ggn-guard', action='store_true', help='T72 GGN-model cap of the first BB2 step (E64)')
    parser.add_argument('--pair', default=None,
                        help='E66/E87/E88: input pair name from a pre-registered pair set (E11 native pairs, the 19 Learn2Reg '
                             'OASIS validation pairs or the 12 T0 tuning pairs); default: the development pair')
    parser.add_argument('--curvature-weight', type=float, default=0.,
                        help='E78 T76: alpha of the curvature regularizer in F (model change; 0 = absent)')
    parser.add_argument('--geometry-weight', type=float, default=None,
                        help='E79 T77: explicit mu = nu (model parameter change; default: strong 1, weak .001)')
    parser.add_argument('--geometry-weight-levels', default=None,
                        help='E80 T78: four explicit mu = nu values, one per level, e.g. 1,1,1,0.001 (model parameter change)')
    parser.add_argument('--pg-metric-policy-levels', default=None,
                        help='E80 T78: four PG metric policies, one per level (metric only); exclusive with the coarse/native options')
    parser.add_argument('--data-weight', default=None,
                        help='E80 T80: brain-shell-v1:beta=0.1,sigma_mm=2 — per-node data-term weights from the fixed image (model change)')
    parser.add_argument('--contrast-mix', default=None,
                        help='E80 T79: a,sigma,eps_frac — inputs replaced by (1-a) I + a N(I), N = local-contrast normalisation')
    parser.add_argument('--level-shares', default=None,
                        help='T67 cumulative level time shares, e.g. 0.15,0.30,0.50,1 (E56); default .05,.15,.35,1')
    parser.add_argument('--first-trial-cap-levels', default=None,
                        help='E84 T81: four per-level caps c in level voxels (none = off), e.g. none,none,none,0.2; '
                             'trial tau <= c/max_i |p_i|_h (path only)')
    parser.add_argument('--momentum-first-levels', default=None,
                        help='E111 T85: four per-level switches 0/1, e.g. 0,0,0,1 (native only); the first inner step of '
                             'each outer follows the last outer increment when it descends (path only)')
    parser.add_argument('--native-start-scale', type=float, default=None,
                        help='E90 T83: beta in [0.1, 1]; the native level starts from beta*.9^k times the prolonged L2 field (default off)')
    parser.add_argument('--practical-stop', default=None,
                        help='H17: epsilon=E[,window_seconds=30][,min_seconds=30][,min_commits=5]; the native level ends '
                             'when the projected ratio gain over the remaining budget is <= E (never convergence)')
    parser.add_argument('--curvature-weight-levels', default=None,
                        help='G2: four per-level curvature weights alpha_L >= 0, e.g. 0,0,1e-3,1e-3 (exclusive with --curvature-weight)')
    parser.add_argument('--transfer-smoothing', default=None,
                        help='A19: kernel:sigma_mm[:scope], kernel heat_dirichlet|gaussian_zero_faces, sigma in (0, 10] mm, '
                             'scope native (default) or all; the transfer certifies alpha_k * S_sigma(P u) (path only)')
    parser.add_argument('--inner-steps-levels', default=None,
                        help='E85 T82: four per-level inner step limits (none = 5), e.g. none,1,1,1 (mainline parameter, explicit option)')
    parser.add_argument('--first-trial-scope', choices=('initial', 'all'), default=None,
                        help='E84 T81: cap the first step of each inner solve (initial, default) or every step (all)')
    parser.add_argument('--active-set', default=None,
                        help='S30-G4 T89: d1,d2 (voxels; w = 1 within d1 of the support union, linear to 0 at d2): the native PG direction '
                             'is restricted to the active nodes; with --throughput v28 the frozen part is cached on the device')
    parser.add_argument('--pre-optimizer', default=None,
                        help='S30-G3 T88: fp32-v1[:iterations=80/80/60/40,lr=.25,cap=.5,sigma_grad=1,mu=1/1/.1/.03,alpha=1e-3/1e-3/1e-3/0,'
                             'kappa_pre=10,j0=.2,seconds=8,taper=4]; the float32 pre-optimizer replaces the coarse levels (uncertified initializer)')
    parser.add_argument('--native-transfer-alphas', choices=('fine09',), default=None,
                        help='AP2: with --pre-optimizer, the native transfer tries 1, .99, .98, .96, .94, .92, then .9^k (default .9^k)')
    parser.add_argument('--ggn-diagonal-reuse', default=None,
                        help='GG2: theta=T,period=K -- with --pre-optimizer, the native PG metric reuses the GGN diagonal locals of '
                             'cells whose E94 minimum J >= T * activation (and were so at the previous build); full build every K-th')
    parser.add_argument('--native-geometry-weights', default=None,
                        help='T92 D2-a/b (ablation): mu=M,nu=N on the native level, exactly one 0 and the other the native '
                             '--geometry-weight-levels value (e.g. mu=0,nu=0.03 removes the JD change penalty)')
    parser.add_argument('--native-reference', choices=('fixed', 'identity'), default=None,
                        help='T92 D2-c/c\' (ablation): the native reference is the native start (fixed) or u = 0 (identity), never '
                             'refreshed; a commit needs only E\'s certified decrease')
    parser.add_argument('--native-penalty-scale', type=float, default=None,
                        help='T92 D2-c\' down-weighted arm: with --native-reference identity, the native lam, mu, nu times s in '
                             '[1e-6, 1); the metric follows (a = s lam + shift, b = s (mu + nu)/2)')
    parser.add_argument('--native-metric-ablation', choices=('diagonal', 'identity'), default=None,
                        help='T92 D2-d (ablation): native P^-1 = S diag(M)^-1 S (diagonal) or I (identity; needs a native '
                             '--first-trial-cap-levels entry)')
    parser.add_argument('--native-inner', choices=('lm',), default=None,
                        help='T92 D2-e (ablation): the native level\'s inner solver is the library barrier-GGN LM '
                             '(F29 GGN-diagonal-equilibrated Poisson PCG, the same accepted steps per outer); pre-optimizer route only')
    parser.add_argument('--export-reserve', type=float, default=None,
                        help='FX1: explicit export reserve in seconds (default: 3 s for budgets up to 120 s, else 30 s)')
    parser.add_argument('--resident-checks', choices=('per_pair',), default='per_pair',
                        help='S30-G1 serve: first-use checks on real inputs are repeated per pair (the only policy)')
    a = parser.parse_args()
    if a.action == 'prepare':
        prepare(a.output, a.strength, a.seconds, throughput=False if a.throughput == 'none' else a.throughput,
                stage=a.stage, first_level=a.first_level, lean_commits=a.lean_commits,
                pg_metric_policy=a.pg_metric_policy, pg_shift_factor=a.pg_shift_factor,
                hierarchy_policy=a.hierarchy_policy, resolution_exit=a.resolution_exit,
                pg_jd_affine_activation=a.pg_jd_affine_activation, jd_step_global=a.jd_step_global,
                level_shares=None if a.level_shares is None else [float(x) for x in a.level_shares.split(',')],
                barrier_basis=a.barrier_basis, pg_native_metric_policy=a.pg_native_metric_policy,
                bb2_ggn_guard=a.bb2_ggn_guard, pair=a.pair, curvature_weight=a.curvature_weight,
                geometry_weight=a.geometry_weight,
                geometry_weight_levels=None if a.geometry_weight_levels is None else [float(x) for x in a.geometry_weight_levels.split(',')],
                pg_metric_policy_levels=None if a.pg_metric_policy_levels is None else a.pg_metric_policy_levels.split(','),
                contrast_mix=a.contrast_mix, data_weight=a.data_weight,
                first_trial_cap_levels=None if a.first_trial_cap_levels is None else
                    [None if x.strip().lower() == 'none' else float(x) for x in a.first_trial_cap_levels.split(',')],
                first_trial_scope=a.first_trial_scope,
                inner_steps_levels=None if a.inner_steps_levels is None else
                    [None if x.strip().lower() == 'none' else int(x) for x in a.inner_steps_levels.split(',')],
                native_start_scale=a.native_start_scale, transfer_smoothing=a.transfer_smoothing,
                practical_stop=a.practical_stop, pre_optimizer=a.pre_optimizer, active_set=a.active_set,
                export_reserve=a.export_reserve, native_transfer_alphas=a.native_transfer_alphas,
                ggn_diagonal_reuse=a.ggn_diagonal_reuse,
                native_geometry_weights=a.native_geometry_weights, native_reference=a.native_reference,
                native_metric_ablation=a.native_metric_ablation, native_penalty_scale=a.native_penalty_scale,
                native_inner=a.native_inner,
                momentum_first_levels=None if a.momentum_first_levels is None else parse_switches(a.momentum_first_levels),
                curvature_weight_levels=None if a.curvature_weight_levels is None else
                    [float(x) for x in a.curvature_weight_levels.split(',')])
    elif a.action == 'serve':
        serve(a.output, resident_checks=a.resident_checks)
    else:
        run(a.output)
