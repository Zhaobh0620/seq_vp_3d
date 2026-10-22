"""Optional single PG-weak E17 pilot, unchanged inner5/outer1 work budget.

Only the original Sequential solver makes acceptance/stopping decisions.
The caller owns snapshot preparation and a 900-second process watchdog;
this module neither launches itself in the background nor resumes a run.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_stage_3d_e11_short as e11, run_stage_3d_e15_e2e as e15
from scripts.run_stage_3d_c import dump
from jd_curl.three_d.backends.accelerated import AcceleratedHybridBackend
from jd_curl.three_d.boundary import restrict_free
from jd_curl.three_d.nifti import file_sha256
from jd_curl.three_d.residuals import AcceptedState3D
from jd_curl.three_d.sequential import solve_sequential, SequentialReuseConfig3D
from jd_curl.three_d.solvers.poisson_gradient import PGConfig3D

INNER_STEPS = 5
OUTER_STEPS = 1
WATCHDOG_SECONDS = 900
ROUTE = 'pg_weak_baseline'
PAIR = 'OASIS_OAS1_0001_MR1__OASIS_OAS1_0002_MR1'


class RecordingBackend(AcceleratedHybridBackend):
    """Save derivative-ready states; promotion alone is NOT an optimizer accept."""
    def __init__(self, device, on_promote, *, reuse_prepared=False, fused_cpu=False, near_boundary_cpu=True, reuse_reference=False, device_batch=None, selective_kernels=False, overlap_certificate=False, fused_kernels=False, fused_jd_root=False, fused_diagonal=False, cpu_gradient_band=None, fused_hvp=False, lean_host=False, cpu_gradient_margin=None, lean_identity=False, boundary_reuse=False, screen_prefilter=False, fused_node_jd=False, lean_checks=False, step_extrema=False, early_warped=False, device_candidate=False, root_screen=False, lean_certificate=False, barrier_skip=False, laplacian_memo=False, tuned_launch=False, fused_spline=False, skip_reference_first=False, estimator_overlap=False, top4_select=False, outer_reuse=False, gpu_edge_hull=False, commit_recheck=False, laplacian_cache=False, fused_laplacian=False, metric_prestart=False, device_scalars=False, fused_shift=False, direction_prefetch=False, fp32_poisson=False, fused_recheck=False, device_forcing=False, active_set_skip=False, fp32_screen=False, early_trial_hash=False):
        super().__init__(device, reuse_prepared=reuse_prepared, fused_cpu=fused_cpu, near_boundary_cpu=near_boundary_cpu, reuse_reference=reuse_reference, device_batch=device_batch, selective_kernels=selective_kernels,
                         **(dict(overlap_certificate=True) if overlap_certificate else {}),
                         **(dict(fused_kernels=True) if fused_kernels else {}),
                         **(dict(fused_jd_root=True) if fused_jd_root else {}),
                         **(dict(fused_diagonal=True) if fused_diagonal else {}),
                         **(dict(cpu_gradient_band=cpu_gradient_band) if cpu_gradient_band is not None else {}),
                         **(dict(fused_hvp=True) if fused_hvp else {}),
                         **(dict(lean_host=True) if lean_host else {}),
                         **(dict(cpu_gradient_margin=cpu_gradient_margin) if cpu_gradient_margin is not None else {}),
                         **(dict(lean_identity=True) if lean_identity else {}),
                         **(dict(boundary_reuse=True) if boundary_reuse else {}),
                         **(dict(screen_prefilter=True) if screen_prefilter else {}),
                         **(dict(fused_node_jd=True) if fused_node_jd else {}),
                         **(dict(lean_checks=True) if lean_checks else {}),
                         **{name: True for name, value in (('step_extrema', step_extrema), ('early_warped', early_warped),
                            ('device_candidate', device_candidate), ('root_screen', root_screen),
                            ('lean_certificate', lean_certificate), ('barrier_skip', barrier_skip),
                            ('laplacian_memo', laplacian_memo), ('tuned_launch', tuned_launch),
                            ('fused_spline', fused_spline), ('skip_reference_first', skip_reference_first),
                            ('estimator_overlap', estimator_overlap), ('top4_select', top4_select),
                            ('outer_reuse', outer_reuse), ('gpu_edge_hull', gpu_edge_hull),
                            ('commit_recheck', commit_recheck), ('laplacian_cache', laplacian_cache),
                            ('fused_laplacian', fused_laplacian), ('metric_prestart', metric_prestart),
                            ('device_scalars', device_scalars), ('fused_shift', fused_shift),
                            ('direction_prefetch', direction_prefetch), ('fp32_poisson', fp32_poisson),
                            ('fused_recheck', fused_recheck), ('device_forcing', device_forcing),
                            ('active_set_skip', active_set_skip), ('fp32_screen', fp32_screen),   # T89 / T90: HS1 review (2026-10-17) -- never wired here before
                            ('early_trial_hash', early_trial_hash))                                 # v41 (H2)
                            if value})   # E97, E98, E99, E100, E101, E103, E104, E105
        self.on_promote = on_promote

    def promote_trial(self, problem, trial):
        state = super().promote_trial(problem, trial)
        if isinstance(state, AcceptedState3D):
            self.on_promote(problem, state)
        return state


def worker(out, data, *, device='cuda', reuse_prepared=False, checkpoint_compressed=True):
    """Small fixtures may replace e11.run.inputs; production uses copied inputs.

    Worker wall includes load/construction, device initialization, solver,
    every checkpoint/log and native label evaluation. Import/process startup
    is outside this function and must be included by the parent's wall clock.
    """
    import torch
    if not isinstance(reuse_prepared, bool) or not isinstance(checkpoint_compressed, bool):
        raise ValueError('reuse_prepared and checkpoint_compressed must be boolean')
    started = perf_counter()
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    promoted, attempts, commits, history = [], [], [], []
    io = dict(promoted_checkpoints=0., inner_observer=0., outer_observer=0., final_exports=0.)
    first_inner_record_seconds = None
    solve_started = None

    def progress(stage, **values):
        dump(out/'progress.json', dict(stage=stage, utc=datetime.now(timezone.utc).isoformat(),
             worker_elapsed_seconds=perf_counter()-started, pid=os.getpid(), **values))

    def save_arrays(name, **arrays):
        temp = out/(name+'.partial.npz')
        writer = np.savez_compressed if checkpoint_compressed else np.savez
        writer(temp, **arrays)
        temp.replace(out/name)
        return dict(path=name, sha256=file_sha256(out/name))

    def save_promoted(problem, state):
        tick = perf_counter()
        record = save_arrays(f'promoted_{len(promoted)+1:04d}_NONCOMMIT.npz',
                             u=state.u, warped=state.warped)
        record.update(index=len(promoted)+1, role='promoted_noncommit; derivative-ready only, not optimizer acceptance',
                      source='RecordingBackend.promote_trial; CPU gradient-recheck temporaries are not checkpoints',
                      eval_id=state.eval_id, problem_id=state.problem_id, controls_id=problem.controls.controls_id,
                      certificate=asdict(state.certificate), energy=asdict(state.energy),
                      worker_visible_seconds=perf_counter()-started)
        promoted.append(record); dump(out/'promoted.json', promoted)
        progress('promoted_noncommit_saved', promoted=len(promoted), formal_commits=len(commits))
        io['promoted_checkpoints'] += perf_counter()-tick

    def inner_observer(row):
        nonlocal first_inner_record_seconds
        tick = perf_counter()
        if first_inner_record_seconds is None:
            first_inner_record_seconds = tick-started
        # The solver supplies a deep copy including tau, GE, target, BB2 and
        # the actual fallback/rejection path. Never infer acceptance from files.
        attempts.append(dict(attempt_index=len(attempts)+1, worker_visible_seconds=tick-started, **row))
        dump(out/'inner_attempts.json', attempts)
        progress('inner_attempt_recorded', attempts=len(attempts), accepted=row['accepted'],
                 reason=row['reason'], formal_commits=len(commits))
        io['inner_observer'] += perf_counter()-tick

    def outer_observer(problem, start, inner, row, candidate):
        tick = perf_counter(); history.append(row)
        if row['outer_accepted']:
            record = save_arrays(f'committed_{len(commits)+1:03d}.npz', u=candidate.u,
                                 warped=candidate.warped, before_u=start.u)
            record.update(index=len(commits)+1, role='formal_outer_commit',
                          ratio=row['committed_mse_ratio'], GF=row['committed_GF'],
                          min_jd_lower_bound=candidate.certificate.min_jd_lower_bound,
                          certificate=asdict(candidate.certificate), eval_id=candidate.eval_id,
                          solver_commit_seconds=row['elapsed_seconds'],
                          worker_visible_seconds=perf_counter()-started,
                          solve_visible_seconds=perf_counter()-solve_started)
            commits.append(record)
        dump(out/'history.json', history); dump(out/'commits.json', commits)
        progress('outer_checked', formal_commits=len(commits), reason=row['reason'])
        io['outer_observer'] += perf_counter()-tick

    progress('load_and_construction_started')
    tick = perf_counter()
    with e15.execution():
        problem, arrays, manifest = e11.load_problem(Path(data), 'weak', e11.CERTIFICATE)
    if np.any(arrays['initial_u']) or np.any(problem.controls.u_ref):
        raise ValueError('pilot requires original zero initial field and reference')
    load_seconds = perf_counter()-tick
    tick = perf_counter()
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    backend = RecordingBackend(device, save_promoted, reuse_prepared=reuse_prepared)
    backend_seconds = perf_counter()-tick
    inner = PGConfig3D(max_iterations=INNER_STEPS)
    outer = SequentialReuseConfig3D(max_iterations=OUTER_STEPS, eta0=.01)
    config = dict(stage='E17_pilot', route=ROUTE, device=device, inner=asdict(inner), outer=asdict(outer),
                  weights=asdict(problem.weights), memory=asdict(problem.memory),
                  certificate=asdict(problem.certificate_config), barrier_location=problem.barrier_location,
                  grid_id=problem.grid.grid_id, input_sha256=manifest['pair_sha256'],
                  threads={k: os.environ.get(k) for k in e11.run.THREADS},
                  watchdog_seconds=WATCHDOG_SECONDS, watchdog_owner='external parent process, mandatory for native CLI',
                  baseline_policy='solver initial_state, GF_initial and native_mse; no extra pre-solver CPU baseline',
                  termination_policy='original solver GF success; no ratio stop',
                  evidence_scope='single inner5/outer1 execution-speed gate, not full registration/convergence',
                  labels_policy='seg4 primary; seg35 audit-only; never used in loss',
                  checkpoint_scope='ALL successful backend.promote_trial calls, including preparation; excludes temporary CPU gradient rechecks; not comparable to E15 optimizer-promotion-only IO',
                  evaluation_backend=backend.evaluation_backend_id)
    if reuse_prepared or not checkpoint_compressed:
        config.update(stage='E18_reuse_io', reuse_prepared=reuse_prepared,
                      checkpoint_encoding='deflate' if checkpoint_compressed else 'stored',
                      reuse_scope='same immutable problem and byte-identical prepared state only')
    dump(out/'config.json', config)
    dump(out/'promoted.json', promoted); dump(out/'inner_attempts.json', attempts)
    dump(out/'commits.json', commits); dump(out/'history.json', history)
    progress('solve_started')
    solve_started = perf_counter()
    result = solve_sequential(problem, restrict_free(arrays['initial_u'], problem.grid), solver='pg',
                              inner_config=inner, config=outer, backend=backend,
                              observer=outer_observer, inner_observer=inner_observer)
    backend.device.synchronize()
    solve_wall = perf_counter()-solve_started
    tick = perf_counter()
    initial = result.initial_state
    if initial is not None:
        record = save_arrays('initial.npz', u=initial.u, warped=initial.warped)
        record.update(role='solver_original_initial_state', native_mse=result.native_mse,
                      GF=result.GF_initial, GF_threshold=result.GF_threshold, ratio=1. if result.native_mse else None,
                      certificate=asdict(initial.certificate), eval_id=initial.eval_id)
        dump(out/'initial.json', record)
    dump(out/'history.json', list(result.history)); dump(out/'commits.json', commits)
    io['final_exports'] += perf_counter()-tick
    progress('native_label_evaluation_started', formal_commits=len(commits))
    tick = perf_counter()
    final_metrics = None if result.final_state is None else e11.label_evaluation(arrays, problem.grid, result.final_state.u)
    label_seconds = perf_counter()-tick
    tick = perf_counter()
    role = 'formal_outer_commit' if commits else 'initial_no_outer_commit' if result.final_state is not None else 'no_valid_state'
    dump(out/'labels.json', dict(role=role, primary='seg4', audit_only='seg35', labels_not_in_loss=True,
                                 baseline=manifest['baseline_label_metrics'], final=final_metrics))
    io['final_exports'] += perf_counter()-tick
    summary = dict(route=ROUTE, device=device, status=result.status, success=result.success, reason=result.reason,
                   native_mse=result.native_mse, final_mse=result.final_mse, ratio=result.mse_ratio,
                   best_ratio=result.best_mse_ratio, GF0=result.GF_initial, GF=result.GF_final,
                   GF_threshold=result.GF_threshold, GE=result.GE_final, GD=result.GD_final,
                   energy=None if result.final_state is None else asdict(result.final_state.energy),
                   certificate=None if result.final_state is None else asdict(result.final_state.certificate),
                   counts=result.counts, commits=len(commits), promoted_noncommit_snapshots=len(promoted),
                   attempted_inner_candidates=len(attempts), solver_seconds=result.elapsed_seconds,
                   solve_call_wall_seconds=solve_wall, load_make_problem_seconds=load_seconds,
                   backend_construction_seconds=backend_seconds, native_label_seconds=label_seconds,
                   time_to_first_inner_record_seconds=first_inner_record_seconds,
                   measured_io_seconds=io, measured_io_total_seconds=sum(io.values()),
                   io_scope='checkpoint/observer/final export measurements; small startup/status/summary writes excluded',
                   worker_seconds=perf_counter()-started,
                   peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   device_peak_allocated_bytes=torch.cuda.max_memory_allocated() if device == 'cuda' else None,
                   device_peak_reserved_bytes=torch.cuda.max_memory_reserved() if device == 'cuda' else None,
                   backend_timings=backend.timings,
                   backend_timing_scope='inclusive nested operation timers; NEVER sum or subtract from wall time',
                   evaluation_counts=backend.evaluation_counts, comparison_counts=backend.comparison_counts,
                   cpu_inner_gradient_rechecks=backend.cpu_gradient_checks, transfers=asdict(backend.device.transfers),
                   resolved_config=result.resolved_config)
    if reuse_prepared or not checkpoint_compressed:
        summary.update(reuse_counts=dict(backend.reuse_counts), reuse_prepared=reuse_prepared,
                       checkpoint_encoding='deflate' if checkpoint_compressed else 'stored')
    dump(out/'summary.json', summary)
    progress('worker_completed', formal_commits=len(commits), status=result.status, success=result.success)
    dump(out/'worker_completion.json', dict(complete=True, worker_total_seconds=perf_counter()-started,
             commits=len(commits), metrics_role=role, device=device, independent_audit_pending=True))
    return result


def verify_frozen_run(run_root):
    """Root prepares/freezes the run and owns the actual 900-second watchdog."""
    import jd_curl
    run_root = Path(run_root).resolve()
    if ROOT != (run_root/'snapshot').resolve() or not Path(jd_curl.__file__).resolve().is_relative_to(ROOT):
        raise ValueError('CLI must execute the new run snapshot, not workspace sources')
    schedule = e11.read(run_root/'schedule.json')
    if (schedule['inner_steps'], schedule['outer_steps'], schedule['watchdog_seconds'], schedule['device']) != (5, 1, 900, 'cuda'):
        raise ValueError('pilot requires CUDA inner5/outer1 and parent watchdog900')
    tasks = schedule['tasks']
    if (len(tasks) != 1 or tasks[0]['route'] != ROUTE or tasks[0]['pair'] != PAIR
            or tasks[0]['index'] != 0):
        raise ValueError('pilot only accepts one PG weak task')
    for name, digest in e11.read(run_root/'source_sha256.json').items():
        if file_sha256(ROOT/name) != digest:
            raise ValueError('frozen source SHA mismatch: '+name)
    for name, digest in e11.read(run_root/'input_sha256.json').items():
        if file_sha256(run_root/name) != digest:
            raise ValueError('copied input SHA mismatch: '+name)
    task = tasks[0]
    manifest = e11.read(run_root/'inputs'/task['pair']/'manifest.json')
    if (manifest['shape_zyx'] != [192, 224, 160] or manifest['spacing_xyz'] != [1., 1., 1.]
            or manifest['unit'] != 'mm' or manifest['pair'] != PAIR):
        raise ValueError('CLI pilot requires the first native 1 mm pair')
    return run_root/task['name'], run_root/'inputs'/task['pair']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    args = parser.parse_args()
    out, data = verify_frozen_run(args.run_root)
    worker(out, data, device='cuda')


if __name__ == '__main__':
    main()
