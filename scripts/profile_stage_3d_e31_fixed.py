"""User-operated fixed native replay; no optimizer, restart, or new commit."""
import argparse
from dataclasses import asdict, replace
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
from time import perf_counter

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jd_curl.three_d.backends.accelerated import AcceleratedHybridBackend
from jd_curl.three_d.backends.cpu_comparison import fused_cpu_reductions
from jd_curl.three_d.backends.device_batch import DeviceBatchConfig
from jd_curl.three_d.boundary import restrict_free
from jd_curl.three_d.feasibility import EnclosureCertificateConfig, CompiledEnclosureCertificateConfig
from jd_curl.three_d.grid import make_grid_3d
from jd_curl.three_d.linearization import build_linearization
from jd_curl.three_d.nifti import file_sha256
from jd_curl.three_d.problem import ParallelMemoryConfig, Weights3D, array_hash, make_problem, make_controls, with_controls
from jd_curl.three_d.residuals import AcceptedState3D, TrialState3D
from jd_curl.three_d.work_control import WorkBudgetExceeded, checkpoint, work_session


def read(path):
    return json.loads(path.read_text())


def dump(path, value):
    temporary = path.with_suffix('.partial.json')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def one(rows, message):
    if len(rows) != 1:
        raise ValueError(message+' must identify exactly one record')
    return rows[0]


def contained(root, name):
    path = (root/name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('record path escapes its input directory')
    return path


def load_case(campaign, task, outer_index, inner_iteration):
    """Resolve three distinct roles from recorded identities, never filenames alone."""
    folder = contained(campaign, task)
    names = ('config.json', 'history.json', 'commits.json', 'promoted.json', 'inner_attempts.json')
    metadata = {name: read(folder/name) for name in names}
    config = metadata['config.json']
    row = one([x for x in metadata['history.json'] if x['level'] == 3
               and x['outer_i'] == outer_index and x['outer_accepted']], 'native outer record')
    attempt = one([x for x in row['inner_history'] if x['iteration'] == inner_iteration
                   and x['accepted']], 'accepted inner record')
    recorded = one([x for x in metadata['inner_attempts.json'] if x['level'] == 3
                    and x['controls_id'] == row['controls_id']
                    and x['base_eval_id'] == attempt['base_eval_id']
                    and x['trial_eval_id'] == attempt['trial_eval_id']], 'recorded inner attempt')
    if any(recorded[k] != v for k, v in attempt.items()):
        raise ValueError('inner history disagrees with independent attempt log')
    commit = one([x for x in metadata['commits.json'] if x['level'] == 3 and x['native']
                  and x['eval_id'] == row['candidate_eval_id']], 'native formal commit')
    if commit['role'] != 'formal_outer_commit':
        raise ValueError('reference checkpoint is not a formal outer commit')
    records = {}
    for role, key in (('base', 'base_eval_id'), ('candidate', 'trial_eval_id')):
        matching = [x for x in metadata['promoted.json'] if x['level'] == 3
                    and x['eval_id'] == attempt[key] and x['controls_id'] == row['controls_id']]
        if not matching or any(x['sha256'] != matching[0]['sha256'] for x in matching):
            raise ValueError('missing or inconsistent derivative-ready '+role)
        record = matching[0]
        if record['role'] != 'derivative_ready_NOT_outer_commit':
            raise ValueError('inner snapshot role must remain NONCOMMIT')
        records[role] = record
    required = -config['inner']['c1']*attempt['tau']*attempt['gradient_dot_direction']
    if (not math.isfinite(required) or required < 0 or
            not math.isclose(required, attempt['comparison_diagnostics']['required_reduction'],
                             rel_tol=1e-13, abs_tol=0.)):
        raise ValueError('Armijo requirement does not match frozen metadata')
    hashes = {str(folder/name): file_sha256(folder/name) for name in names}
    fields = {}
    for role, record, key in [('reference', commit, 'before_u')]+[
            (role, record, 'u') for role, record in records.items()]:
        checkpoint('snapshot_read', role=role)
        path = contained(folder, record['path'])
        digest = file_sha256(path)
        if digest != record['sha256']:
            raise ValueError(role+' checkpoint SHA mismatch')
        hashes[str(path)] = digest
        with np.load(path, allow_pickle=False) as saved:
            fields[role] = saved[key]
        if role != 'reference' and array_hash(fields[role]) != record['certificate']['state_hash']:
            raise ValueError(role+' map/certificate identity mismatch')
    data = contained(campaign/'inputs', config['task']['pair'])
    manifest = read(data/'manifest.json')
    digest = file_sha256(data/'pair.npz')
    if digest != manifest['pair_sha256'] or digest != config['input_sha256']:
        raise ValueError('input pair SHA mismatch')
    hashes[str(data/'pair.npz')] = digest
    hashes[str(data/'manifest.json')] = file_sha256(data/'manifest.json')
    checkpoint('input_read')
    with np.load(data/'pair.npz', allow_pickle=False) as arrays:
        fixed, moving = arrays['fixed'], arrays['moving']
    nz, ny, nx = manifest['shape_zyx']; hx, hy, hz = manifest['spacing_xyz']
    x0, y0, z0 = manifest['origin_xyz']
    grid = make_grid_3d(nx=nx, ny=ny, nz=nz, hx=hx, hy=hy, hz=hz,
                        x0=x0, y0=y0, z0=z0, unit=manifest['unit'])
    barrier = config['barrier_model']
    if barrier['policy'] != 'localized-c2-v1' or barrier['location'] != config['barrier_location']:
        raise ValueError('unsupported or inconsistent barrier model')
    certificate_type = CompiledEnclosureCertificateConfig if 'kernel' in config['certificate'] else EnclosureCertificateConfig
    problem = make_problem(grid=grid, fixed_image=fixed, moving_image=moving,
        u_ref=fields['reference'], weights=Weights3D(**config['weights']),
        memory=ParallelMemoryConfig(**config['memory']),
        certificate_config=certificate_type(**config['certificate']),
        barrier_location=config['barrier_location'], barrier_activation=barrier['activation'])
    if (problem.problem_id != row['problem_id'] or problem.controls.controls_id != row['controls_id']
            or grid.grid_id != config['native_grid_id']):
        raise ValueError('reconstructed native model/reference identity mismatch')
    evidence = dict(input_sha256=hashes, configuration=config, frozen_attempt=attempt,
        original_problem_id=row['problem_id'],
        controls_id=row['controls_id'], grid_id=grid.grid_id, outer_index=outer_index,
        inner_iteration=inner_iteration, attempt=attempt['attempt'], required_reduction=required,
        roles=dict(reference=dict(path=commit['path'], array='before_u', role='frozen_outer_reference',
                                   eval_id=row.get('start_eval_id')),
                   **{role: dict(path=x['path'], role=x['role'], eval_id=x['eval_id'])
                      for role, x in records.items()}),
        map_sha256={role: array_hash(value) for role, value in fields.items()},
        historical_E_reduction=attempt['actual_reduction'],
        saved_candidates_are_not_new_commits=True)
    return problem, fields['base'], fields['candidate'], evidence


def reduction_record(values, required):
    return dict(E=dict(actual=float(values[0].actual), resolution=float(values[0].resolution),
                       certified=float(values[0].certified), armijo_passed=bool(values[0].certified >= required)),
                F=dict(actual=float(values[1].actual), resolution=float(values[1].resolution),
                       certified=float(values[1].certified), descent_passed=bool(values[1].certified >= 0.)))


def freeze_source(output):
    hashes = {}
    for directory in ('jd_curl','scripts'):
        for path in sorted((ROOT/directory).rglob('*.py')):
            if '__pycache__' in path.parts: continue
            key = str(path.relative_to(ROOT)); payload = path.read_bytes()
            target = output/'snapshot'/key; target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(payload)
            hashes[key] = hashlib.sha256(payload).hexdigest()
    return hashes


def verify_source(output, hashes):
    if any(file_sha256(ROOT/key) != digest or file_sha256(output/'snapshot'/key) != digest
           for key,digest in hashes.items()):
        raise RuntimeError('live or frozen source changed during fixed replay')


def run(campaign, output, *, task='pg_weak__four__3600s', outer_index=1,
        inner_iteration=1, tiles=(32768,), device='cuda', wall_seconds=600., certificate_comparison=False):
    if not math.isfinite(wall_seconds) or not 0 < wall_seconds <= 600:
        raise ValueError('wall_seconds must be in (0,600]')
    if not tiles or len(set(tiles)) != len(tiles) or any(x not in (8192,32768,65536) for x in tiles):
        raise ValueError('tiles must be distinct registered sizes: 8192,32768,65536')
    if not isinstance(certificate_comparison, bool) or (certificate_comparison and tuple(tiles)!=(32768,)):
        raise ValueError('certificate comparison requires boolean and fixed GPU tile32768')
    output, campaign = Path(output).resolve(), Path(campaign).resolve()
    if output.is_relative_to(campaign):
        raise ValueError('output must not modify the source campaign')
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    variants = [('E29', None, False)]+[(f'{name}-{tile}', tile, selective)
        for tile in tiles for name, selective in (('E30',False), ('E31',True))]
    if certificate_comparison:
        variants = [('E32-cert2048',32768,True),('E33-cert2048',32768,True),('E33-cert8192',32768,True)]
    result = dict(certificate_comparison=certificate_comparison, status='running', completed=False, completed_variants=0,
        requested_variants=[x[0] for x in variants], cases=[], independent_cpu=None,
        device=device, torch_version=torch.__version__, cuda_version=torch.version.cuda, dtype='float64',
        environment=dict(python=sys.version, numpy=np.__version__, torch=torch.__version__,
            cuda=torch.version.cuda, requested_device=device, torch_threads=torch.get_num_threads(),
            thread_environment={key:os.environ.get(key) for key in
                ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')}),
        optimization_runs=0, new_commits=0, wall_limit_seconds=wall_seconds,
        deadline_policy='cooperative checkpoints; external watchdog required for a hard wall limit',
        timing_scope='synchronized wall time; backend subphases nest, do not sum',
        rss_scope='process lifetime high-water RSS, not per-variant allocation',
        source_sha256=freeze_source(output))
    active = {'variant':None, 'phase':'setup'}
    last_progress = [float('-inf')]

    def progress(event, **values):
        record = dict(event=event, elapsed_seconds=perf_counter()-started, **active, **values)
        with (output/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(record, allow_nan=False)+'\n')
        dump(output/'progress.json', record)
        print('fixed_probe '+json.dumps(record, allow_nan=False), flush=True)

    def work(event):
        if event['event'] == 'work_progress' and perf_counter()-last_progress[0] >= 5.:
            last_progress[0] = perf_counter()
            progress('working', work_phase=event['phase'])

    def save():
        result['elapsed_seconds'] = perf_counter()-started
        result['skipped_variants'] = result['requested_variants'][result['completed_variants']:]
        dump(output/'result.json', result)

    def stage(name, fn, backend=None, row=None):
        active['phase'] = name; progress('stage_started'); checkpoint(name)
        if backend is not None: backend.device.synchronize()
        tick = perf_counter()
        value = fn()
        if backend is not None: backend.device.synchronize()
        seconds = perf_counter()-tick
        if row is not None: row['wall_seconds'][name] = seconds
        progress('stage_completed', seconds=seconds); save(); checkpoint(name+'_finished')
        return value

    baseline = None
    baseline_proofs = baseline_energy = None
    warmed = set()
    try:
        verify_source(output,result['source_sha256'])
        with work_session(lambda: perf_counter()-started >= wall_seconds, work):
            problem, base_u, candidate_u, evidence = stage('input_and_problem_setup',
                lambda: load_case(campaign, task, outer_index, inner_iteration))
            result['evidence'] = evidence
            original_problem = problem
            theta_base, theta_candidate = [restrict_free(u,problem.grid) for u in (base_u,candidate_u)]
            required = evidence['required_reduction']
            for name, tile, selective in variants:
                active['variant'] = name
                row = dict(name=name, wall_seconds={})
                result['current_case'] = row
                backend = a = b = gradients = value = refreshed = None
                try:
                    if certificate_comparison:
                        from jd_curl.three_d.compiled_enclosure import LEGACY_KERNEL, HOISTED_KERNEL, runtime_info
                        from jd_curl.three_d.feasibility import compiled_jacobian_enclosure
                        kernel = LEGACY_KERNEL if name.startswith('E32') else HOISTED_KERNEL
                        cpu_tile = 8192 if name.endswith('8192') else 2048
                        cfg = CompiledEnclosureCertificateConfig(**dict(asdict(original_problem.certificate_config),
                            tile_size=cpu_tile, kernel=kernel))
                        row['certificate_config'] = asdict(cfg)
                        row['certificate_runtime'] = runtime_info(kernel)
                        if kernel not in warmed:
                            stage('cold_compile', lambda:compiled_jacobian_enclosure(
                                np.zeros((1,8,3)), original_problem.grid.spacing, kernel=kernel), row=row)
                            warmed.add(kernel)
                        # Reuse immutable images/operators, but rebuild and validate
                        # controls and problem identity for this proof configuration.
                        problem = stage('problem_setup', lambda:with_controls(
                            replace(original_problem, certificate_config=cfg),
                            make_controls(original_problem.grid, original_problem.controls.u_ref,
                                          certificate_config=cfg)), row=row)
                    def construct():
                        if device == 'cuda': torch.cuda.reset_peak_memory_stats()
                        return AcceleratedHybridBackend(device, reuse_prepared=True,
                            reuse_reference=True, fused_cpu=True, near_boundary_cpu=False,
                            device_batch=None if tile is None else DeviceBatchConfig(tile),
                            selective_kernels=selective)
                    backend = stage('backend_setup', construct, row=row)
                    result['environment'].update(actual_device=str(backend.device.device),
                        gpu_name=torch.cuda.get_device_name(backend.device.device)
                            if backend.device.device.type=='cuda' else None)
                    a = stage('base_evaluate', lambda:backend.evaluate_trial(problem,theta_base),backend,row)
                    a = stage('base_promote', lambda:backend.promote_trial(problem,a),backend,row)
                    b = stage('candidate_evaluate', lambda:backend.evaluate_trial(problem,theta_candidate),backend,row)
                    if not isinstance(a,AcceptedState3D) or not isinstance(b,TrialState3D):
                        raise ValueError('recorded valid state failed fresh evaluation')
                    values = stage('stable_E_F', lambda:backend.stable_energy_reductions(
                        problem,a,b,required_reduction=required),backend,row)
                    row['reductions'] = reduction_record(values,required)
                    np.testing.assert_allclose(values[0].actual,evidence['historical_E_reduction'],
                                               rtol=2e-9,atol=2e-9)
                    row['comparison'] = backend.last_comparison
                    row['energy'] = asdict(b.energy)
                    row['certificate'] = asdict(b.certificate)
                    if certificate_comparison:
                        normalize = lambda c:{k:v for k,v in asdict(c).items() if k not in ('config','certificate_version')}
                        proofs = (normalize(a.certificate),normalize(b.certificate))
                        if baseline_proofs is None:
                            baseline_proofs, baseline_energy = proofs, dict(row['energy'])
                        if proofs != baseline_proofs:
                            raise AssertionError('certificate comparison disagreement')
                        np.testing.assert_allclose(list(row['energy'].values()),list(baseline_energy.values()),
                                                   rtol=2e-12,atol=2e-12)
                        row['proof_energy_comparison_passed'] = True
                    if result['independent_cpu'] is None:
                        cpu = stage('independent_cpu_E_F', lambda:fused_cpu_reductions(
                            problem,a,b,joint=True),backend,row)
                        result['independent_cpu'] = reduction_record(cpu,required)
                        result['independent_cpu']['scope'] = 'one independently resampled CPU geometric reduction; existing exact CPU whole-map proofs'
                    for term, key in (('E','armijo_passed'),('F','descent_passed')):
                        expected = result['independent_cpu'][term]
                        if row['reductions'][term][key] != expected[key]:
                            raise AssertionError('device/independent CPU acceptance disagreement')
                        np.testing.assert_allclose(row['reductions'][term]['actual'],expected['actual'],
                                                   rtol=2e-9,atol=2e-9)
                    b = stage('candidate_promote', lambda:backend.promote_trial(problem,b),backend,row)
                    gradients = stage('outer_gradients', lambda:backend.outer_gradient_vectors(
                        build_linearization(problem,b)),backend,row)
                    if baseline is None:
                        baseline = tuple(x.copy() for x in gradients)
                    row['gradient_comparison'] = []
                    for value, expected in zip(gradients,baseline,strict=True):
                        error = float(np.max(np.abs(value-expected)))
                        np.testing.assert_allclose(value,expected,rtol=2e-8,atol=2e-9)
                        row['gradient_comparison'].append(dict(max_abs_error=error,
                            norm=float(np.linalg.norm(value)), reference=result['requested_variants'][0]+'_same_state'))
                    if certificate_comparison:
                        refreshed = stage('reference_refresh', lambda:backend.refresh_reference(problem,b),backend,row)
                        if refreshed is None: raise AssertionError('expected explicit reference reuse')
                        fresh_problem,fresh_state = refreshed
                        assert fresh_state.certificate is b.certificate
                        assert (fresh_state.energy.proximity,fresh_state.energy.jd,fresh_state.energy.curl)==(0.,0.,0.)
                        fresh_gradients = stage('refreshed_gradients', lambda:backend.outer_gradient_vectors(
                            build_linearization(fresh_problem,fresh_state)),backend,row)
                        for fresh_vector, expected in zip(fresh_gradients,
                                                         (gradients[0],gradients[1],gradients[0]),strict=True):
                            np.testing.assert_allclose(fresh_vector,expected,rtol=2e-8,atol=2e-9)
                        row['reference_refresh_passed'] = True
                        row['refreshed_GF_max_abs_error'] = float(np.max(np.abs(fresh_gradients[0]-gradients[0])))
                        fresh_problem = fresh_state = fresh_gradients = refreshed = None
                    row.update(policy=backend.evaluation_backend_id, backend_timings=dict(backend.timings),
                        transfers=asdict(backend.device.transfers),
                        evaluation_counts=dict(backend.evaluation_counts),
                        comparison_counts=dict(backend.comparison_counts),
                        fusion_counts=dict(getattr(backend,'fusion_counts',{})),
                        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0,
                        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved() if device=='cuda' else 0,
                        resident_pipeline_seconds=sum(row['wall_seconds'][key] for key in
                            ('candidate_evaluate','stable_E_F','candidate_promote','outer_gradients')))
                    if certificate_comparison:
                        row['base_candidate_work_seconds'] = sum(row['wall_seconds'][key] for key in
                            ('base_evaluate','base_promote','candidate_evaluate','stable_E_F','candidate_promote','outer_gradients'))
                        row['with_refresh_work_seconds'] = row['base_candidate_work_seconds']+row['wall_seconds']['reference_refresh']+row['wall_seconds']['refreshed_gradients']
                    checkpoint('variant_validation_finished')
                    result['cases'].append(row);result['completed_variants'] += 1
                    result.pop('current_case')
                    save();progress('variant_completed')
                finally:
                    # Do not retain device models or native state fields across cases.
                    backend = a = b = gradients = value = refreshed = None
                    gc.collect()
                    if device == 'cuda': torch.cuda.empty_cache()
            result.update(status='completed',completed=True)
    except WorkBudgetExceeded as error:
        result.update(status='time_budget', interrupted_phase=error.phase)
    except KeyboardInterrupt:
        result.update(status='user_interrupted',interrupted_phase=active['phase'])
    except Exception as error:
        result.update(status='failed',error_type=type(error).__name__,error=str(error))
        save();progress('failed');raise
    finally:
        try:
            verify_source(output,result['source_sha256'])
        except Exception as error:
            result.update(status='failed',completed=False,error_type=type(error).__name__,error=str(error))
            save();raise
    save();progress('finished',status=result['status'],completed_variants=result['completed_variants'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--task',default='pg_weak__four__3600s')
    parser.add_argument('--outer-index',type=int,default=1)
    parser.add_argument('--inner-iteration',type=int,default=1)
    parser.add_argument('--tiles',type=int,nargs='+',default=[32768],choices=[8192,32768,65536])
    parser.add_argument('--certificate-comparison',action='store_true',help='E34 fixed E32/E33 CPU certificate policies; GPU tile32768')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--wall-seconds',type=float,default=600.)
    args=parser.parse_args()
    torch.set_num_threads(1)
    result=run(args.campaign,args.output,task=args.task,outer_index=args.outer_index,
        inner_iteration=args.inner_iteration,tiles=args.tiles,device=args.device,wall_seconds=args.wall_seconds,certificate_comparison=args.certificate_comparison)
    return 0 if result['completed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
