"""One native pair, six unchanged short budgets, opt-in E13/E14 execution.

Reuse E11's worker, outputs, labels and signal-safe serial supervisor. The
historical E11 campaign is read-only; no interrupted candidate is resumed.
"""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_stage_3d_e11_short as e11
from scripts.run_stage_3d_c import dump
from jd_curl.three_d.problem import ParallelMemoryConfig
from jd_curl.three_d.sequential import SequentialReuseConfig3D
from jd_curl.three_d.nifti import file_sha256

OLD = ROOT/'results/mri_3d/stage_E11_short_v1'
DEST = ROOT/'results/mri_3d/stage_E15_e2e_v1'
MEMORY = ParallelMemoryConfig(8192, 65536, cpu_workers=4)


@contextmanager
def execution():
    original_dump = e11.run.dump

    def emit(path, value):
        if path.name == 'config.json':
            value = dict(value, stage='E15_e2e', execution_policy='E13_parallel4_E14_reuse',
                         historical_baseline='historical_baseline.json; not contemporaneous timings')
            assert value['memory'] == asdict(MEMORY)
            assert value['outer']['reuse_initial_reference'] and value['outer']['joint_outer_gradients']
        original_dump(path, value)

    with patch.object(e11, 'MEMORY', MEMORY), \
         patch.object(e11.run, 'SequentialConfig3D', SequentialReuseConfig3D), \
         patch.object(e11.run, 'dump', emit):
        yield


def worker(task, out, data, *, device='cuda'):
    with execution():
        e11.worker(task, out, data, device=device)


def tasks(old_schedule):
    first = old_schedule['tasks'][0]['pair']
    rows = [dict(r) for r in old_schedule['tasks'] if r['pair'] == first]
    assert len(rows) == 6 and [r['index'] for r in rows] == list(range(6))
    assert [r['route'] for r in rows] == e11.run.ROUTES
    return rows


def prepare(out):
    status = e11.read(OLD/'status.json')
    assert status['status'] == 'interrupted', 'old campaign must remain stopped'
    rows = tasks(e11.read(OLD/'schedule.json'))
    out.mkdir(parents=True, exist_ok=False)
    data = out/'inputs'/rows[0]['pair']; data.mkdir(parents=True)
    hashes = {}
    for name in ('pair.npz', 'manifest.json'):
        source = OLD/'inputs'/rows[0]['pair']/name
        target = data/name; shutil.copy2(source, target)
        digest = file_sha256(source)
        assert file_sha256(target) == digest
        hashes[str(target.relative_to(out))] = digest
    manifest = e11.read(data/'manifest.json')
    assert hashes[str((data/'pair.npz').relative_to(out))] == manifest['pair_sha256']
    assert manifest['shape_zyx'] == [192, 224, 160] and manifest['spacing_xyz'] == [1., 1., 1.]
    assert manifest['unit'] == 'mm' and manifest['pair'] == rows[0]['pair']
    old_runs = [r for r in e11.read(OLD/'runs.json') if r['pair'] == rows[0]['pair']]
    assert len(old_runs) == 6 and sum(r['process_status'] == 'completed' for r in old_runs) == 5
    baseline = dict(source=str(OLD), runs=old_runs, status=status,
                    timing_scope='historical only; different dates/background load, not controlled paired repeats',
                    missing_completed_baseline=['lm_weak_balanced'], no_resume=True)
    dump(out/'historical_baseline.json', baseline)
    hashes['historical_baseline.json'] = file_sha256(out/'historical_baseline.json')
    dump(out/'input_sha256.json', hashes)
    dump(out/'schedule.json', dict(tasks=rows, serial=True, device='cuda', inner_steps=e11.INNER_STEPS,
         outer_steps=e11.OUTER_STEPS, watchdog_seconds=e11.WATCHDOG, certificate=asdict(e11.CERTIFICATE),
         memory=asdict(MEMORY), outer=asdict(SequentialReuseConfig3D(max_iterations=1)),
         baseline_type='historical_E11_first_pair', implementation='E13_E14_explicit', created_utc=e11.now()))
    files = [p for d in ('jd_curl', 'scripts', 'tests', 'configs', 'docs') for p in (ROOT/d).rglob('*')
             if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc', '.pyo')]
    files += [ROOT/'README.md', ROOT/'pyproject.toml']
    source_hashes = {str(p.relative_to(ROOT)): file_sha256(p) for p in files}
    dump(out/'source_sha256.json', source_hashes)
    for p in files:
        target = out/'snapshot'/p.relative_to(ROOT); target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
        assert file_sha256(target) == source_hashes[str(p.relative_to(ROOT))]
    dump(out/'status.json', dict(status='prepared_not_started', scheduled=6, completed=0,
                                 failed=0, timed_out=0, utc=e11.now()))
    print('Prepared six new frozen tasks; E11 remains stopped:', out, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEST)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--prepare', action='store_true')
    actions.add_argument('--campaign', action='store_true')
    actions.add_argument('--worker', type=int)
    args = parser.parse_args(); out = args.output.resolve()
    if args.prepare:
        return prepare(out)
    schedule = e11.read(out/'schedule.json')
    assert len(schedule['tasks']) == 6 and schedule['memory'] == asdict(MEMORY)
    assert schedule['outer'] == asdict(SequentialReuseConfig3D(max_iterations=1))
    # Parent and every child must import only this campaign's frozen source.
    import jd_curl
    assert ROOT == (out/'snapshot').resolve()
    assert Path(jd_curl.__file__).resolve().is_relative_to(ROOT)
    if args.campaign:
        return e11.campaign(out, worker_script=Path(__file__).resolve())
    task = schedule['tasks'][args.worker]; assert task['index'] == args.worker
    data = out/'inputs'/task['pair']
    assert file_sha256(data/'pair.npz') == e11.read(data/'manifest.json')['pair_sha256']
    return worker(task, out/task['name'], data)


if __name__ == '__main__':
    sys.exit(main())
