"""Frozen 5x6 native campaign; adapters only, original Sequential decisions."""
import argparse
from dataclasses import asdict,replace
from datetime import datetime,timezone
import json,os,shutil,signal,subprocess,sys
from pathlib import Path
from time import perf_counter
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_stage_3d_e as run
from scripts.run_stage_3d_c import dump
from jd_curl.three_d.feasibility import EnclosureCertificateConfig
from jd_curl.three_d.problem import make_problem,Weights3D,MemoryConfig
from jd_curl.three_d.nifti import file_sha256,warp_labels,dice_labels
from jd_curl.three_d.residuals import AcceptedState3D
from jd_curl.three_d.solvers import poisson_gradient as pg,lm
from jd_curl.types import InvalidTrial

CERTIFICATE=EnclosureCertificateConfig(delta=0.,tile_size=8192)
MEMORY=MemoryConfig(8192,65536)
WATCHDOG=7200
INNER_STEPS=5
OUTER_STEPS=1
DEST=ROOT/'results/mri_3d/stage_E11_short_v1'
DATA=ROOT/'data/three_d/stage_E11_native_v1'


def now():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(path.read_text()) if path.exists() else None


def tasks(selection):
    return [dict(index=i*len(run.ROUTES)+j,pair=pair['pair'],route=route,
                 name=pair['pair']+'__'+route) for i,pair in enumerate(selection['pairs']) for j,route in enumerate(run.ROUTES)]


def load_problem(data,strength,certificate_config):
    with patch.object(run,'DATA',data):arrays,grid,manifest=run.inputs()
    mu=1. if strength=='strong' else .001
    p=make_problem(grid=grid,fixed_image=arrays['fixed'],moving_image=arrays['moving'],u_ref=arrays['initial_u'],
        weights=Weights3D(manifest['weights']['lam'],mu,mu,.002),memory=MEMORY,
        certificate_config=certificate_config,barrier_location='closed27_uniform')
    return p,arrays,manifest


def label_evaluation(arrays,grid,u,label_sets=None):
    """label_sets: None = the OASIS sets seg4 (1..4) and seg35 (1..35), unchanged; D1 (2026-10-14): an explicit mapping
    name -> label ids from the pair manifest (e.g. LPBA40 'lpba56', 'brain'), keys fixed_<name>/moving_<name>."""
    result={}
    sets=[(kind,list(range(1,n+1))) for kind,n in [('seg4',4),('seg35',35)]] if label_sets is None else [(k,[int(i) for i in v]) for k,v in label_sets.items()]
    for kind,ids in sets:
        warped=warp_labels(arrays['moving_'+kind],u,grid)
        result[kind]=dice_labels(arrays['fixed_'+kind],warped,labels=ids)
    return result


def worker(task,out,data,*,device='cuda'):
    """Device argument is exposed for tiny CPU equivalence tests; CLI uses CUDA."""
    started=perf_counter();events=[];promoted=[];holder={};trials=0
    # Shared worker owns out.mkdir, so first progress is written by its factory.
    def event(stage,**kw):
        row=dict(stage=stage,utc=now(),elapsed_seconds=perf_counter()-started,**kw)
        events.append(row);dump(out/'progress.json',row);dump(out/'events.json',events)
    old_dump=run.dump
    def emit(path,value):
        if path.name=='config.json':
            value=dict(value,stage='E11_short',pair=task['pair'],watchdog_seconds=WATCHDOG,
                time_windows_seconds=[600,1800,3600,7200],barrier_location='closed27_uniform',
                labels_policy='seg4 primary; seg35 audit-only',termination_policy='original_full_GF_only',
                limited_budget_not_convergence=True,source_manifest='campaign source_sha256.json')
            assert value['inner']['max_iterations']==INNER_STEPS and value['outer']['max_iterations']==OUTER_STEPS
        old_dump(path,value)
        if path.name in ('initial.json','config.json','commits.json','summary.json'):
            event('saved_'+path.stem)
    def factory(strength,*,certificate_config=None):
        event('native_load_and_construction_started')
        assert certificate_config==CERTIFICATE
        result=load_problem(data,strength,certificate_config)
        holder['p'],holder['arrays'],holder['manifest']=result
        dump(out/'registered_protocol.json',dict(task=task,device=device,inner_steps=INNER_STEPS,outer_steps=OUTER_STEPS,
            watchdog_seconds=WATCHDOG,memory=asdict(MEMORY),certificate=asdict(CERTIFICATE),weights=asdict(result[0].weights),
            pair_sha256=result[2]['pair_sha256']))
        event('construction_completed_baseline_started');return result
    solver=pg if task['route'].startswith('pg_') else lm
    original_evaluate,original_promote=solver.evaluate_trial,solver.promote_trial
    def evaluate(*args,**kw):
        nonlocal trials
        trials+=1;event('inner_trial_started',trial=trials)
        result=original_evaluate(*args,**kw)
        event('inner_trial_evaluated',trial=trials,valid=not isinstance(result,InvalidTrial),
              reason=result.reason if isinstance(result,InvalidTrial) else 'CPU_CERTIFIED_ENERGY_READY')
        return result
    def promote(*args,**kw):
        event('inner_promotion_started',trial=trials)
        result=original_promote(*args,**kw)
        if isinstance(result,AcceptedState3D):
            name=f'inner_promoted_{len(promoted)+1:03d}_NOT_OUTER_COMMITTED.npz'
            temp=out/(name+'.partial.npz');np.savez_compressed(temp,u=result.u,warped=result.warped);temp.replace(out/name)
            promoted.append(dict(index=len(promoted)+1,path=name,sha256=file_sha256(out/name),
                role='inner accepted/promoted only, NOT an outer commit',eval_id=result.eval_id,
                certificate=asdict(result.certificate),energy=asdict(result.energy),trial=trials))
            dump(out/'inner_promoted.json',promoted)
        event('inner_promotion_completed',trial=trials,saved=len(promoted));return result
    PG,LM,SQ=run.PGConfig3D,run.LMConfig3D,run.SequentialConfig3D
    with patch.object(run,'problem_for',factory),patch.object(run,'dump',emit),\
         patch.object(run,'PGConfig3D',lambda **kw:replace(PG(**kw),max_iterations=INNER_STEPS)),\
         patch.object(run,'LMConfig3D',lambda **kw:replace(LM(**kw),max_iterations=INNER_STEPS)),\
         patch.object(run,'SequentialConfig3D',lambda **kw:replace(SQ(**kw),max_iterations=OUTER_STEPS)),\
         patch.object(solver,'evaluate_trial',evaluate),patch.object(solver,'promote_trial',promote):
        run.worker(task['route'],device,out,certificate_config=CERTIFICATE)
    event('native_label_evaluation_started')
    commits=read(out/'commits.json') or []
    if commits:
        commit=commits[-1];assert file_sha256(out/commit['path'])==commit['sha256']
        with np.load(out/commit['path']) as archive:u=archive['u']
        role='formal_outer_commit';metrics=label_evaluation(holder['arrays'],holder['p'].grid,u)
    else:
        role='initial_no_outer_commit';metrics=holder['manifest']['baseline_label_metrics']
    dump(out/'labels.json',dict(role=role,baseline=holder['manifest']['baseline_label_metrics'],final=metrics,
        primary='seg4',audit_only='seg35',labels_not_in_loss=True))
    event('worker_completed',commits=len(commits))
    dump(out/'worker_completion.json',dict(complete=True,worker_total_seconds=perf_counter()-started,commits=len(commits),
        metrics_role=role,device=device))


def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    selection=read(DATA/'selection.json');schedule=tasks(selection)
    assert len(schedule)==30 and len({t['name'] for t in schedule})==30
    (out/'inputs').mkdir();shutil.copy2(DATA/'selection.json',out/'inputs/selection.json')
    input_hashes={}
    for pair in selection['pairs']:
        dest=out/'inputs'/pair['pair'];dest.mkdir()
        for name in ('manifest.json','pair.npz'):
            source=DATA/pair['pair']/name;target=dest/name;shutil.copy2(source,target)
            assert file_sha256(source)==file_sha256(target)
            input_hashes[str(target.relative_to(out))]=file_sha256(target)
        manifest=read(dest/'manifest.json')
        assert file_sha256(dest/'pair.npz')==manifest['pair_sha256']
        assert manifest['shape_zyx']==[192,224,160] and manifest['spacing_xyz']==[1.,1.,1.]
        assert manifest['unit']=='mm' and manifest['pair']==pair['pair']
    dump(out/'input_sha256.json',input_hashes)
    dump(out/'schedule.json',dict(tasks=schedule,serial=True,device='cuda',inner_steps=INNER_STEPS,outer_steps=OUTER_STEPS,
        watchdog_seconds=WATCHDOG,certificate=asdict(CERTIFICATE),memory=asdict(MEMORY),created_utc=now()))
    files=[p for d in ['jd_curl','scripts','tests','configs','docs'] for p in (ROOT/d).rglob('*')
        if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc']+[ROOT/'README.md']
    hashes={str(p.relative_to(ROOT)):file_sha256(p) for p in files};dump(out/'source_sha256.json',hashes)
    for p in files:
        target=out/'snapshot'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
        assert file_sha256(target)==hashes[str(p.relative_to(ROOT))]
    dump(out/'status.json',dict(status='prepared_not_started',scheduled=30,completed=0,failed=0,timed_out=0,utc=now()))
    print('Prepared 30 frozen tasks:',out,flush=True)


def stop_child(proc):
    if proc is None or proc.poll() is not None:return
    try:os.killpg(proc.pid,signal.SIGTERM)
    except ProcessLookupError:
        proc.wait();return
    try:proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid,signal.SIGKILL);proc.wait()


def collect(task,folder,status,code,seconds):
    return dict(**task,process_status=status,returncode=code,parent_wall_seconds=seconds,
        summary=read(folder/'summary.json'),commits=read(folder/'commits.json') or [],
        labels=read(folder/'labels.json'),completion=read(folder/'worker_completion.json'),last_progress=read(folder/'progress.json'))


class CampaignResourceError(RuntimeError):
    """Execution must stop; never an optimization/convergence result."""


def campaign(out, *, worker_script=None, task_watchdog=None, resource_check=None):
    schedule=read(out/'schedule.json');assert schedule['inner_steps']==INNER_STEPS and schedule['outer_steps']==OUTER_STEPS
    assert not (out/'runs.json').exists(),'campaign already started; do not overwrite or silently resume'
    for name,digest in read(out/'source_sha256.json').items():assert file_sha256(ROOT/name)==digest,name
    for name,digest in read(out/'input_sha256.json').items():assert file_sha256(out/name)==digest,name
    import torch
    assert torch.cuda.is_available(),'CUDA required; no implicit CPU fallback'
    dump(out/'environment.json',dict(utc=now(),gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
        pid=os.getpid(),source_root=str(ROOT),device_count=torch.cuda.device_count(),threads={k:os.environ.get(k) for k in run.THREADS}))
    records=[];proc=None;current=None;task_started=False;started=perf_counter()
    def status(state):
        timed = {} if task_watchdog is None else dict(
            native_formal_commits=sum(c.get('native', False) for r in records for c in r['commits']),
            formal_commits_scope='all levels of finished tasks; native count separate')
        monitored = {} if resource_check is None else dict(
            resource_stopped=sum(r['process_status']=='resource_stopped' for r in records),
            remaining_not_started=len(schedule['tasks'])-len(records)-int(
                task_started and current is not None and len(records)<=current['index']))
        dump(out/'status.json',dict(status=state,pid=os.getpid(),utc=now(),elapsed_seconds=perf_counter()-started,
            scheduled=len(schedule['tasks']),finished=len(records),completed=sum(r['process_status']=='completed' for r in records),
            failed=sum(r['process_status']=='process_failed' for r in records),timed_out=sum(r['process_status']=='timeout' for r in records),
            formal_commits=sum(len(r['commits']) for r in records),current=current,worker_pid=proc.pid if proc and proc.poll() is None else None,**timed,**monitored))
    def interrupt(signum,frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    dump(out/'runs.json',records);status('running')
    try:
        for current in schedule['tasks']:
            task_started=False;proc=None
            if resource_check is not None:resource_check(current, None)
            folder=out/current['name'];tick=perf_counter()
            command=[sys.executable,'-B',str(Path(__file__).resolve() if worker_script is None else worker_script),'--worker',str(current['index']),'--output',str(out)]
            with (out/(current['name']+'.stdout')).open('w') as stdout,(out/(current['name']+'.stderr')).open('w') as stderr:
                execution = {} if task_watchdog is None else {
                    'env': dict(os.environ, JD_CURL_TASK_STARTED=str(tick))}
                limit = WATCHDOG if task_watchdog is None else task_watchdog(current)
                proc=subprocess.Popen(command,cwd=ROOT,stdout=stdout,stderr=stderr,start_new_session=True,**execution)
                task_started=True
                status('running');print(now(),'START',current['index']+1,current['name'],proc.pid,flush=True)
                try:
                    timeout = limit if task_watchdog is None else max(.001, limit-(perf_counter()-tick))
                    if resource_check is None:
                        code=proc.wait(timeout=timeout)
                    else:
                        end=perf_counter()+timeout
                        while True:
                            code=proc.poll()
                            if code is not None:break
                            resource_check(current, proc)
                            remaining=end-perf_counter()
                            if remaining <= 0:raise subprocess.TimeoutExpired(command,timeout)
                            try:
                                code=proc.wait(timeout=min(5.,remaining));break
                            except subprocess.TimeoutExpired:
                                if perf_counter() >= end:raise
                    state='completed' if code==0 else 'process_failed'
                except subprocess.TimeoutExpired:
                    stop_child(proc);code=proc.returncode;state='timeout'
            if resource_check is not None:
                failure=read(folder/'resource_failure.json')
                if failure is not None:raise CampaignResourceError(failure['reason'])
            record=collect(current,folder,state,code,perf_counter()-tick)
            if task_watchdog is not None:
                record['within_total_budget'] = state == 'completed' and record['parent_wall_seconds'] <= limit
            if state=='completed':assert record['completion'] and record['completion']['complete']
            records.append(record);dump(out/'runs.json',records);status('running')
            print(now(),'END',current['index']+1,state,record['parent_wall_seconds'],'commits',len(record['commits']),flush=True)
    except CampaignResourceError as error:
        stop_child(proc)
        if task_started and len(records)<=current['index']:
            record=collect(current,out/current['name'],'resource_stopped',proc.returncode,perf_counter()-tick)
            record['within_total_budget']=False
            records.append(record);dump(out/'runs.json',records)
        dump(out/'resource_failure.json',dict(reason=str(error),utc=now(),task=current,
            task_started=task_started,automatic_restart=False))
        status('resource_stopped');return 75
    except KeyboardInterrupt:
        stop_child(proc)
        if task_started and current is not None and len(records)<=current['index']:
            records.append(collect(current,out/current['name'],'interrupted',proc.returncode if proc else None,perf_counter()-tick))
            dump(out/'runs.json',records)
        status('interrupted');return 130
    finally:stop_child(proc)
    for name,digest in read(out/'source_sha256.json').items():assert file_sha256(ROOT/name)==digest,name
    for name,digest in read(out/'input_sha256.json').items():assert file_sha256(out/name)==digest,name
    current=None;status('finished_all_tasks')
    dump(out/'completion.json',dict(all_tasks_attempted=len(records)==len(schedule['tasks']),sources_unchanged=True,inputs_unchanged=True,
        independent_result_audit_pending=True,finished_utc=now()))
    return 0


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=DEST)
    actions=parser.add_mutually_exclusive_group(required=True);actions.add_argument('--prepare',action='store_true')
    actions.add_argument('--campaign',action='store_true');actions.add_argument('--worker',type=int)
    a=parser.parse_args();out=a.output.resolve()
    if a.prepare:return prepare(out)
    if a.campaign:return campaign(out)
    task=read(out/'schedule.json')['tasks'][a.worker];assert task['index']==a.worker
    # This is the frozen snapshot package, not subsequently edited working code.
    import jd_curl
    assert Path(jd_curl.__file__).resolve().is_relative_to(ROOT)
    data=out/'inputs'/task['pair'];assert file_sha256(data/'pair.npz')==read(data/'manifest.json')['pair_sha256']
    worker(task,out/task['name'],data)

if __name__=='__main__':sys.exit(main())
