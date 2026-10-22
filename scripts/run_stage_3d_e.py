"""E2 registered OASIS small-volume CPU/CUDA prefixes and frozen-state profiling."""
import argparse
import cProfile
from dataclasses import asdict
import json
import os
from pathlib import Path
import pstats
import resource
import shutil
import subprocess
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_stage_3d_c import dump
from jd_curl.three_d.nifti import file_sha256
from jd_curl.three_d.grid import make_grid_3d
from jd_curl.three_d.boundary import restrict_free
from jd_curl.three_d.problem import make_problem, Weights3D, MemoryConfig, make_self_reference, with_controls
from jd_curl.three_d.feasibility import CertificateConfig, certify_map
from jd_curl.three_d.residuals import evaluate_trial, promote_trial, AcceptedState3D
from jd_curl.three_d.interpolation import sample_value_gradient
from jd_curl.three_d.linearization import build_linearization, energy_gradient, barrier_ggn_hvp
from jd_curl.three_d.poisson import build_poisson_inverse
from jd_curl.three_d.energy_comparison import stable_energy_reduction
from jd_curl.three_d.sequential import solve_sequential, SequentialConfig3D, mse
from jd_curl.three_d.solvers.poisson_gradient import PGConfig3D
from jd_curl.three_d.solvers.lm import LMConfig3D

DATA = ROOT/'data/three_d/stage_E1_v1'
THREADS = ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
ROUTES = [f'{solver}_{strength}_{variant}' for solver,variant in
          (('pg','baseline'),('lm','baseline'),('lm','balanced')) for strength in ('strong','weak')]


def inputs():
    manifest = json.loads((DATA/'manifest.json').read_text())
    assert file_sha256(DATA/'pair.npz') == manifest['pair_sha256']
    with np.load(DATA/'pair.npz') as archive:
        arrays = {k:archive[k] for k in archive.files}
    nz,ny,nx = manifest['shape_zyx']; hx,hy,hz = manifest['spacing_xyz']; x0,y0,z0=manifest['origin_xyz']
    grid = make_grid_3d(nx=nx,ny=ny,nz=nz,hx=hx,hy=hy,hz=hz,x0=x0,y0=y0,z0=z0,unit='mm')
    return arrays,grid,manifest


def problem_for(strength, *, certificate_config=None):
    arrays,grid,manifest = inputs()
    mu = 1. if strength=='strong' else .001
    p = make_problem(grid=grid,moving_image=arrays['moving'],fixed_image=arrays['fixed'],
        u_ref=arrays['initial_u'],weights=Weights3D(manifest['weights']['lam'],mu,mu,.001),
        memory=MemoryConfig(cell_tile_size=512,spline_query_chunk=4096),
        certificate_config=CertificateConfig(tile_size=512) if certificate_config is None else certificate_config)
    return p,arrays,manifest


def ready(p,u):
    state = promote_trial(p,evaluate_trial(p,restrict_free(u,p.grid)))
    if not isinstance(state,AcceptedState3D):
        raise RuntimeError('state did not pass full CPU certification: '+str(state))
    return state


def worker(route,device,out, *, certificate_config=None):
    started=perf_counter()
    out.mkdir(parents=True,exist_ok=False)
    backend=torch=None
    device_init=0.
    if device=='cuda':
        tick=perf_counter()
        import torch
        from jd_curl.three_d.backends.hybrid import HybridBackend
        backend=HybridBackend('cuda')
        backend.device.synchronize()
        torch.cuda.reset_peak_memory_stats()
        device_init=perf_counter()-tick
    solver,strength,variant=route.split('_')
    problem,arrays,manifest=problem_for(strength,certificate_config=certificate_config)
    inner=PGConfig3D() if solver=='pg' else LMConfig3D(preconditioner='ggn_diagonal_poisson' if variant=='balanced' else 'poisson')
    outer=SequentialConfig3D(max_iterations=3,eta0=.01)
    # Independent native baseline exported BEFORE the timed solve, including for timeout.
    initial=ready(problem,arrays['initial_u'])
    native=mse(initial,problem.grid)
    from jd_curl.three_d.solvers.common import gradient_norm_mass
    GF0=gradient_norm_mass(energy_gradient(build_linearization(problem,initial),term='outer'),problem.grid.omega)
    np.savez_compressed(out/'initial.npz',u=initial.u,warped=initial.warped)
    dump(out/'initial.json',dict(native_mse=native,ratio=1.,GF=GF0,GF_threshold=1e-8+1e-6*GF0,
        certificate=asdict(initial.certificate),path='initial.npz',sha256=file_sha256(out/'initial.npz')))
    config=dict(route=route,device=device,weights=asdict(problem.weights),inner=asdict(inner),outer=asdict(outer),
        memory=asdict(problem.memory),certificate=asdict(problem.certificate_config),
        grid_id=problem.grid.grid_id,input_sha256=manifest['pair_sha256'],threads={k:os.environ.get(k) for k in THREADS},
        watchdog_seconds=120,time_windows_seconds=[5,15,30,60],labels_not_in_loss=True)
    dump(out/'config.json',config)
    construction=perf_counter()-started
    rows=[]; commits=[]; io_seconds=0.
    best_ratio=1.; best_index=0
    solve_t0=perf_counter()
    def observe(p,start,inner_result,row,candidate):
        nonlocal io_seconds,best_ratio,best_index
        tick=perf_counter()
        rows.append(row)
        if row['outer_accepted']:
            index=len(commits)+1
            name=f'committed_{index:03d}.npz'
            np.savez_compressed(out/name,u=candidate.u,warped=candidate.warped,before_u=start.u)
            ratio=mse(candidate,p.grid)/native
            if ratio < best_ratio:
                best_ratio,best_index=ratio,index
            entry=dict(index=index,path=name,sha256=file_sha256(out/name),ratio=ratio,best_ratio=best_ratio,best_index=best_index,
                       solver_commit_seconds=row['elapsed_seconds'],GF=row['GF_candidate'],
                       min_jd_lower_bound=candidate.certificate.min_jd_lower_bound,
                       visible_commit_seconds=perf_counter()-solve_t0)
            commits.append(entry)
        dump(out/'history.json',rows)
        # Atomic commit index only references fully written field files.
        dump(out/'commits.json',commits)
        io_seconds+=perf_counter()-tick
    result=solve_sequential(problem,initial.theta,solver=solver,inner_config=inner,config=outer,observer=observe,backend=backend)
    if backend: backend.device.synchronize()
    elapsed=perf_counter()-solve_t0
    dump(out/'history.json',list(result.history)); dump(out/'commits.json',commits)
    summary=dict(route=route,device=device,status=result.status,success=result.success,reason=result.reason,
        native_mse=result.native_mse,final_mse=result.final_mse,ratio=result.mse_ratio,best_ratio=result.best_mse_ratio,
        GF=result.GF_final,GF0=result.GF_initial,GF_threshold=result.GF_threshold,GE=result.GE_final,GD=result.GD_final,
        energy=asdict(result.final_state.energy),certificate=asdict(result.final_state.certificate),counts=result.counts,
        commits=len(commits),solver_seconds=result.elapsed_seconds,experiment_solver_seconds=elapsed,
        observer_io_seconds=io_seconds,construction_seconds=construction,gpu_initialization_seconds=device_init,
        worker_seconds=perf_counter()-started,peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        device_peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch else None,
        device_peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch else None,
        device_path_seconds=backend.timings if backend else None,
        cpu_controller_remainder_seconds=result.elapsed_seconds-sum(backend.timings.values()) if backend else None,
        cpu_inner_gradient_rechecks=backend.cpu_gradient_checks if backend else 0,
        transfers=asdict(backend.device.transfers) if backend else None,resolved_config=result.resolved_config)
    dump(out/'summary.json',summary)
    print(json.dumps(summary,allow_nan=False),flush=True)


def profile(out):
    """Single native MRI state, declared deformation probe, no optimized candidate selection."""
    import torch
    from jd_curl.three_d.backends.hybrid import HybridBackend
    p,arrays,_=problem_for('strong')
    base=ready(p,arrays['initial_u']); lin=build_linearization(p,base)
    backend=HybridBackend('cuda')
    dl=backend.linearize(lin)
    g=p.grid
    z,y,x=np.meshgrid(*(np.linspace(0,1,n) for n in g.shape),indexing='ij')
    u=(.1*np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z))[...,None]*np.array([1.,-.6,.8])
    u[[0,-1],:,:]=u[:,[0,-1],:]=u[:,:,[0,-1]]=0
    trial=ready(p,u)
    rng=np.random.default_rng(321); v=rng.normal(size=base.theta.size)
    tv=backend.device.upload(v)
    inverse=build_poisson_inverse(g,a=p.weights.lam,b=(p.weights.mu+p.weights.nu)/2,metric='q1')
    dinverse=dl.model.poisson(a=p.weights.lam,b=(p.weights.mu+p.weights.nu)/2,metric='q1')
    points=g.identity()+u; tpoints=backend.device.upload(points)
    cpu=dict(spline=lambda:sample_value_gradient(p.sampler,points,chunk_size=4096),
        certification=lambda:certify_map(u,g,config=p.certificate_config),
        trial=lambda:evaluate_trial(p,trial.theta),gradient=lambda:energy_gradient(lin),
        hvp=lambda:barrier_ggn_hvp(lin,v),poisson=lambda:inverse.apply(v),
        stable_difference=lambda:stable_energy_reduction(p,base,trial))
    gpu=dict(spline=lambda:dl.model.spline.sample(tpoints,chunk_size=4096,strict=dl.model._strict),
        gradient=lambda:dl.gradient(),hvp=lambda:dl.hvp(tv),poisson=lambda:dinverse.apply(tv))
    timings={}
    for name,cpu_fn in cpu.items():
        cpu_fn()
        if name in gpu: gpu[name]()
        runs={'cpu':[],'cuda':[]}
        for _ in range(3):
            for device,fn in [('cpu',cpu_fn)]+([('cuda',gpu[name])] if name in gpu else []):
                backend.device.synchronize(); tick=perf_counter(); fn(); backend.device.synchronize()
                runs[device].append(perf_counter()-tick)
        timings[name]=dict(seconds=runs,median_seconds={k:float(np.median(v)) for k,v in runs.items() if v})
    def errors(a,b):
        value=float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-12))
        assert value < 2e-10
        return value
    equivalence={name:errors(backend.device.download(gpu[name]()),cpu[name]()) for name in ('gradient','hvp','poisson')}
    val,grad=gpu['spline'](); expected=cpu['spline']()
    equivalence['spline_value']=errors(backend.device.download(val),expected.value)
    equivalence['spline_gradient']=errors(backend.device.download(grad),expected.gradient)
    pr=cProfile.Profile();pr.enable()
    for _ in range(3):
        for fn in cpu.values(): fn()
    pr.disable();pr.dump_stats(str(out/'cpu_frozen_profile.pstats'))
    with (out/'cpu_frozen_profile.txt').open('w') as stream:
        pstats.Stats(pr,stream=stream).strip_dirs().sort_stats('cumulative').print_stats(45)
    dump(out/'profile.json',dict(timings=timings,equivalence=equivalence,
        scope='fixed MRI native state + prespecified 0.1 mm deformation probe; not a solve or representative of later boundary states',
        profiler_excluded_from_main_times=True,device=str(backend.device.device),torch_version=torch.__version__,cuda_version=torch.version.cuda,
        device_name=torch.cuda.get_device_name(),shape_zyx=g.shape))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--worker',choices=ROUTES)
    parser.add_argument('--device',choices=['cpu','cuda'])
    parser.add_argument('--profile',action='store_true')
    args=parser.parse_args();out=Path(args.output).resolve()
    if args.worker: return worker(args.worker,args.device,out)
    if args.profile: return profile(out)
    out.mkdir(parents=True,exist_ok=False)
    files=[p for folder in ('jd_curl','tests','scripts','configs') for p in (ROOT/folder).rglob('*')
           if p.is_file() and '__pycache__' not in p.parts]
    files += list((ROOT/'docs/stage_3d_E').rglob('*.md'))
    hashes={str(p.relative_to(ROOT)):file_sha256(p) for p in sorted(files)}
    dump(out/'source_sha256.json',hashes)
    for p in files:
        target=out/'snapshot'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    shutil.copyfile(DATA/'manifest.json',out/'input_manifest.json')
    dump(out/'schedule.json',dict(routes=ROUTES,devices=['cpu','cuda'],watchdog_seconds=120,serial=True))
    env=os.environ.copy();env.update({k:'1' for k in THREADS});env['PYTHONDONTWRITEBYTECODE']='1'
    records=[]
    for route in ROUTES:
        for device in ('cpu','cuda'):
            name=route+'_'+device; tick=perf_counter()
            command=[sys.executable,'-B',__file__,'--worker',route,'--device',device,'--output',str(out/name)]
            with (out/(name+'.stdout')).open('w') as stdout,(out/(name+'.stderr')).open('w') as stderr:
                try:
                    process=subprocess.run(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,timeout=120)
                    status='completed' if process.returncode==0 else 'process_failed'
                    returncode=process.returncode
                except subprocess.TimeoutExpired:
                    status,returncode='timeout',None
            folder=out/name
            row=dict(name=name,route=route,device=device,process_status=status,returncode=returncode,parent_wall_seconds=perf_counter()-tick,
                summary=json.loads((folder/'summary.json').read_text()) if (folder/'summary.json').exists() else None,
                commits=json.loads((folder/'commits.json').read_text()) if (folder/'commits.json').exists() else [])
            records.append(row);dump(out/'runs.json',records)
            print(name,status,round(row['parent_wall_seconds'],3),'ratio',row['summary']['ratio'] if row['summary'] else 'checkpoint only',flush=True)
    with (out/'profile.stdout').open('w') as stdout,(out/'profile.stderr').open('w') as stderr:
        proc=subprocess.run([sys.executable,'-B',__file__,'--profile','--output',str(out)],cwd=ROOT,env=env,stdout=stdout,stderr=stderr,timeout=120)
    stable=all(file_sha256(ROOT/p)==h for p,h in hashes.items())
    dump(out/'verification.json',dict(sources_unchanged=stable,profile_returncode=proc.returncode,
        completed=sum(r['process_status']=='completed' for r in records),timeout=sum(r['process_status']=='timeout' for r in records)))
    assert stable and proc.returncode==0


if __name__=='__main__':
    main()
