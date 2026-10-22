"""Registered 3D-C synthetic CPU matrix: isolated serial workers, no MRI/GPU."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jd_curl.three_d.grid import make_grid_3d
from jd_curl.three_d.interpolation import build_image_sampler, sample_value_gradient
from jd_curl.three_d.feasibility import certify_map
from jd_curl.three_d.problem import make_problem, Weights3D, MemoryConfig
from jd_curl.three_d.sequential import solve_sequential, SequentialConfig3D, mse
from jd_curl.three_d.solvers.poisson_gradient import PGConfig3D
from jd_curl.three_d.solvers.lm import LMConfig3D


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False,
                               default=lambda obj: obj.item() if isinstance(obj, np.generic) else str(obj))+"\n")
    temp.replace(path)


def grid():
    return make_grid_3d(nx=7, ny=6, nz=5, hx=.2, hy=.25, hz=.3, x0=-.3, y0=.2, z0=-.7, unit="mm")


def manufacture(out):
    g = grid()
    points = g.identity()
    X = [(points[..., j]-a[0])/(a[-1]-a[0]) for j, a in enumerate(g.axes)]
    x, y, z = X
    moving = (.8*np.exp(-((x-.35)**2/.05+(y-.46)**2/.08+(z-.55)**2/.06))
              +.5*np.exp(-((x-.72)**2/.03+(y-.63)**2/.04+(z-.32)**2/.07))
              +.3*np.exp(-((x-.48)**2/.09+(y-.22)**2/.03+(z-.72)**2/.04)))
    envelope = np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z)
    scales = np.random.default_rng(17).uniform(.8, 1.2, 3)
    u = envelope[..., None]*np.stack((.035*np.cos(np.pi*y), -.025*np.cos(np.pi*z), .03*np.cos(np.pi*x)), axis=-1)*scales
    u[[0, -1], :, :] = u[:, [0, -1], :] = u[:, :, [0, -1]] = 0.
    certificate = certify_map(u, g)
    if certificate.status != "certified_feasible":
        raise RuntimeError("predeclared manufactured map was not certified; stop rather than tune data")
    sampled = sample_value_gradient(build_image_sampler(moving, g), points+u)
    fixed = sampled.value+.001*np.cos(2*np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z)
    path = out / "synthetic_input.npz"
    np.savez_compressed(path, moving=moving, fixed=fixed, manufactured_u=u, initial_u=np.zeros_like(u))
    dump(out / "input_manifest.json", dict(path=path.name, sha256=sha(path), shape_zyx=g.shape,
          spacing_xyz=g.spacing, origin_xyz=[float(a[0]) for a in g.axes], unit=g.unit,
          seed=17, intensity_reference=1., dataset="manufactured 3D-C development case, not MRI",
          formula_source="scripts/run_stage_3d_c.py:manufacture", no_additional_normalization=True,
          manufactured_map_certificate=asdict(certificate)))


def configuration(name):
    solver, strength, variant = name.split("_")
    if solver not in ("pg", "lm") or strength not in ("strong", "weak") or variant not in ("baseline", "candidate", "identity"):
        raise ValueError("unknown registered run")
    eta = .3 if solver == "pg" and variant == "candidate" else .01
    inner = (PGConfig3D() if solver == "pg" else
             LMConfig3D(preconditioner="ggn_diagonal_poisson" if variant == "candidate" else "poisson"))
    return solver, strength, variant, inner, SequentialConfig3D(max_iterations=3, eta0=eta)


def worker(name, inputs, out):
    started = perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    solver, strength, variant, inner, outer = configuration(name)
    g = grid()
    with np.load(inputs) as data:
        moving, fixed = data["moving"], data["fixed"]
        initial_u = data["initial_u"]
    if variant == "identity":
        fixed = moving
    L = float(np.prod([a[-1]-a[0] for a in g.axes]))**(1/3)
    geometry_weight = 1. if strength == "strong" else .001
    weights = Weights3D(.1/L**2, geometry_weight, geometry_weight, .001)
    problem = make_problem(grid=g, fixed_image=fixed, moving_image=moving, u_ref=initial_u,
                           weights=weights, memory=MemoryConfig(cell_tile_size=64, spline_query_chunk=128))
    construction = perf_counter()-started
    dump(out / "resolved_config.json", dict(run=name, solver=solver, weights=asdict(weights),
         inner=asdict(inner), outer=asdict(outer), memory=asdict(problem.memory),
         certificate=asdict(problem.certificate_config), input_sha256=sha(inputs), grid_id=g.grid_id,
         identity_control=variant == "identity", timeout_seconds=120,
         threads={k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}))
    history = []
    io_seconds = 0.
    best = None
    best_commit_seconds = 0.
    native_mse = None

    def observe(p, start, inner_result, row, candidate):
        nonlocal io_seconds, best, best_commit_seconds, native_mse
        io_start = perf_counter()
        if native_mse is None:
            native_mse = mse(start, g)
        history.append(row)
        dump(out / "history.json", history)
        if row["outer_accepted"]:
            if best is None:
                best = start
            if candidate.energy.image < best.energy.image:
                best = candidate
                best_commit_seconds = row["elapsed_seconds"]
            checkpoint = out / f"committed_{len(history):03d}.npz"
            np.savez_compressed(checkpoint, latest_u=candidate.u, best_u=best.u)
            dump(out / "checkpoint.json", dict(path=checkpoint.name, sha256=sha(checkpoint),
                 outer_i=row["outer_i"], latest_ratio=row["committed_mse_ratio"], GF=row["committed_GF"],
                 GF_initial=row["GF_initial"], GF_threshold=row["GF_threshold"],
                 native_mse=native_mse, best_mse_ratio=mse(best, g)/native_mse if native_mse else None,
                 best_commit_seconds=best_commit_seconds,
                 last_commit_seconds=row["elapsed_seconds"], min_jd_lower_bound=candidate.certificate.min_jd_lower_bound,
                 outer_accepted=row["counts_after"]["outer_accepted"],
                 note="latest exported committed state, never an inner trial"))
        io_seconds += perf_counter()-io_start

    result = solve_sequential(problem, np.zeros(3*g.n_free_scalar), solver=solver,
                               inner_config=inner, config=outer, observer=observe)
    io_start = perf_counter()
    state, best_state = result.final_state, result.best_state
    if state is not None:
        np.savez_compressed(out / "states.npz", initial_u=result.initial_state.u,
                            latest_u=state.u, best_u=best_state.u)
    dump(out / "history.json", list(result.history))
    io_seconds += perf_counter()-io_start
    summary = dict(run=name, status=result.status, success=result.success, reason=result.reason,
                   native_mse=result.native_mse, final_mse=result.final_mse, mse_ratio=result.mse_ratio,
                   best_mse_ratio=result.best_mse_ratio, GF_initial=result.GF_initial, GF=result.GF_final,
                   GF_threshold=result.GF_threshold, GD=result.GD_final, GE=result.GE_final,
                   D=state.energy.image if state else None, B=state.energy.barrier if state else None,
                   F=state.energy.outer if state else None, E=state.energy.total if state else None,
                   certificate=asdict(state.certificate) if state else None,
                   forcing_passed_rounds=sum(r["forcing_passed"] for r in result.history),
                   attempted_outer_rounds=len(result.history), counts=result.counts,
                   construction_seconds=construction, solver_seconds=result.elapsed_seconds,
                   io_seconds=io_seconds, worker_seconds=perf_counter()-started,
                   last_commit_seconds=result.last_commit_seconds, best_commit_seconds=result.best_commit_seconds,
                   committed_state_age_seconds=result.elapsed_seconds-result.last_commit_seconds,
                   peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   scope="single registered CPU manufactured case; not MRI/average registration speed",
                   timing_note="solver includes observer I/O; parent wall time includes interpreter/import startup")
    dump(out / "summary.json", summary)
    print(json.dumps(summary, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--worker")
    parser.add_argument("--input")
    args = parser.parse_args()
    out = Path(args.output).resolve()
    if args.worker:
        worker(args.worker, Path(args.input).resolve(), out)
        return
    out.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    manufacture(out)
    paths = [p for name in ("jd_curl", "scripts", "tests", "configs") for p in (ROOT/name).rglob("*")
             if p.is_file() and "__pycache__" not in p.parts]
    paths += [ROOT/name for name in ("docs/numerical_validation_3d_C.md", "docs/AGENTS.md",
              "docs/04_JD_Curl_3D_Poisson_LM_Formulas_ZH.md", "docs/05_JD_Curl_3D_Python_Architecture_ZH.md")]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}
    dump(out/"source_sha256.json", hashes)
    for p in paths:
        target = out/"snapshot"/p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
    env = os.environ.copy()
    env.update({k: "1" for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    schedule = [f"{solver}_{strength}_{variant}" for variant in ("identity", "baseline", "candidate")
                for solver in ("pg", "lm") for strength in ("strong", "weak")]
    dump(out/"schedule.json", dict(order=schedule, per_process_timeout_seconds=120, serial=True))
    rows = []
    for name in schedule:
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", name,
                   "--input", str(out/"synthetic_input.npz"), "--output", str(out/name)]
        start = perf_counter()
        with (out/(name+".stdout")).open("w") as stdout, (out/(name+".stderr")).open("w") as stderr:
            try:
                proc = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, timeout=120)
                status = "completed" if proc.returncode == 0 else "process_failed"
                returncode = proc.returncode
            except subprocess.TimeoutExpired:
                status, returncode = "timeout", None
        summary_path, checkpoint = out/name/"summary.json", out/name/"checkpoint.json"
        row = dict(run=name, process_status=status, returncode=returncode, parent_wall_seconds=perf_counter()-start,
                   summary=json.loads(summary_path.read_text()) if summary_path.exists() else None,
                   latest_exported_checkpoint=json.loads(checkpoint.read_text()) if checkpoint.exists() else None)
        rows.append(row)
        dump(out/"runs.json", rows)
        print(name, status, round(row["parent_wall_seconds"], 3),
              None if row["summary"] is None else row["summary"]["mse_ratio"], flush=True)
    stable = all(sha(ROOT/p) == value for p, value in hashes.items())
    dump(out/"run_verification.json", dict(inputs_unchanged_during_runs=stable, total_wall_seconds=perf_counter()-started,
                                           completed=sum(r["process_status"] == "completed" for r in rows),
                                           timeout=sum(r["process_status"] == "timeout" for r in rows)))
    dump(out/"manifest.json", {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file()})
    if not stable:
        raise SystemExit("Inputs changed during registered runs; do not call this frozen-protocol evidence")


if __name__ == "__main__":
    main()
