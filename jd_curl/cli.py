"""Auditable 2D runs. Existing output directories are never reused."""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
from time import perf_counter

from .diagnostics import high_frequency_diagnostics
from .experiments.compare_solvers import _iter_solutions
from .experiments.synthetic import make_synthetic_case
from .io import load_case_npz, load_solver_config, result_summary, save_case_npz, save_result, write_json
from .provenance import file_provenance, runtime_metadata, source_snapshot
from .validation import InputValidationError


def _comparison_report(rows: list[dict]) -> str:
    def number(value):
        return "—" if value is None else f"{value:.9g}"
    lines = ["# 2D registration comparison", "",
        "Same case, initial displacement, fixed controls, weights and stopping tolerances. "
        "Costs include rejected attempts; timing is a single serial run, not a speed ranking.", "",
        "| Solver | Status | E initial | E final | G final | Threshold | Accepted | Rejected attempts | CG | Seconds |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        s = row["summary"]
        if s["initial"] is None:
            lines.append(f"| {row['solver']} | {s['status']} | — | — | — | — | — | — | — | — |")
        else:
            counts = s["counts"]
            lines.append(f"| {row['solver']} | {s['status']} | {s['initial']['energy']['total']:.9g} | "
                f"{s['final']['energy']['total']:.9g} | {number(s['final']['G'])} | {number(s['gradient_threshold'])} | "
                f"{counts['n_accepted']} | {counts['n_rejected_attempts']} | {counts['total_cg_iterations']} | {s['wall_seconds']:.6g} |")
    lines += ["", "Only `converged_gradient` is success. See each solution's summary.json for energy terms, "
        "full costs and node-JD diagnostics; history.jsonl retains all attempts. "
        "Exit code 0 certifies solver stopping only, not the user's 70% real-pair MSE gate. "
        "Positive node JD is not a cell-wise topology guarantee. A supplied manufactured displacement "
        "is not necessarily the minimizer because proximity uses a zero reference.", ""]
    if any("registration_metrics" in row for row in rows):
        lines += ["## MRI diagnostics (not additional objective terms)", "",
            "| Solver | Image MSE initial | Image MSE final | MSE ratio (lower is better; <=0.30) | MSE target passed | MSE reduction | Cell min JD | Nonpositive cells | Displacement RMS initial | Displacement RMS final |",
            "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|"]
        for row in rows:
            metrics = row["registration_metrics"]
            if not metrics["available"]:
                lines.append(f"| {row['solver']} | — | — | — | False (unavailable) | — | — | — | — | — |")
                continue
            error = metrics["displacement_error"]
            initial = None if error is None else error["initial"]["rms"]
            final = None if error is None else error["final"]["rms"]
            cell = metrics["bilinear_cell_geometry"]["final"]
            lines.append(f"| {row['solver']} | {number(metrics['image_mse']['initial'])} | "
                f"{number(metrics['image_mse']['final'])} | {number(metrics['mse_ratio'])} | {metrics['mse_target_passed']} | "
                f"{number(metrics['mse_reduction_fraction'])} | "
                f"{number(cell['min_corner_jd'])} | {number(cell['nonpositive_cell_fraction'])} | {number(initial)} | {number(final)} |")
        lines += ["", "Displacement errors use mm and supplied manufactured truth only. The known-warp case "
            "supplies nodal JD/curl truth controls, oracle only for the nodal objective; the real pair has no displacement ground truth. "
            "The image gate is mse_ratio = MSE_final / MSE_initial <= 0.30 (lower is better), "
            "equivalent to at least 70% MSE reduction, NOT mse_ratio <= 0.70. "
            "Its denominator is this solve's initial state; outer/multiscale application gates must instead use "
            "their fixed whole-run native-grid initial reference. Undefined ratios do not pass. "
            "MSE target passed is an image-only gate, not solver convergence or geometry acceptance. Cell determinants "
            "refer to the bilinear interpolant, distinct from the saved nodal central-difference JD; "
            "the Q1 objective uses cell Gauss geometry. Intensity MSE "
            "is not anatomical accuracy. See registration_metrics.json and case metadata.", ""]
    return "\n".join(lines)


def _frequency_report(rows: list[dict]) -> str:
    lines = ["# Fixed-base frequency / grid refinement diagnostic", "",
        "Identity geometry, constant images; not nonlinear registration or a multiscale solver. "
        "Qgeom uses weighted JD/curl JVP blocks only. QL = omega * v.T L v. "
        "PCG solves (H_GN + omega*gamma*I)p = A v with exact solution v; gamma=0.001.", "",
        "| n | Mode | Qgeom / QL | PCG valid | Reason | CG iterations | True relative residual | Solution relative error |",
        "|---:|---|---:|---|---|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['n']} | {r['mode']} | {r['ratio']:.9g} | {r['linear_valid']} | "
            f"{r['linear_reason']} | {r['cg_iterations']} | {r['true_relative_residual']} | {r['solution_relative_error']} |")
    return "\n".join(lines)+"\n"


def _copy_input(source: Path, target: Path, run: dict) -> None:
    source = Path(source)
    shutil.copyfile(source, target)
    entry = file_provenance(source)
    copied = file_provenance(target)
    if entry["sha256"] != copied["sha256"]:
        raise InputValidationError(f"Input changed while copying: {source}")
    entry["copy"] = str(target.relative_to(Path(run["output"])))
    run["inputs"].append(entry)


def _execute(args, output: Path, run: dict) -> int:
    if args.command == "prepare-mri":
        from .nifti import load_nifti_slice
        from .experiments.mri import make_mri_case
        from importlib.metadata import version
        if (args.mode == "pair") != (args.fixed is not None):
            raise InputValidationError("prepare-mri: --fixed required only for mode=pair")
        run["environment"]["versions"]["nibabel"] = version("nibabel")
        inputs = output/"inputs"
        inputs.mkdir()
        slices = {}
        for role, path in (("moving", args.moving), ("fixed", args.fixed)):
            if path is None:
                continue
            suffix = ".nii.gz" if path.name.endswith(".nii.gz") else ".nii" if path.suffix == ".nii" else None
            if suffix is None:
                raise InputValidationError("prepare-mri: expected .nii or .nii.gz")
            target = inputs/(role+suffix)
            _copy_input(path, target, run)
            slices[role] = load_nifti_slice(target)
        case = make_mri_case(**slices)
        metadata = dict(case.metadata)
        metadata["provenance"] = dict(metadata["provenance"], original_inputs=run["inputs"])
        case = replace(case, metadata=metadata)
        save_case_npz(case, output/"case.npz")
        run["generated"] = [file_provenance(output/name) for name in ("case.npz", "case.json")]
        run["case_metadata"] = case.metadata
        return 0
    if args.command == "make-synthetic":
        case = make_synthetic_case(nx=args.nx, ny=args.ny, identity=args.identity)
        save_case_npz(case, output/"case.npz")
        run["generated"] = [file_provenance(output/name) for name in ("case.npz", "case.json")]
        run["case_metadata"] = case.metadata
        return 0
    if args.command == "diagnose":
        rows = high_frequency_diagnostics(tuple(args.sizes))
        write_json(output/"diagnostics.json", rows)
        (output/"report.md").write_text(_frequency_report(rows))
        return 0 if all(r["linear_valid"] for r in rows) else 2

    inputs = output/"inputs"
    inputs.mkdir()
    _copy_input(args.case, inputs/"case.npz", run)
    _copy_input(args.case.with_suffix(".json"), inputs/"case.json", run)
    paths = args.configs if args.command == "compare" else [args.config]
    specifications = []
    for i, path in enumerate(paths):
        target = inputs/f"config_{i:02d}.json"
        _copy_input(path, target, run)
        specifications.append(load_solver_config(target))
    if any(w != specifications[0][0] for w, _ in specifications):
        raise InputValidationError("comparison: weights must match in every config")
    configs = tuple(c for _, c in specifications)
    if any(c.stop != configs[0].stop for c in configs):
        raise InputValidationError("comparison: StopConfig must match in every config")
    case = load_case_npz(inputs/"case.npz")
    weights = specifications[0][0]
    rows = []
    for i, result in enumerate(_iter_solutions(case, weights, configs)):
        solver = result.resolved_config["solver"]
        directory = f"{i:02d}_{solver}"
        metadata = dict(weights=asdict(weights), case_metadata=case.metadata,
            environment=run["environment"], input_provenance=run["inputs"],
            code_identity="../source.tar.gz + ../sha256.json")
        save_result(result, output/directory, metadata)
        rows.append(dict(solver=solver, directory=directory, summary=result_summary(result)))
        if case.metadata["data_kind"] in ("mri_pair", "mri_known_warp"):
            from .experiments.metrics import registration_metrics
            metrics = registration_metrics(case, result)
            rows[-1]["registration_metrics"] = metrics
            write_json(output/directory/"registration_metrics.json", metrics)
        # Preserve each completed solve, even if a later solve raises a programming error.
        write_json(output/"comparison.json", rows)
        (output/"report.md").write_text(_comparison_report(rows))
        run["completed_solvers"] = len(rows)
        write_json(output/"run.json", run)
    return 0 if all(row["summary"]["success"] for row in rows) else 2


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("make-synthetic")
    make.add_argument("--nx", type=int, default=33)
    make.add_argument("--ny", type=int, default=29)
    make.add_argument("--identity", action="store_true")
    mri = commands.add_parser("prepare-mri")
    mri.add_argument("--mode", choices=("pair", "known-warp"), required=True)
    mri.add_argument("--moving", type=Path, required=True)
    mri.add_argument("--fixed", type=Path)
    solve = commands.add_parser("solve")
    solve.add_argument("--case", type=Path, required=True)
    solve.add_argument("--config", type=Path, required=True)
    compare = commands.add_parser("compare")
    compare.add_argument("--case", type=Path, required=True)
    compare.add_argument("--configs", type=Path, nargs="+", required=True)
    diagnose = commands.add_parser("diagnose")
    diagnose.add_argument("--sizes", type=int, nargs="+", default=[17, 33, 65, 129])
    for subparser in (make, mri, solve, compare, diagnose):
        subparser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=False)  # Outside exception handler: never touch an old run.
    start = perf_counter()
    run = dict(command=[sys.executable, "-B", "-m", "jd_curl.cli", *argv],
        cwd=str(Path.cwd()), output=str(output), status="running", inputs=[], completed_solvers=0)
    write_json(output/"run.json", run)
    try:
        run["environment"] = runtime_metadata()
        source_snapshot(output)
        write_json(output/"run.json", run)
        code = _execute(args, output, run)
        run.update(status="completed" if code == 0 else "numerical_nonconvergence", exit_code=code)
        print(f"{run['status']}: {output}")
        return code
    except BaseException as exc:
        run.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                   exception_type=type(exc).__name__, exception_message=str(exc))
        raise
    finally:
        run.update(wall_seconds=perf_counter()-start, finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(output/"run.json", run)


if __name__ == "__main__":
    raise SystemExit(main())
