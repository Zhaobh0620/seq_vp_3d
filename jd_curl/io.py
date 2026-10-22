"""Explicit NPZ/JSON contracts and result export; no hidden preprocessing."""

from dataclasses import asdict, fields
import json
from pathlib import Path

import numpy as np

from .grid import make_grid_2d
from .solvers import SolveOutcome, SolverConfig
from .solvers.common import InitializationFailure, LMConfig, PoissonGradientConfig, StopConfig, geometry_diagnostics
from .jacobian_barrier import JacobianBarrierConfig, JacobianFeasibilityConfig
from .types import Controls2D, RegistrationCase2D, Weights
from .validation import InputValidationError, as_float64_array


def write_json(path: Path, value) -> None:
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n")


def read_json(path: Path) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise InputValidationError(f"JSON: duplicate key {key!r}")
            result[key] = value
        return result
    def bad_constant(value):
        raise InputValidationError(f"JSON: nonfinite constant {value}")
    try:
        value = json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=bad_constant)
        json.dumps(value, allow_nan=False)  # Also catches numeric overflow such as 1e999.
    except (json.JSONDecodeError, ValueError) as exc:
        raise InputValidationError(f"{path}: invalid finite JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise InputValidationError(f"{path}: expected JSON object")
    return value


def _keys(value: dict, *, allowed: set, required: set, name: str) -> None:
    if not isinstance(value, dict):
        raise InputValidationError(f"{name}: expected an object")
    missing, unknown = required-set(value), set(value)-allowed
    if missing or unknown:
        raise InputValidationError(f"{name}: missing {sorted(missing)}, unknown {sorted(unknown)}")


def load_solver_config(path: Path) -> tuple[Weights, SolverConfig]:
    data = read_json(path)
    _keys(data, allowed={"solver", "weights", "parameters"}, required={"solver", "weights"}, name="config")
    _keys(data["weights"], allowed={"lam", "mu", "nu"}, required={"lam", "mu", "nu"}, name="weights")
    weights = Weights(**data["weights"])
    if data["solver"] not in ("poisson_gradient", "lm"):
        raise InputValidationError("solver: expected poisson_gradient or lm")
    cls = PoissonGradientConfig if data["solver"] == "poisson_gradient" else LMConfig
    parameters = data.get("parameters", {})
    _keys(parameters, allowed={f.name for f in fields(cls)}, required=set(), name="parameters")
    parameters = parameters.copy()
    if "stop" in parameters:
        _keys(parameters["stop"], allowed={f.name for f in fields(StopConfig)}, required=set(), name="stop")
        parameters["stop"] = StopConfig(**parameters["stop"])
    if parameters.get("jacobian_barrier") is not None:
        _keys(parameters["jacobian_barrier"], allowed={f.name for f in fields(JacobianBarrierConfig)},
              required={"weight", "min_jd"}, name="jacobian_barrier")
        parameters["jacobian_barrier"] = JacobianBarrierConfig(**parameters["jacobian_barrier"])
    if parameters.get("jacobian_feasibility") is not None:
        _keys(parameters["jacobian_feasibility"], allowed={f.name for f in fields(JacobianFeasibilityConfig)},
              required=set(), name="jacobian_feasibility")
        parameters["jacobian_feasibility"] = JacobianFeasibilityConfig(**parameters["jacobian_feasibility"])
    return weights, cls(**parameters)


def save_case_npz(case: RegistrationCase2D, path: Path) -> None:
    path = Path(path)
    if path.suffix != ".npz":
        raise InputValidationError("case path: expected .npz extension")
    sidecar = path.with_suffix(".json")
    if any(p.exists() or p.is_symlink() for p in (path, sidecar)):
        raise FileExistsError(f"Case output already exists: {path}")
    if {"schema_version", "controls_id"} & set(case.metadata):
        raise InputValidationError("metadata: schema_version and controls_id are reserved sidecar fields")
    metadata = dict(case.metadata, schema_version=1, controls_id=case.controls.controls_id)
    # Validate serializability before creating either file.
    json.dumps(metadata, allow_nan=False)
    grid, controls = case.grid, case.controls
    arrays = dict(fixed_image=case.fixed_image, moving_image=case.moving_image, u0=case.u0,
        u_ref=controls.u_ref, f=controls.f, g=controls.g, hx=grid.hx, hy=grid.hy,
        x0=grid.x[0], y0=grid.y[0])
    if case.u_true is not None:
        arrays["u_true"] = case.u_true
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    write_json(sidecar, metadata)


def load_case_npz(path: Path) -> RegistrationCase2D:
    path = Path(path)
    metadata = read_json(path.with_suffix(".json"))
    version = metadata.pop("schema_version", None)
    if type(version) is not int or version != 1:
        raise InputValidationError("schema_version: expected integer 1")
    controls_id = metadata.pop("controls_id", None)
    required = {"fixed_image", "moving_image", "u0", "u_ref", "f", "g", "hx", "hy", "x0", "y0"}
    with np.load(path, allow_pickle=False) as arrays:
        missing, unknown = required-set(arrays.files), set(arrays.files)-required-{"u_true"}
        if missing or unknown:
            raise InputValidationError(f"case arrays: missing {sorted(missing)}, unknown {sorted(unknown)}")
        fixed = as_float64_array(arrays["fixed_image"], "fixed_image")
        if fixed.ndim != 2:
            raise InputValidationError("fixed_image: expected (ny,nx)")
        grid_parameters = {key: float(as_float64_array(arrays[key], key, ())) for key in ("hx", "hy", "x0", "y0")}
        grid = make_grid_2d(nx=fixed.shape[1], ny=fixed.shape[0], **grid_parameters)
        controls = Controls2D(arrays["u_ref"], arrays["f"], arrays["g"], controls_id)
        return RegistrationCase2D(grid, fixed, arrays["moving_image"], arrays["u0"], controls, metadata,
                                  arrays["u_true"] if "u_true" in arrays else None)


def result_summary(result: SolveOutcome) -> dict:
    if isinstance(result, InitializationFailure):
        return dict(success=False, status=result.status, message=result.message, invalid_count=result.invalid_count,
                    initial=None, final=None, counts=None, wall_seconds=None, gradient_threshold=None)
    first = result.history[0] if result.history else None
    final_geometry = geometry_diagnostics(result.final_state)
    initial_geometry = {key: getattr(first, key) for key in final_geometry} if first else final_geometry
    counts = {key: getattr(result, key) for key in (
        "n_accepted", "n_trial_evaluations", "n_domain_rejections", "n_energy_evaluations",
        "n_gradient_evaluations", "n_hvp", "total_cg_iterations", "n_preconditioner_applications",
        "n_feasibility_checks", "n_jacobian_rejections")}
    counts.update(n_attempts=len(result.history), n_rejected_attempts=sum(not r.accepted for r in result.history),
                  n_rejected_candidates=sum(not r.accepted and r.trial_eval_id is not None for r in result.history),
                  n_feasible_bound_rejections=sum(r.reason == "FEASIBLE_STEP_BOUND" for r in result.history))
    return dict(success=result.success, status=result.status, message=result.message,
        initial=dict(energy=asdict(first.energy if first else result.final_state.energy),
                     G=result.initial_gradient_norm_mass, geometry=initial_geometry),
        final=dict(energy=asdict(result.final_state.energy), G=result.final_gradient_norm_mass, geometry=final_geometry),
        gradient_threshold=result.gradient_threshold, counts=counts, wall_seconds=result.wall_seconds,
        problem_id=result.final_state.problem_id, controls_id=result.final_state.controls_id,
        geometry_discretization=result.resolved_config["geometry_discretization"],
        jacobian_barrier=result.resolved_config.get("jacobian_barrier"),
        jacobian_feasibility=result.resolved_config.get("jacobian_feasibility"),
        geometry_convention="Full-node F06 difference JD, not an interpolation-cell topology guarantee.")


def save_result(result: SolveOutcome, output_dir: Path, metadata: dict) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir/"resolved_config.json", dict(solver_config=result.resolved_config, **metadata))
    write_json(output_dir/"summary.json", result_summary(result))
    with (output_dir/"history.jsonl").open("x") as stream:
        if not isinstance(result, InitializationFailure):
            for row in result.history:
                stream.write(json.dumps(asdict(row), allow_nan=False)+"\n")
    if not isinstance(result, InitializationFailure):
        state = result.final_state
        cells = {} if state.cell_geometry is None else dict(
            cell_gauss_J=state.cell_geometry.J, cell_gauss_curl=state.cell_geometry.curl)
        if state.corner_geometry is not None:
            cells["cell_corner_J"] = state.corner_geometry.J
        np.savez_compressed(output_dir/"result.npz", z=result.final_z, u=state.u, phi=state.phi,
                            J=state.geometry.J, curl=state.geometry.curl, warped_image=state.warped_image, **cells)
