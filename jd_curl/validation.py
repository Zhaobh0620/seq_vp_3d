"""Shared real/finite/shape validation; never repairs or mutates input arrays."""

from __future__ import annotations

from numbers import Integral, Real
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .types import Controls2D, FloatArray, Grid2D, Weights


class InputValidationError(ValueError):
    """A named input violates the numerical or layout contract."""


class StateMismatchError(ValueError):
    """A cached evaluation does not belong to this problem/controls pair."""


class CoordinateConventionError(InputValidationError):
    """Case metadata does not state the required axes, direction or units."""


def validate_geometry_discretization(value: str) -> None:
    if not isinstance(value, str) or value not in ("nodal", "q1"):
        raise InputValidationError("geometry_discretization: expected nodal or q1")


def validate_case_metadata(metadata: dict) -> None:
    if not isinstance(metadata, dict):
        raise InputValidationError("metadata: expected a JSON object")
    for name, expected in (("axis_order", "yx"), ("component_order", "xy"), ("mapping", "fixed_to_moving")):
        if metadata.get(name) != expected:
            raise CoordinateConventionError(f"metadata.{name}: expected {expected!r}")
    for name in ("displacement_unit", "data_kind", "preprocessing"):
        if not isinstance(metadata.get(name), str) or not metadata[name].strip():
            raise CoordinateConventionError(f"metadata.{name}: explicit nonempty text required")
    if metadata.get("coordinate_unit", metadata["displacement_unit"]) != metadata["displacement_unit"]:
        raise CoordinateConventionError("metadata.coordinate_unit: must match displacement_unit")
    if not isinstance(metadata.get("provenance"), (str, dict)) or not metadata["provenance"]:
        raise InputValidationError("metadata.provenance: explicit source required")


def require_finite_result(value, name: str) -> None:
    """Numerical output failure, distinct from an invalid original input."""
    if isinstance(value, np.ndarray) and value.dtype == np.float64 and value.size >= (1 << 20):
        from .three_d import fast_checks  # E51 exact parallel all-finite for large arrays
        if fast_checks.enabled(value.size):
            if not fast_checks.all_finite(value):
                raise FloatingPointError(f"{name}: nonfinite numerical result")
            return
    if not np.all(np.isfinite(value)):
        raise FloatingPointError(f"{name}: nonfinite numerical result")


def finite_real(value: float, name: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise InputValidationError(f"{name}: expected a real scalar")
    result = float(value)
    if not np.isfinite(result) or (positive and result <= 0):
        qualifier = "positive and finite" if positive else "finite"
        raise InputValidationError(f"{name}: must be {qualifier}")
    return result


def grid_size(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 5:
        raise InputValidationError(f"{name}: expected an integer >= 5")
    return int(value)


def as_float64_array(
    value: FloatArray, name: str, shape: tuple[int, ...] | None = None,
    *, check_finite: bool = True,
) -> FloatArray:
    """Check layout/type; trial evaluation may defer finiteness to InvalidTrial."""
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise InputValidationError(f"{name}: expected a regular real array") from exc
    if array.dtype.kind not in "iuf":
        raise InputValidationError(f"{name}: expected real integer or floating values")
    if shape is not None and array.shape != shape:
        raise InputValidationError(f"{name}: expected shape {shape}, got {array.shape}")
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.asarray(array, dtype=np.float64)
    if check_finite and not np.all(np.isfinite(array)):
        raise InputValidationError(f"{name}: all values must be finite float64")
    return array


def readonly_copy(value: FloatArray) -> FloatArray:
    array = np.array(value, dtype=np.float64, copy=True, order="C")
    array.flags.writeable = False
    return array


def require_zero_boundary(u: FloatArray, name: str) -> None:
    """Check a validated (ny,nx,2) field without altering its boundary."""
    if np.any(u[[0, -1], :, :] != 0) or np.any(u[:, [0, -1], :] != 0):
        raise InputValidationError(f"{name}: both components must have exactly zero boundary")


def validate_problem_inputs(
    *, grid: Grid2D, weights: Weights,
    fixed_image: FloatArray, moving_image: FloatArray,
    u0: FloatArray, controls: Controls2D,
) -> None:
    """Validate data on one grid; owning constructors, not this function, copy it.

    Sampling-domain checks are owned by sample_value_gradient, not this validator.
    """
    shape = (grid.ny, grid.nx)
    for name in ("lam", "mu", "nu"):
        finite_real(getattr(weights, name), f"weights.{name}", positive=True)
    as_float64_array(fixed_image, "fixed_image", shape)
    as_float64_array(moving_image, "moving_image", shape)
    initial = as_float64_array(u0, "u0", shape + (2,))
    reference = as_float64_array(controls.u_ref, "controls.u_ref", shape + (2,))
    as_float64_array(controls.f, "controls.f", shape)
    as_float64_array(controls.g, "controls.g", shape)
    require_zero_boundary(initial, "u0")
    require_zero_boundary(reference, "controls.u_ref")
