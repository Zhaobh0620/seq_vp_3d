"""2D types; scalar arrays use (y,x), vector components use (x,y)."""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import json
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .validation import (
    InputValidationError, as_float64_array, finite_real,
    readonly_copy, require_zero_boundary, validate_case_metadata,
)

FloatArray: TypeAlias = NDArray[np.float64]

if TYPE_CHECKING:
    from .jacobian_barrier import JacobianBarrierConfig, JacobianFeasibilityConfig
    from .cell_q1 import CellQ1Operators, CellGeometry
    from .differences import DifferenceOperators2D
    from .interpolation import ImageSampler2D

RESIDUAL_NAMES = ("image", "proximity_x", "proximity_y", "jd", "curl")


@dataclass(frozen=True)
class GeometryState:
    """Read-only arrays produced by compute_geometry; curl is before subtracting g."""

    F: FloatArray
    A: FloatArray
    J: FloatArray
    curl: FloatArray


@dataclass(frozen=True)
class SampleResult:
    """Read-only values and physical (x,y) gradients from one spline."""

    value: FloatArray
    gradient: FloatArray


@dataclass(frozen=True)
class InvalidTrial:
    """Numerical trial rejection, not a substitute for input/programming errors."""

    reason: str
    message: str
    invalid_count: int


@dataclass(frozen=True)
class Grid2D:
    """Grid data produced by make_grid_2d; do not hand-build inconsistent fields."""

    nx: int
    ny: int
    hx: float
    hy: float
    x: FloatArray
    y: FloatArray
    X: FloatArray
    Y: FloatArray
    q: FloatArray
    sqrt_q: FloatArray
    omega: float
    n_free_scalar: int


@dataclass(frozen=True)
class Weights:
    lam: float
    mu: float
    nu: float

    def __post_init__(self) -> None:
        for name in ("lam", "mu", "nu"):
            object.__setattr__(self, name, finite_real(getattr(self, name), name, positive=True))


@dataclass(frozen=True)
class Controls2D:
    """Owned, read-only snapshots; f and g are not clipped or normalized."""

    u_ref: FloatArray
    f: FloatArray
    g: FloatArray
    controls_id: str

    def __post_init__(self) -> None:
        f = as_float64_array(self.f, "controls.f")
        if f.ndim != 2 or min(f.shape) < 5:
            raise InputValidationError("controls.f: expected (ny,nx) with ny,nx >= 5")
        g = as_float64_array(self.g, "controls.g", f.shape)
        u_ref = as_float64_array(self.u_ref, "controls.u_ref", f.shape + (2,))
        require_zero_boundary(u_ref, "controls.u_ref")
        if not isinstance(self.controls_id, str) or not self.controls_id.strip():
            raise InputValidationError("controls_id: expected a nonempty string")
        for name, value in (("u_ref", u_ref), ("f", f), ("g", g)):
            object.__setattr__(self, name, readonly_copy(value))


@dataclass(frozen=True)
class Q1Controls2D:
    """Actual four-Gauss references, never interpolated from nodal controls."""

    u_ref: FloatArray
    f_gauss: FloatArray
    g_gauss: FloatArray
    controls_id: str

    def __post_init__(self) -> None:
        u = as_float64_array(self.u_ref, "q1_controls.u_ref")
        if u.ndim != 3 or u.shape[-1] != 2 or min(u.shape[:2]) < 5:
            raise InputValidationError("q1_controls.u_ref: expected (ny,nx,2), ny,nx >= 5")
        require_zero_boundary(u, "q1_controls.u_ref")
        shape = (u.shape[0]-1, u.shape[1]-1, 4)
        if not isinstance(self.controls_id, str) or not self.controls_id.strip():
            raise InputValidationError("controls_id: expected a nonempty string")
        for name, value in (("u_ref", u),
                            ("f_gauss", as_float64_array(self.f_gauss, "f_gauss", shape)),
                            ("g_gauss", as_float64_array(self.g_gauss, "g_gauss", shape))):
            object.__setattr__(self, name, readonly_copy(value))


@dataclass(frozen=True)
class RegistrationCase2D:
    grid: Grid2D
    fixed_image: FloatArray
    moving_image: FloatArray
    u0: FloatArray
    controls: Controls2D
    metadata: dict
    u_true: FloatArray | None = None

    def __post_init__(self) -> None:
        shape = (self.grid.ny, self.grid.nx)
        for name in ("fixed_image", "moving_image", "u0", "u_true"):
            value = getattr(self, name)
            if name == "u_true" and value is None:
                continue
            vector = name in ("u0", "u_true")
            value = as_float64_array(value, name, shape+(2,) if vector else shape)
            if vector:
                require_zero_boundary(value, name)
            object.__setattr__(self, name, readonly_copy(value))
        if not isinstance(self.controls, Controls2D):
            raise InputValidationError("controls: expected Controls2D")
        as_float64_array(self.controls.f, "controls.f", shape)
        object.__setattr__(self, "controls", Controls2D(self.controls.u_ref, self.controls.f,
                                                     self.controls.g, self.controls.controls_id))
        validate_case_metadata(self.metadata)
        try:
            json.dumps(self.metadata, allow_nan=False)
        except (ValueError, TypeError) as exc:
            raise InputValidationError("metadata: expected finite JSON values") from exc
        object.__setattr__(self, "metadata", deepcopy(self.metadata))


@dataclass(frozen=True)
class Problem2D:
    """Created by build_problem_2d; do not replace fields while retaining an ID."""

    grid: Grid2D
    weights: Weights
    controls: Controls2D | Q1Controls2D
    fixed_image: FloatArray
    sampler: ImageSampler2D
    diff: DifferenceOperators2D
    problem_id: str
    geometry_discretization: str = "nodal"
    cell_operators: CellQ1Operators | None = None
    cell_f: FloatArray | None = None
    cell_g: FloatArray | None = None
    jacobian_barrier: JacobianBarrierConfig | None = None
    corner_operators: CellQ1Operators | None = None
    jacobian_feasibility: JacobianFeasibilityConfig | None = None


@dataclass(frozen=True)
class ResidualBlocks:
    """Weighted blocks: first three nodal; geometry either nodal or Q1 Gauss."""

    image: FloatArray
    proximity_x: FloatArray
    proximity_y: FloatArray
    jd: FloatArray
    curl: FloatArray

    barrier: FloatArray | None = None

    def __post_init__(self) -> None:
        image = as_float64_array(self.image, "residuals.image")
        if image.ndim != 2:
            raise InputValidationError("residuals.image: expected (ny,nx)")
        jd = as_float64_array(self.jd, "residuals.jd")
        cell_shape = (image.shape[0]-1, image.shape[1]-1, 4)
        if jd.shape not in (image.shape, cell_shape):
            raise InputValidationError("residuals.jd: expected nodal or Q1 quadrature shape")
        for index, name in enumerate(RESIDUAL_NAMES):
            shape = image.shape if index < 3 else jd.shape
            value = as_float64_array(getattr(self, name), "residuals." + name, shape)
            object.__setattr__(self, name, readonly_copy(value))

        if self.barrier is not None:
            value = as_float64_array(self.barrier, "residuals.barrier", cell_shape)
            if jd.shape != cell_shape:
                raise InputValidationError("barrier residual requires Q1 geometry blocks")
            object.__setattr__(self, "barrier", readonly_copy(value))


@dataclass(frozen=True)
class EnergyTerms:
    image: float
    proximity: float
    jd: float
    curl: float
    total: float
    barrier: float = 0.


@dataclass(frozen=True)
class EvalState:
    """Consistent read-only snapshot produced only by evaluate_state."""

    problem_id: str
    controls_id: str
    eval_id: int
    z: FloatArray
    u: FloatArray
    phi: FloatArray
    geometry: GeometryState
    warped_image: FloatArray
    image_grad: FloatArray
    residuals: ResidualBlocks
    energy: EnergyTerms
    cell_geometry: CellGeometry | None = None
    corner_geometry: CellGeometry | None = None
    barrier_jacobian_scale: FloatArray | None = None
