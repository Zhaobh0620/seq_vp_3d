"""Q1 cell geometry at four Gauss points; no quadrature weights in operators.

Local coordinates lie in [0,1]^2. Point order is (-,-), (+,-), (-,+), (+,+),
where the two coordinates are (1 +/- 1/sqrt(3))/2. Each point has physical
weight hx*hy/4, applied by the residual layer, not by these Euclidean actions.
"""

from dataclasses import dataclass

import numpy as np

from .types import FloatArray, Grid2D
from .validation import InputValidationError, as_float64_array, readonly_copy, require_finite_result


@dataclass(frozen=True)
class CellGeometry:
    """Read-only cell-quadrature arrays produced by compute_cell_geometry."""

    F: FloatArray
    A: FloatArray
    J: FloatArray
    curl: FloatArray


class CellQ1Operators:
    """Physical derivatives and actual transposes of four-corner Q1 fields."""

    def __init__(self, grid: Grid2D, *, location: str = "gauss"):
        self._shape = (grid.ny, grid.nx)
        self._quadrature_shape = (grid.ny-1, grid.nx-1, 4)
        low, high = (1.-1./np.sqrt(3.))/2., (1.+1./np.sqrt(3.))/2.
        if location == "corners":
            low, high = 0., 1.
        elif location != "gauss":
            raise InputValidationError("Q1 location: expected gauss or corners")
        xi, eta = np.array([low, high, low, high]), np.array([low, low, high, high])
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            values = np.stack(((1.-xi)*(1.-eta), xi*(1.-eta), (1.-xi)*eta, xi*eta), axis=-1)
            dx = np.stack((eta-1., 1.-eta, -eta, eta), axis=-1)/grid.hx
            dy = np.stack((xi-1., -xi, 1.-xi, xi), axis=-1)/grid.hy
        require_finite_result(dx, "Q1 x derivative coefficients")
        require_finite_result(dy, "Q1 y derivative coefficients")
        self._values, self._dx, self._dy = (readonly_copy(a) for a in (values, dx, dy))

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    @property
    def quadrature_shape(self) -> tuple[int, int, int]:
        return self._quadrature_shape

    def _apply(self, s: FloatArray, coefficients: FloatArray, name: str) -> FloatArray:
        s = as_float64_array(s, name, self.shape)
        corners = np.stack((s[:-1, :-1], s[:-1, 1:], s[1:, :-1], s[1:, 1:]), axis=-1)
        with np.errstate(over="raise", invalid="raise"):
            out = corners @ coefficients.T
        require_finite_result(out, name+" result")
        return out

    def _transpose(self, t: FloatArray, coefficients: FloatArray, name: str) -> FloatArray:
        t = as_float64_array(t, name, self.quadrature_shape)
        with np.errstate(over="raise", invalid="raise"):
            corners = t @ coefficients
            out = np.zeros(self.shape)
            out[:-1, :-1] += corners[..., 0]
            out[:-1, 1:] += corners[..., 1]
            out[1:, :-1] += corners[..., 2]
            out[1:, 1:] += corners[..., 3]
        require_finite_result(out, name+" result")
        return out

    def interpolate(self, s: FloatArray) -> FloatArray:
        return self._apply(s, self._values, "Q1 interpolate.s")

    def dx(self, s: FloatArray) -> FloatArray:
        return self._apply(s, self._dx, "Q1 dx.s")

    def dy(self, s: FloatArray) -> FloatArray:
        return self._apply(s, self._dy, "Q1 dy.s")

    def dx_T(self, t: FloatArray) -> FloatArray:
        return self._transpose(t, self._dx, "Q1 dx_T.t")

    def dy_T(self, t: FloatArray) -> FloatArray:
        return self._transpose(t, self._dy, "Q1 dy_T.t")

    def squared_derivative_diagonal(self, ax, ay, weight) -> FloatArray:
        """diag(B.T W B), B=diag(ax) Dx+diag(ay) Dy, before restriction.

        Coefficients are scalars or fields at this operator's own four points.
        Scatter each cell's four local columns; never average shared-node JD.
        """
        values = []
        for name, value in (("ax", ax), ("ay", ay), ("weight", weight)):
            value = as_float64_array(value, "Q1 diagonal."+name)
            if value.shape not in ((), self.quadrature_shape):
                raise InputValidationError("Q1 diagonal."+name+": expected scalar or quadrature shape")
            values.append(value)
        ax, ay, weight = values
        if np.any(weight < 0):
            raise InputValidationError("Q1 diagonal.weight: must be nonnegative")
        out = np.zeros(self.shape)
        slices = ((slice(None, -1), slice(None, -1)), (slice(None, -1), slice(1, None)),
                  (slice(1, None), slice(None, -1)), (slice(1, None), slice(1, None)))
        with np.errstate(over="raise", invalid="raise"):
            for k, nodes in enumerate(slices):
                column = ax*self._dx[:, k]+ay*self._dy[:, k]
                out[nodes] += np.sum(weight*column**2, axis=-1)
        require_finite_result(out, "Q1 squared derivative diagonal")
        return out


def compute_cell_geometry(u: FloatArray, ops: CellQ1Operators) -> CellGeometry:
    """Full determinant/cofactor and raw curl; nonpositive JD is not repaired.

    Affine fields are allowed here. The solver's injection imposes zero edges.
    """
    u = as_float64_array(u, "cell_geometry.u", ops.shape+(2,))
    with np.errstate(over="raise", invalid="raise"):
        F = np.empty(ops.quadrature_shape+(2, 2))
        F[..., 0, 0] = 1.+ops.dx(u[..., 0])
        F[..., 0, 1] = ops.dy(u[..., 0])
        F[..., 1, 0] = ops.dx(u[..., 1])
        F[..., 1, 1] = 1.+ops.dy(u[..., 1])
        A = np.empty_like(F)
        A[..., 0, 0], A[..., 0, 1] = F[..., 1, 1], -F[..., 1, 0]
        A[..., 1, 0], A[..., 1, 1] = -F[..., 0, 1], F[..., 0, 0]
        J = F[..., 0, 0]*F[..., 1, 1]-F[..., 0, 1]*F[..., 1, 0]
        curl = F[..., 1, 0]-F[..., 0, 1]
    for name, value in (("F", F), ("A", A), ("J", J), ("curl", curl)):
        require_finite_result(value, "cell geometry "+name)
        # These are fresh owning arrays, never views of u or of each other.
        value.flags.writeable = False
    return CellGeometry(F, A, J, curl)
