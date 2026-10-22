"""T02 polynomial determinant/cofactor and raw vector curl; never clip JD."""
from dataclasses import dataclass

import numpy as np

from ..validation import InputValidationError, as_float64_array, require_finite_result
from .cell_q1 import cell_node_indices
from .work_control import checkpoint


def cofactor(F):
    F = as_float64_array(F, "F")
    if F.shape[-2:] != (3, 3):
        raise InputValidationError("F: last two axes must be (3,3)")
    with np.errstate(over="raise", invalid="raise"):
        # Same cross-product products/subtraction, without np.cross's input
        # copies and three temporary row arrays followed by a stack copy.
        out = np.empty(F.shape, dtype=np.float64)
        for r, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            for c, l, m in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
                out[..., r, c] = F[..., j, l]*F[..., k, m]-F[..., j, m]*F[..., k, l]
    require_finite_result(out, "cofactor")
    return out


def determinant(F):
    F = as_float64_array(F, "F")
    A = cofactor(F)
    with np.errstate(over="raise", invalid="raise"):
        result = np.sum(F[..., 0, :] * A[..., 0, :], axis=-1)
    require_finite_result(result, "determinant")
    return result


def cofactor_jvp(F, V):
    F = as_float64_array(F, "F")
    V = as_float64_array(V, "V", F.shape)
    if F.shape[-2:] != (3, 3):
        raise InputValidationError("F: last two axes must be (3,3)")
    with np.errstate(over="raise", invalid="raise"):
        out = np.stack([np.cross(V[..., j, :], F[..., k, :]) + np.cross(F[..., j, :], V[..., k, :])
                        for j, k in ((1, 2), (2, 0), (0, 1))], axis=-2)
    require_finite_result(out, "cofactor JVP")
    return out


@dataclass(frozen=True)
class CellGeometry3D:
    F: np.ndarray
    A: np.ndarray
    J: np.ndarray
    curl: np.ndarray


def compute_cell_geometry(u, ops, *, tile):
    u = as_float64_array(u, "geometry.u", ops.grid.shape + (3,))
    return _compute_cell_geometry(u, ops, tile=tile)


def _compute_cell_geometry(u, ops, *, tile):
    """Internal complete-volume streams: check each gathered tile, not all u.

    Public single-tile calls above retain their full-field input validation.
    A stream must visit every cell before returning an energy/derivative.
    """
    checkpoint("cpu_cell_geometry", completed=tile.start, total=ops.grid.n_cells)
    u = as_float64_array(u, "geometry.u", ops.grid.shape + (3,), check_finite=False)
    indices = cell_node_indices(ops.grid, tile)
    gathered = as_float64_array(u.reshape(-1, 3)[indices], "geometry.local_u")
    with np.errstate(over="raise", invalid="raise"):
        # Gather each component once, not once per derivative. Keep the same
        # (cells,8) @ (8,points) products and stacking/reduction order as the
        # scalar Q1 operator; no all-volume geometry cache or fused reduction.
        local = [gathered[..., a].copy(order="C") for a in range(3)]
        F = np.empty((len(indices), len(ops.points), 3, 3), dtype=np.float64)
        for r, v in enumerate(local):
            for j, coefficients in enumerate(ops.coefficients):
                F[..., r, j] = v @ coefficients.T
        F += np.eye(3)
        A = cofactor(F)
        J = np.sum(F[..., 0, :] * A[..., 0, :], axis=-1)
        curl = np.stack((F[..., 2, 1] - F[..., 1, 2], F[..., 0, 2] - F[..., 2, 0],
                         F[..., 1, 0] - F[..., 0, 1]), axis=-1)
    for name, value in (("F", F), ("A", A), ("J", J), ("curl", curl)):
        require_finite_result(value, name)
        # Every output already owns newly allocated C-order storage; freezing
        # it requires no second copy and never freezes caller-owned input.
        value.flags.writeable = False
    return CellGeometry3D(F, A, J, curl)
