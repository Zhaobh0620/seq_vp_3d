"""T02/T03 tiled Q1 values, physical derivatives and Euclidean transposes."""
from itertools import product

import numpy as np

from ..validation import InputValidationError, as_float64_array, readonly_copy, require_finite_result
from .validation import integer_at_least

# Labels xyz, stored in z/y/x C-order (x fastest).
VERTICES = tuple((x, y, z) for z, y, x in product((0, 1), repeat=3))


def cell_tiles(grid, size=8192):
    size = integer_at_least(size, "tile size")
    for start in range(0, grid.n_cells, size):
        yield slice(start, min(start + size, grid.n_cells))


def cell_node_indices(grid, tile):
    if not isinstance(tile, slice) or tile.step not in (None, 1):
        raise InputValidationError("tile: explicit contiguous cell slice required")
    start = integer_at_least(tile.start, "tile.start", 0)
    stop = integer_at_least(tile.stop, "tile.stop")
    if not start < stop <= grid.n_cells:
        raise InputValidationError("tile: require 0 <= start < stop <= n_cells")
    cz, cy, cx = np.unravel_index(np.arange(start, stop), grid.cell_shape)
    _, ny, nx = grid.shape
    return np.stack([((cz + z) * ny + cy + y) * nx + cx + x for x, y, z in VERTICES], axis=-1)


class CellQ1Operators3D:
    def __init__(self, grid, *, location="gauss27"):
        self.grid = grid
        self.location = location
        if location == "gauss27":
            nodes = (.5 - np.sqrt(15.) / 10, .5, .5 + np.sqrt(15.) / 10)
            weights = (5 / 18, 4 / 9, 5 / 18)
        elif location == "corners8":
            nodes, weights = (0., 1.), (.5, .5)
        elif location == "closed27_uniform":
            nodes, weights = (0., .5, 1.), (1/3, 1/3, 1/3)
        else:
            raise InputValidationError("location: expected gauss27, corners8 or closed27_uniform")
        self.points = readonly_copy([(x, y, z) for z, y, x in product(nodes, repeat=3)])
        self.weights = readonly_copy([grid.omega * x * y * z for z, y, x in product(weights, repeat=3)])
        if location == "closed27_uniform":
            self.weights = readonly_copy(np.full(27, grid.omega/27))
        basis = np.empty((len(self.points), 8))
        derivative = np.empty((3, len(self.points), 8))
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            for a, vertex in enumerate(VERTICES):
                factors = [self.points[:, j] if vertex[j] else 1 - self.points[:, j] for j in range(3)]
                basis[:, a] = factors[0] * factors[1] * factors[2]
                for j in range(3):
                    others = [k for k in range(3) if k != j]
                    derivative[j, :, a] = ((1 if vertex[j] else -1) / grid.spacing[j]
                                           * factors[others[0]] * factors[others[1]])
        require_finite_result(derivative, "Q1 derivative coefficients")
        self.basis, self.coefficients = readonly_copy(basis), readonly_copy(derivative)

    def _coefficients(self, axis):
        if axis is None:
            return self.basis
        if axis not in ("x", "y", "z"):
            raise InputValidationError("derivative axis must be x, y or z")
        return self.coefficients[("x", "y", "z").index(axis)]

    def _apply(self, field, tile, coefficients):
        field = as_float64_array(field, "Q1 field", self.grid.shape)
        indices = cell_node_indices(self.grid, tile)
        with np.errstate(over="raise", invalid="raise"):
            out = field.ravel()[indices] @ coefficients.T
        require_finite_result(out, "Q1 action")
        return out

    def interpolate(self, field, *, tile):
        return self._apply(field, tile, self.basis)

    def derivative(self, field, axis, *, tile):
        return self._apply(field, tile, self._coefficients(axis))

    def derivative_transpose(self, values, axis, *, tile, accumulate_into):
        """Add D.T values to caller-owned float64 output, including shared nodes.

        One local vertex has unique global destinations within a tile; no
        duplicate-index advanced assignment or atomics within each update.
        """
        indices = cell_node_indices(self.grid, tile)
        values = as_float64_array(values, "Q1 transpose values", (len(indices), len(self.points)))
        out = accumulate_into
        if (not isinstance(out, np.ndarray) or out.dtype != np.float64 or out.shape != self.grid.shape
                or not out.flags.c_contiguous or not out.flags.writeable):
            raise InputValidationError("transpose output: writable C-order float64 node array required")
        require_finite_result(out, "transpose initial output")
        self._transpose_add(values, self._coefficients(axis), indices, out)
        require_finite_result(out, "Q1 transpose output")
        return out

    @staticmethod
    def _transpose_local(values, coefficients):
        """E108: _transpose_add's local product (same check, error state and product), for a worker thread."""
        require_finite_result(values, "Q1 transpose values")
        with np.errstate(over="raise", invalid="raise"):
            return values @ coefficients

    @staticmethod
    def _transpose_scatter(local, indices, out, *, also_into=None):
        """E108: _transpose_add's accumulation of a _transpose_local product (same order and error state)."""
        with np.errstate(over="raise", invalid="raise"):
            for a in range(8):
                out.ravel()[indices[:, a]] += local[:, a]
                if also_into is not None:
                    also_into.ravel()[indices[:, a]] += local[:, a]

    @staticmethod
    def _transpose_add(values, coefficients, indices, out, *, also_into=None):
        """Internal validated stream accumulator; caller checks full out once."""
        require_finite_result(values, "Q1 transpose values")
        with np.errstate(over="raise", invalid="raise"):
            local = values @ coefficients
            for a in range(8):
                out.ravel()[indices[:, a]] += local[:, a]
                if also_into is not None:
                    also_into.ravel()[indices[:, a]] += local[:, a]
