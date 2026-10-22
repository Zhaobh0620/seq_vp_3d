"""T01 rectangular physical grid; no permanently stored dense coordinates."""
from dataclasses import dataclass
import hashlib

import numpy as np

from ..validation import InputValidationError, finite_real, grid_size, readonly_copy


@dataclass(frozen=True)
class Grid3D:
    """Factory-built; axes/spacing are xyz, array shape is zyx."""
    shape: tuple[int, int, int]
    spacing: tuple[float, float, float]
    axes: tuple[np.ndarray, np.ndarray, np.ndarray]
    trapezoid: tuple[np.ndarray, np.ndarray, np.ndarray]
    omega: float
    unit: str
    grid_id: str

    @property
    def interior_shape(self):
        return tuple(n - 2 for n in self.shape)

    @property
    def cell_shape(self):
        return tuple(n - 1 for n in self.shape)

    @property
    def n_free_scalar(self):
        return int(np.prod(self.interior_shape))

    @property
    def n_cells(self):
        return int(np.prod(self.cell_shape))

    def node_weights(self):
        tx, ty, tz = self.trapezoid
        return self.omega * tz[:, None, None] * ty[None, :, None] * tx[None, None, :]

    def identity(self):
        """Allocate only on explicit request; components xyz, spatial axes zyx."""
        x, y, z = self.axes
        out = np.empty(self.shape + (3,))
        out[..., 0], out[..., 1], out[..., 2] = x, y[:, None], z[:, None, None]
        return out


def make_grid_3d(*, nx, ny, nz, hx, hy, hz, x0=0., y0=0., z0=0., unit="unit"):
    sizes = tuple(grid_size(n, name) for n, name in zip((nx, ny, nz), ("nx", "ny", "nz")))
    spacing = tuple(finite_real(h, name, positive=True)
                    for h, name in zip((hx, hy, hz), ("hx", "hy", "hz")))
    origins = tuple(finite_real(o, name) for o, name in zip((x0, y0, z0), ("x0", "y0", "z0")))
    if not isinstance(unit, str) or not unit.strip():
        raise InputValidationError("unit: explicit nonempty length unit required")
    axes, mass = [], []
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        omega = float(np.prod(spacing))
        for n, h, origin in zip(sizes, spacing, origins):
            axis = origin + h * np.arange(n, dtype=np.float64)
            if not np.all(np.isfinite(axis)) or not np.all(np.diff(axis) > 0):
                raise InputValidationError("coordinates must be finite and strictly increasing")
            axes.append(readonly_copy(axis))
            weights = np.ones(n)
            weights[[0, -1]] = .5
            mass.append(readonly_copy(weights))
    if not np.isfinite(omega) or omega / 8 <= 0:
        raise InputValidationError("grid mass must be positive finite float64, including corners")
    digest = hashlib.sha256(repr((sizes, spacing, origins, unit)).encode()).hexdigest()
    return Grid3D(sizes[::-1], spacing, tuple(axes), tuple(mass), omega, unit, digest)
