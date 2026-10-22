"""T76 (E78) curvature regularizer R(u) = 1/2 sum_i sum_a alpha q_i (L u_a)_i^2.

L is the seven-point Laplacian with the grid's physical spacing, evaluated at interior nodes
(zero at boundary nodes; boundary neighbours enter with their zero Dirichlet value). For
zero-boundary inputs L^T restricted to the interior equals L, so gradient, GGN action and
diagonal are all expressed with the same stencil. Every function here is CPU/NumPy; the device
mirror lives in backends/torch_ops.py and must keep the same operation order per axis.
"""
import numpy as np

from ..validation import InputValidationError, require_finite_result


def inverse_square_spacing(grid):
    """(1/hz^2, 1/hy^2, 1/hx^2) in array-axis order (z, y, x)."""
    hx, hy, hz = grid.spacing
    return (1./np.float64(hz))**2, (1./np.float64(hy))**2, (1./np.float64(hx))**2


def laplacian_nodes(field, grid):
    """Seven-point Laplacian of a zero-boundary node field; zero on the boundary nodes."""
    field = np.asarray(field, dtype=np.float64)
    if field.shape != grid.shape+(3,):
        raise InputValidationError("laplacian: field must have the grid shape + (3,)")
    iz, iy, ix = inverse_square_spacing(grid)
    out = np.zeros_like(field)
    c = field[1:-1, 1:-1, 1:-1]
    with np.errstate(over="raise", invalid="raise"):
        out[1:-1, 1:-1, 1:-1] = ((field[1:-1, 1:-1, 2:]-2.*c+field[1:-1, 1:-1, :-2])*ix
                                 + (field[1:-1, 2:, 1:-1]-2.*c+field[1:-1, :-2, 1:-1])*iy
                                 + (field[2:, 1:-1, 1:-1]-2.*c+field[:-2, 1:-1, 1:-1])*iz)
    return out


def transpose_laplacian_nodes(z, grid):
    """L^T z for a node block z: only interior rows of L exist, so boundary entries of z are ignored."""
    z = np.array(z, dtype=np.float64, copy=True)
    z[[0, -1], :, :] = 0.
    z[:, [0, -1], :] = 0.
    z[:, :, [0, -1]] = 0.
    return laplacian_nodes(z, grid)


def residual_weight(problem):
    """sqrt(alpha q) on the nodes, shape grid.shape + (1,)."""
    q = problem.grid.node_weights()
    with np.errstate(over="raise", invalid="raise"):
        return np.sqrt(problem.curvature_weight*q)[..., None]


def curvature_residual(problem, u):
    """r_R = sqrt(alpha q) L u (node block, zero on boundary nodes)."""
    with np.errstate(over="raise", invalid="raise"):
        return residual_weight(problem)*laplacian_nodes(u, problem.grid)


def curvature_energy(problem, u):
    """R(u) as a longdouble sum of the float64 residual squares."""
    r = curvature_residual(problem, u)
    with np.errstate(over="raise", invalid="raise"):
        return .5*np.sum(r*r, dtype=np.longdouble)


def curvature_gradient_nodes(problem, u):
    """L^T (alpha q L u) on the nodes (zero on the boundary), shape grid.shape + (3,)."""
    q = problem.grid.node_weights()
    with np.errstate(over="raise", invalid="raise"):
        return laplacian_nodes((problem.curvature_weight*q)[..., None]*laplacian_nodes(u, problem.grid), problem.grid)


def curvature_hvp_nodes(problem, v):
    """Same operator as the gradient (R is quadratic): L^T (alpha q L v)."""
    return curvature_gradient_nodes(problem, v)


def curvature_diagonal_nodes(problem):
    """Exact diagonal sum_i alpha q_i L_ia^2 per node (same for the three components)."""
    grid = problem.grid
    iz, iy, ix = inverse_square_spacing(grid)
    q = grid.node_weights()
    interior = np.zeros(grid.shape)
    interior[1:-1, 1:-1, 1:-1] = 1.
    weighted = problem.curvature_weight*q*interior         # alpha q_i on interior rows i, 0 elsewhere
    c0 = 2.*(ix+iy+iz)
    out = np.zeros(grid.shape)
    with np.errstate(over="raise", invalid="raise"):
        out += weighted*c0*c0                                # the row of node a itself
        out[:, :, :-1] += weighted[:, :, 1:]*ix*ix          # rows of the +x / -x neighbours
        out[:, :, 1:] += weighted[:, :, :-1]*ix*ix
        out[:, :-1, :] += weighted[:, 1:, :]*iy*iy
        out[:, 1:, :] += weighted[:, :-1, :]*iy*iy
        out[:-1, :, :] += weighted[1:, :, :]*iz*iz
        out[1:, :, :] += weighted[:-1, :, :]*iz*iz
    out[[0, -1], :, :] = 0.
    out[:, [0, -1], :] = 0.
    out[:, :, [0, -1]] = 0.
    require_finite_result(out, "curvature diagonal")
    return np.repeat(out[..., None], 3, axis=-1)
