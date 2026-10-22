"""Deformation smoothness report (E78): SDlogJ and J-range fractions at cell centres.

Diagnostic only: J here is the Q1 derivative at the cell centre (the average of the four
parallel edge differences per axis), not the certified whole-cell minimum. Never an acceptance
or stopping rule. SDlogJ follows the Learn2Reg convention (std of log J); "brain" means cells
whose lower-corner node carries a nonzero fixed seg4 label.
"""
import numpy as np

from ..validation import InputValidationError, require_finite_result

SMOOTHNESS_POLICY = "cell-centre-q1-sdlogj-v1"


def cell_jacobian_centres(u, grid):
    """det(I + grad u) at every cell centre; u is a (nz, ny, nx, 3) node field, components xyz."""
    u = np.asarray(u, dtype=np.float64)
    if u.shape != grid.shape+(3,):
        raise InputValidationError("cell_jacobian_centres: u must have the grid shape + (3,)")
    hx, hy, hz = grid.spacing

    def d(axis, comp):
        a = u[..., comp]
        if axis == 0:      # d/dx: x-edge differences, averaged over the 4 edges of each cell
            diff = (a[:, :, 1:]-a[:, :, :-1])/hx
            return .25*(diff[:-1, :-1]+diff[1:, :-1]+diff[:-1, 1:]+diff[1:, 1:])
        if axis == 1:
            diff = (a[:, 1:, :]-a[:, :-1, :])/hy
            return .25*(diff[:-1, :, :-1]+diff[1:, :, :-1]+diff[:-1, :, 1:]+diff[1:, :, 1:])
        diff = (a[1:, :, :]-a[:-1, :, :])/hz
        return .25*(diff[:, :-1, :-1]+diff[:, 1:, :-1]+diff[:, :-1, 1:]+diff[:, 1:, 1:])
    with np.errstate(over="raise", invalid="raise"):
        F = [[(1. if r == j else 0.)+d(j, r) for j in range(3)] for r in range(3)]
        J = (F[0][0]*(F[1][1]*F[2][2]-F[1][2]*F[2][1]) - F[0][1]*(F[1][0]*F[2][2]-F[1][2]*F[2][0])
             + F[0][2]*(F[1][0]*F[2][1]-F[1][1]*F[2][0]))
    require_finite_result(J, "cell-centre Jacobian determinant")
    return J


def smoothness_report(u, grid, brain_nodes=None):
    """SDlogJ over all cells and over brain cells, J < .5 / J > 2 fractions, min centre J."""
    J = cell_jacobian_centres(u, grid)
    with np.errstate(divide="ignore", invalid="ignore"):
        lj = np.log(np.clip(J, 1e-12, None))
    report = dict(policy=SMOOTHNESS_POLICY, n_cells=int(J.size), min_J_centre=float(J.min()),
                  sdlogj_all=float(lj.std()), pct_J_lt_half_all=float(100*np.mean(J < .5)),
                  pct_J_gt_two_all=float(100*np.mean(J > 2)))
    if brain_nodes is not None:
        mask = np.asarray(brain_nodes, dtype=bool)
        if mask.shape != grid.shape:
            raise InputValidationError("smoothness_report: brain_nodes must have the grid shape")
        cells = mask[:-1, :-1, :-1]
        if cells.any():
            report.update(n_brain_cells=int(cells.sum()), sdlogj_brain=float(lj[cells].std()),
                          pct_J_lt_half_brain=float(100*np.mean(J[cells] < .5)),
                          pct_J_gt_two_brain=float(100*np.mean(J[cells] > 2)))
        else:
            report.update(n_brain_cells=0, sdlogj_brain=None, pct_J_lt_half_brain=None, pct_J_gt_two_brain=None)
    return report
