"""T80 (E80 M): per-node weights m_i of the data term D = 1/2 sum_i q_i m_i (W_i - F_i)^2.

Only the image term uses q m; proximity, JD/curl, barrier and the curvature regularizer keep q.
The policy brain-shell-v1 derives m from the fixed image alone, one-sided (review T80): m = 1 on the
support {F > 0}; outside, m = beta + (1 - beta) s with s = G_sigma * 1{F > 0} (sigma in mm, nearest boundary),
so the weight decays from about (1 + beta)/2 at the boundary to beta far away. Coarse pyramid levels use the
native weight downsampled with the image pipeline (nifti.downsample_array), never a weight recomputed from the
anti-aliased coarse image (whose positive tails would dilate the support).
"""
import numpy as np
from scipy.ndimage import gaussian_filter

from ..validation import InputValidationError, readonly_copy, require_finite_result
from .nifti import downsample_array

DATA_WEIGHT_POLICY = "brain-shell-v1"


def data_mass(problem):
    """q_i m_i on the nodes (q itself when the problem has no data weight)."""
    q = problem.grid.node_weights()
    return q if problem.data_weight is None else q*problem.data_weight


def brain_shell_weight(fixed, grid, *, beta, sigma_mm):
    fixed = np.asarray(fixed, dtype=np.float64)
    if fixed.shape != grid.shape:
        raise InputValidationError("brain_shell_weight: fixed must have the grid shape")
    beta, sigma_mm = float(beta), float(sigma_mm)
    if not (np.isfinite(beta) and 0. <= beta <= 1. and np.isfinite(sigma_mm) and sigma_mm > 0.):
        raise InputValidationError("brain_shell_weight: 0 <= beta <= 1 and sigma_mm > 0 required")
    hx, hy, hz = grid.spacing
    support = fixed > 0.
    smooth = gaussian_filter(support.astype(np.float64), sigma=(sigma_mm/hz, sigma_mm/hy, sigma_mm/hx), mode="nearest")
    weight = np.where(support, 1., beta+(1.-beta)*np.clip(smooth, 0., 1.))
    require_finite_result(weight, "brain-shell data weight")
    return weight


def data_weight_from_policy(policy, fixed, grid):
    """policy = dict(policy='brain-shell-v1', beta=..., sigma_mm=...) -> node weight array (native level)."""
    if policy.get("policy") != DATA_WEIGHT_POLICY:
        raise InputValidationError("unknown data weight policy")
    return brain_shell_weight(fixed, grid, beta=policy["beta"], sigma_mm=policy["sigma_mm"])


def data_weight_levels(policy, fixed, grid, level_grids):
    """One weight per pyramid level: the native weight itself on the native grid, the native weight downsampled
    with the image pipeline (Gaussian anti-aliasing + linear resampling, values stay within [beta, 1]) elsewhere."""
    native = data_weight_from_policy(policy, fixed, grid)
    out = []
    for level_grid in level_grids:
        if level_grid.grid_id == grid.grid_id:
            out.append(readonly_copy(native))
            continue
        values, coarse, _ = downsample_array(native, grid, level_grid.shape)
        if coarse.grid_id != level_grid.grid_id:
            raise InputValidationError("data_weight_levels: pyramid grid mismatch")
        out.append(readonly_copy(np.clip(values, min(float(policy["beta"]), 1.), 1.)))
    return tuple(out)
