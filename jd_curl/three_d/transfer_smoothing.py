"""A19 (path only; explicit option): low-pass smoothing of the prolonged displacement
before a level transfer is certified.

The transfer seed v = P u_L (T38g) is replaced by w = S_sigma v, and the transfer tries alpha_k * w with the T38g /
T83 scales (alpha_k = beta * .9^k, each relative to w). S_sigma is linear and keeps the six zero faces:

- heat_dirichlet: the discrete heat semigroup exp(t L_h), t = sigma^2/2, L_h the seven-point Laplacian with the
  grid's physical spacing on the interior nodes and zero Dirichlet faces (T76's L). It is diagonal in the DST-I basis:
  mode k_j along axis j decays by exp(-t (4/h_j^2) sin^2(pi k_j / (2 (n_j - 1)))). Computed with scipy.fft.dstn
  (type 1, orthonormal, its own inverse), float64, single-threaded: deterministic.
- gaussian_zero_faces: scipy.ndimage.gaussian_filter per component (sigma in voxels per axis, mode 'constant',
  truncate 4), then the six faces set to zero.

No acceptance, certificate, energy or schedule changes: the smoothed seed is certified like any transfer seed."""
import numbers

import numpy as np

from ..validation import InputValidationError, as_float64_array, readonly_copy

TRANSFER_SMOOTHING_POLICY = "transfer-lowpass-v1"
KERNELS = ("heat_dirichlet", "gaussian_zero_faces")
SCOPES = ("native", "all")
SIGMA_RANGE_MM = (0., 10.)      # open below, closed above


def validate_transfer_smoothing(spec):
    """spec: dict(kernel, sigma_mm[, scope='native']) -> a validated copy with the policy name."""
    if not isinstance(spec, dict) or set(spec) - {"kernel", "sigma_mm", "scope", "policy"}:
        raise InputValidationError("transfer smoothing: dict(kernel, sigma_mm, scope) required")
    kernel, sigma, scope = spec.get("kernel"), spec.get("sigma_mm"), spec.get("scope", "native")
    if spec.get("policy", TRANSFER_SMOOTHING_POLICY) != TRANSFER_SMOOTHING_POLICY:
        raise InputValidationError("transfer smoothing: unknown policy")
    if kernel not in KERNELS:
        raise InputValidationError(f"transfer smoothing kernel must be one of {KERNELS}")
    if scope not in SCOPES:
        raise InputValidationError(f"transfer smoothing scope must be one of {SCOPES}")
    if not isinstance(sigma, numbers.Real) or isinstance(sigma, (bool, np.bool_)):
        raise InputValidationError("transfer smoothing sigma_mm must be a real number")
    sigma = float(sigma)
    if not np.isfinite(sigma) or not SIGMA_RANGE_MM[0] < sigma <= SIGMA_RANGE_MM[1]:
        raise InputValidationError("transfer smoothing sigma_mm must be in (0, 10] mm")
    return dict(policy=TRANSFER_SMOOTHING_POLICY, kernel=kernel, sigma_mm=sigma, scope=scope)


def applies(spec, target_level):
    """Whether the transfer INTO target_level (1, 2 or 3) is smoothed."""
    return spec is not None and (target_level == 3 or spec["scope"] == "all")


def _heat_dirichlet(u, grid, sigma):
    from scipy.fft import dstn, idstn
    t = .5*sigma*sigma
    hx, hy, hz = grid.spacing
    interior = np.ascontiguousarray(u[1:-1, 1:-1, 1:-1, :])
    decay = np.ones(interior.shape[:3])
    for axis, (m, h) in enumerate(zip(interior.shape[:3], (hz, hy, hx))):
        k = np.arange(1, m+1, dtype=np.float64)
        factor = np.exp(-t*(4./(h*h))*np.sin(np.pi*k/(2.*(m+1)))**2)
        shape = [1, 1, 1]
        shape[axis] = m
        decay = decay*factor.reshape(shape)
    coefficients = dstn(interior, type=1, axes=(0, 1, 2), norm="ortho", workers=1)
    coefficients *= decay[..., None]
    out = np.zeros_like(u)
    out[1:-1, 1:-1, 1:-1, :] = idstn(coefficients, type=1, axes=(0, 1, 2), norm="ortho", workers=1)
    return out


def _gaussian_zero_faces(u, grid, sigma):
    from scipy.ndimage import gaussian_filter
    hx, hy, hz = grid.spacing
    voxels = (sigma/hz, sigma/hy, sigma/hx)          # array axis order z, y, x
    out = np.empty_like(u)
    for a in range(3):
        out[..., a] = gaussian_filter(u[..., a], sigma=voxels, mode="constant", cval=0., truncate=4.)
    out[[0, -1]] = 0.
    out[:, [0, -1]] = 0.
    out[:, :, [0, -1]] = 0.
    return out


def smooth_transfer_seed(u, grid, spec):
    """S_sigma u for a zero-face displacement u on grid (read-only result)."""
    spec = validate_transfer_smoothing(spec)
    u = as_float64_array(u, "transfer seed", grid.shape+(3,))
    if any(np.any(face != 0) for face in (u[[0, -1]], u[:, [0, -1]], u[:, :, [0, -1]])):
        raise InputValidationError("transfer smoothing requires zero displacement on all six faces")
    with np.errstate(over="raise", invalid="raise"):
        out = (_heat_dirichlet if spec["kernel"] == "heat_dirichlet" else _gaussian_zero_faces)(u, grid,
                                                                                               spec["sigma_mm"])
    if not np.all(np.isfinite(out)):
        raise FloatingPointError("transfer smoothing produced nonfinite values")
    return readonly_copy(out)
