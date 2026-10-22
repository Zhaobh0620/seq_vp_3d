"""E1 axis-aligned, millimetre NIfTI adapter; no registration dependencies.

Scalar native xyz -> right-handed RAS xyz -> core zyx. Vector components are
always RAS physical xyz, never inferred from array axes. Unsupported geometry
is rejected instead of silently resampled. Coarse preprocessing is explicit.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from ..validation import InputValidationError, as_float64_array, readonly_copy
from ..types import InvalidTrial
from .grid import Grid3D, make_grid_3d
from .interpolation import check_map_domain
from .validation import zero_boundary


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _geometry(image):
    if image.header.get_xyzt_units()[0] != "mm":
        raise InputValidationError("NIfTI spatial units must explicitly be mm")
    q, qc = image.get_qform(coded=True)
    s, sc = image.get_sform(coded=True)
    if not qc or not sc or not np.isfinite(q).all() or not np.isfinite(s).all():
        raise InputValidationError("finite, coded qform AND sform required")
    if not np.allclose(q, s, rtol=0, atol=1e-5):
        raise InputValidationError("qform/sform disagreement")
    original = nib.orientations.io_orientation(s)
    ras = nib.orientations.axcodes2ornt(("R", "A", "S"))
    transform = nib.orientations.ornt_transform(original, ras)
    native_from_ras = nib.orientations.inv_ornt_aff(transform, image.shape[:3])
    affine = s@native_from_ras
    spacing = np.diag(affine[:3, :3])
    # Pure signed permutations only; a nonzero obliquity/shear is not a spacing.
    if (not np.all(spacing > 0)
            or not np.allclose(affine[:3, :3], np.diag(spacing), rtol=0, atol=1e-8)
            or not np.array_equal(affine[3], [0., 0., 0., 1.])):
        raise InputValidationError("oblique/sheared/non-axis-aligned NIfTI unsupported")
    shape_xyz = tuple(np.asarray(image.shape[:3])[np.argsort(transform[:, 0])])
    grid = make_grid_3d(nx=shape_xyz[0], ny=shape_xyz[1], nz=shape_xyz[2],
        hx=spacing[0], hy=spacing[1], hz=spacing[2],
        x0=affine[0, 3], y0=affine[1, 3], z0=affine[2, 3], unit="mm")
    return grid, transform, native_from_ras, affine


@dataclass(frozen=True)
class ScalarVolume:
    values: np.ndarray
    grid: Grid3D
    native_shape_xyz: tuple
    to_ras_orientation: np.ndarray
    metadata: dict


def _labels(values):
    if not np.isfinite(values).all() or np.any(values < 0) or not np.array_equal(values, np.rint(values)):
        raise InputValidationError("labels must be finite nonnegative integers")


def load_scalar(path, *, labels=False):
    path = Path(path).resolve(strict=True)
    image = nib.load(path)
    if len(image.shape) != 3 or any(n < 5 for n in image.shape):
        raise InputValidationError("expected scalar 3D NIfTI, each axis at least five nodes")
    grid, transform, index_map, affine = _geometry(image)
    values = image.get_fdata(dtype=np.float64)
    if not np.isfinite(values).all():
        raise InputValidationError("nonfinite NIfTI intensity")
    if labels:
        _labels(values)
    ras_values = nib.orientations.apply_orientation(values, transform)
    data = readonly_copy(ras_values.transpose(2, 1, 0))
    metadata = dict(source=str(path), sha256=file_sha256(path), native_shape_xyz=list(image.shape),
        native_affine=image.affine.tolist(), native_axcodes=list(nib.aff2axcodes(image.affine)),
        native_from_ras_index=index_map.tolist(), ras_affine=affine.tolist(),
        shape_zyx=list(grid.shape), spacing_xyz=list(grid.spacing), unit="mm", labels=labels,
        qform_code=int(image.header["qform_code"]), sform_code=int(image.header["sform_code"]),
        min=float(data.min()), max=float(data.max()))
    return ScalarVolume(data, grid, tuple(image.shape), readonly_copy(transform), metadata)


def native_scalar(values, volume):
    """Restore scalar storage orientation, without interpolation or intensity change."""
    values = as_float64_array(values, "native scalar values", volume.grid.shape)
    inverse = np.empty_like(volume.to_ras_orientation)
    for native_axis, (ras_axis, sign) in enumerate(volume.to_ras_orientation):
        inverse[int(ras_axis)] = native_axis, sign
    native = nib.orientations.apply_orientation(values.transpose(2, 1, 0), inverse)
    return readonly_copy(native)


def require_common_grid(*volumes):
    first = volumes[0].grid
    for volume in volumes[1:]:
        if (volume.grid.shape != first.shape or volume.grid.unit != first.unit
                or any(not np.array_equal(a, b) for a, b in zip(first.axes, volume.grid.axes))):
            raise InputValidationError("source images/labels must share exactly the same canonical grid")
    return first


def coarse_grid(grid, shape_zyx):
    if len(shape_zyx) != 3 or any(isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) for n in shape_zyx):
        raise InputValidationError("integer zyx target shape required")
    if any(n < 5 or n > old for n, old in zip(shape_zyx, grid.shape)):
        raise InputValidationError("target axes must be >=5 and no larger than source")
    spacing = tuple((axis[-1]-axis[0])/(n-1) for axis, n in zip(grid.axes, shape_zyx[::-1]))
    return make_grid_3d(nx=shape_zyx[2], ny=shape_zyx[1], nz=shape_zyx[0],
        hx=spacing[0], hy=spacing[1], hz=spacing[2],
        x0=grid.axes[0][0], y0=grid.axes[1][0], z0=grid.axes[2][0], unit=grid.unit)


def downsample(volume, shape_zyx, *, labels=False):
    """Fixed Gaussian AA + linear preprocessing; optimizer remains cubic spline."""
    return downsample_array(volume.values, volume.grid, shape_zyx, labels=labels)


def downsample_array(values, source_grid, shape_zyx, *, labels=False):
    """Array entry for the same E1 preprocessing; no fake NIfTI metadata."""
    values = as_float64_array(values, "source values", source_grid.shape)
    grid = coarse_grid(source_grid, shape_zyx)
    axes = [np.linspace(0, old-1, new) for old, new in zip(source_grid.shape, shape_zyx)]
    ratios = (np.asarray(source_grid.shape)-1)/(np.asarray(shape_zyx)-1)
    sigma = .5*np.sqrt(ratios**2-1)
    if labels:
        _labels(values)
        indices = [np.floor(a+.5).astype(np.intp) for a in axes]
        values = values[np.ix_(*indices)]
    else:
        smooth = gaussian_filter(values, sigma=sigma, mode="nearest", truncate=4.)
        coords = np.array(np.meshgrid(*axes, indexing="ij"))
        values = map_coordinates(smooth, coords, order=1, mode="nearest", prefilter=False)
    metadata = dict(shape_zyx=list(grid.shape), spacing_xyz=list(grid.spacing), origin_xyz=[float(a[0]) for a in grid.axes],
        source_shape_zyx=list(source_grid.shape), preserve_node_center_endpoints=True,
        intensity_rescale=False, labels=labels, ratios_zyx=ratios.tolist(),
        gaussian_sigma_source_voxels_zyx=None if labels else sigma.tolist(),
        gaussian_mode=None if labels else "nearest", gaussian_truncate=None if labels else 4.,
        resampling="nearest_floor_index_plus_half" if labels else "linear_after_gaussian")
    return readonly_copy(values), grid, metadata


def warp_labels(values, u, grid):
    """Q1 nodal pullback positions, nearest label; no clipping/outside fallback."""
    values = as_float64_array(values, "labels", grid.shape)
    _labels(values)
    u = as_float64_array(u, "label warp", grid.shape+(3,))
    if not zero_boundary(u):
        raise InputValidationError("six boundary faces must have zero displacement")
    invalid = check_map_domain(u, grid)
    if isinstance(invalid, InvalidTrial):
        raise InputValidationError("label sampling domain: "+invalid.reason)
    points = grid.identity()+u
    indices = [np.floor((points[..., r]-grid.axes[r][0])/grid.spacing[r]+.5).astype(np.intp) for r in range(3)]
    return readonly_copy(values[indices[2], indices[1], indices[0]])


def dice_labels(fixed, warped, *, labels):
    _labels(fixed)
    _labels(warped)
    if fixed.shape != warped.shape:
        raise InputValidationError("Dice arrays must share shape")
    rows = []
    for label in labels:
        a, b = fixed == label, warped == label
        na, nb = int(a.sum()), int(b.sum())
        value = None if na+nb == 0 else float(2*np.count_nonzero(a & b)/(na+nb))
        rows.append(dict(label=int(label), fixed_count=na, warped_count=nb, dice=value))
    defined = [r["dice"] for r in rows if r["dice"] is not None]
    return dict(per_label=rows, macro_dice=float(np.mean(defined)) if defined else None,
                defined_classes=len(defined), classes=list(map(int, labels)))


def ras_affine(grid):
    affine = np.eye(4)
    affine[:3, :3] = np.diag(grid.spacing)
    affine[:3, 3] = [a[0] for a in grid.axes]
    return affine


def _save(path, array_xyz, grid, *, vector=False, nifti_version=1):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    if grid.unit != "mm":
        raise InputValidationError("export requires mm physical grid")
    if isinstance(nifti_version, bool) or nifti_version not in (1, 2):
        raise InputValidationError("nifti_version must explicitly be 1 or 2")
    affine = ras_affine(grid)
    image_type = nib.Nifti1Image if nifti_version == 1 else nib.Nifti2Image
    image = image_type(array_xyz, affine)
    image.set_qform(affine, code=1)
    image.set_sform(affine, code=1)
    image.header.set_xyzt_units("mm")
    if vector:
        image.header.set_intent("vector", (), name="RAS_mm_Q1")
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, path)


def save_scalar(path, values, grid, *, nifti_version=1):
    values = as_float64_array(values, "export scalar", grid.shape)
    _save(path, values.transpose(2, 1, 0), grid, nifti_version=nifti_version)


def save_displacement(path, u, grid, *, nifti_version=1):
    """Export nodes on this grid only; caller must certify original and readback."""
    u = as_float64_array(u, "export displacement", grid.shape+(3,))
    if not zero_boundary(u):
        raise InputValidationError("six boundary faces must have zero displacement")
    _save(path, u.transpose(2, 1, 0, 3)[..., None, :], grid, vector=True, nifti_version=nifti_version)


def load_displacement(path):
    """Read our RAS_mm_Q1 vector convention; do not infer arbitrary NIfTI vectors."""
    image = nib.load(path)
    if (len(image.shape) != 5 or image.shape[-2:] != (1, 3)
            or image.header.get_intent()[0] != "vector" or image.header.get_intent()[2] != "RAS_mm_Q1"):
        raise InputValidationError("expected explicit RAS_mm_Q1 vector NIfTI (x,y,z,1,3)")
    grid, transform, _, _ = _geometry(image)
    if not np.array_equal(transform, [[0, 1], [1, 1], [2, 1]]):
        raise InputValidationError("vector reader accepts RAS lattice only; reorientation must be explicit")
    u = image.get_fdata(dtype=np.float64)[..., 0, :].transpose(2, 1, 0, 3)
    u = as_float64_array(u, "import displacement", grid.shape+(3,))
    if not zero_boundary(u):
        raise InputValidationError("six boundary faces must have zero displacement")
    return readonly_copy(u), grid
