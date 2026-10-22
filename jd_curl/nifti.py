"""Explicit native 2D NIfTI-to-local-mm conversion; no resampling or reorientation.

NiBabel is an optional preparation dependency, not a solver dependency.
"""

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

from .grid import make_grid_2d
from .provenance import file_provenance
from .types import FloatArray, Grid2D
from .validation import InputValidationError, as_float64_array, readonly_copy


@dataclass(frozen=True)
class NiftiSlice2D:
    """Factory-owned local image; world = origin + basis @ [x,y]."""

    grid: Grid2D
    image: FloatArray
    metadata: dict


def load_nifti_slice(path: Path) -> NiftiSlice2D:
    image = nib.load(str(path))
    if not isinstance(image, (nib.Nifti1Image, nib.Nifti2Image)):
        raise InputValidationError("MRI input: expected NIfTI image")
    shape = image.shape
    if len(shape) not in (2, 3) or (len(shape) == 3 and shape[2] != 1):
        raise InputValidationError("MRI shape: require (nx,ny) or (nx,ny,1); no implicit volume slicing")
    if image.header.get_xyzt_units()[0] != "mm":
        raise InputValidationError("MRI spatial unit: explicit mm required")
    if image.get_data_dtype().kind not in "iuf":
        raise InputValidationError("MRI intensities: real numeric dtype required")
    qform, qcode = image.get_qform(coded=True)
    sform, scode = image.get_sform(coded=True)
    if not qcode and not scode:
        raise InputValidationError("MRI affine: no coded qform/sform; fallback geometry is not accepted")
    if qcode and scode and not np.allclose(qform, sform, rtol=0., atol=1e-5):
        raise InputValidationError("MRI affine: conflicting qform and sform")
    affine = as_float64_array(image.affine, "MRI affine", (4, 4))
    if not np.array_equal(affine[3], [0, 0, 0, 1]):
        raise InputValidationError("MRI affine: invalid homogeneous row")
    spacing = np.linalg.norm(affine[:3, :2], axis=0)
    if not np.all(spacing > 0):
        raise InputValidationError("MRI affine: degenerate in-plane axis")
    basis = affine[:3, :2]/spacing
    if not np.allclose(basis.T@basis, np.eye(2), rtol=0., atol=1e-6):
        raise InputValidationError("MRI affine: in-plane shear is incompatible with a rectangular grid")
    zooms = np.asarray(image.header.get_zooms(), dtype=float)
    if not np.allclose(spacing, zooms[:2], rtol=1e-6, atol=1e-6):
        raise InputValidationError("MRI spacing: header zooms and affine disagree")
    grid = make_grid_2d(nx=shape[0], ny=shape[1], hx=float(spacing[0]), hy=float(spacing[1]))
    data = as_float64_array(image.get_fdata(dtype=np.float64, caching="unchanged"), "MRI intensities")
    local = data.T if len(shape) == 2 else data[:, :, 0].T
    metadata = dict(source=file_provenance(path), original_shape=list(shape),
        original_dtype=str(image.get_data_dtype()), affine=affine.tolist(), header_zooms=zooms.tolist(),
        spatial_unit="mm", axis_codes=list(nib.aff2axcodes(affine)),
        qform_code=int(qcode), sform_code=int(scode), selected_form="sform" if scode else "qform",
        qform=None if qform is None else qform.tolist(), sform=None if sform is None else sform.tolist(),
        array_transform="source.T" if len(shape) == 2 else "source[:,:,0].T",
        slice_axis=None if len(shape) == 2 else 2, slice_index=0,
        world_origin_mm=affine[:3, 3].tolist(), local_basis_world=basis.tolist(),
        world_mapping="world_mm = world_origin_mm + local_basis_world @ [x_mm,y_mm]",
        world_displacement="du_world_mm = local_basis_world @ [ux_mm,uy_mm]",
        intensity_scaling=dict(slope=float(image.dataobj.slope), intercept=float(image.dataobj.inter),
                               applied_by="NiBabel get_fdata; no additional intensity transformation"),
        statistics=dict(min=float(np.min(local)), max=float(np.max(local)),
                        nonzero_fraction=float(np.mean(local != 0))), nibabel_version=nib.__version__)
    return NiftiSlice2D(grid, readonly_copy(local), metadata)


def require_same_slice_grid(fixed: NiftiSlice2D, moving: NiftiSlice2D) -> None:
    if (fixed.grid.nx, fixed.grid.ny) != (moving.grid.nx, moving.grid.ny):
        raise InputValidationError("MRI pair: different shapes; explicit external preprocessing required")
    if not np.allclose(fixed.metadata["affine"], moving.metadata["affine"], rtol=0., atol=1e-6):
        raise InputValidationError("MRI pair: different affines; no implicit alignment/resampling")
