"""Optional device point screen: CPU-verified rejection, never feasibility.

Closed27 float64 device values nominate one physical cell/point at a time.
Only the existing CPU outward interval upper bound can prove J <= delta.
No witness (including arithmetic uncertainty) means the caller MUST continue
with its original whole-cell certification. A full no-witness device scan is
extra work; this precheck does not promise an end-to-end speed improvement.
"""
import numpy as np
import torch

from ...types import InvalidTrial
from ...validation import InputValidationError, as_float64_array
from .. import interval as iv
from ..cell_q1 import CellQ1Operators3D, cell_node_indices
from ..feasibility import POINTS
from ..stationarity import point_jd_interval
from ..validation import integer_at_least
from .torch_ops import TorchModel, geometry


def screen(problem, u, model, *, device_u=None, witness_observer=None, batch_tiles=1, finite_known=False,
           prefilter=None):
    """Return a proved InvalidTrial or None; None is NOT a positive certificate.

    ``u`` is the authoritative CPU candidate array. An optional already-uploaded
    ``device_u`` only selects points: even stale/rounded device values cannot
    reject a valid CPU map. Callers retain boundary/domain guards and the full
    certification/observer path. No full-map certificate is emitted here.

    E93: ``prefilter`` is an optional callable (device_u, D) -> one bool per tile or None. A False
    entry must PROVE that the legacy minimum of that tile is > delta; such tiles are skipped and
    every other tile runs the code below unchanged, so the result is identical.
    """
    if not isinstance(model, TorchModel) or model.problem is not problem:
        raise InputValidationError("witness screen requires the current problem's TorchModel")
    batch_tiles = integer_at_least(batch_tiles, 'screen batch_tiles')
    u = as_float64_array(u, "witness candidate", problem.grid.shape+(3,), check_finite=False)
    # E89: finite_known=True only when the caller has just proven every value finite on this same
    # immutable array (check_map_domain counts nonfinite nodes); the scan would return True.
    if (not finite_known and not np.all(np.isfinite(u))) or not iv.arithmetic_supported():
        return None
    device = model.device
    if device_u is None:
        device_u = device.upload(u)
    else:
        device.check(device_u, u.shape)
    # The screen is closed27 even when the energy uses an eight-corner barrier.
    D = (model._Dc if problem.barrier_location == "closed27_uniform" else
         device.upload(CellQ1Operators3D(problem.grid, location="closed27_uniform").coefficients))
    delta = problem.certificate_config.delta
    select = prefilter(device_u, D) if prefilter is not None else None
    if batch_tiles > 1:
        if select is not None:
            return _batched_screen_selected(problem, u, model, device_u, D, batch_tiles, witness_observer, select)
        return _batched_screen(problem, u, model, device_u, D, batch_tiles, witness_observer)
    for (start, _), indices in (model.tiles() if select is None else model.tiles(select)):
        if indices is None:            # E93: proven legacy minimum > delta; nothing to do
            continue
        _, _, J, _ = (model.geometry(device_u.reshape(-1, 3)[indices], D, cofactor=False, with_curl=False)
                      if model._selective_kernels else geometry(device_u.reshape(-1, 3)[indices], D))
        # NaN/Inf are never witnesses. Finite underflow/rounding can only
        # nominate a point, and is independently checked on the CPU below.
        values = torch.where(torch.isfinite(J), J, torch.full_like(J, torch.inf))
        minimum, flat_index = values.reshape(-1).min(dim=0)
        if device.scalar(minimum) > delta:
            continue
        selected = int(device.scalar(flat_index))
        invalid = _verify(problem, u, start+selected//27, POINTS[selected % 27], witness_observer)
        if invalid is not None:
            return invalid
    return None


def _verify(problem, u, cell, point, observer):
    nodes = cell_node_indices(problem.grid, slice(cell, cell+1))[0]
    try:
        upper = float(point_jd_interval(u.reshape(-1, 3)[nodes], problem.grid.spacing, point).hi)
    except FloatingPointError:
        return None
    delta = problem.certificate_config.delta
    if upper <= delta:
        if observer is not None:
            observer(cell, point)
        return InvalidTrial("JACOBIAN_VIOLATION",
                            f"CPU interval point witness: cell={cell}, reference_xyz={tuple(point)}, "
                            f"physical_J_upper={upper.hex()} <= delta={delta.hex()}", 1)
    return None


def _batched_screen(problem, u, model, device_u, D, batch_tiles, observer):
    pending, starts = [], []
    def flush():
        # Only two scalars per tile cross the device boundary, in tile order.
        values = model.device.download(torch.stack(pending))
        for start, (minimum, selected) in zip(starts, values):
            if minimum <= problem.certificate_config.delta:
                selected = int(selected)
                invalid = _verify(problem, u, start+selected//27, POINTS[selected % 27], observer)
                if invalid is not None:
                    return invalid
        return None
    for (start, _), indices in model.tiles():
        _, _, J, _ = (model.geometry(device_u.reshape(-1, 3)[indices], D, cofactor=False, with_curl=False)
                      if model._selective_kernels else geometry(device_u.reshape(-1, 3)[indices], D))
        values = torch.where(torch.isfinite(J), J, torch.full_like(J, torch.inf))
        minimum, selected = values.reshape(-1).min(dim=0)
        pending.append(torch.stack((minimum, selected.to(dtype=torch.float64))))
        starts.append(start)
        if len(pending) == batch_tiles:
            invalid = flush()
            if invalid is not None:
                return invalid
            pending.clear(); starts.clear()
    return flush() if pending else None


def _batched_screen_selected(problem, u, model, device_u, D, batch_tiles, observer, select):
    """E93: _batched_screen with proven-skippable tiles. Groups are the same batch_tiles consecutive tiles
    as in _batched_screen and are flushed after the same tile (so the same checkpoints are passed before
    any return); a skipped tile contributes nothing, exactly as its legacy minimum > delta would."""
    pending, starts = [], []
    def flush():
        values = model.device.download(torch.stack(pending))
        for start, (minimum, selected) in zip(starts, values):
            if minimum <= problem.certificate_config.delta:
                selected = int(selected)
                invalid = _verify(problem, u, start+selected//27, POINTS[selected % 27], observer)
                if invalid is not None:
                    return invalid
        return None
    seen = 0
    for (start, _), indices in model.tiles(select):
        if indices is not None:
            _, _, J, _ = (model.geometry(device_u.reshape(-1, 3)[indices], D, cofactor=False, with_curl=False)
                          if model._selective_kernels else geometry(device_u.reshape(-1, 3)[indices], D))
            values = torch.where(torch.isfinite(J), J, torch.full_like(J, torch.inf))
            minimum, selected = values.reshape(-1).min(dim=0)
            pending.append(torch.stack((minimum, selected.to(dtype=torch.float64))))
            starts.append(start)
        seen += 1
        if seen == batch_tiles:
            if pending:
                invalid = flush()
                if invalid is not None:
                    return invalid
            pending.clear(); starts.clear()
            seen = 0
    return flush() if pending else None
