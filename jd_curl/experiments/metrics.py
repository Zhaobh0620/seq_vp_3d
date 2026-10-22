"""Read-only diagnostics derived from accepted outputs; no extra objective evaluations."""

import numpy as np

from ..io import result_summary
from ..geometry import bilinear_cell_jacobians
from ..solvers import SolveOutcome
from ..solvers.common import InitializationFailure
from ..types import RegistrationCase2D
from ..validation import InputValidationError, finite_real


def mse_acceptance(mse_initial: float | None, mse_final: float | None, *, reference: str) -> dict:
    """Image gate only: remaining MSE ratio <= 0.30 means >= 70% reduction.

    No rounded percentages or acceptance tolerance. The caller fixes the
    reference state; outer/scale transitions must not reset this denominator.
    Undefined ratios never count as an image-gate pass.
    """
    if not isinstance(reference, str) or not reference.strip():
        raise InputValidationError('mse reference: require an explicit nonempty label')
    for name, value in (('mse_initial', mse_initial), ('mse_final', mse_final)):
        if value is not None and finite_real(value, name) < 0:
            raise InputValidationError(name+': must be nonnegative')
    initial = None if mse_initial is None else float(mse_initial)
    final = None if mse_final is None else float(mse_final)
    ratio = None
    if initial is None or final is None:
        status = 'unavailable'
    elif initial == 0.:
        status = 'zero_initial_mse'
    else:
        ratio = final/initial
        status = 'valid' if np.isfinite(ratio) else 'nonfinite_ratio'
        if status != 'valid':
            ratio = None
    return dict(mse_initial=initial, mse_final=final, mse_ratio=ratio,
        mse_ratio_max=0.30, mse_ratio_status=status, mse_reference=reference,
        acceptance_metric='mse_ratio', acceptance_direction='minimize',
        mse_target_passed=bool(ratio is not None and ratio <= 0.30),
        mse_reduction_fraction=None if ratio is None else 1.-ratio)


def registration_metrics(case: RegistrationCase2D, result: SolveOutcome) -> dict:
    if isinstance(result, InitializationFailure):
        return dict(available=False, reason=result.status, displacement_error=None,
                    **mse_acceptance(None, None, reference='solve_initial_state'))
    summary = result_summary(result)
    q = case.grid.q
    area = float(np.sum(q))
    mse = {key: float(2*summary[key]["energy"]["image"]/area) for key in ("initial", "final")}
    rmse = {key: float(np.sqrt(2*summary[key]["energy"]["image"]/area)) for key in ("initial", "final")}
    cell_geometry = {}
    for name, u in (("initial", case.u0), ("final", result.final_state.u)):
        cell_jd = bilinear_cell_jacobians(u, case.grid)
        cell_geometry[name] = dict(min_corner_jd=float(np.min(cell_jd)),
            nonpositive_cell_fraction=float(np.mean(np.any(cell_jd <= 0., axis=-1))))
    error = None
    if case.u_true is not None:
        error = {}
        for name, u in (("initial", case.u0), ("final", result.final_state.u)):
            delta = u-case.u_true
            epe = np.hypot(delta[..., 0], delta[..., 1])
            error[name] = dict(rms=float(np.sqrt(np.sum(q*epe**2)/area)),
                              mean_epe=float(np.sum(q*epe)/area), max_epe=float(np.max(epe)))
    return dict(available=True, data_kind=case.metadata["data_kind"],
        geometry_discretization=result.resolved_config["geometry_discretization"],
        length_unit=case.metadata["displacement_unit"], area=area,
        image_mse=mse, primary_image_metric="mse",
        **mse_acceptance(mse['initial'], mse['final'], reference='solve_initial_state'),
        image_rmse=rmse,
        rmse_reduction_fraction=1.-rmse["final"]/rmse["initial"] if rmse["initial"] > 0 else None,
        bilinear_cell_geometry=cell_geometry,
        cell_geometry_note="Corners of the bilinear interpolant of the saved nodal map; not central-difference JD or a global injectivity proof.",
        displacement_error=error,
        displacement_error_note="q-weighted RMS/mean and full-node max relative to supplied manufactured truth; nodal truth controls are not Q1-oracle controls; not an optimizer success criterion."
                                if error is not None else "No ground truth supplied; intensity improvement is not displacement accuracy.",
        grid_note="All nodes with the same trapezoidal q; no foreground-only metric or objective mask.")
