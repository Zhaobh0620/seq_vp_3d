"""T37 four-level continuation around the unchanged Sequential/Poisson solvers.

Coarse objectives are not the native objective. Only a native-level result
can claim whole-run convergence or supply the reported native MSE ratio.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np

from ..validation import InputValidationError, as_float64_array, readonly_copy
from .boundary import restrict_free
from .feasibility import EnclosureCertificateConfig
from .grid import Grid3D
from .nifti import coarse_grid, downsample_array
from .data_weight import data_weight_levels
from .problem import Weights3D, make_problem, array_hash
from .residuals import validate_state
from .sequential import SequentialConfig3D, SequentialResult3D, solve_sequential
from .solvers.lm import LMConfig3D
from .solvers.poisson_gradient import PGConfig3D
from .validation import zero_boundary

FACTORS = (8, 4, 2, 1)
TRANSFER_STATUSES = ("outer_max_iterations", "converged_regularized_gradient")


# Search exhaustion is not convergence or a failure of the committed mapping.
# Unknown/numerical/linear-solve failures must still stop the hierarchy.
def _coarse_search_stalled(result):
    statuses = ("outer_stagnated", "outer_no_progress", "outer_precision_stagnated",
                "outer_domain_blocked", "outer_feasibility_blocked", "outer_feasibility_uncertain")
    if result.status not in statuses and not (
            result.status == "outer_inner_failed" and result.history
            and result.history[-1]["inner_status"] in ("line_search_failed", "trial_budget_exhausted")):
        return False
    safe_reasons = {"ACCEPTED", "INVALID_DOMAIN", "JACOBIAN_VIOLATION", "FEASIBILITY_UNCERTAIN", "BARRIER_DOMAIN",
                    "ENERGY_RESOLUTION", "NO_STRICT_DECREASE", "ARMIJO_REJECTED",
                    "JD_COMPRESSION_PRESSURE", "MODEL_GAIN_OR_ARMIJO_REJECTED"}
    attempts = result.history[-1].get("inner_history", ()) if result.history else ()
    return all(row["reason"] in safe_reasons for row in attempts)


def four_level_grids(native):
    """Preserve physical node-center FOV, including the original native grid."""
    if not isinstance(native, Grid3D) or native.unit != "mm":
        raise InputValidationError("four-level registration requires a physical mm Grid3D")
    shapes = [tuple((n-2+s)//s+1 for n in native.shape) for s in FACTORS]
    if any(n < 5 for shape in shapes for n in shape):
        raise InputValidationError("all four levels need >=5 nodes on every axis")
    return tuple(coarse_grid(native, shape) for shape in shapes[:-1])+(native,)


def prolong_displacement(u, source, target):
    """Physical Q1 interpolation only; caller MUST certify the new Q1 map.

    Separable interpolation avoids allocating a native dense query-coordinate
    volume. No displacement amplitude scaling or boundary clipping is used.
    """
    if source.unit != target.unit or source.unit != "mm":
        raise InputValidationError("transfer requires matching physical mm units")
    u = as_float64_array(u, "transfer displacement", source.shape+(3,))
    if not zero_boundary(u):
        raise InputValidationError("transfer requires zero displacement on all six faces")
    result = u
    for axis, (old, new, h) in enumerate(zip(source.axes[::-1], target.axes[::-1], source.spacing[::-1])):
        tolerance = 32*np.finfo(float).eps*max(1., np.max(np.abs(old)), np.max(np.abs(new)))
        if len(new) < len(old) or not np.allclose(old[[0, -1]], new[[0, -1]], rtol=0., atol=tolerance):
            raise InputValidationError("transfer must refine the same physical FOV")
        t = (new-old[0])/h
        # Identical analytical FOV endpoints can differ by construction roundoff.
        # Pin just their interpolation coordinates, not any physical displacement.
        t[0], t[-1] = 0., len(old)-1
        if np.any(t[1:-1] <= 0) or np.any(t[1:-1] >= len(old)-1):
            raise InputValidationError("target interior is not inside the source FOV")
        left = np.floor(t).astype(np.intp)
        left[-1] = len(old)-2
        weight = t-left
        shape = [1]*4
        shape[axis] = len(new)
        weight = weight.reshape(shape)
        result = (1-weight)*np.take(result, left, axis=axis)+weight*np.take(result, left+1, axis=axis)
    if not zero_boundary(result):
        raise InputValidationError("Q1 transfer did not preserve the six zero faces")
    return readonly_copy(result)


TRANSFER_BACKTRACK_ALPHAS = tuple(.9**k for k in range(7))
TRANSFER_BACKTRACK_POLICY = "q1-transfer-scale09-six-retries-v1"
# AP2 (explicit option, the native transfer of the pre-optimizer route only):
# AP1's finer list above .9, then the same .9^k tail (the floor .9^6 is unchanged). Each
# scale is still relative to the original seed and the first certified one is used; the default list is untouched.
TRANSFER_FINE_ALPHAS = (1., .99, .98, .96, .94, .92)+TRANSFER_BACKTRACK_ALPHAS[1:]
TRANSFER_FINE_POLICY = "q1-transfer-fine-above-09-then-scale09-v1"
TRANSFER_ALPHA_LISTS = {"fine09": (TRANSFER_FINE_POLICY, TRANSFER_FINE_ALPHAS)}
NATIVE_START_SCALE_RANGE = (.1, 1.)     # T83 (E90): native transfer tries beta * .9^k; beta = 1 is T38g exactly


def validate_start_scale(beta):
    """T83: a finite real beta in [0.1, 1] (numbers.Real, not bool); returns it as float."""
    import numbers
    if not isinstance(beta, numbers.Real) or isinstance(beta, (bool, np.bool_)):
        raise InputValidationError("start scale must be a real number")
    value = float(beta)
    if not np.isfinite(value) or not NATIVE_START_SCALE_RANGE[0] <= value <= NATIVE_START_SCALE_RANGE[1]:
        raise InputValidationError("start scale must be in [0.1, 1]")
    return value


def prepare_transfer_problem(*, seed, grid, fixed_image, moving_image, weights,
                             memory=None, certificate_config=None, budget=None, observer=None, barrier_activation=None,
                             barrier_basis="samples", curvature_weight=0., data_weight=None, start_scale=1.,
                             active_weights=None, alpha_list=None):
    """Restore a transfer initializer, never an outer commit or a clipped map.

    T83 (E90): with start_scale = beta < 1 the scales tried are beta * .9^k (each relative to the original seed);
    beta = 1 is the T38g sequence exactly (the first attempt uses the seed itself).
    AP2: alpha_list = "fine09" tries TRANSFER_FINE_ALPHAS instead (start_scale must be 1); None keeps T38g exactly.

    Each scale is relative to the original seed. No monotonicity of JD in alpha
    is assumed; the first fully certified trial is used, not a claimed maximum.
    Returns (problem or None, last attempted field, diagnostic report).
    """
    if budget is not None and not callable(budget):
        raise InputValidationError("transfer budget must be callable")
    config = EnclosureCertificateConfig(delta=0.) if certificate_config is None else certificate_config
    if not isinstance(config, EnclosureCertificateConfig) or config.delta != 0.:
        raise InputValidationError("transfer recovery requires whole-cell strict JD>0 certification")
    start_scale = validate_start_scale(start_scale)
    if alpha_list is not None and (not isinstance(alpha_list, str) or alpha_list not in TRANSFER_ALPHA_LISTS):
        raise InputValidationError("alpha_list must be None or one of " + ", ".join(sorted(TRANSFER_ALPHA_LISTS)))
    if alpha_list is not None and start_scale != 1.:
        raise InputValidationError("alpha_list and start_scale are exclusive")
    original = readonly_copy(as_float64_array(seed, "transfer seed", grid.shape+(3,)))
    field = original
    started = perf_counter()
    attempts = []
    alphas = TRANSFER_BACKTRACK_ALPHAS if start_scale == 1. else tuple(start_scale*a for a in TRANSFER_BACKTRACK_ALPHAS)
    policy = TRANSFER_BACKTRACK_POLICY
    if alpha_list is not None:
        policy, alphas = TRANSFER_ALPHA_LISTS[alpha_list]

    def finish(status, problem=None, alpha=None):
        return problem, field, dict(policy=policy, status=status,
            attempts=attempts, accepted_alpha=alpha,
            displacement_loss_fraction=None if alpha is None else 1-alpha,
            elapsed_seconds=perf_counter()-started, role="initializer_only_NOT_outer_commit",
            **({} if start_scale == 1. else {"start_scale": start_scale}),
            **({} if alpha_list is None else {"alpha_list": alpha_list}))

    for index, alpha in enumerate(alphas):
        if budget is not None and budget():
            return finish("time_budget")
        field = original if index == 0 and start_scale == 1. else readonly_copy(alpha*original)
        proofs = []
        tick = perf_counter()
        try:
            problem = make_problem(grid=grid, fixed_image=fixed_image, moving_image=moving_image,
                u_ref=field, weights=weights, memory=memory, certificate_config=config,
                barrier_location="closed27_uniform", reference_certificate_observer=proofs.append,
                **({"barrier_activation": barrier_activation} if barrier_activation is not None else {}),
                **({"barrier_basis": barrier_basis} if barrier_basis != "samples" else {}),
                **({"curvature_weight": curvature_weight} if curvature_weight else {}),
                **({"data_weight": data_weight} if data_weight is not None else {}),   # T80
                **({"active_weights": active_weights} if active_weights is not None else {}))   # T89 (S30-G4)
        except InputValidationError:
            if not proofs or proofs[0].status == "certified_feasible":
                raise  # bad configuration, image or implementation; do not mask it
            problem = None
        proof = proofs[0]
        domain = None
        if problem is not None and barrier_basis != "samples":
            # T68: a certified initializer must also lie inside the Bernstein barrier domain;
            # otherwise it is treated like a failed proof and the next alpha is tried.
            from .bernstein_barrier import min_coefficient
            domain = min_coefficient(problem, field, fast=True)   # same decision as the exact check
            if not domain > config.delta:
                problem = None
        row = dict(attempt_index=index, alpha=alpha, certificate=asdict(proof),
                   certificate_passed=proof.status == "certified_feasible" if domain is not None else problem is not None,
                   elapsed_seconds=perf_counter()-tick)
        if domain is not None:
            row["bernstein_min_coefficient"] = domain
            row["barrier_domain_passed"] = bool(domain > config.delta)
        attempts.append(row)
        if observer is not None:
            observer(deepcopy(row), field)
        if budget is not None and budget():
            return finish("time_budget")  # even a certified initializer may be too late
        if problem is not None:
            return finish("ready", problem, alpha)
        if domain is None and proof.reason not in ("JACOBIAN_VIOLATION", "subdivision_budget", "INVALID_DOMAIN"):
            return finish("rejected")
    return finish("rejected")


@dataclass(frozen=True)
class FourLevelResult3D:
    success: bool
    status: str
    reason: str
    levels: tuple[dict, ...]
    last_level: int | None
    last_result: SequentialResult3D | None
    elapsed_seconds: float

    @property
    def native_result(self):
        return self.last_result if self.last_level == 3 else None


def _curvature_levels(curvature_weight):
    """G2: a scalar alpha (T76: the same on every level; default 0.) or a tuple of four finite alpha >= 0, one per
    level. The scalar is used as is (same object on every level), so the default path is unchanged."""
    if isinstance(curvature_weight, (tuple, list)):
        import numbers
        if len(curvature_weight) != 4 or any(not isinstance(a, numbers.Real) or isinstance(a, (bool, np.bool_))
                                             for a in curvature_weight):
            raise InputValidationError("curvature_weight: a number or a tuple of four numbers (one per level)")
        levels = tuple(float(a) for a in curvature_weight)
        if not all(np.isfinite(a) and a >= 0 for a in levels):
            raise InputValidationError("curvature weights must be finite and nonnegative")
        return levels
    return (curvature_weight,)*4


def solve_four_level(*, grid, fixed_image, moving_image, weights, inner_configs, outer_configs,
                     solver="pg", memory=None, certificate_config=None, backend=None,
                     observer=None, inner_observer=None, level_observer=None,
                     level_budgets=None, budget=None, advance_stalled_levels=False, transfer_backtracking=False, barrier_activation=None,
                     first_level=0, barrier_basis="samples", curvature_weight=0., data_weight_policy=None,
                     native_start_scale=1., transfer_smoothing=None, native_practical_stop=None,
                     preoptimizer=None, preoptimizer_taper=4, active_set=None, native_outer_gate=None,
                     native_transfer_alphas=None, native_reference=None, native_solver=None):
    """Four explicit per-level budgets; zero start, strict JD>0, fixed weights per level.

    native_solver (T92 D2-e, ablation switch): None (every level uses `solver`, unchanged) or "lm" -- the native level's
    inner solver is the library barrier-GGN LM (inner_configs[3] an LMConfig3D); only with solver "pg" and the
    pre-optimizer route (levels 0-2 are not solved). F, E, the certificate, E/F acceptance, forcing and GF are unchanged.

    native_transfer_alphas (AP2, explicit option): None (T38g .9^k, unchanged) or "fine09" -- the native transfer of the
    pre-optimizer seed tries TRANSFER_FINE_ALPHAS; requires preoptimizer (and so transfer_backtracking) and no native start scale.

    preoptimizer (T88, S30-G3, explicit option): a callable (fixed, moving, grid) -> (u float64 with zero faces, record),
    an UNCERTIFIED initializer of the native level. Levels 0-2 are then not solved (skipped_by_preoptimizer); the
    native seed is the P0c boundary-band taper (preoptimizer_taper nodes) of that field, certified by the transfer
    routine (alpha back-off, whole-cell JD > 0; requires transfer_backtracking), never an outer commit.

    data_weight_policy (T80): None, or a dict for data_weight_from_policy; the weight of every level is
    computed from that level's fixed image, so the data term D uses q m on every level.

    weights: one Weights3D for every level (unchanged default), or a tuple of four Weights3D
    (E80 T78: explicit per-level geometry weights); level L's problem, including its transfer
    certification, uses weights[L].

    observer(level, problem, start, inner, row, candidate) follows the original
    outer acceptance lifecycle; inner_observer(level, row) is diagnostic only.
    level_observer(event, grid, readonly_u) may save prepared/finished/rejected
    transfer fields. Transfer and zero-baseline probes are NOT formal commits.
    By default all failed levels abort. Explicit advance_stalled_levels lets
    exhausted coarse searches transfer only their last certified formal state
    (or certified initializer if no commit). Failed candidates are never used;
    native failures, numerical failures and rejected transfers still stop.
    Explicit transfer_backtracking restores a failed transfer by certified
    displacement scaling, without salvaging any failed optimization candidate.
    Optional monotone level_budgets allow time-ended layers to transfer their
    last formal state; the global budget stops all further level work.
    """
    started = perf_counter()
    # E48 explicit hierarchy probe: levels below first_level are not solved; the
    # first solved level starts from the original zero field (T37 zero start).
    if type(first_level) is not int or first_level not in (0, 1, 2):
        raise InputValidationError("first_level must be 0 (default), 1 or 2")
    if not isinstance(transfer_backtracking, bool):
        raise InputValidationError("transfer_backtracking must be boolean")
    if not isinstance(advance_stalled_levels, bool):
        raise InputValidationError("advance_stalled_levels must be boolean")
    native_start_scale = validate_start_scale(native_start_scale)     # T83 (E90)
    if preoptimizer is not None:
        if not callable(preoptimizer) or not transfer_backtracking or first_level != 0 or transfer_smoothing is not None:
            raise InputValidationError("preoptimizer: a callable, with transfer_backtracking, first_level 0 and no transfer_smoothing")
        if isinstance(preoptimizer_taper, bool) or not isinstance(preoptimizer_taper, (int, np.integer)) or preoptimizer_taper < 1:
            raise InputValidationError("preoptimizer_taper must be a positive integer")
    if native_start_scale != 1. and not transfer_backtracking:
        raise InputValidationError("native_start_scale requires transfer_backtracking (T38g route)")
    if native_transfer_alphas is not None:                           # AP2: the pre-optimizer route's native transfer only
        if not isinstance(native_transfer_alphas, str) or native_transfer_alphas not in TRANSFER_ALPHA_LISTS:
            raise InputValidationError("native_transfer_alphas must be None or one of " + ", ".join(sorted(TRANSFER_ALPHA_LISTS)))
        if preoptimizer is None or native_start_scale != 1.:
            raise InputValidationError("native_transfer_alphas requires the pre-optimizer route and no native start scale")
    if native_practical_stop is not None:                            # H17
        from .practical_stop import PracticalStop
        if not isinstance(native_practical_stop, PracticalStop):
            raise InputValidationError("native_practical_stop must be a PracticalStop")
    if native_outer_gate is not None:                                # FX1 F4 (throughput-v35): the native level only
        from .outer_gate import OuterGate
        if not isinstance(native_outer_gate, OuterGate) or budget is None:
            raise InputValidationError("native_outer_gate must be an OuterGate and requires a budget")
    if native_reference is not None:                                 # T92 (D2-c/c'): the native level only
        if native_reference not in ("fixed", "identity") or preoptimizer is None or active_set is not None \
                or native_practical_stop is not None:
            raise InputValidationError("native_reference is 'fixed' or 'identity' on the pre-optimizer route "
                                       "(no active set, no practical stop)")
    if native_solver is not None and (native_solver != "lm" or solver != "pg" or preoptimizer is None
                                      or native_reference is not None or active_set is not None):
        raise InputValidationError("native_solver: only 'lm', with solver pg and the pre-optimizer route "
                                   "(no native reference mode, no active set)")   # T92 D2-e
    if active_set is not None:                                       # T89 (S30-G4): the native level only
        from .active_set import parse_active_set
        active_set = parse_active_set(active_set)
    if transfer_smoothing is not None:                               # A19
        from .transfer_smoothing import validate_transfer_smoothing
        transfer_smoothing = validate_transfer_smoothing(transfer_smoothing)
        if not transfer_backtracking:
            raise InputValidationError("transfer_smoothing requires transfer_backtracking (T38g route)")
    if budget is not None and not callable(budget):
        raise InputValidationError("global budget must be callable")
    curvature_by_level = _curvature_levels(curvature_weight)          # G2: one alpha per level (default: the same)
    timed_levels = level_budgets is not None
    level_budgets = (None,)*4 if level_budgets is None else tuple(level_budgets)
    if len(level_budgets) != 4 or any(b is not None and not callable(b) for b in level_budgets):
        raise InputValidationError("four optional callable level budgets required")
    grids = four_level_grids(grid)
    fixed = readonly_copy(as_float64_array(fixed_image, "native fixed", grid.shape))
    moving = readonly_copy(as_float64_array(moving_image, "native moving", grid.shape))
    if isinstance(weights, Weights3D):
        weights_by_level = (weights,)*4
    elif isinstance(weights, tuple) and len(weights) == 4 and all(isinstance(w, Weights3D) for w in weights):
        weights_by_level = weights
    else:
        raise InputValidationError("expected Weights3D (or a tuple of four Weights3D, one per level)")
    if solver not in ("pg", "lm", "jd_qp"):
        raise InputValidationError("expected solver pg, lm or jd_qp")
    inner_configs, outer_configs = tuple(inner_configs), tuple(outer_configs)
    from .solvers.sequential_qp import SequentialQPConfig3D
    inner_type = {"pg":PGConfig3D,"lm":LMConfig3D,"jd_qp":SequentialQPConfig3D}[solver]
    level_types = (inner_type,)*3+(LMConfig3D if native_solver == "lm" else inner_type,)
    if (len(inner_configs) != 4 or len(outer_configs) != 4
            or any(not isinstance(c, t) for c, t in zip(inner_configs, level_types))
            or any(not isinstance(c, SequentialConfig3D) for c in outer_configs)):
        raise InputValidationError("four matching inner and Sequential outer configurations required")
    certificate_config = EnclosureCertificateConfig(delta=0.) if certificate_config is None else certificate_config
    if not isinstance(certificate_config, EnclosureCertificateConfig) or certificate_config.delta != 0.:
        raise InputValidationError("four-level branch requires full-cell enclosure certification with delta=0")
    records, last, last_level = [], None, None
    # T80: one data weight per level, all derived from the native fixed image (review: a weight recomputed from the
    # anti-aliased coarse image would dilate the support and voxelise sigma).
    level_weights = None if data_weight_policy is None else data_weight_levels(data_weight_policy, fixed, grid, grids)

    def finish(status, reason):
        return FourLevelResult3D(status == "converged_native_gradient", status, reason,
                                 tuple(records), last_level, last, perf_counter()-started)

    def emit(event, level_grid, field):
        if level_observer is not None:
            level_observer(deepcopy(event), level_grid, field)

    for level, level_grid in enumerate(grids):
        if level < first_level:
            records.append(dict(level=level, shrink_factor=FACTORS[level], shape_zyx=level_grid.shape,
                                grid_id=level_grid.grid_id, status="skipped_by_explicit_first_level",
                                reason=f"first_level={first_level}", formal_commits=0))
            continue
        if preoptimizer is not None and level < 3:
            # T88: the coarse levels are replaced by the uncertified pre-optimizer (nothing solved here)
            records.append(dict(level=level, shrink_factor=FACTORS[level], shape_zyx=level_grid.shape,
                                grid_id=level_grid.grid_id, status="skipped_by_preoptimizer",
                                reason="T88 preoptimizer initializer", formal_commits=0))
            continue
        if budget is not None and budget():
            return finish("time_budget_exhausted", "Global budget; no later level attempted")
        tick = perf_counter()
        if level == 3:
            images = fixed, moving
            preprocessing = dict(policy="native_unchanged", smoothing=False, intensity_rescale=False)
        else:
            a, ga, metadata = downsample_array(fixed, grid, level_grid.shape)
            b, gb, other = downsample_array(moving, grid, level_grid.shape)
            if ga.grid_id != level_grid.grid_id or gb.grid_id != ga.grid_id or metadata != other:
                raise InputValidationError("fixed/moving pyramid grids or policies differ")
            images = a, b
            preprocessing = metadata
        seed = (readonly_copy(np.zeros(level_grid.shape+(3,))) if last is None else
                prolong_displacement(last.final_state.u, last.final_problem.grid, level_grid))
        smoothing = None
        pre_record = None
        if preoptimizer is not None and level == 3:
            # T88: the native seed is the tapered pre-optimizer field; its certification follows below (alpha back-off)
            from .preopt import boundary_taper
            tick_pre = perf_counter()
            raw, pre_record = preoptimizer(fixed, moving, grid)
            raw = readonly_copy(as_float64_array(raw, "preoptimizer field", grid.shape+(3,)))
            if not zero_boundary(raw) or not np.all(np.isfinite(raw)):
                raise InputValidationError("preoptimizer field must be finite with zero faces")
            seed, _ = boundary_taper(raw, grid, preoptimizer_taper)
            pre_record = dict(pre_record, raw_hash=array_hash(raw), taper_L=int(preoptimizer_taper), seed_hash=array_hash(seed),
                              seconds_total=perf_counter()-tick_pre, role="uncertified_preoptimizer_initializer_NOT_outer_commit")
            emit(dict(level=level, event="preoptimizer_done", role="uncertified_preoptimizer_initializer", preoptimizer=pre_record),
                 level_grid, seed)
        if last is not None and transfer_smoothing is not None:
            from .transfer_smoothing import applies, smooth_transfer_seed
            if applies(transfer_smoothing, level):
                # A19: the transfer certifies alpha_k * S_sigma(P u) (the T38g/T83 scales, each relative to S_sigma P u)
                tick_smooth = perf_counter()
                smoothed = smooth_transfer_seed(seed, level_grid, transfer_smoothing)
                seconds = perf_counter()-tick_smooth                     # the smoothing only, not the hashes
                smoothing = dict(transfer_smoothing, seed_hash=array_hash(seed), smoothed_hash=array_hash(smoothed),
                                 seconds=seconds)
                seed = smoothed
        level_data_weight = {} if level_weights is None else {"data_weight": level_weights[level]}   # T80
        active_record = None
        if active_set is not None and level == 3:
            # T89 (S30-G4): the native level's fixed node weights of the PG direction (from the native images only)
            from .active_set import active_weights as _active_weights
            tick_active = perf_counter()
            w_active, active_record = _active_weights(fixed, moving, grid, active_set)
            active_record = dict(active_record, seconds=perf_counter()-tick_active)
            level_data_weight = dict(level_data_weight, active_weights=w_active)
        row = dict(level=level, shrink_factor=FACTORS[level], shape_zyx=level_grid.shape,
                   spacing_xyz=level_grid.spacing, grid_id=level_grid.grid_id,
                   preprocessing=preprocessing, weights=asdict(weights_by_level[level]),
                   input_role=("uncertified_preoptimizer_initializer" if pre_record is not None else
                               "original_zero" if last is None else "previous_level_final_committed_or_initial"),
                   **({} if pre_record is None else dict(preoptimizer=pre_record)),
                   inherited_outer_commits=0 if last is None else last.counts["outer_accepted"],
                   native_metrics=level == 3, inner=asdict(inner_configs[level]), outer=asdict(outer_configs[level]),
                   **({} if active_record is None else dict(active_set=active_record)))   # T89 (S30-G4)

        if transfer_backtracking and (last is not None or pre_record is not None):
            problem, seed, recovery = prepare_transfer_problem(seed=seed, grid=level_grid,
                fixed_image=images[0], moving_image=images[1], weights=weights_by_level[level], memory=memory,
                certificate_config=certificate_config, budget=budget,
                **({"barrier_activation": barrier_activation} if barrier_activation is not None else {}),
                **({"barrier_basis": barrier_basis} if barrier_basis != "samples" else {}),
                **({"curvature_weight": curvature_by_level[level]} if curvature_by_level[level] else {}),
                **level_data_weight,
                **({"start_scale": native_start_scale} if level == 3 and native_start_scale != 1. else {}),   # T83
                **({"alpha_list": native_transfer_alphas} if level == 3 and native_transfer_alphas is not None else {}),   # AP2
                observer=lambda attempt, field: emit(dict(attempt, level=level,
                    event="transfer_attempt", role="uncommitted_transfer_attempt"), level_grid, field))
            if smoothing is not None:
                recovery["smoothing"] = smoothing                    # A19
            row["transfer_recovery"] = recovery
            if recovery["status"] != "ready":
                timed_out = recovery["status"] == "time_budget"
                reason = ("GLOBAL_WALL_BUDGET_DURING_TRANSFER" if timed_out else
                          "BARRIER_DOMAIN" if recovery["attempts"] and recovery["attempts"][-1].get("barrier_domain_passed") is False
                          else recovery["attempts"][-1]["certificate"]["reason"])
                row.update(status="time_budget_before_solver" if timed_out else "transfer_rejected",
                           reason=reason, elapsed_seconds=perf_counter()-tick)
                if recovery["attempts"]:
                    row["transfer_certificate"] = recovery["attempts"][-1]["certificate"]
                records.append(row)
                if not timed_out:
                    emit(dict(row, event="transfer_rejected", role="uncommitted_transfer"), level_grid, seed)
                return finish("time_budget_exhausted" if timed_out else "transfer_rejected", f"level {level}: {reason}")
            proofs = [problem.controls.certificate]
            row["input_role"] = ("certified_scaled_preoptimizer_initializer_NOT_outer_commit" if pre_record is not None
                                 else "certified_scaled_transfer_initializer_NOT_outer_commit")
        else:
            proofs = []
            try:
                problem = make_problem(grid=level_grid, fixed_image=images[0], moving_image=images[1],
                    u_ref=seed, weights=weights_by_level[level], memory=memory, certificate_config=certificate_config,
                    barrier_location="closed27_uniform", reference_certificate_observer=proofs.append,
                **({"barrier_activation": barrier_activation} if barrier_activation is not None else {}),
                **({"barrier_basis": barrier_basis} if barrier_basis != "samples" else {}),
                **({"curvature_weight": curvature_by_level[level]} if curvature_by_level[level] else {}),
                **level_data_weight)
            except InputValidationError:
                if not proofs or proofs[0].status == "certified_feasible":
                    raise  # configuration/programming errors are not feasibility failures
                row.update(status="transfer_rejected", reason=proofs[0].reason,
                           transfer_certificate=asdict(proofs[0]), elapsed_seconds=perf_counter()-tick)
                records.append(row)
                emit(dict(row, event="transfer_rejected", role="uncommitted_transfer"), level_grid, seed)
                return finish("transfer_rejected", f"level {level}: {proofs[0].reason}")
        row.update(transfer_certificate=asdict(proofs[0]), problem_id=problem.problem_id,
                   controls_id=problem.controls.controls_id, prepare_seconds=perf_counter()-tick)
        if isinstance(curvature_weight, (tuple, list)):
            row["curvature_weight"] = problem.curvature_weight              # G2: the built problem's alpha
        emit(dict(row, event="level_prepared", role="certified_level_initializer_not_outer_commit"), level_grid, seed)
        if budget is not None and budget():
            row.update(status="time_budget_before_solver", reason="GLOBAL_WALL_BUDGET",
                       elapsed_seconds=perf_counter()-tick)
            records.append(row)
            return finish("time_budget_exhausted", "Global budget during level preparation")
        # A fresh solver call discards BB2 history and LM damping/PCG state.
        # The backend itself invalidates its caches when problem/grid changes.
        last = solve_sequential(problem, restrict_free(seed, level_grid),
            solver=native_solver if level == 3 and native_solver is not None else solver,   # T92 D2-e
            inner_config=inner_configs[level], config=outer_configs[level], backend=backend,
            baseline_at_zero=True,
            **({"budget": lambda: ((budget is not None and budget()) or
                (level_budgets[level] is not None and level_budgets[level]()))}
               if timed_levels or budget is not None else {}),
            **({"practical_stop": native_practical_stop} if level == 3 and native_practical_stop is not None else {}),   # H17
            **({"outer_gate": native_outer_gate} if level == 3 and native_outer_gate is not None else {}),   # FX1 F4
            **({"reference_mode": native_reference} if level == 3 and native_reference is not None else {}),   # T92
            observer=None if observer is None else lambda p, s, i, r, c: observer(level, p, s, i, deepcopy(r), c),
            inner_observer=None if inner_observer is None else lambda r: inner_observer(level, r))
        last_level = level
        row.update(status=last.status, reason=last.reason, success=last.success,
                   formal_commits=last.counts["outer_accepted"], counts=last.counts,
                   GF0=last.GF_initial, GF=last.GF_final, GF_threshold=last.GF_threshold,
                   zero_mse=last.native_mse, final_mse=last.final_mse, mse_ratio=last.mse_ratio,
                   history=last.history, resolved_config=last.resolved_config,
                   final_certificate=None if last.final_state is None else asdict(last.final_state.certificate),
                   elapsed_seconds=perf_counter()-tick)
        coarse_stalled = bool(advance_stalled_levels and level < 3
                              and last.final_state is not None and _coarse_search_stalled(last))
        if coarse_stalled:
            validate_state(last.final_problem, last.final_state)
            proof = last.final_state.certificate
            if (proof.state_hash != array_hash(last.final_state.u)
                    or proof.min_jd_lower_bound is None or proof.min_jd_lower_bound <= 0):
                raise InputValidationError("stalled level needs a matching strictly positive committed certificate")
        if advance_stalled_levels:
            row.update(transition_policy="advance_certified_coarse_search_stall_v1",
                       coarse_search_stalled=coarse_stalled,
                       transfer_source="final_committed_or_certified_initializer_never_inner_candidate")
        records.append(row)
        field = seed if last.final_state is None else last.final_state.u
        emit(dict(row, event="level_completed", role="level_final_not_necessarily_native"), level_grid, field)
        if last.status == "gpu_certificate_mismatch":
            # E101 (T84 plan D, D-a): an abnormal end of the run, not a failure of the level's search
            return finish("gpu_certificate_mismatch", f"level {level}: {last.reason}")
        allowed = TRANSFER_STATUSES + (("outer_time_budget",) if timed_levels or budget is not None else ()) \
            + (("practical_stop",) if level == 3 and native_practical_stop is not None else ())   # H17
        if (last.status not in allowed and not coarse_stalled) or last.final_state is None:
            return finish("level_failed", f"level {level}: {last.status}: {last.reason}")
        if budget is not None and budget() and not (level == 3 and last.status == "practical_stop"):
            return finish("time_budget_exhausted", "Global budget; last completed level only")
    if last.success:
        return finish("converged_native_gradient", "Original native zero-baseline GF criterion; not global optimality")
    if last.status == "practical_stop":
        return finish("native_practical_stop", last.reason)          # H17: success false, never convergence
    return finish("native_budget_exhausted", "Four levels completed, native gradient has not converged")
