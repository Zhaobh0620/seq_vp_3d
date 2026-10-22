"""Explicit GPU trial/stable-reduction policy, with CPU hard certification.

The 4096-eps guard is a conservative numerical policy, NOT an enclosure of
all spline/geometry/roundoff error. Critical comparisons are independently
re-evaluated by the legacy CPU longdouble path. No sampled GPU JD proves
feasibility, and no change to the mathematical objective is made here.
"""
from dataclasses import dataclass
from copy import copy
import copy as _copy_module
from contextlib import contextmanager
import hashlib
import weakref
from time import perf_counter

import numpy as np
import torch

from ...energy_comparison import EnergyReduction
from ...types import InvalidTrial
from ...validation import (InputValidationError, StateMismatchError, as_float64_array,
                           finite_real, readonly_copy, require_finite_result)
from ..boundary import inject_free
from ..energy_comparison import (stable_energy_reduction as cpu_reduction,
                                 stable_energy_reductions as cpu_reductions)
from ..feasibility import certify_map
from ..interpolation import check_map_domain
from ..linearization import build_linearization, cpu_outer_recheck, energy_gradient
from ..problem import array_hash, array_hashes, Controls3D, QUADRATURE_ID, with_controls
from .. import problem as _identity

SCALAR_BAND = 1e-6       # E104: decision band of the device BB2 tail (relative to |y||iy|)
DEVICE_FORCING_BAND = 1e-6     # E109 (B14): |GE_device - target| <= band * target -> CPU recheck
DEVICE_FORCING_CHECKS = 3      # E109: forcing-only triggers per level grid also rechecked on the CPU and compared
DEVICE_FORCING_RECHECK = 50    # E109: and every 50th after them
DEVICE_FORCING_LIMIT = 1e-8    # E109: relative device-vs-CPU difference of a norm above which the option is switched off
                               # (band/100: outside the band a GE this close to the CPU GE gives the same forcing decision;
                               # rounding alone measured up to ~3e-12 where GE is a small, cancellation-dominated norm)
SCALAR_CHECKS = 3        # E104: first uses per (grid, kind) also compared with the host product
SHIFT_CHECKS = 2         # E105: first uses per level grid compared bit for bit with the NumPy expression
FP32_CHECKS = 3          # E106: first uses per level grid with the float32 direction compared to the float64 one
FP32_RECHECK = 50        # E106 (review S3): and every 50th float32 use per level grid after those
FP32_LIMIT = 1e-4        # E106: relative l2/max/slope difference above which float32 is switched off for the run
FP32_GUARD_FACTOR, FP32_GUARD_FLOOR = 100., 1e-5   # E106: descent guard c_d = max(100 * measured l2 rel, 1e-5)
NORM_FLOOR, NORM_CEILING = 2.**-450, 2.**450   # E104 review: each norm of the band away from under/overflow
from ..residuals import (AcceptedState3D, EnergyTerms3D, TrialState3D,
                        evaluate_trial as cpu_evaluate, promote_trial as cpu_promote,
                        validate_state)
from ..solvers.common import gradient_norm_mass
from ..work_control import WorkBudgetExceeded, checkpoint, operation
from .hybrid import HybridBackend, _PendingDiagonal, timed
from .torch_ops import TorchLinearization, TorchModel, barrier, geometry
from .witness_screen import screen
from .cpu_comparison import fused_cpu_reductions, _validate_bindings, bind_theta_u, POLICY as CPU_FUSED_POLICY
from .device_batch import DeviceBatchConfig

POLICY_ID = "gpu-stable4096-cpu-critical-v1"
GGN_REUSE_POLICY = "ggn-diagonal-reuse-v1"           # GG2 (explicit option, metric only)
GPU_NEAR_BOUNDARY_POLICY = "gpu-near-boundary-guarded-v1"
REFERENCE_REUSE_POLICY = "certified-self-reference-reuse-v1"
SELECTIVE_KERNEL_POLICY = "selective-kernels-v1"
OVERLAP_POLICY = "overlap-certificate-v1"
FUSED_POLICY = "triton-fused-q1-v1"
GUARD_FACTOR = 4096.
BERNSTEIN_GUARD = 1e-11   # T68: device-vs-CPU rounding band of the Bernstein coefficients
CRITICAL_GUARDS = 8.
# T90 (S30-G5): the float32 screening band -- per term, relative to that term's values of the two states compared:
# FP32_BAND_SOFT * 2^-24 for the JD/curl penalties (measured on T0C native states: <= 2 ulps), FP32_BAND_BARRIER * 2^-24 for
# the barrier (measured <= 732 ulps at the transfer state, <= 30 later: the near-activation logs cancel). The first
# FP32_SCREEN_CHECKS screened evaluations per level grid and then every FP32_SCREEN_RECHECK-th are also computed in
# float64 and every term and difference must lie within its band (else the screening is switched off for the run).
FP32_BAND_SOFT = 64.
FP32_BAND_BARRIER = 4096.
FP32_EPS = 2.**-24
FP32_SCREEN_CHECKS = 3
FP32_SCREEN_RECHECK = 50
MICROSTEP_RELATIVE = 1e-8
NEAR_BOUNDARY_RELATIVE = 1e-7


@dataclass(frozen=True)
class AcceleratedTrialState3D(TrialState3D):
    evaluation_backend_id: str


@dataclass(frozen=True)
class AcceleratedAcceptedState3D(AcceptedState3D):
    evaluation_backend_id: str


def _value_geometry(model, local, D, *, with_curl):
    if model._selective_kernels:
        return model.geometry(local, D, cofactor=False, with_curl=with_curl)
    return geometry(local, D)


def _sample_fingerprint(warped):
    return warped.shape, warped.dtype.str, array_hash(warped)


class _PendingFingerprint:
    """E89: the same fingerprint tuple, hashed on the shared hash pool and joined at first use.
    Only for immutable arrays (callers check the writeable flags synchronously first)."""
    def __init__(self, arrays, *, single=False):
        self._heads = tuple((value.shape, value.dtype.str) for value in arrays)
        self._futures = tuple(_identity.submit_hash(value) for value in arrays)
        self._single, self._value = single, None

    def get(self):
        if self._value is None:
            value = tuple(head+(future.result(),) for head, future in zip(self._heads, self._futures))
            self._value = value[0] if self._single else value
        return self._value


def _resolved(fingerprint):
    return fingerprint.get() if isinstance(fingerprint, _PendingFingerprint) else fingerprint


class _DeviceState(TorchLinearization):
    def __init__(self, model, state, u, warped, gradient, sample_fingerprint=None):
        self.model, self.state = model, state
        self._u, self._warped, self._b = u, warped, gradient
        self.sample_fingerprint = sample_fingerprint


def _barrier_difference(a, b, delta, activation=None):
    """Same cancellation-avoiding branches as T13, using device float64."""
    if activation is not None:
        a = torch.minimum(a, torch.full_like(a, activation))
        b = torch.minimum(b, torch.full_like(b, activation))
        scale = activation-delta
        ra, rb = (activation-a)/scale, (activation-b)/scale
        h = (b-a)/scale
        loga = torch.where(ra < .5, torch.log1p(-ra), torch.log((a-delta)/scale))
        x = (b-a)/(a-delta)
        logratio = torch.where(x >= -.5, torch.log1p(x), torch.log((b-delta)/(a-delta)))
        return h*(ra+rb)*loga-rb*rb*logratio
    change = b-a
    x = change/(a-delta)
    polynomial = torch.full_like(x, 1/14)
    for k in range(13, 1, -1):
        polynomial = (-1.)**k/k+x*polynomial
    small = x*(a-1)/(1-delta)+x*x*polynomial
    far = change/(1-delta)-torch.log1p(x)
    negative = change/(1-delta)-torch.log((b-delta)/(a-delta))
    return torch.where(x.abs() < .01, small, torch.where(x >= -.5, far, negative))


def _mass_norm_checked(gradient, omega):
    """E95: solvers.common.gradient_norm_mass(gradient, omega) for a C-contiguous float64 array whose values were
    just checked finite: the same BLAS nrm2 and the same checks on its result, without rescanning the array."""
    from scipy.linalg import norm as _norm
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        value = float(_norm(gradient.ravel(), check_finite=False)/np.sqrt(omega))
    require_finite_result(value, "mass gradient norm")
    if value == 0. and np.any(gradient != 0):
        raise FloatingPointError("mass gradient norm: nonzero result underflowed")
    return value


MISMATCH_DETAILS = 8          # E101: plan D mismatch records kept in full


class NodeEstimates:
    """E100: per tile of one node_min_closed27_jd call (tiles of `size` cells in order), the candidate cells (device),
    their closed27 rows (host) and the estimator's values. gather() makes the device calls of the call's loop,
    estimate() its host estimator calls, in the same order on the same rows; the first exception either meets is
    kept with its tile number (later tiles are not computed) and at() raises it at that tile."""

    def __init__(self, keep_rows=False):
        self.tiles, self.error, self.keep_rows, self.n_tiles = [], None, keep_rows, 0

    def gather(self, cand_mask, j27, size, n):
        self.n_tiles = -(-n//size)
        for number, start in enumerate(range(0, n, size)):
            end = min(start+size, n)
            try:
                cand = torch.nonzero(cand_mask[start:end]).flatten()
                rows = j27(cand+start).cpu().numpy() if cand.numel() else None
            except Exception as error:
                self.error = (number, error)
                return
            self.tiles.append([start, end, cand, rows, None])

    def estimate(self, estimator):
        for number, tile in enumerate(self.tiles):
            if tile[3] is None:
                continue
            try:
                tile[4] = estimator(tile[3])
            except Exception as error:
                self.error = (number, error)
                del self.tiles[number:]
                return
            if not self.keep_rows:
                tile[3] = None

    def at(self, number, start, end):
        if self.error is not None and number == self.error[0]:
            raise self.error[1]
        if number >= len(self.tiles) or self.tiles[number][:2] != [start, end]:
            raise InputValidationError("E100 prepared node estimates do not match this call's tiles")
        tile = self.tiles[number]
        return tile[2], tile[4]

    def candidates(self):
        return sum(int(tile[2].numel()) for tile in self.tiles)

    def informative(self):
        """A first-use comparison that can show a wrong row offset: candidates on a tile after the first one (or a
        single-tile grid with candidates). Review E100: candidates in tile 0 alone pass with `j27(cand)` for
        `j27(cand+start)`."""
        with_candidates = [tile[0] for tile in self.tiles if tile[2].numel()]
        return bool(with_candidates) and (self.n_tiles == 1 or any(start > 0 for start in with_candidates))

    def same_as(self, trace):
        """Every tile's candidate cells, rows and estimator values equal, bit for bit, those the v17 loop recorded in
        `trace` (rows kept only). The values themselves are compared: a changed value need not move the node
        minimum or the stats (review: an ulp change of one value passed a node-and-stats-only check)."""
        if not self.keep_rows or self.error is not None or len(trace) != len(self.tiles):
            return False
        bits = lambda a: np.ascontiguousarray(a, dtype=np.float64).view(np.uint64)
        for tile, entry in zip(self.tiles, trace):
            if entry["start"] != tile[0] or not torch.equal(entry["cand"], tile[2]):
                return False
            if (entry["rows"] is None) != (tile[3] is None) or (entry.get("mu") is None) != (tile[4] is None):
                return False
            if tile[3] is not None and not (np.array_equal(bits(entry["rows"].cpu().numpy()), bits(tile[3]))
                                            and tile[4].shape == tuple(entry["mu"].shape)
                                            and np.array_equal(bits(entry["mu"].cpu().numpy()), bits(tile[4]))):
                return False
        return True


def parse_ggn_reuse(spec):
    """GG2: 'theta=T,period=K' (both keys, nothing else) -> (theta, period); theta in [1.25, 100], K an integer in
    [1, 1000]. Contract draft: docs/draft_GG2_ggn_reuse_contract.md (theta = 2, K = 4 pre-set)."""
    if not isinstance(spec, str):
        raise InputValidationError("GG2: the spec is a string 'theta=T,period=K'")
    parts = {}
    for item in spec.split(","):
        key, sep, value = item.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or key not in ("theta", "period") or key in parts or not value:
            raise InputValidationError("GG2: the spec is 'theta=T,period=K'")
        parts[key] = value
    if set(parts) != {"theta", "period"}:
        raise InputValidationError("GG2: the spec needs both theta and period")
    try:
        theta = float(parts["theta"])
        period = int(parts["period"])
    except ValueError:
        raise InputValidationError("GG2: theta must be a number and period an integer") from None
    if not (np.isfinite(theta) and 1.25 <= theta <= 100.) or not 1 <= period <= 1000:
        raise InputValidationError("GG2: theta in [1.25, 100] and period in [1, 1000] required")
    return theta, period


class _PendingReuseDiagonal(_PendingDiagonal):
    """GG2: a partial build whose device checks fail (non-finite or <= 0) is replaced by the default diagonal of the same
    linearization (counted; the kept locals are dropped) and checked again as before."""

    def __init__(self, backend, tensor, fallback):
        super().__init__(backend, tensor)
        self._fallback = fallback

    def result(self):
        fallback, self._fallback = self._fallback, None
        try:
            return super().result()
        except FloatingPointError:
            if fallback is None:
                raise
        self._tensor = fallback()
        return super().result()


class _NodePrepared:
    """E100: node_min_prepare's outcome for one node_min_jd call (its identity, the fused pass's outcome, estimates)."""

    def __init__(self, dl, activation, compiled, fused, record, estimator):
        self.dl, self.u, self.version = dl, dl._u, dl._u._version
        self.activation, self.compiled, self.fused = activation, bool(compiled), fused
        self._record, self._estimator = record, estimator
        self.failure = self.minima = self.cand = self.estimates = None
        self.nonfinite = self.estimated = False

    def estimate(self):
        """The host estimator over the gathered rows (the caller runs it while the GGN diagonal kernel runs)."""
        if self.estimates is not None:
            tick = perf_counter()
            self.estimates.estimate(self._estimator)
            self._record["estimate_seconds"] += perf_counter()-tick
        self.estimated = True

    def matches(self, dl, activation, compiled):
        """Only for the call it was prepared for, after estimate() (review E100: rows without values are never used)."""
        return (self.estimated and self.dl is dl and self.u is dl._u and self.version == dl._u._version
                and activation is not None and self.activation == activation and self.compiled == bool(compiled))

    def rebind(self, dl):
        """E103: the next outer's device state of the same map (reference reuse keeps the very device u tensor, unchanged);
        False when the tensor or its version differ (then the prepared outcome is not used)."""
        if self.u is not dl._u or self.version != dl._u._version:
            return False
        self.dl = dl
        return True


def _same_prepared(a, b):
    """E103: two first parts of one node_min_jd call agree bit for bit -- kind (failure type / nonfinite / values),
    activation and compiled flag, node minima (int64 bits), candidate mask, every tile's range, candidate cells and
    estimator values, and the tile at which an estimator error stopped (with its type)."""
    kind = lambda p: "failure" if p.failure is not None else ("nonfinite" if p.nonfinite else "values")
    if kind(a) != kind(b) or a.activation != b.activation or a.compiled != b.compiled:
        return False
    if kind(a) == "failure":
        return type(a.failure) is type(b.failure)
    if kind(a) == "nonfinite":
        return True
    if not (a.minima.shape == b.minima.shape and torch.equal(a.minima.view(torch.int64), b.minima.view(torch.int64))):
        return False
    if (a.cand is None) != (b.cand is None) or (a.cand is not None and not torch.equal(a.cand, b.cand)):
        return False
    ea, eb = a.estimates, b.estimates
    if (ea is None) != (eb is None):
        return False
    if ea is None:
        return True
    if ea.n_tiles != eb.n_tiles or len(ea.tiles) != len(eb.tiles) or (ea.error is None) != (eb.error is None):
        return False
    if ea.error is not None and (ea.error[0] != eb.error[0] or type(ea.error[1]) is not type(eb.error[1])):
        return False
    bits = lambda v: np.ascontiguousarray(v, dtype=np.float64).view(np.uint64)
    for ta, tb in zip(ea.tiles, eb.tiles):
        if ta[:2] != tb[:2] or not torch.equal(ta[2], tb[2]) or (ta[4] is None) != (tb[4] is None):
            return False
        if ta[4] is not None and not (np.shape(ta[4]) == np.shape(tb[4]) and np.array_equal(bits(ta[4]), bits(tb[4]))):
            return False
    return True


class _FusedNodeFailure(Exception):
    """E94: a failure of the fused node JD kernel itself (not of the torch code around it)."""
    def __init__(self, error):
        super().__init__(repr(error))
        self.error = error


def _float_bits(value):
    """Records compared exactly: floats by their bits (-0.0 != 0.0), containers recursively."""
    import struct
    if isinstance(value, float):
        return ("f", struct.pack("<d", value))
    if isinstance(value, dict):
        return {k: _float_bits(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_float_bits(v) for v in value]
    return value


class AcceleratedHybridBackend(HybridBackend):
    """Opt-in evaluation policy; old HybridBackend remains unchanged.

    At most one base linearization and one candidate live on the device
    (E89 boundary_reuse: plus the outer start, kept until its E/F comparison).
    Frozen Gauss references are recomputed by tile, never cached full-volume.
    Diagnostic counters distinguish actual CPU fallbacks from GPU decisions.
    Optional prepared reuse retains one factory-owned readonly host state;
    same problem identity and exact content fingerprints are checked on hits.
    near_boundary_cpu=False skips ONLY the certificate-bound forced CPU
    comparison. All hard proofs, microstep/critical/numeric fallbacks and
    near-target CPU gradient checks remain; GPU guards are not error proofs.
    reuse_reference=True rebinds an owned same-map proof/sampling cache after
    self-reference refresh, with a new model view and new evaluation identity.
    """
    def __init__(self, device="cuda", *, witness_screen=True, reuse_prepared=False, fused_cpu=False, near_boundary_cpu=True, reuse_reference=False, device_batch=None, selective_kernels=False, overlap_certificate=False, fused_kernels=False, fused_jd_root=False, fused_diagonal=False, cpu_gradient_band=None, fused_hvp=False, lean_host=False, cpu_gradient_margin=None, lean_identity=False, boundary_reuse=False, screen_prefilter=False, fused_node_jd=False, lean_checks=False, step_extrema=False, early_warped=False, device_candidate=False, root_screen=False, lean_certificate=False, barrier_skip=False, laplacian_memo=False, fused_spline=False, tuned_launch=False, skip_reference_first=False, estimator_overlap=False, top4_select=False, outer_reuse=False, gpu_edge_hull=False, commit_recheck=False, laplacian_cache=False, fused_laplacian=False, metric_prestart=False, device_scalars=False, fused_shift=False, direction_prefetch=False, fp32_poisson=False, fused_recheck=False,
                 device_forcing=False, active_set_skip=False, fp32_screen=False, early_trial_hash=False):
        if not isinstance(witness_screen, bool):
            raise InputValidationError("witness_screen must be boolean")
        if not isinstance(reuse_prepared, bool):
            raise InputValidationError("reuse_prepared must be boolean")
        if not isinstance(fused_cpu, bool):
            raise InputValidationError("fused_cpu must be boolean")
        if not isinstance(near_boundary_cpu, bool):
            raise InputValidationError("near_boundary_cpu must be boolean")
        if not isinstance(reuse_reference, bool) or (reuse_reference and not reuse_prepared):
            raise InputValidationError("reuse_reference requires boolean and reuse_prepared=True")
        if device_batch is not None and not isinstance(device_batch, DeviceBatchConfig):
            raise InputValidationError("device_batch must be DeviceBatchConfig or None")
        if not isinstance(selective_kernels, bool) or (selective_kernels and device_batch is None):
            raise InputValidationError("selective_kernels requires boolean and device_batch")
        if not isinstance(overlap_certificate, bool):
            raise InputValidationError("overlap_certificate must be boolean")
        if not isinstance(fused_kernels, bool):
            raise InputValidationError("fused_kernels must be boolean")
        self.fused_kernels = fused_kernels
        if not isinstance(fused_jd_root, bool):
            raise InputValidationError("fused_jd_root must be boolean")
        self.fused_jd_root = fused_jd_root   # E55b: T66 helper only; evaluation/comparison ids unchanged
        if not isinstance(fused_diagonal, bool) or (fused_diagonal and not fused_kernels):
            raise InputValidationError("fused_diagonal must be boolean and requires fused_kernels")
        self.fused_diagonal = fused_diagonal   # E65: metric-only GGN diagonal kernel
        if not isinstance(fused_hvp, bool) or (fused_hvp and not fused_kernels):
            raise InputValidationError("fused_hvp must be boolean and requires fused_kernels")
        self.fused_hvp = fused_hvp             # E72: GGN action kernel (T72 curvature, LM)
        # E69: below band*target the device gradient norm decides the inner target directly (roundoff
        # ~1e-12 relative cannot cross a 1% band); within (band, 1.01]*target the CPU recheck is unchanged.
        if cpu_gradient_band is not None and not (isinstance(cpu_gradient_band, float) and .5 <= cpu_gradient_band < 1.):
            raise InputValidationError("cpu_gradient_band must be None or a float in [.5, 1)")
        self.cpu_gradient_band = cpu_gradient_band
        self.device_gradient_targets = 0
        # E76: CPU recheck only within a relative margin m of the target (device-vs-CPU gradient
        # norms differ by ~1e-12 relative, so outside 1 +- m the decision cannot differ).
        if cpu_gradient_margin is not None and not (isinstance(cpu_gradient_margin, float)
                                                    and 0. < cpu_gradient_margin <= .01):
            raise InputValidationError("cpu_gradient_margin must be None or a float in (0, .01]")
        if cpu_gradient_margin is not None and cpu_gradient_band is not None:
            raise InputValidationError("cpu_gradient_margin replaces cpu_gradient_band; give one")
        self.cpu_gradient_margin = cpu_gradient_margin
        self.device_gradient_above = 0
        # Review E76: every CPU recheck also measures |G_cpu/G_device - 1|; above margin/100 the
        # margin is abandoned for the rest of the run (v8 band) and the event is recorded.
        self.gradient_margin_monitor = dict(rechecks=0, max_relative_difference=0., abandoned=False)
        if not isinstance(lean_host, bool):
            raise InputValidationError("lean_host must be boolean")
        self.lean_host = lean_host               # E76: bit-identical host-side shortcuts
        if not isinstance(lean_identity, bool) or (lean_identity and not lean_host):
            raise InputValidationError("lean_identity must be boolean and requires lean_host")
        # E89 (P01): identity facts recorded where they are established (certificate digest and
        # predicates, theta/u binding at construction), copy-free owned trial arrays and downloads,
        # background fingerprints. Same values and decisions; the checks become memo lookups.
        self.lean_identity = lean_identity
        if not isinstance(boundary_reuse, bool) or (boundary_reuse and (not lean_identity or device_batch is None)):
            raise InputValidationError("boundary_reuse must be boolean and requires lean_identity and device_batch")
        # E89 (P05): outer/inner boundary reuse of exactly equal device results: the last inner gradient
        # is computed as the joint outer triple (kept for the outer norms), the first inner gradient after
        # a refresh is the outer vector of the same device data (reference terms vanish exactly), and the
        # outer start's device state stays resident for the outer E/F comparison.
        self.boundary_reuse = boundary_reuse
        self._joint_stash = self._outer_stash = self._first_gradient = self._outer_start = None
        if not isinstance(screen_prefilter, bool) or (screen_prefilter and not witness_screen):
            raise InputValidationError("screen_prefilter must be boolean and requires witness_screen")
        # E93: a fused Triton pass with an a-priori error bound skips the witness-screen tiles whose
        # legacy closed27 minimum is proven > delta; all other tiles run the legacy screen unchanged,
        # so results are identical. CUDA with Triton only (otherwise the legacy screen runs). The first
        # use on every level grid is also checked against the legacy per-cell minima; any violation of
        # the bound switches the prefilter off for the rest of the run and is recorded.
        self.screen_prefilter = screen_prefilter
        self._prefilter = None
        self._prefilter_verified = set()
        if not isinstance(fused_node_jd, bool):
            raise InputValidationError("fused_node_jd must be boolean")
        if not isinstance(lean_checks, bool) or (lean_checks and not lean_identity):
            raise InputValidationError("lean_checks must be boolean and requires lean_identity")
        # E95: no rescans of arrays a compiled pass already proved finite (trial entry, gradient norm), and the
        # device gradient kept for the device-side scalings of the equilibrated Poisson direction. Bit-identical.
        self.lean_checks = lean_checks
        self._device_gradient = None
        # E97 (throughput-v15): each switch requires lean_checks; every value, decision and record equals v14's.
        for name, value in (("step_extrema", step_extrema), ("early_warped", early_warped),
                            ("device_candidate", device_candidate), ("root_screen", root_screen)):
            if not isinstance(value, bool) or (value and not lean_checks):
                raise InputValidationError(name+" must be boolean and requires lean_checks")
        self.step_extrema, self.early_warped = step_extrema, early_warped
        self.device_candidate, self.root_screen = device_candidate, root_screen
        self._step_extrema = self._staged_warped = self._axpy_note = self._direction_twin = None
        self._device_candidate_verified = set()
        self.step_extrema_record = dict(policy="fused-axpy-step-extrema-v1", noted=0, microstep_hits=0)
        self.early_warped_record = dict(policy="early-warped-staging-v1", staged=0, published=0)
        self.device_candidate_record = dict(policy="device-candidate-u-v1", built=0, verified_grids=0, mismatches=0,
                                            fallbacks=0, disabled=False)
        # E98 (throughput-v16): each switch requires lean_checks; every value, decision and record equals v15's.
        for name, value in (("lean_certificate", lean_certificate), ("barrier_skip", barrier_skip),
                            ("laplacian_memo", laplacian_memo)):
            if not isinstance(value, bool) or (value and not lean_checks):
                raise InputValidationError(name+" must be boolean and requires lean_checks")
        if lean_certificate and not overlap_certificate:
            raise InputValidationError("lean_certificate requires overlap_certificate")
        self.lean_certificate, self.barrier_skip, self.laplacian_memo = lean_certificate, barrier_skip, laplacian_memo
        self.barrier_skip_record = dict(policy="closed27-inactive-point-skip-v1", calls=0, compared=0, verified=[],
                                        mismatches=0, errors=0, disabled=False, unsupported=0)
        self.laplacian_memo_record = dict(policy="curvature-laplacian-memo-v1", hits=0, misses=0)
        # E99 (throughput-v17): each switch requires lean_checks (skip_reference_first also barrier_skip); every value,
        # decision and record equals v16's. The checks against the v16 paths are recorded per kernel and grid.
        for name, value in (("fused_spline", fused_spline), ("tuned_launch", tuned_launch),
                            ("skip_reference_first", skip_reference_first)):
            if not isinstance(value, bool) or (value and not lean_checks):
                raise InputValidationError(name+" must be boolean and requires lean_checks")
        if skip_reference_first and not barrier_skip:
            raise InputValidationError("skip_reference_first requires barrier_skip")
        self.fused_spline, self.tuned_launch, self.skip_reference_first = fused_spline, tuned_launch, skip_reference_first
        if fused_spline:
            from .spline_fused import new_record as _spline_record
            self.fused_spline_record = _spline_record()
        if tuned_launch:
            self.tuned_launch_record = dict(policy="tuned-launch-v1", calls=0, compared=0, verified=[], mismatches=0,
                                            errors=0, disabled=False, reference_only=0, registers={}, low_registers=0)
        if skip_reference_first:
            self.barrier_skip_record["reference_only"] = 0      # E99: unverified calls that ran the v15 kernel only
        # E100 (throughput-v18): each switch requires lean_checks; every value, decision and record equals v17's.
        for name, value in (("estimator_overlap", estimator_overlap), ("top4_select", top4_select),
                            ("outer_reuse", outer_reuse)):
            if not isinstance(value, bool) or (value and not lean_checks):
                raise InputValidationError(name+" must be boolean and requires lean_checks")
        if estimator_overlap and not fused_node_jd:
            raise InputValidationError("estimator_overlap requires fused_node_jd")
        if outer_reuse and not boundary_reuse:
            raise InputValidationError("outer_reuse requires boundary_reuse")
        self.estimator_overlap, self.top4_select, self.outer_reuse = estimator_overlap, top4_select, outer_reuse
        self._node_prepared, self._overlap_verified = None, set()
        self.estimator_overlap_record = dict(policy="node-estimator-overlap-v1", prepared=0, used=0, discarded=0,
                                             compared=0, verified=[], mismatches=0, disabled=False, prepare_errors=0,
                                             first_part_seconds=0., estimate_seconds=0.)
        self.top4_select_record = None
        if top4_select:
            from ..solvers.jd_affine_compiled import new_top4_record, top4_selfcheck
            self.top4_select_record = new_top4_record()
            top4_selfcheck(self.top4_select_record)            # synthetic rows: ties, NaN, +-0, +-inf, subnormals
        self._outer_reuse_verified, self._outer_norm = set(), None
        self.outer_reuse_record = dict(policy="outer-vector-reuse-v1", device_copies=0, verified_grids=0,
                                       mismatches=0, errors=0, disabled=False, scans_skipped=0, norms_reused=0)
        # E101 (throughput-v19): T84 (the certificate's edge-hull margins on the GPU, bit for bit, inside the inner
        # solve's scope only; plan D: every outer candidate re-certified on the CPU) and the Laplacian memo with a
        # reference slot (C1/C2) and fused kernel (C3). Every value, decision and recorded identity equals v18's.
        for name, value in (("gpu_edge_hull", gpu_edge_hull), ("commit_recheck", commit_recheck),
                            ("laplacian_cache", laplacian_cache), ("fused_laplacian", fused_laplacian)):
            if not isinstance(value, bool) or (value and not lean_checks):
                raise InputValidationError(name+" must be boolean and requires lean_checks")
        if gpu_edge_hull and not lean_certificate:
            raise InputValidationError("gpu_edge_hull requires lean_certificate (the owned certificate front end)")
        if commit_recheck and not gpu_edge_hull:
            # T84 plan D re-certifies what the GPU margins certified; E102 (user 2026-10-09): the margins may run
            # without it (throughput-v20), never the other way round
            raise InputValidationError("commit_recheck (T84 plan D) requires gpu_edge_hull")
        if (laplacian_cache or fused_laplacian) and not laplacian_memo:
            raise InputValidationError("laplacian_cache and fused_laplacian require laplacian_memo")
        self.gpu_edge_hull, self.commit_recheck = gpu_edge_hull, commit_recheck
        self.laplacian_cache, self.fused_laplacian = laplacian_cache, fused_laplacian
        self._gpu_scope, self._gpu_hull, self._recheck_pool, self._prestart = 0, None, None, None
        if gpu_edge_hull:
            from .edge_hull_gpu import new_record as _hull_record
            self.gpu_certificate_record = _hull_record()
            if not commit_recheck:
                self.gpu_certificate_record["commit_recheck"] = False     # E102: plan D off (absent in v19 = on)
        if fused_laplacian:
            from .laplacian_fused import new_record as _laplacian_record
            self.fused_laplacian_record = _laplacian_record()
        # E103 (throughput-v21, M1): the next metric build's host parts (the T73/T69 estimator over the prepared rows
        # and the T59 shift alpha) start in a background thread at the inner step budget; the next build uses them
        # when its base is the same frozen u and image gradient. Every value, decision and record equals v20's.
        if not isinstance(metric_prestart, bool) or (metric_prestart and not (estimator_overlap and boundary_reuse)):
            raise InputValidationError("metric_prestart must be boolean and requires estimator_overlap and boundary_reuse")
        # E104 (throughput-v22, G3b narrowed): the inner slope g.p and the BB2 tail y.(p_old - p) from the device copies
        # of g and p (round-off level, with 1e-6 decision bands); the gradient's finite check on its device copy (exact).
        if not isinstance(device_scalars, bool) or (device_scalars and not (lean_checks and device_candidate
                                                                            and lean_identity)):
            raise InputValidationError("device_scalars must be boolean and requires lean_checks, device_candidate and "
                                       "lean_identity")
        self.device_scalars, self._twins, self._scalar_checks = device_scalars, [], {}
        # E105 (throughput-v23): the T59 shift alpha of the metric build by host_kernels.shift_mean, NumPy's own
        # reduction order bit for bit; the first SHIFT_CHECKS uses per level grid also evaluate the NumPy expression
        # and compare the bits (a mismatch switches it off for the run; the NumPy value is used).
        if not isinstance(fused_shift, bool) or (fused_shift and metric_prestart):
            # review E105 (M2): with metric_prestart the E103 alpha path (and its first-use check) would be skipped
            raise InputValidationError("fused_shift must be boolean and is not combined with metric_prestart")
        self.fused_shift, self._shift_checks = fused_shift, {}
        # E105 (throughput-v23): after the gradient kernels of an inner step the same gradient's equilibrated Poisson
        # direction (s*g, M^-1, -(s*x), the two finite flags: v22's kernels in v22's order) is queued on the main
        # stream, and the gradient's download and finite check run on a side stream meanwhile. Same values.
        if not isinstance(direction_prefetch, bool) or (direction_prefetch and not lean_checks):
            raise InputValidationError("direction_prefetch must be boolean and requires lean_checks")
        self.direction_prefetch = direction_prefetch
        self._prefetch, self._prefetch_poisson, self._prefetch_stream, self._prefetch_flag = None, None, None, None
        self._prefetch_flags32 = None
        self.direction_prefetch_record = dict(policy="poisson-prefetch-v1", enqueued=0, used=0, nonfinite_pre=0,
                                              dropped={}, unsupported=0, errors=0, error_types=[], disabled=False)
        # E106 (throughput-v24, C14 phase A; user 2026-10-11): the Poisson metric's spectral solve M^-1 in float32 (the
        # direction only); everything else in float64. Range, finiteness, first-use and descent guards (docs/04 E106).
        if not isinstance(fp32_poisson, bool) or (fp32_poisson and not lean_checks):
            raise InputValidationError("fp32_poisson must be boolean and requires lean_checks")
        self.fp32_poisson, self._fp32_checks, self._fp32_uses, self._fp32_note = fp32_poisson, {}, {}, None
        self.fp32_poisson_record = dict(policy="fp32-poisson-spectral-v1", uses=0, fp32_used=0, range_fallbacks=0,
                                        nonfinite_fallbacks=0, descent_reruns=0, descent_rerun_failures=0, checks=0,
                                        max_l2_rel=0., max_max_rel=0., max_slope_rel=0., violations=0,
                                        violation_details=[], errors=0, error_types=[], disabled=False)
        # E108 (throughput-v26): the outer CPU gradient recheck of sequential.norms (backend.cpu_state, then the CPU
        # outer gradient vectors) in one pass: one sampling, each tile's geometry once for the energy and the vectors,
        # local products on the tile threads, the sampling while the first tiles run. The same vectors bit for bit.
        if not isinstance(fused_recheck, bool):
            raise InputValidationError("fused_recheck must be boolean")
        self.fused_recheck = fused_recheck
        self.fused_recheck_record = dict(policy="fused-cpu-outer-recheck-v1", rechecks=0)
        # T89 (S30-G4, throughput-v28): with a problem that carries active weights (the native active set), the
        # energy blocks, gradient cells/nodes and sampled nodes of the frozen part are cached instead of recomputed;
        # every value equals the full computation's (first uses compared; a mismatch switches it off for the run).
        if not isinstance(active_set_skip, bool) or (active_set_skip and not (lean_checks and fused_kernels and fused_spline)):
            raise InputValidationError("active_set_skip must be boolean and requires lean_checks, fused_kernels and fused_spline")
        self.active_set_skip = active_set_skip
        if active_set_skip:
            from .triton_q1 import new_active_set_record
            self.active_set_skip_record = new_active_set_record()
        # T90 (S30-G5, throughput-v29/v30): the trial's soft-geometry and
        # barrier energy terms from the float32 kernels; the comparison guard is widened to the float64 band and any
        # comparison inside the band is redone from the float64 device evaluation; FP64 energies are recorded for every
        # outer candidate (exact_energy). Decisions outside the band equal the float64 decisions.
        if not isinstance(fp32_screen, bool) or (fp32_screen and not (lean_checks and fused_kernels)):
            raise InputValidationError("fp32_screen must be boolean and requires lean_checks and fused_kernels")
        self.fp32_screen = fp32_screen
        self._last_band, self._screen_uses = None, {}
        self.fp32_screen_record = dict(policy="fp32-energy-screen-v1", band_soft=FP32_BAND_SOFT, band_barrier=FP32_BAND_BARRIER,
                                       screened=0, fp64_full=0,
                                       fp64_rechecks=0, band_hits=0, checks=0, verified=[], max_rel=0., mismatches=0,
                                       mismatch_details=[], errors=0, disabled=False, unsupported=0, exact_energies=0)
        # v41 (H2, throughput-v41, implementation only): the SHA-256 of a trial's
        # frozen u starts on its own pool as soon as u is injected (before the domain check, the device u and the screen);
        # the certificate uses that digest instead of hashing u itself -- the same function on the same bytes, so every
        # certificate field is identical. A witness-rejected trial's digest is computed and dropped.
        if not isinstance(early_trial_hash, bool) or (early_trial_hash and not (lean_certificate and lean_identity)):
            raise InputValidationError("early_trial_hash must be boolean and requires lean_certificate and lean_identity")
        self.early_trial_hash = early_trial_hash
        self._trial_digest = self._digest_pool = None
        self.early_trial_hash_record = dict(policy="trial-u-sha256-at-inject-v1", submitted=0, used=0, unused=0, errors=0)
        # E109 (throughput-v27, B14, rounding level; user 2026-10-12): an outer recheck triggered by the forcing target
        # alone (GF not within 1.01x of the threshold) keeps the device norms when |GE - target| > 1e-6 target; the
        # first DEVICE_FORCING_CHECKS such uses per level grid and every DEVICE_FORCING_RECHECK-th are rechecked on the
        # CPU and compared (relative 1e-8 = band/100; a violation switches the option off for the run).
        if not isinstance(device_forcing, bool) or (device_forcing and not lean_checks):
            raise InputValidationError("device_forcing must be boolean and requires lean_checks")
        self.device_forcing, self._forcing_uses, self.device_forcing_last = device_forcing, {}, None
        self.device_forcing_record = dict(policy="device-forcing-decision-v1", band=DEVICE_FORCING_BAND,
                                          limit=DEVICE_FORCING_LIMIT, triggers=0, device_used=0, band_cpu=0,
                                          disabled_cpu=0, checks=0, checked_grids={}, max_rel=[0., 0., 0.],
                                          violations=0, violation_details=[], disabled=False)
        self.fused_shift_record = dict(policy="numpy-order-shift-mean-v1", numpy=np.__version__, uses=0, used=0,
                                       small=0, checks=0, fallbacks=0, mismatches=0, mismatch_details=[], errors=0,
                                       error_types=[], disabled=False)
        self.device_scalars_record = dict(policy="device-bb2-tail-v1", tails=0, yys=0, yy_band=0, finite_device=0,
                                          fallbacks=0, stale=0, checks=0, max_yy_rel=0., violations=0, violation_details=[],
                                          errors=0, error_types=[], disabled=False)
        self.metric_prestart, self._metric_prestart, self._prestart_pool = metric_prestart, None, None
        self._prestart_verified, self._prestart_taken = set(), None
        self.metric_prestart_record = dict(policy="next-metric-host-prestart-v1", prestarts=0, used=0, dropped=0,
                                           verified_grids=[], compared=0, mismatches=0, mismatch_details=[],
                                           errors=0, error_types=[], disabled=False, wait_seconds=0.,
                                           first_part_seconds=0., alpha_used=0, estimates_used=0,
                                           uninformative_comparisons=0)
        # E94 (G2/C4): the T62/T69 per-cell measure of node_min_jd from one fused Triton pass that reproduces the
        # torch path bit for bit (measured cuBLAS/torch orders written out); candidate cells, the estimator, the
        # scatter and the stats stay the torch code. CUDA with Triton only. Every level grid's first use is
        # compared bit for bit with the torch path (node array and stats); a mismatch, a failure of the kernel
        # itself, or an unexpected table switches it off for the rest of the run and is recorded.
        self.fused_node_jd = fused_node_jd
        self._fused_node = None
        self._fused_node_verified = set()
        self.fused_node_jd_record = dict(policy="triton-fused-node-jd-v1", calls=0, verified=[], mismatches=0,
                                         nonfinite_fallbacks=0, unavailable=0, unsupported_calls=0, errors=0,
                                         error_types=[], disabled=False)
        self.screen_prefilter_record = dict(policy="triton-screen-prefilter-v1", calls=0, tiles=0, tiles_selected=0,
                                            verified_grids=0, mismatches=0, unavailable=0, errors=0, error_types=[],
                                            disabled=False)
        # Review E89: on the fused (Triton) path the vanishing reference terms of (a) need J == J_r bitwise from two
        # differently written expressions, which depends on the compiler's contraction choices. So the first reuse
        # of (a) and of (b) on every level grid is checked on the device against the vector it replaces; on any
        # mismatch that reuse is switched off for the rest of the run (the computed vector is used), and recorded.
        self.boundary_reuse_record = {name: dict(reused=0, verified_grids=0, mismatches=0, nonfinite=0, disabled=False)
                                      for name in ("first_gradient", "last_gradient")}
        self._boundary_verified = {"first_gradient": set(), "last_gradient": set()}
        self.fused_status = {}
        super().__init__(device)
        self.device.lean_host = lean_host
        # E50: CPU whole-cell proof runs concurrently with device sampling/energy
        # of the SAME candidate; results are used only if the proof certifies.
        self.overlap_certificate = overlap_certificate
        self._certificate_pool = None
        self.selective_kernels = selective_kernels
        self.witness_screen = witness_screen
        self.reuse_prepared = reuse_prepared
        self.fused_cpu = fused_cpu
        self.near_boundary_cpu = near_boundary_cpu
        self.reuse_reference = reuse_reference
        self.device_batch = device_batch
        self._fused_comparison = None
        if device_batch is not None:
            self.fusion_counts = dict(energy_differences=0, comparison_hits=0, joint_gradients=0)
        self.comparison_policy_id = POLICY_ID if near_boundary_cpu else POLICY_ID+"-"+GPU_NEAR_BOUNDARY_POLICY
        self.reuse_counts = {"trial_hits": 0, "promotion_hits": 0}
        if reuse_reference:
            self.reuse_counts.update(reference_hits=0, reference_misses=0)
        self._prepared = None
        self.backend_id = "torch-float64-"+str(self.device.device)+"-"+POLICY_ID
        if fused_cpu:
            self.backend_id += "-"+CPU_FUSED_POLICY
        if not near_boundary_cpu:
            self.backend_id += "-"+GPU_NEAR_BOUNDARY_POLICY
        if reuse_reference:
            self.backend_id += "-"+REFERENCE_REUSE_POLICY
        if device_batch is not None:
            self.backend_id += "-"+device_batch.policy_id
            self.comparison_policy_id += "-"+device_batch.policy_id
        if selective_kernels:
            self.backend_id += "-"+SELECTIVE_KERNEL_POLICY
            self.comparison_policy_id += "-"+SELECTIVE_KERNEL_POLICY
        if overlap_certificate:
            self.backend_id += "-"+OVERLAP_POLICY
        if fused_kernels:
            self.backend_id += "-"+FUSED_POLICY
            self.comparison_policy_id += "-"+FUSED_POLICY
        self.evaluation_backend_id = self.backend_id
        self._candidate = None
        self.comparison_counts = {"gpu": 0, "cpu": 0}
        self.evaluation_counts = {"gpu": 0, "cpu_fallback": 0, "promotions": 0,
                                  "cpu_state": 0, "witness_rejections": 0,
                                  "witness_scans": 0, "witness_numeric_failures": 0,
                                  "full_certifications": 0}
        self.last_comparison = None

    def _model_for(self, problem):
        if self._model is None or self._model.problem is not problem:
            self._model = TorchModel(self.device, problem,
                **(dict(tile_size=self.device_batch.tile_size) if self.device_batch is not None else {}),
                **(dict(selective_kernels=True) if self.selective_kernels else {}),
                **(dict(fused_kernels=True) if self.fused_kernels else {}),
                **(dict(fused_jd_root=True) if self.fused_jd_root else {}),
                **(dict(fused_diagonal=True) if self.fused_diagonal else {}),
                **(dict(fused_hvp=True) if self.fused_hvp else {}),
                **(dict(barrier_skip=self.barrier_skip_record) if self.barrier_skip else {}),
                **(dict(laplacian_memo=self.laplacian_memo_record) if self.laplacian_memo else {}),   # E98
                **(dict(fused_spline=self.fused_spline_record) if self.fused_spline else {}),        # E99
                **(dict(tuned_launch=self.tuned_launch_record) if self.tuned_launch else {}),
                **(dict(skip_reference_first=True) if self.skip_reference_first else {}),
                **(dict(laplacian_cache=True) if self.laplacian_cache else {}),                       # E101
                **(dict(fused_laplacian=self.fused_laplacian_record) if self.fused_laplacian else {}),
                **(dict(active_set_skip=self.active_set_skip_record) if self.active_set_skip else {}))   # T89
            if self.fused_diagonal:
                key = "fused_diagonal_models" if self._model._fused_diagonal else "fused_diagonal_fallback_models"
                self.fused_status[key] = self.fused_status.get(key, 0)+1
            if self.fused_hvp:
                key = "fused_hvp_models" if self._model._fused_hvp else "fused_hvp_fallback_models"
                self.fused_status[key] = self.fused_status.get(key, 0)+1
                self._model.fused_counts = self.fused_status       # E72: fused_hvp_calls accumulate here
            if self.fused_jd_root:
                key = "fused_jd_root_models" if self._model._jd_root is not None else "fused_jd_root_fallback_models"
                self.fused_status[key] = self.fused_status.get(key, 0)+1
            if self.fused_kernels:
                # Record what actually ran: the label alone does not prove fused execution.
                key = "fused_models" if self._model._fused is not None else "fused_fallback_models"
                self.fused_status[key] = self.fused_status.get(key, 0)+1
            self._lin = self._candidate = None
            self._prepared = None
            self._fused_comparison = None
            self._joint_stash = self._outer_stash = self._first_gradient = self._outer_start = None
            if self.lean_checks:
                self._device_gradient = self._outer_proven = None      # E96: no device copy outlives its model
                self._step_extrema = self._staged_warped = self._axpy_note = self._direction_twin = None   # E97
                self._node_prepared = self._outer_norm = None       # E100
                self._twins = []                                   # E104: no device copy outlives its model
                if self._prefetch is not None:
                    self.prefetch_drop("model_reset")              # E105
                self._fp32_note = None                             # E106
                self._prefetch_poisson = None
        return self._model

    @contextmanager
    def _stage(self, name):
        """Opt-in synchronized wall-time scopes, including failed operations."""
        if self.device_batch is None:
            yield
            return
        self.device.synchronize()
        started = perf_counter()
        try:
            with operation(name):
                yield
                self.device.synchronize()
        finally:
            self.device.synchronize()
            self.timings[name] = self.timings.get(name, 0.)+perf_counter()-started

    @contextmanager
    def _timed_only(self, name):
        """E89: the synchronized wall time of _stage, added to the same timings key, without a work operation."""
        self.device.synchronize()
        started = perf_counter()
        try:
            yield
            self.device.synchronize()
        finally:
            self.device.synchronize()
            self.timings[name] = self.timings.get(name, 0.)+perf_counter()-started

    @staticmethod
    def _prepared_fingerprint(problem, state):
        arrays = (state.theta, state.u, state.warped, state.image_grad, problem.controls.u_ref)
        if any(value.flags.writeable for value in arrays):
            raise StateMismatchError("prepared reuse requires factory-owned readonly arrays")
        digests = array_hashes(arrays)  # E48: concurrent, identical SHA-256 digests
        return tuple((value.shape, value.dtype.str, digest) for value, digest in zip(arrays, digests))

    def _prepared_match(self, problem, theta):
        if not self.reuse_prepared or self._prepared is None or self._prepared[0] is not problem:
            return None
        _, state, fingerprint = self._prepared
        if not (self._same_theta(theta, state.theta) if self.lean_identity else np.array_equal(theta, state.theta)):
            return None
        self._validate(problem, state)
        fingerprint = _resolved(fingerprint)
        if self._prepared_fingerprint(problem, state) != fingerprint:
            raise StateMismatchError("prepared state/reference modified since promotion")
        # Array equality alone treats +0 and -0 alike. Only the exact bytes
        # already certified/evaluated may reuse a proof and its image samples.
        if array_hash(theta) != fingerprint[0][2]:
            return None
        return state

    @staticmethod
    def _same_theta(theta, other):
        """E89: np.array_equal(theta, other). The same object with a recorded finite fact has no NaN, so it
        equals itself; otherwise the compiled == comparison stops at the first difference."""
        from .. import host_kernels
        if theta is other and _identity.has_fact(theta, _identity.FACT_FINITE):
            return True
        if host_kernels.enabled(theta.size) and host_kernels._flat_ok(theta, other):
            return host_kernels.equal(theta, other)
        return np.array_equal(theta, other)

    def _pending_prepared_fingerprint(self, problem, state):
        """E89: _prepared_fingerprint hashed in the background (same tuple, joined in _prepared_match);
        the readonly requirement is still checked here, before anything is stored."""
        arrays = (state.theta, state.u, state.warped, state.image_grad, problem.controls.u_ref)
        if any(value.flags.writeable for value in arrays):
            raise StateMismatchError("prepared reuse requires factory-owned readonly arrays")
        return _PendingFingerprint(arrays)

    def _validate(self, problem, state):
        validate_state(problem, state)
        policy = getattr(state, "evaluation_backend_id", None)
        if policy is not None:
            expected = hashlib.sha256((problem.problem_id+state.certificate.state_hash
                                       +self.backend_id).encode()).hexdigest()
            if policy != self.backend_id or state.eval_id != expected:
                raise StateMismatchError("device evaluation policy/state identity mismatch")

    def refresh_reference(self, problem, state):
        """Rebind one owned, unchanged map; None requests the old fresh path.

        Proof/sample reuse is about this map, NOT about old-reference energies.
        Sharing immutable device constants never mutates the previous model.
        RecordingBackend still sees the new promoted identity, not a commit.
        """
        if not self.reuse_reference:
            raise InputValidationError("reference reuse must be explicitly enabled")
        self._fused_comparison = None
        checkpoint("reference_reuse_binding")
        self._validate(problem, state)
        data = next((d for d in (self._lin, self._candidate)
                     if d is not None and d.state is state and d.model.problem is problem), None)
        if (self._prepared is None or self._prepared[0] is not problem
                or self._prepared[1] is not state or data is None):
            self.reuse_counts["reference_misses"] += 1
            return None
        self._prepared_match(problem, state.theta)  # detects mutated owned arrays
        _validate_bindings(problem, (state,))  # same full-map proof contract as CPU reuse
        identity = hashlib.sha256((problem.grid.grid_id+QUADRATURE_ID
                                   +state.certificate.state_hash).encode()).hexdigest()
        controls = Controls3D(state.u, state.certificate, identity, problem.grid.grid_id,
                              QUADRATURE_ID, problem._gauss)
        refreshed = with_controls(problem, controls)
        model = copy(data.model)
        model.problem, model._ref = refreshed, data._u
        energy = EnergyTerms3D(state.energy.image, 0., 0., 0., state.energy.barrier, state.energy.curvature)
        identity = hashlib.sha256((refreshed.problem_id+state.certificate.state_hash
                                   +self.backend_id).encode()).hexdigest()
        rebound = AcceleratedAcceptedState3D(state.theta, state.u, state.warped, energy,
            state.certificate, refreshed.problem_id, identity, state.image_grad, self.backend_id)
        self._model = model
        self._lin = _DeviceState(model, rebound, data._u, data._warped, data._b, data.sample_fingerprint)
        self._candidate = None
        self._prepared = (refreshed, rebound, self._pending_prepared_fingerprint(refreshed, rebound) if self.lean_identity
                          else self._prepared_fingerprint(refreshed, rebound))
        promoted = self.promote_trial(refreshed, rebound)
        if self.boundary_reuse:
            # E89 (P05c): the outer start's device state stays resident until the outer comparison.
            self._outer_start = self._lin if self._lin is not None and self._lin.state is promoted else None
            # E89 (P05a): with u_ref = u on the device (model._ref is data._u) the proximity and soft terms of
            # grad E_r are exact zeros added to accumulators that never hold -0, so the first inner gradient
            # of this state is the outer vector already computed from the same device data.
            stash, self._outer_stash = self._outer_stash, None
            self._first_gradient = None
            if (stash is not None and stash[0] is state and stash[1] is data.model and stash[2] is data._u
                    and stash[3] is data._warped and stash[4] is data._b):
                self._first_gradient = (promoted, refreshed, model, stash[5], stash[6] if len(stash) > 6 else None)
        self.reuse_counts["reference_hits"] += 1
        return refreshed, promoted

    def _sample(self, model, u):
        warped, gradient = model.sample_u(u)                           # T89: the active-node sampling when enabled
        if not bool(torch.isfinite(warped).all()) or not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError("nonfinite device spline output")
        return warped, gradient

    def _device_state(self, problem, state):
        self._validate(problem, state)
        model = self._model_for(problem)
        for cached in (self._candidate, self._lin) if not self.boundary_reuse else (self._candidate, self._lin, self._outer_start):
            if cached is not None and cached.state is state:
                return cached
        u = self.device.upload(state.u)
        warped, gradient = self._sample(model, u)
        if getattr(state, "evaluation_backend_id", None) is not None:
            if not np.array_equal(self.device.download(warped), state.warped):
                raise StateMismatchError("repeated device sampling changed evaluated trial")
        return _DeviceState(model, state, u, warped, gradient,
                            _sample_fingerprint(state.warped) if self.device_batch is not None else None)

    def linearize(self, lin):
        build_linearization(lin.problem, lin.state)
        self._lin = self._device_state(lin.problem, lin.state)
        return self._lin

    @timed("gradient")
    def gradient(self, lin, target):
        if self._prefetch is not None:
            self.prefetch_drop("superseded")
        if self.device_candidate:
            self._direction_twin = self._axpy_note = None     # E97: a new gradient starts a new direction
        dl = self.linearize(lin)
        first = self._first_gradient if self.boundary_reuse else None
        if first is not None:
            self._first_gradient = None
            if first[0] is lin.state and first[1] is lin.problem and first[2] is dl.model and dl.model._ref is dl._u:
                vector = self._verified_reuse("first_gradient", lin.problem.grid, first[3], dl)   # E89 (P05a)
                if vector is not None:
                    if not (self.outer_reuse and vector is first[3] and not self.outer_reuse_record["disabled"]):
                        return self._decide_gradient(vector, lin, target)
                    # E100: the outer vector, proven finite and frozen; its device tensor serves the first direction
                    decided = self._decide_gradient(vector, lin, target, proven=True)
                    if decided is vector and first[4] is not None:
                        self._device_gradient = self._outer_device_copy(lin.problem.grid, vector, first[4])
                        if self._device_gradient is not None:
                            self._note_twin(*self._device_gradient)          # E104
                    return decided
        if self.lean_checks:
            device_gradient = dl.gradient()
            try:
                fetched = self._prefetch_direction(device_gradient) if self.direction_prefetch else None
                host, known = fetched if fetched is not None else (self.device.download(device_gradient), None)
                decided = self._decide_gradient(host, lin, target, device=device_gradient, known_finite=known)
            except BaseException:
                if self._prefetch is not None:
                    self.prefetch_drop("gradient_error")        # E105: nothing waits for it
                raise
            if self._prefetch is not None:
                if decided is host:
                    self._prefetch["host"] = host                 # E105: the direction queued for this gradient
                else:
                    self.prefetch_drop("not_decided")
            # E95: the device copy of this exact (frozen) host vector, for the device Poisson scalings
            self._device_gradient = (decided, device_gradient) if decided is host else None
            if decided is host:
                self._note_twin(host, device_gradient)                      # E104
            return decided
        return self._decide_gradient(self.device.download(dl.gradient()), lin, target)

    def device_gradient_for(self, gradient):
        """E95: the device tensor whose download is this very host gradient (frozen, unchanged), else None.
        Handed out once: the memo is released (one direction per gradient)."""
        memo, self._device_gradient = self._device_gradient, None
        if memo is not None and memo[0] is gradient and not gradient.flags.writeable:
            return memo[1]
        return None

    def release_device_gradient(self):
        """E96: drop the device gradient memo (end of an inner solve; the next direction uploads afresh).
        E97: also the direction twin and the axpy note of the solve (device_candidate). E104: and the device copies
        kept for the slope and BB2 tail (BB2 never spans inner solves)."""
        self._device_gradient = None
        self._twins = []
        if self._prefetch is not None:
            self.prefetch_drop("release")                       # E105
        self._fp32_note = None                                  # E106
        if self.device_candidate:
            self._axpy_note = self._direction_twin = None
        if self.early_warped:
            self._staged_warped = None

    def poisson(self, inverse):
        if not self.lean_checks:
            return super().poisson(inverse)
        from .hybrid import _HostPoisson
        poisson = _HostPoisson(self.device, inverse, self.timings, device_gradient=self.device_gradient_for,
                               **(dict(keep_direction=self.keep_direction) if self.device_candidate else {}),
                               **(dict(prefetch=self) if self.direction_prefetch else {}),
                               **(dict(fp32=self) if self.fp32_poisson else {}))
        if self.direction_prefetch:
            import weakref
            self.prefetch_drop("new_poisson")
            self._prefetch_poisson = weakref.ref(poisson)
        return poisson

    # ---------------------------------------------------------------- E97 (throughput-v15)

    def note_step_extrema(self, a, b, top, theta_max):
        """E97: max|a-b| and max|b| of these two frozen arrays (one entry, weak references; finite maxima only)."""
        if (self.step_extrema and np.isfinite(top) and np.isfinite(theta_max)
                and _identity._immutable(a) and _identity._immutable(b)):
            import weakref
            self._step_extrema = (weakref.ref(a), weakref.ref(b), top, theta_max)
            self.step_extrema_record["noted"] += 1

    def known_step_extrema(self, a, b):
        memo = self._step_extrema
        if memo is None or not self.step_extrema:
            return None
        ra, rb, top, theta_max = memo
        if ra() is a and rb() is b and _identity._immutable(a) and _identity._immutable(b):
            return top, theta_max
        return None

    def keep_direction(self, host, device):
        """E97: the frozen host direction a device Poisson just downloaded and its device source tensor."""
        self._direction_twin = (host, device)
        self._note_twin(host, device)                                       # E104

    # ---------------------------------------------------------------- E104 (throughput-v22, G3b narrowed)

    def _note_twin(self, host, device):
        """E104: remember the device tensor a frozen host vector (gradient or direction) was downloaded from -- the same
        values, bit for bit -- for the device slope and BB2 tail; the 4 most recent (this and the previous step). The
        gradient's host copy is a frozen view of the download's own CPU buffer (no other owner), the direction's an
        owned frozen array; the first uses per grid compare the products with the host ones (E104 item 4)."""
        if self.device_scalars and isinstance(host, np.ndarray) and not host.flags.writeable:
            import weakref
            self._twins.insert(0, (weakref.ref(host), device, device._version))
            del self._twins[4:]

    def forget_twin(self, host):
        """E106: drop the device copy registered for this host array (a float32 direction replaced by the float64 one
        before use), so the registry holds what v23's would (the 4 most recent used vectors)."""
        self._twins = [t for t in self._twins if t[0]() is not host]

    def _twin(self, host):
        """The device copy of this very (still frozen) host array, unless the tensor was written in place since."""
        for ref, device, version in self._twins:
            if ref() is host:
                if device._version == version and not host.flags.writeable:
                    return device
                self.device_scalars_record["stale"] += 1          # review E104 (round 2): counted apart
                return None
        return None

    def _scalar_error(self, error):
        """E104: a device failure (e.g. out of memory) switches the device scalars off for the run; the host path runs."""
        record = self.device_scalars_record
        record["errors"] += 1
        record["disabled"] = True
        if len(record["error_types"]) < 8:
            record["error_types"].append(type(error).__name__)

    def _scalar_check(self, grid, kind, device_value, host_value, abs_sum, n):
        """E104 first uses per (grid, kind): |device - host| <= 2 gamma_n sum|a_i b_i| (times 1.01 for the device's own
        sum); any dot product, in any order with or without FMA, is within gamma_n sum|a_i b_i| of the exact one."""
        record = self.device_scalars_record
        eps = 2.**-53
        gamma = n*eps/(1-n*eps)
        diff = abs(device_value-host_value)
        bound = 2*gamma*abs_sum*1.01
        rel = diff/abs(host_value) if host_value != 0 else (0. if diff == 0 else float("inf"))
        record["checks"] += 1
        record["max_"+kind+"_rel"] = max(record["max_"+kind+"_rel"], rel)
        key = (grid.grid_id, kind)
        self._scalar_checks[key] = self._scalar_checks.get(key, 0)+1
        if not diff <= bound:
            record["violations"] += 1
            record["disabled"] = True
            if len(record["violation_details"]) < 8:
                record["violation_details"].append(dict(kind=kind, device=float(device_value).hex(),
                                                        host=float(host_value).hex(), bound=bound))
            return False
        return True

    def _device_finite(self, device, host, known=None):
        """E104: all values of the device copy finite (an exact reduction); False sends the caller to the host scan.
        known (E105): the same reduction's value, computed on the prefetch side stream."""
        from ..work_control import WorkBudgetExceeded
        if self.device_scalars_record["disabled"] or device.numel() != host.size:
            return False
        if known is not None:
            return known
        try:
            return bool(torch.isfinite(device).all())
        except WorkBudgetExceeded:
            raise
        except Exception as error:
            self._scalar_error(error)
            return False

    def _scalar_checking(self, grid, kind):
        return self._scalar_checks.get((grid.grid_id, kind), 0) < SCALAR_CHECKS

    # ---------------------------------------------------------------- E105 direction_prefetch

    def _prefetch_direction(self, device_gradient):
        """E105: when this solve's device Poisson holds its scale S (the solve's later directions), queue on the main
        stream exactly what prepare_equilibrated and apply_equilibrated queue for this gradient -- pre = S*g, its finite
        flag (copied to pinned memory), x = M^-1 pre, out = -(S*x), out's finite flag -- and download g (and, with
        device_scalars, reduce its finite flag) on a side stream that waits only for the gradient kernels.
        Returns (host gradient, finite flag or None), or None (nothing queued; the caller downloads as in v22)."""
        record = self.direction_prefetch_record
        poisson = self._prefetch_poisson() if self._prefetch_poisson is not None else None
        if record["disabled"] or poisson is None or poisson._scale is None:
            return None
        if not device_gradient.is_cuda:
            record["unsupported"] += 1
            return None
        from ..work_control import WorkBudgetExceeded
        try:
            if self._prefetch_stream is None:
                self._prefetch_stream = torch.cuda.Stream(device=device_gradient.device)
                self._prefetch_flag = torch.empty((), dtype=torch.bool, pin_memory=True)
            main = torch.cuda.current_stream(device_gradient.device)
            gradient_done = torch.cuda.Event()
            gradient_done.record(main)
            scale, device_scale = poisson._scale
            pre = device_scale*device_gradient
            use32 = self.fp32_active()
            if use32:
                # E106: the finite and range flags of the fp32_poisson prepare, then the float32 solve
                from .hybrid import FP32_RANGE
                if self._prefetch_flags32 is None:
                    self._prefetch_flags32 = torch.empty(2, dtype=torch.bool, pin_memory=True)
                self._prefetch_flags32.copy_(torch.stack((torch.isfinite(pre).all(), pre.abs().amax() <= FP32_RANGE)),
                                             non_blocking=True)
            else:
                self._prefetch_flag.copy_(torch.isfinite(pre).all(), non_blocking=True)
            pre_done = torch.cuda.Event()
            pre_done.record(main)
            x = poisson.inverse.apply_fp32(pre) if use32 else poisson.inverse.apply(pre)
            out = -(device_scale*x)
            out_finite = torch.isfinite(out).all()
            self._prefetch = dict(poisson=poisson, scale=scale, device_scale=device_scale, gradient=device_gradient,
                                  version=device_gradient._version, host=None, x=x, out=out, out_finite=out_finite,
                                  pre_done=pre_done, taken=False, fp32=use32, pre=pre if use32 else None)
            record["enqueued"] += 1
            side = self._prefetch_stream
            known = None
            # The side-stream work below is synchronous (the copy and bool() block until done), so no side-stream read
            # of device_gradient outlives this call; an asynchronous download would need record_stream or an event.
            with torch.cuda.stream(side):
                side.wait_event(gradient_done)
                host = self.device.download(device_gradient)
                if self.device_scalars and not self.device_scalars_record["disabled"]:
                    known = bool(torch.isfinite(device_gradient).all())
            return host, known
        except WorkBudgetExceeded:
            raise
        except Exception as error:      # noqa: BLE001 - recorded; v22's path runs from here on
            if self._prefetch is not None:
                self.prefetch_drop("error")
            record["errors"] += 1
            record["disabled"] = True
            if type(error).__name__ not in record["error_types"]:
                record["error_types"].append(type(error).__name__)
            return None

    def prefetch_take(self, poisson, scale, gradient):
        """E105 (from prepare_equilibrated): the queued (x, out, out_finite) for this very frozen gradient, scale and
        Poisson, after handing out the device gradient memo as v22 does; 'nonfinite' when s*g is not finite (v22 then
        returns None); None when nothing matches (the prefetch is dropped and v22's path runs)."""
        pf = self._prefetch
        if pf is None:
            return None
        if not (pf["poisson"] is poisson and pf["scale"] is scale and pf["host"] is gradient
                and not gradient.flags.writeable and poisson._scale is not None and poisson._scale[0] is scale
                and poisson._scale[1] is pf["device_scale"] and pf["gradient"]._version == pf["version"]
                and self._device_gradient is not None and self._device_gradient[0] is gradient
                and self._device_gradient[1] is pf["gradient"]):
            self.prefetch_drop("mismatch")
            return None
        self.device_gradient_for(gradient)                       # the memo is handed out once, as in v22
        pf["pre_done"].synchronize()
        if pf["fp32"]:
            finite, in_range = self._prefetch_flags32.tolist()   # E106: the fp32_poisson prepare's two flags
            if not finite:
                self.direction_prefetch_record["nonfinite_pre"] += 1
                self.prefetch_drop("nonfinite_pre")
                return "nonfinite"
            pf["taken"] = True
            return dict(scale=pf["device_scale"], pre=pf["pre"], gradient=pf["gradient"], use32=bool(in_range),
                        queued=(pf["x"], pf["out"], pf["out_finite"]) if in_range else None, from_prefetch=True)
        if not bool(self._prefetch_flag):
            self.direction_prefetch_record["nonfinite_pre"] += 1
            self.prefetch_drop("nonfinite_pre")
            return "nonfinite"
        pf["taken"] = True
        return pf["x"], pf["out"], pf["out_finite"]

    def prefetch_taken(self, poisson):
        pf = self._prefetch
        return pf is not None and pf["taken"] and pf["poisson"] is poisson

    def prefetch_used(self, poisson):
        if self.prefetch_taken(poisson):
            self.direction_prefetch_record["used"] += 1
            self._prefetch = None

    def prefetch_detach(self):
        """T92 (D2-d): a metric build without a device Poisson; drop any queued direction and forget the last Poisson,
        so no later gradient queues a solve into an earlier build's Poisson (wasted device work, never used)."""
        if self.direction_prefetch:
            self.prefetch_drop("metric_ablation")
            self._prefetch_poisson = None

    def prefetch_drop(self, reason, owner=None):
        pf = self._prefetch
        if pf is not None and (owner is None or pf["poisson"] is owner):
            dropped = self.direction_prefetch_record["dropped"]
            dropped[reason] = dropped.get(reason, 0)+1
            self._prefetch = None

    def _sync_deferred(self, name, when):
        """E105: the 'gradient' operation that queued a direction returns without waiting for it."""
        pf = self._prefetch
        return pf is not None and not pf["taken"] and name == "gradient" and when == "end"

    # ---------------------------------------------------------------- E106 fp32_poisson (C14 phase A)

    def fp32_active(self):
        return self.fp32_poisson and not self.fp32_poisson_record["disabled"]

    def fp32_checking(self, grid):
        """The first FP32_CHECKS float32 uses per level grid, and every FP32_RECHECK-th one after them (review E106 S3:
        later directions can be more high-frequency, where the spectral division amplifies rounding most)."""
        self.fp32_poisson_record["uses"] += 1
        uses = self._fp32_uses[grid.grid_id] = self._fp32_uses.get(grid.grid_id, 0)+1
        return self._fp32_checks.get(grid.grid_id, 0) < FP32_CHECKS or uses % FP32_RECHECK == 0

    def fp32_check_result(self, grid, l2_rel, max_rel, slope_rel):
        """E106: one first-use comparison of the float32 direction with the float64 one (relative l2 and max
        differences of out, relative difference of g.out); above FP32_LIMIT switches float32 off for the run."""
        import math
        record = self.fp32_poisson_record
        self._fp32_checks[grid.grid_id] = self._fp32_checks.get(grid.grid_id, 0)+1
        record["checks"] += 1
        # review E106 (S2): the record must stay JSON-serializable (allow_nan=False): a nonfinite measurement is
        # stored as 1e300 in the maxima (it is a violation anyway) and as repr() in the details
        finite = lambda v: v if math.isfinite(v) else 1e300
        record["max_l2_rel"] = max(record["max_l2_rel"], finite(l2_rel))
        record["max_max_rel"] = max(record["max_max_rel"], finite(max_rel))
        record["max_slope_rel"] = max(record["max_slope_rel"], finite(slope_rel))
        if not (l2_rel <= FP32_LIMIT and max_rel <= FP32_LIMIT and slope_rel <= FP32_LIMIT):
            record["violations"] += 1
            record["disabled"] = True
            if len(record["violation_details"]) < 8:
                record["violation_details"].append(dict(grid=grid.grid_id, l2_rel=repr(l2_rel), max_rel=repr(max_rel),
                                                        slope_rel=repr(slope_rel)))
            return False
        return True

    def fp32_error(self, error):
        record = self.fp32_poisson_record
        record["errors"] += 1
        record["disabled"] = True
        if type(error).__name__ not in record["error_types"]:
            record["error_types"].append(type(error).__name__)

    def fp32_note(self, host, gradient_norm, direction_norm, rerun):
        self._fp32_note = (host, gradient_norm, direction_norm, rerun)

    def fp32_descent_guard(self, gradient, direction, slope):
        """E106: for a float32 direction whose slope g.p (the solver's float64 host product) is not below
        -c_d |g| |p| (c_d = FP32_GUARD_FACTOR times the largest first-use relative l2 difference, at least
        FP32_GUARD_FLOOR), the float64 direction of the same gradient (v23's values), else None. So a direction is
        never judged non-descent on account of float32 rounding."""
        note, self._fp32_note = self._fp32_note, None
        if note is None or note[0] is not direction:
            return None
        record = self.fp32_poisson_record
        c_d = max(FP32_GUARD_FACTOR*record["max_l2_rel"], FP32_GUARD_FLOOR)
        if slope <= -c_d*note[1]*note[2]:
            return None
        record["descent_reruns"] += 1
        again, finite = note[3]()
        if finite:
            record["fp32_used"] -= 1                  # review E106 (M6): fp32_used counts float32 directions used
        else:
            record["descent_rerun_failures"] += 1     # the solver's direction check then raises, as in v23
        return again

    def shift_alpha(self, problem, base, factor, reference):
        """E105: image_curvature_shift(problem, base, factor, cached=True) with the mean from host_kernels.shift_mean
        (the same IEEE operations in NumPy's order). reference() evaluates the NumPy expression: it is used (and
        raises as before) when the kernel declines (layout, nonfinite partial sums), errs, or is switched off, and
        for the first SHIFT_CHECKS uses per level grid, whose kernel value must equal it bit for bit."""
        import math
        record = self.fused_shift_record
        record["uses"] += 1
        if record["disabled"]:
            return reference()
        from .. import host_kernels
        from ..solvers.poisson_gradient import _node_quotient
        try:
            key = problem.grid.grid_id
            q = _node_quotient(problem.grid)
            small = not host_kernels.enabled(q.size)
            mean = None if small else host_kernels.shift_mean(base.image_grad, q)
        except Exception as error:      # noqa: BLE001 - recorded; the NumPy expression decides
            record["errors"] += 1
            record["disabled"] = True
            if type(error).__name__ not in record["error_types"]:
                record["error_types"].append(type(error).__name__)
            return reference()
        if small:
            # review E105 (M1): below MIN_SIZE (coarse levels), counted apart; reference() outside the try (round 2):
            # a raising NumPy expression is the run's own error, not a kernel error
            record["small"] += 1
            return reference()
        alpha = None if mean is None else factor*mean/3
        if alpha is None or not math.isfinite(alpha):
            record["fallbacks"] += 1
            return reference()
        if self._shift_checks.get(key, 0) < SHIFT_CHECKS:
            self._shift_checks[key] = self._shift_checks.get(key, 0)+1
            record["checks"] += 1
            expected = reference()
            if float(expected).hex() != alpha.hex():
                record["mismatches"] += 1
                record["disabled"] = True
                if len(record["mismatch_details"]) < 8:
                    record["mismatch_details"].append(dict(kernel=alpha.hex(), numpy=float(expected).hex()))
            return expected
        record["used"] += 1
        return alpha

    def device_bb2_tail(self, gradient, old_gradient, direction, old_direction, grid):
        """E104: iy = p_old - p and y = g - g_old on the device (elementwise the host values), then iy's finiteness
        (exact), y.iy and |iy|; None without the four copies. Returns dict(finite, yy, niy, abs_sum)."""
        import math
        from ..work_control import WorkBudgetExceeded
        record = self.device_scalars_record
        if not self.device_scalars or record["disabled"]:
            return None
        twins = [self._twin(v) for v in (gradient, old_gradient, direction, old_direction)]
        if any(t is None or t.numel() != gradient.size for t in twins):
            record["fallbacks"] += 1
            return None
        try:
            gd, god, pd, pod = (t.reshape(-1) for t in twins)
            iy = pod-pd
            if not bool(torch.isfinite(iy).all()):
                return dict(finite=False)
            y = gd-god
            checking = self._scalar_checking(grid, "yy")
            parts = [torch.dot(y, iy), torch.linalg.vector_norm(iy)]
            if checking:
                parts.append((y*iy).abs().sum())
            values = torch.stack(parts).cpu().tolist()
        except WorkBudgetExceeded:
            raise
        except Exception as error:
            self._scalar_error(error)
            return None
        record["tails"] += 1
        out = dict(finite=True, yy=values[0], niy=values[1], abs_sum=values[2] if checking else None)
        if not (math.isfinite(out["yy"]) and math.isfinite(out["niy"])):
            out["yy"] = None
        return out

    def device_yy_decide(self, tail, ny, host_yy, grid, n):
        """E104: the device y.iy when clearly positive (> 1e-6 |y||iy|): the host one is then positive too and the
        solver uses this value; otherwise None (band: the host product). host_yy() computes the v21 value."""
        record = self.device_scalars_record
        yy, niy = tail.get("yy"), tail.get("niy")
        record["yys"] += 1
        # review E104 (M1): each norm away from under/overflow (the device norm is unscaled), then the band
        if (yy is None or record["disabled"] or not (NORM_FLOOR <= ny <= NORM_CEILING and NORM_FLOOR <= niy <= NORM_CEILING)
                or not yy > SCALAR_BAND*ny*niy):
            record["yy_band"] += 1
            return None
        if tail.get("abs_sum") is not None:
            if not self._scalar_check(grid, "yy", yy, host_yy(), tail["abs_sum"], n):
                return None
        return yy

    def note_axpy(self, candidate, base_theta, tau, direction):
        """E97: candidate = base_theta + fl(tau*direction) by the compiled axpy (both roundings)."""
        if self.device_candidate:
            import weakref
            self._axpy_note = (weakref.ref(candidate), weakref.ref(base_theta), float(tau), weakref.ref(direction))

    def _device_candidate_u(self, problem, model, theta, u):
        """E97: the device u of this trial built as tau*direction (one kernel) placed in the interior, plus the
        base's device u (one kernel) -- the host axpy's two roundings -- with +0 faces, or None (upload)."""
        note, twin, lin = self._axpy_note, self._direction_twin, self._lin
        record = self.device_candidate_record
        if note is None or twin is None or lin is None or record["disabled"]:
            return None
        candidate, base_theta, tau, direction = note[0](), note[1](), note[2], note[3]()
        host_dir, dev_dir = twin
        if (candidate is not theta or direction is not host_dir or base_theta is None
                or getattr(lin.state, "theta", None) is not base_theta or lin.model is not model
                or not (_identity._immutable(theta) and _identity._immutable(base_theta)
                        and _identity._immutable(host_dir) and _identity.has_fact(theta, _identity.FACT_FINITE))):
            record["fallbacks"] += 1
            return None
        grid = problem.grid
        base_u = lin._u
        if tuple(base_u.shape) != grid.shape+(3,) or dev_dir.numel() != theta.size:
            record["fallbacks"] += 1
            return None
        nz, ny, nx = grid.shape
        t = dev_dir.reshape(3, nz-2, ny-2, nx-2)*tau
        du = torch.zeros_like(base_u)
        du[1:-1, 1:-1, 1:-1].copy_(t.permute(1, 2, 3, 0))
        du[1:-1, 1:-1, 1:-1].add_(base_u[1:-1, 1:-1, 1:-1])
        if grid.grid_id not in self._device_candidate_verified:
            host = self.device.download(du)
            if not np.array_equal(np.ascontiguousarray(host).view(np.uint64), np.ascontiguousarray(u).view(np.uint64)):
                record["mismatches"] += 1
                record["disabled"] = True
                return None
            self._device_candidate_verified.add(grid.grid_id)
            record["verified_grids"] += 1
        record["built"] += 1
        return du

    def _verified_reuse(self, name, grid, vector, dl):
        """E89 (review): the reused host vector, or None when this reuse is off. The first reuse on each level grid
        is compared bitwise (uint64 view) with dl.gradient() computed on the device; a mismatch switches the reuse
        off for the rest of the run and the computed vector is returned instead."""
        record = self.boundary_reuse_record[name]
        if record["disabled"]:
            return None
        if grid.grid_id not in self._boundary_verified[name]:
            fresh = self.device.download(dl.gradient())     # an exception here propagates exactly as in v10
            if not bool(np.all(np.isfinite(fresh))):
                record["nonfinite"] += 1                     # v10's own vector is nonfinite: the caller raises
                return fresh
            self._boundary_verified[name].add(grid.grid_id)
            record["verified_grids"] += 1
            if not (fresh.shape == vector.shape
                    and np.array_equal(np.ascontiguousarray(fresh).view(np.uint64), np.ascontiguousarray(vector).view(np.uint64))):
                record["mismatches"] += 1
                record["disabled"] = True
                return fresh
        record["reused"] += 1                                # vectors returned (an unused triple still counts)
        return vector

    @timed("gradient")
    def final_gradient(self, lin, target):
        """E89 (P05b): gradient(lin, target) at the last iterate of an inner solve. The vector is the total
        vector of the joint outer triple (same terms in the same order as dl.gradient(), bit for bit); the
        device triple is kept for outer_gradient_vectors of this same state, problem and model."""
        if self._prefetch is not None:
            self.prefetch_drop("final_gradient")
        if not self.boundary_reuse:
            return self.gradient(lin, target)
        if self.device_candidate:
            self._direction_twin = self._axpy_note = None     # E97
        dl = self.linearize(lin)
        first, self._first_gradient = self._first_gradient, None
        if (first is not None and first[0] is lin.state and first[1] is lin.problem and first[2] is dl.model
                and dl.model._ref is dl._u):
            vector = self._verified_reuse("first_gradient", lin.problem.grid, first[3], dl)
            if vector is not None:
                return self._decide_gradient(vector, lin, target)
        if self.boundary_reuse_record["last_gradient"]["disabled"]:
            return self._decide_gradient(self.device.download(dl.gradient()), lin, target)
        # Review E89: timed without an operation scope, so no work checkpoint or event is added inside the inner
        # solve; outer_gradient_vectors records its joint_outer_geometry operation where v10 does.
        with self._timed_only('joint_outer_geometry'):
            vectors = dl.joint_outer_gradients()
        hosts = tuple(self.device.download(value) for value in vectors)
        # The joint pass is counted where the outer vectors are used (outer_gradient_vectors), as in v10.
        total = self._verified_reuse("last_gradient", lin.problem.grid, hosts[2], dl)
        if total is not hosts[2]:                       # verification mismatch: no reuse of this triple
            return self._decide_gradient(total, lin, target)
        result = self._decide_gradient(hosts[2], lin, target)      # raises exactly as gradient() would
        from .. import fast_checks
        if all(fast_checks.all_finite(v) if fast_checks.enabled(v.size) else bool(np.all(np.isfinite(v)))
               for v in hosts[:2]):
            for value in hosts:
                value.flags.writeable = False
            # E100 (outer_reuse): the device tensor hosts[0] was downloaded from (a fresh tensor nothing writes into)
            self._joint_stash = (lin.state, lin.problem, dl.model, hosts,
                                 vectors[0] if self.outer_reuse and not self.outer_reuse_record["disabled"] else None)
        return result

    def _outer_device_copy(self, grid, host, device):
        """E100: (host, device) for device_gradient_for -- the device tensor host was downloaded from. The first use
        on each level grid downloads it again and compares bit for bit (uint64 view); a mismatch switches the copy
        off for the rest of the run (the first direction then uploads host, as in v17)."""
        record = self.outer_reuse_record
        if record["disabled"] or not self.lean_checks:
            return None
        if grid.grid_id not in self._outer_reuse_verified:
            from ..work_control import WorkBudgetExceeded
            try:
                again = self.device.download(device)     # extra checking work (not in v17): its failure only switches off
            except WorkBudgetExceeded:
                raise
            except Exception:
                record["errors"] += 1
                record["disabled"] = True
                return None
            if not (again.shape == host.shape and np.array_equal(np.ascontiguousarray(again).view(np.uint64),
                                                                 np.ascontiguousarray(host).view(np.uint64))):
                record["mismatches"] += 1
                record["disabled"] = True
                return None
            self._outer_reuse_verified.add(grid.grid_id)
            record["verified_grids"] += 1
        record["device_copies"] += 1
        return (host, device)

    def _known_outer_norm(self, gradient, omega):
        """E100: the norm outer_vector_norms computed for this very frozen array (same function), once; else None."""
        memo, self._outer_norm = self._outer_norm, None
        if memo is not None and memo[0] is gradient and memo[1] == omega and not gradient.flags.writeable:
            self.outer_reuse_record["norms_reused"] += 1
            return memo[2]
        return None

    def _decide_gradient(self, result, lin, target, *, proven=False, device=None, known_finite=None):
        if proven:
            self.outer_reuse_record["scans_skipped"] += 1      # E100: proven finite and frozen (the outer vector)
        elif device is not None and self.device_scalars and self._device_finite(device, result, known_finite):
            # E104: result is the exact download of device, so its device finite check decides the same
            self.device_scalars_record["finite_device"] += 1
        else:
            require_finite_result(result, "device gradient")
        # E95: gradient_norm_mass without its second finite scan of this just-checked array (same nrm2, same checks)
        known = self._known_outer_norm(result, lin.problem.grid.omega) if proven else None
        norm = (known if known is not None else _mass_norm_checked(result, lin.problem.grid.omega) if self.lean_checks
                else gradient_norm_mass(result, lin.problem.grid.omega))
        if self.lean_identity:
            # E89 (P03): the solver's G is this same norm of this same array; frozen so it cannot change.
            result.flags.writeable = False
            self._gradient_norm_memo = (result, lin.problem.grid.omega, norm)
        margin_active = self.cpu_gradient_margin is not None and not self.gradient_margin_monitor['abandoned']
        if margin_active:
            if norm <= (1-self.cpu_gradient_margin)*target:
                self.device_gradient_targets += 1
                return result
            if norm > (1+self.cpu_gradient_margin)*target:
                self.device_gradient_above += 1
                return result
            self.cpu_gradient_checks += 1
            state = self.cpu_state(lin.problem, lin.state)
            cpu = energy_gradient(build_linearization(lin.problem, state))
            self._monitor_margin(norm, gradient_norm_mass(cpu, lin.problem.grid.omega))
            return cpu
        if self.cpu_gradient_margin is not None:      # abandoned: the v8 band from here on
            if norm <= .99*target:
                self.device_gradient_targets += 1
                return result
            if norm <= 1.01*target:
                self.cpu_gradient_checks += 1
                state = self.cpu_state(lin.problem, lin.state)
                cpu = energy_gradient(build_linearization(lin.problem, state))
                self._monitor_margin(norm, gradient_norm_mass(cpu, lin.problem.grid.omega))
                return cpu
            return result
        if self.cpu_gradient_band is not None and norm <= self.cpu_gradient_band*target:
            self.device_gradient_targets += 1
            return result
        if norm <= 1.01*target:
            self.cpu_gradient_checks += 1
            state = self.cpu_state(lin.problem, lin.state)
            result = energy_gradient(build_linearization(lin.problem, state))
        return result

    def _energy(self, model, u, warped, *, base=None, screen=False):
        """Optional E/F residual differences share candidate/reference geometry.
        screen (T90): the soft-geometry and barrier terms from the float32 kernels; self._energy_band is then the
        float64 band of the comparison (relative to those terms of both states), else None.

        Difference validity is checked at comparison, not trial evaluation:
        a nonfinite difference must use the existing CPU comparison fallback.
        """
        w, q, qd = model.w, model._q, model._qd   # T80: qd = q m on the image term
        values = {"image": .5*(qd*(warped-model._fixed)**2).sum(),
                  "proximity": .5*w.lam*(q[..., None]*(u-model._ref)**2).sum()}
        for name in ("jd", "curl", "barrier", "curvature"):
            values[name] = torch.zeros((), dtype=torch.float64, device=u.device)
        if model._alpha:
            values["curvature"] = .5*(model.curvature_residual(u)**2).sum()   # T76
        minimum = torch.full((), float("inf"), dtype=torch.float64, device=u.device)
        if base is not None:
            x, y = qd.sqrt()*(base._warped-model._fixed), qd.sqrt()*(warped-model._fixed)
            total = .5*((y-x)*(y+x)).sum()
            if model._alpha:
                x, y = model.curvature_residual(base._u), model.curvature_residual(u)
                total = total+.5*((y-x)*(y+x)).sum()   # T76: an outer (F) block
            outer = total.clone()
            weight = (w.lam*q).sqrt()[..., None]
            x, y = weight*(base._u-model._ref), weight*(u-model._ref)
            total += .5*((y-x)*(y+x)).sum()
            min_gap = minimum.clone()
        fused = model._fused
        self._energy_band = None
        if fused is not None:
            # E51 fused geometry: same terms, block-ordered sums (roundoff-level differences only).
            if screen and not fused.bernstein:                                    # T90: the float32 screening kernels
                soft = fused.soft_energy_f32(u, model._ref, None if base is None else base._u)
                bar = fused.barrier_energy_f32(u, None if base is None else base._u)
                s_soft = float(self.device.scalar(soft[0]+soft[1])); s_bar = float(self.device.scalar(bar[0]))
                if base is not None:
                    e = base.state.energy
                    s_soft += e.jd+e.curl; s_bar += e.barrier
                self._energy_band = FP32_EPS*(FP32_BAND_SOFT*max(s_soft, 0.)+FP32_BAND_BARRIER*max(s_bar, 0.))
            else:
                soft = fused.soft_energy(u, model._ref, None if base is None else base._u)
                bar = fused.barrier_energy(u, None if base is None else base._u)
            values["jd"], values["curl"], values["barrier"] = soft[0], soft[1], bar[0]
            minimum = bar[1]
            if base is not None:
                change = model.problem.barrier_weight*bar[2]
                total = total+soft[2]+change
                outer = outer+change
                min_gap = torch.minimum(bar[1], bar[3])-model.problem.certificate_config.delta
        for _, indices in (() if fused is not None else model.tiles()):
            local = u.reshape(-1, 3)[indices]
            _, _, J, C = _value_geometry(model, local, model._D, with_curl=True)
            _, _, Jr, Cr = _value_geometry(model, model._ref.reshape(-1, 3)[indices], model._D, with_curl=True)
            values["jd"] += .5*w.mu*(model._qw*(J-Jr)**2).sum()
            values["curl"] += .5*w.nu*(model._qw[None, :, None]*(C-Cr)**2).sum()
            _, _, Jb, _ = _value_geometry(model, local, model._Dc, with_curl=False)
            Jb = model.barrier_vals(Jb)   # T68: Bernstein coefficients when selected (identity otherwise)
            minimum = torch.minimum(minimum, Jb.min())
            values["barrier"] += (model.barrier_term(Jb, order=0) if self.selective_kernels else
                barrier(Jb, model.problem.certificate_config.delta, model.problem.barrier_weight,
                        model.problem.barrier_activation)[0]).sum()
            if base is not None:
                al = base._u.reshape(-1, 3)[indices]
                _, _, Ja, Ca = _value_geometry(model, al, model._D, with_curl=True)
                x, y = (w.mu*model._qw).sqrt()*(Ja-Jr), (w.mu*model._qw).sqrt()*(J-Jr)
                total += .5*((y-x)*(y+x)).sum()
                weight = (w.nu*model._qw).sqrt()[None, :, None]
                x, y = weight*(Ca-Cr), weight*(C-Cr)
                total += .5*((y-x)*(y+x)).sum()
                _, _, Ja, _ = _value_geometry(model, al, model._Dc, with_curl=False)
                Ja = model.barrier_vals(Ja)
                delta = model.problem.certificate_config.delta
                min_gap = torch.minimum(min_gap, torch.minimum(Ja.min(), Jb.min())-delta)
                change = model.problem.barrier_weight*_barrier_difference(
                    Ja, Jb, delta, model.problem.barrier_activation).sum()
                total += change
                outer += change
        # T68: Bernstein coefficients have no certificate; within a small guard band of delta the
        # CPU (authoritative) decides, so a device value can never admit a CPU-rejected state.
        guard = BERNSTEIN_GUARD if model._TB is not None else 0.
        if self.device.scalar(minimum) <= model.problem.certificate_config.delta+guard:
            raise FloatingPointError("device barrier points contradict CPU certificate or are within the T68 guard")
        result = EnergyTerms3D(*(self.device.scalar(values[name]) for name in
                               ("image", "proximity", "jd", "curl", "barrier", "curvature")))
        require_finite_result([result.image, result.proximity, result.jd, result.curl,
                               result.barrier, result.curvature, result.total, result.outer], "device energy")
        if min(result.image, result.proximity, result.jd, result.curl, result.barrier, result.curvature) < 0:
            raise FloatingPointError("negative device energy term")
        if base is not None:
            return result, tuple(self.device.scalar(v) for v in (total, outer, min_gap))
        return result

    @timed("evaluate_trial")
    def evaluate_trial(self, problem, theta, *, certificate_observer=None, witness_observer=None):
        self._fused_comparison = None
        if self.early_warped:
            self._staged_warped = None
        theta = as_float64_array(theta, "theta", (3*problem.grid.n_free_scalar,), check_finite=False)
        if not ((self.lean_checks and _identity.has_fact(theta, _identity.FACT_FINITE))
                or (self._all_finite(theta) if self.lean_identity else np.all(np.isfinite(theta)))):
            return InvalidTrial("NONFINITE_STATE", "nonfinite free displacement", int(np.sum(~np.isfinite(theta))))
        prepared = self._prepared_match(problem, theta)
        if prepared is not None:
            self.reuse_counts["trial_hits"] += 1
            if certificate_observer is not None:
                certificate_observer(prepared.certificate)
            return prepared
        if self.lean_identity:
            theta, u = self._owned_pair(problem, theta)
            if self.early_trial_hash:
                self._submit_trial_digest(u)                      # v41 (H2)
        else:
            theta = readonly_copy(theta)
            u = readonly_copy(inject_free(theta, problem.grid))
        invalid = check_map_domain(u, problem.grid)
        if invalid is not None:
            if self.early_trial_hash:
                self._drop_trial_digest()                         # v41 review: release this u at once
            return invalid
        model = device_u = None
        if self.witness_screen:
            self.evaluation_counts["witness_scans"] += 1
            try:
                with self._stage("trial_upload"):
                    model = self._model_for(problem)
                    device_u = (self._device_candidate_u(problem, model, theta, u) if self.device_candidate
                                else None)
                    if device_u is None:
                        device_u = self.device.upload(u)
                with self._stage("trial_witness_screen"):
                    invalid = screen(problem, u, model, device_u=device_u,
                        **({"witness_observer": witness_observer} if witness_observer is not None else {}),
                        **(dict(batch_tiles=self.device_batch.screen_batch_tiles) if self.device_batch is not None else {}),
                        **(dict(finite_known=True) if self.lean_identity else {}),   # E89: check_map_domain proved it
                        **(dict(prefilter=lambda du, D: self._prefilter_select(problem, model, du, D))
                           if self.screen_prefilter else {}))
            except (FloatingPointError, torch.OutOfMemoryError):
                # A failed screen is not a witness: keep the full CPU proof.
                self.evaluation_counts["witness_numeric_failures"] += 1
                model = device_u = invalid = None
            if invalid is not None:
                self.evaluation_counts["witness_rejections"] += 1
                if self.early_trial_hash:
                    self._drop_trial_digest()                     # v41 review: release this u at once
                return invalid
        self.evaluation_counts["full_certifications"] += 1
        if self.overlap_certificate:
            return self._evaluate_overlapped(problem, theta, u, model, device_u, certificate_observer)
        with self._stage("trial_full_certificate"):
            certificate = certify_map(u, problem.grid, config=problem.certificate_config)
        if certificate_observer is not None:
            certificate_observer(certificate)
        if certificate.status != "certified_feasible":
            reason = "FEASIBILITY_UNCERTAIN" if certificate.status == "uncertain" else certificate.reason
            return InvalidTrial(reason, "trial not certified: "+certificate.reason,
                                certificate.n_uncertain+certificate.n_infeasible)
        return self._finish_trial(problem, theta, u, model, device_u, certificate)

    def _prefilter_select(self, problem, model, device_u, D):
        """E93: one bool per witness-screen tile (True = run the legacy tile), or None (legacy screen).
        Review E93: any failure of the prefilter itself (allocation, compilation, launch, constants) leaves
        the legacy screen in charge for the rest of the run and is recorded; a work deadline propagates."""
        from . import screen_prefilter as sp
        from ..work_control import WorkBudgetExceeded
        record = self.screen_prefilter_record
        if record["disabled"] or not sp.supported(device_u.device):
            return None
        try:
            return self._prefilter_selection(problem, model, device_u, D, sp)
        except WorkBudgetExceeded:
            raise
        except Exception as error:
            record["errors"] += 1
            record["error_types"] = sorted(set(record["error_types"]) | {type(error).__name__})
            record["disabled"] = True
            return None

    def _prefilter_selection(self, problem, model, device_u, D, sp):
        record = self.screen_prefilter_record
        grid = problem.grid
        delta = problem.certificate_config.delta
        key = (tuple(grid.shape), tuple(float(h) for h in grid.spacing), float(delta))
        closed = problem.barrier_location == "closed27_uniform"
        source = problem.barrier_ops.coefficients if closed else None
        prefilter = self._prefilter
        if (prefilter is None or prefilter.key != key or
                (closed and not (prefilter.D_host is source or np.array_equal(prefilter.D_host, source)))):
            if source is None:
                # the screen's own table for other barrier locations: a function of the grid only (cached by key)
                from ..cell_q1 import CellQ1Operators3D
                source = CellQ1Operators3D(grid, location="closed27_uniform").coefficients
            try:
                prefilter = sp.ScreenPrefilter(grid, source, delta, device_u.device)
            except InputValidationError:
                record["unavailable"] += 1
                record["disabled"] = True
                return None
            prefilter.key = key
            self._prefilter = prefilter
        select = prefilter.flagged_tiles(device_u, model._tile_size)
        if key not in self._prefilter_verified:
            if not self._prefilter_check(prefilter, model, device_u, D, select):
                record["mismatches"] += 1
                record["disabled"] = True
                return None
            self._prefilter_verified.add(key)
            record["verified_grids"] += 1
        record["calls"] += 1
        record["tiles"] += int(select.size)
        record["tiles_selected"] += int(select.sum())
        return select

    def _prefilter_check(self, prefilter, model, device_u, D, select):
        """E93 first use on a grid. The screen's device table equals the prefilter's host table; then, on a
        deterministic synthetic field of this grid (rough enough to fold some cells; review E93: the run's own
        first state is often u = 0, where every index error would pass) and on the current state:
        |legacy per-cell minimum - jmin| <= e wherever jmin is finite, and every tile holding a legacy value
        <= delta is selected (the legacy screen's own J expression)."""
        if not torch.equal(D, torch.tensor(np.array(prefilter.D_host), dtype=D.dtype, device=D.device)):
            return False
        generator = torch.Generator(device=device_u.device)
        generator.manual_seed(93)
        spacing = torch.tensor(prefilter.spacing, dtype=torch.float64, device=device_u.device)     # (hx, hy, hz)
        synthetic = (torch.randn(device_u.shape, generator=generator, dtype=torch.float64, device=device_u.device)
                     *(.3*spacing)).contiguous()
        for field, chosen in ((synthetic, None), (device_u, select)):
            if chosen is None:
                chosen = prefilter.flagged_tiles(field, model._tile_size)
            if not self._prefilter_agrees(prefilter, model, field, D, chosen):
                return False
        return True

    @staticmethod
    def _prefilter_agrees(prefilter, model, field, D, select):
        jmin, e = prefilter.cell_values(field, model._tile_size)
        delta = prefilter.delta
        size, n = model._tile_size, model.grid.n_cells
        for number, start in enumerate(range(0, n, size)):     # no checkpoints: v11's sequence is kept
            end = min(start+size, n)
            indices = model.tile_indices(start, end)
            _, _, J, _ = (model.geometry(field.reshape(-1, 3)[indices], D, cofactor=False, with_curl=False)
                          if model._selective_kernels else geometry(field.reshape(-1, 3)[indices], D))
            legacy = torch.where(torch.isfinite(J), J, torch.full_like(J, torch.inf)).amin(dim=1)
            fast, bound = jmin[start:end], e[start:end]
            known = torch.isfinite(fast)
            if bool((known & ~((legacy-fast).abs() <= bound)).any()):
                return False
            if bool((legacy <= delta).any()) and not select[number]:
                return False
        return True

    def _node_min_device(self, lin, extension_activation, stats, compiled):
        """E94: node_min_jd's device array, from the fused per-cell pass when it is on (see __init__)."""
        from . import node_jd_fused as nf
        from ..work_control import WorkBudgetExceeded
        record = self.fused_node_jd_record
        dl = self.linearize(lin)
        legacy = dict(extension_activation=extension_activation, stats=stats, **({"compiled": True} if compiled else {}),
                      **({"top4": self.top4_select_record} if compiled and self.top4_select else {}))   # E100
        # E100: the first part of this very call (node_min_prepare), if the caller ran it; otherwise the v17 order.
        # Every first part ends counted once: used, compared (first use) or discarded.
        overlap = self.estimator_overlap_record
        prepared, self._node_prepared = self._node_prepared, None
        use = prepared if prepared is not None and prepared.matches(dl, extension_activation, compiled) else None
        if prepared is not None and use is None:
            overlap["discarded"] += 1
        if use is not None and (not self.fused_node_jd or record["disabled"] or not nf.supported(dl._u.device)):
            overlap["discarded"] += 1
            use = None
        if not self.fused_node_jd or record["disabled"]:
            return dl.node_min_closed27_jd(**legacy)
        if not nf.supported(dl._u.device):
            record["unsupported_calls"] += 1
            return dl.node_min_closed27_jd(**legacy)
        problem = dl.model.problem
        grid_key = (tuple(problem.grid.shape), tuple(float(h) for h in problem.grid.spacing))
        failure = minima = cand = None
        nonfinite = False
        try:
            fused = self._fused_node
            source = problem.barrier_ops.coefficients
            if (fused is None or fused.key != grid_key
                    or not (fused.D_host is source or np.array_equal(fused.D_host, source))):
                fused = None
                try:
                    fused = nf.FusedNodeJD(problem.grid, source, dl._u.device)
                    fused.key = grid_key
                except InputValidationError:
                    failure = "unavailable"
                self._fused_node = fused
            if use is not None and use.fused is not fused:
                overlap["discarded"] += 1
                use = None
            if fused is not None:
                if use is not None and grid_key+(True,) in self._overlap_verified:
                    # E100: the outcome of this same kernel call on this same u, made before the diagonal was queued
                    overlap["used"] += 1
                    if use.failure is not None:
                        raise use.failure
                    minima, cand, nonfinite = use.minima, use.cand, use.nonfinite
                else:
                    # v17's own call (E100 first use: the prepared outcome is compared with these values below)
                    minima, cand, nonfinite = fused.cell_values(dl._u, extension_activation)
        except WorkBudgetExceeded:
            raise
        except Exception as error:
            failure = error
        if use is not None and grid_key+(True,) not in self._overlap_verified:
            # E100 first use (review E100, round 2): the prepared pass and this call must agree in kind -- failure,
            # nonfinite flag, or values (values are compared in _node_with_estimates); a difference switches the
            # overlap off. Either way this call continues exactly as v17's.
            kind = lambda f, n: "failure" if f is not None else ("nonfinite" if n else "values")
            fresh, early = kind(failure, nonfinite), kind(use.failure, use.nonfinite)
            if fresh != "values" or early != "values":
                overlap["compared"] += 1
                if fresh != early:
                    overlap["mismatches"] += 1
                    overlap["disabled"] = True
                use = None
        if failure is not None:                      # the torch path runs outside the guard (its errors are its own)
            self._fused_node_failed(failure)
            return dl.node_min_closed27_jd(**legacy)
        if nonfinite:
            # The torch path sees a nonfinite J; it runs unchanged (and fails exactly as it would in v12).
            record["nonfinite_fallbacks"] += 1
            return dl.node_min_closed27_jd(**legacy)
        key = grid_key+(extension_activation is not None,)
        if key not in self._fused_node_verified:
            if use is not None:                      # (prepare requires this grid's verified fused pass)
                overlap["discarded"] += 1
            return self._fused_node_first_use(dl, fused, key, legacy)
        snapshot = _copy_module.deepcopy(stats)
        try:
            if use is not None and use.estimates is not None:
                out = self._node_with_estimates(dl, use, legacy, minima, cand, key, snapshot)    # E100
            else:
                out = dl.node_min_closed27_jd(**legacy, fused_values=(minima, cand, self._guarded_j27(fused, dl._u)))
        except _FusedNodeFailure as failure:
            self._restore_stats(stats, snapshot)
            self._fused_node_failed(failure.error)
            return dl.node_min_closed27_jd(**legacy)
        record["calls"] += 1
        return out

    def _fused_node_failed(self, failure):
        record = self.fused_node_jd_record
        if failure == "unavailable":
            record["unavailable"] += 1
        else:
            record["errors"] += 1
            record["error_types"] = sorted(set(record["error_types"]) | {type(failure).__name__})
        record["disabled"] = True

    @staticmethod
    def _guarded_j27(fused, device_u):
        """cell_j27 with the kernel's own failures marked (a work deadline propagates unchanged)."""
        from ..work_control import WorkBudgetExceeded
        def j27(cells):
            try:
                return fused.cell_j27(device_u, cells)
            except WorkBudgetExceeded:
                raise
            except Exception as error:
                raise _FusedNodeFailure(error) from error
        return j27

    @staticmethod
    def _restore_stats(stats, snapshot):
        if stats is not None:
            stats.clear()
            stats.update(snapshot)

    # ---------------------------------------------------------------- E101 (throughput-v19)

    @contextmanager
    def gpu_certificate_scope(self):
        """T84: trial certificates may use the GPU edge-hull margins only inside this scope (the outer loop opens it
        around the inner solve). Every other certificate -- transfer, reference refresh, cpu_state, the initial and
        zero baseline states, plan D -- stays on the CPU. Without gpu_edge_hull the scope does nothing."""
        self._gpu_scope += 1
        try:
            yield
        except BaseException:
            self.commit_recheck_discard()      # an early plan D recheck never outlives the inner solve's exception
            raise
        finally:
            self._gpu_scope -= 1

    def _gpu_margins(self):
        """The GPU margins callable (parallel_margins' signature and result), or None when T84 is off for this run."""
        from . import edge_hull_gpu
        record = self.gpu_certificate_record
        if record["disabled"]:
            return None
        if not edge_hull_gpu.supported(self.device.device):
            record["unsupported"] += 1                       # review E101: never silently unavailable
            return None
        if self._gpu_hull is None:
            self._gpu_hull = edge_hull_gpu.GpuEdgeHull(self.device.device, record)
            seed = getattr(self, "_gpu_hull_seed", None)
            if seed:
                # v38 (gpu_hull_first_use_seed): these (grid id, delta) keys' synthetic folding field matched in the dry pass
                # (first_use_registry.seed_backend); the first use on this run's own input still compares GPU and CPU
                self._gpu_hull._synthetic.update(seed)
        return self._gpu_hull.margins

    def _recheck_submit(self, problem, u):
        from concurrent.futures import ThreadPoolExecutor
        if self._recheck_pool is None:
            self._recheck_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jd-curl-recheck")
        return self._recheck_pool.submit(certify_map, u, problem.grid, config=problem.certificate_config)

    def commit_recheck_prestart(self, problem, state):
        """T84 plan D, started early: the inner solve knows this state is its outer candidate (the accepted-step budget
        is reached, so every exit returns it); the same CPU certificate of the same frozen u starts now."""
        if not self.commit_recheck:
            return
        self.commit_recheck_discard()
        self._prestart = (problem, state.u, self._recheck_submit(problem, state.u))
        self.gpu_certificate_record["recheck_prestarts"] += 1

    def commit_recheck_discard(self):
        """Drop an early recheck no candidate will use (waiting for it: no thread outlives its decision)."""
        prestart, self._prestart = self._prestart, None
        if prestart is not None:
            self.gpu_certificate_record["recheck_prestarts_dropped"] += 1
            self.commit_recheck_drain((None, prestart[2]))

    def commit_recheck_start(self, problem, candidate):
        """T84 plan D: the outer candidate's full CPU certificate in the background (it overlaps the outer norms and
        E/F comparison) -- the early one when it was started for this very array, problem grid and certificate
        configuration; None when plan D is off."""
        if not self.commit_recheck:
            return None
        prestart, self._prestart = self._prestart, None
        if prestart is not None and prestart[1] is candidate.u and prestart[0] is problem:
            self.gpu_certificate_record["recheck_prestarts_used"] += 1
            return (candidate, prestart[2])
        if prestart is not None:
            self.gpu_certificate_record["recheck_prestarts_dropped"] += 1
            self.commit_recheck_drain((None, prestart[2]))
        return (candidate, self._recheck_submit(problem, candidate.u))

    def commit_recheck_join(self, recheck):
        """T84 plan D (D-a): (candidate, verdict). 'same': the CPU certificate equals the candidate's field by field
        (float bits); 'cpu_certified': they differ but the CPU certifies -- the candidate carries the CPU certificate
        from here on and the GPU part is off for the rest of the run; 'cpu_not_certified': they differ and the CPU
        does not certify -- the caller ends the run (gpu_certificate_mismatch) without committing."""
        from dataclasses import asdict, replace
        candidate, future = recheck
        record = self.gpu_certificate_record
        tick = perf_counter()
        try:
            cpu = future.result()
        except Exception as error:
            # review E101: a CPU re-certification that fails is a failed check (D-a: no commit, the run ends)
            record["recheck_wait_seconds"] += perf_counter()-tick
            record["commit_rechecks"] += 1
            record["recheck_mismatches"] += 1
            record["recheck_errors"] += 1
            record["disabled"] = True
            self._keep_mismatch(candidate.certificate, None, "cpu_not_certified", error)
            return candidate, "cpu_not_certified"
        record["recheck_wait_seconds"] += perf_counter()-tick
        record["commit_rechecks"] += 1
        if _float_bits(asdict(cpu)) == _float_bits(asdict(candidate.certificate)):
            return candidate, "same"
        record["recheck_mismatches"] += 1
        record["disabled"] = True
        if cpu.status != "certified_feasible" or cpu.state_hash != candidate.certificate.state_hash:
            self._keep_mismatch(candidate.certificate, cpu, "cpu_not_certified")
            return candidate, "cpu_not_certified"
        self._keep_mismatch(candidate.certificate, cpu, "cpu_certified")
        rebound = replace(candidate, certificate=cpu)
        self._lin = self._candidate = self._prepared = None             # device caches keyed on the old object
        self._outer_start = self._first_gradient = self._joint_stash = self._outer_stash = None
        return rebound, "cpu_certified"

    def commit_recheck_drain(self, recheck):
        """T84 plan D: wait for a recheck whose join will not happen (an exception left the outer checks, or an early
        recheck no candidate used). A recheck that raised is recorded (failure evidence), never silently dropped."""
        try:
            recheck[1].result()
        except Exception as error:
            self.gpu_certificate_record["recheck_dropped_errors"] += 1
            self._keep_mismatch(None, None, "dropped_recheck_error", error)

    def _keep_mismatch(self, gpu, cpu, verdict, error=None):
        """Failure evidence (hard constraint 5): both certificates' fields (float bits as hex) and the verdict, at most
        MISMATCH_DETAILS entries."""
        from dataclasses import asdict
        details = self.gpu_certificate_record["recheck_mismatch_details"]
        if len(details) >= MISMATCH_DETAILS:
            return
        hexed = lambda r: None if r is None else {k: (v.hex() if isinstance(v, float) else repr(v))
                                                  for k, v in asdict(r).items()}
        gpu = gpu if gpu is None or hasattr(gpu, "__dataclass_fields__") else None
        details.append(dict(verdict=verdict, gpu_path=hexed(gpu), cpu=hexed(cpu),
                            error=None if error is None else f"{type(error).__name__}: {error}"))

    # ---------------------------------------------------------------- E100 (throughput-v18)

    def node_min_prepare(self, lin, extension_activation, compiled, *, slot=True, keep_rows=None):
        """E100 (estimator_overlap): the device part of the next node_min_jd(lin, extension_activation=...,
        compiled=...) call made now -- the fused per-cell pass and every tile's candidate cells and closed27 rows,
        the same calls on the same u -- before the caller queues the GGN diagonal; estimate() of the returned object
        then runs the host estimator on those rows, tile by tile in order, while the diagonal kernel runs. That call
        uses these results in place of computing them (its stats, scatter and checkpoints are unchanged). None (the
        v17 order) unless the fused pass is on, verified for this grid with the extension, and current. No operation
        scope or work checkpoint is entered here; a failure is kept and raised where the call would have met it."""
        from . import node_jd_fused as nf
        from ..work_control import WorkBudgetExceeded
        if slot:                            # E103: slot=False (the metric prestart) keeps the E100 slot untouched
            self._node_prepared = None
        record = self.estimator_overlap_record
        if (not self.estimator_overlap or record["disabled"] or extension_activation is None
                or not self.fused_node_jd or self.fused_node_jd_record["disabled"]):
            return None
        tick = perf_counter()
        dl = self.linearize(lin)
        if not nf.supported(dl._u.device):
            return None
        problem = dl.model.problem
        grid_key = (tuple(problem.grid.shape), tuple(float(h) for h in problem.grid.spacing))
        fused, source = self._fused_node, problem.barrier_ops.coefficients
        if (fused is None or fused.key != grid_key or grid_key+(True,) not in self._fused_node_verified
                or not (fused.D_host is source or np.array_equal(fused.D_host, source))):
            return None
        top4 = self.top4_select_record if compiled and self.top4_select else None
        prepared = _NodePrepared(dl, extension_activation, compiled, fused, record,
                                 dl.node_estimator(compiled, top4))
        try:
            minima, cand, nonfinite = fused.cell_values(dl._u, extension_activation)
        except WorkBudgetExceeded:
            raise
        except Exception as error:
            prepared.failure = error
        else:
            prepared.nonfinite = nonfinite
            if not nonfinite:
                try:
                    prepared.minima = minima.clone()                      # minima is the kernel's reused buffer
                except Exception:
                    if slot:
                        record["prepare_errors"] += 1     # review E100: an allocation v17 does not make -> the v17 order
                    else:
                        self.metric_prestart_record["errors"] += 1      # review E103: the prestart's own record
                    return None
                prepared.cand = cand
                estimates = NodeEstimates(keep_rows=(grid_key+(True,) not in self._overlap_verified
                                                     if keep_rows is None else keep_rows))
                estimates.gather(cand, self._guarded_j27(fused, dl._u), dl.model._tile_size, dl.model.grid.n_cells)
                prepared.estimates = estimates
        if slot:
            self._node_prepared = prepared
            record["prepared"] += 1
            record["first_part_seconds"] += perf_counter()-tick
        else:                               # E103: counted in E100's record only when a build adopts it
            self.metric_prestart_record["first_part_seconds"] += perf_counter()-tick
        return prepared

    def discard_node_prepared(self):
        """E100: drop a first part its node_min_jd call did not consume (pg_metric_inverse ended or raised before)."""
        if self._node_prepared is not None:
            self._node_prepared = None
            self.estimator_overlap_record["discarded"] += 1

    # ---------------------------------------------------------------- GG2 (explicit option, metric only)

    ggn_reuse = False

    def configure_ggn_reuse(self, theta, period):
        """GG2 (metric only, explicit option): the PG
        metric's GGN diagonal (the equilibration scale of T59/T62/T73) is built from per-cell locals kept across outers.
        Build n recomputes the cells R_n = N_n | N_{n-1}, N_n = {c : E94 cell minimum of J at u_n < theta * act}, and
        keeps every other cell's last locals; every period-th build of a grid and weights (the first included) is full
        and bit for bit the default build. The image/proximal node part is recomputed every build. With theta > 1 every
        cell outside R_n had all closed27 J >= act when its locals were computed, so its barrier part is zero then and
        now: only the soft-geometry part of far cells is stale. F, E, barrier, certificate, E/F, GF are unchanged."""
        theta, period = float(theta), int(period)
        if not (np.isfinite(theta) and 1.25 <= theta <= 100.) or not 1 <= period <= 1000:
            raise InputValidationError("GG2: theta in [1.25, 100] and period in [1, 1000] required")
        if not (self.fused_diagonal and self.estimator_overlap and self.fused_node_jd):
            raise InputValidationError("GG2 needs the fused diagonal, the E94 fused node JD and the E100 estimator overlap")
        self.ggn_reuse = True
        self._ggn_theta, self._ggn_period, self._ggn_store = theta, period, None
        self.ggn_reuse_record = dict(
            policy=GGN_REUSE_POLICY, theta=theta, period=period, builds=0, full_builds=0, partial_builds=0,
            full_reasons=dict(first=0, key_change=0, period=0, first_use=0, first_use_barrier=0),
            fallbacks=dict(unsupported=0, no_minima=0, invalid_diagonal=0, partial_error=0), errors=[], refresh_cells=[], cells=None,
            first_use=dict(compared=0, matched=0, mismatches=0, errors=0, barrier_exercised=0,
                           soft_verified=[], verified=[], probe_cells=[]),
            launch_seconds=0., disabled=False)

    def diagonal_launch(self, lin):
        """The E89 queued diagonal; GG2 (ggn_reuse) builds it from the kept locals when its conditions hold."""
        if not self.ggn_reuse:
            return super().diagonal_launch(lin)
        tick = perf_counter()
        try:
            return self._ggn_reuse_launch(lin)
        finally:
            self.ggn_reuse_record["launch_seconds"] += perf_counter()-tick    # host side of the launch (syncs included)

    @staticmethod
    def _ggn_barrier_cells(fused, u, cells, act):
        """GG2: whether some listed cell (int32 device tensor) has a closed27 J < act or NaN (E94 cell_j27 values of the
        prepared measure's fused object); None when that object is missing (callers then assume the worst)."""
        if not int(cells.numel()):
            return False
        if fused is None:
            return None
        j = fused.cell_j27(u, cells)
        return bool(((j < act) | torch.isnan(j)).any())

    def _ggn_first_use(self, f, fused, u, local, near, act, vkey):
        """GG2: the indexed kernel on the near cells and the first 3*32+5 cells against this full build's locals, bit
        for bit. A match verifies the soft part for this grid and weights (vkey), and the barrier part once some compared
        cell has a closed27 J < act; a mismatch or an error switches GG2 off for the run (this build is the default one)."""
        rec = self.ggn_reuse_record
        part = rec["first_use"]
        scratch = None
        try:
            probe = near.clone()
            probe[:min(int(probe.numel()), 3*32+5)] = True
            cells = probe.nonzero().view(-1).to(torch.int32)
            scratch = torch.empty_like(local)
            f.diagonal_locals_indexed(u, cells, scratch)
            rows = cells.to(torch.int64)
            same = torch.equal(scratch[rows].view(torch.int64), local[rows].view(torch.int64))
            exercised = self._ggn_barrier_cells(fused, u, cells, act) is True
            part["probe_cells"].append(int(cells.numel()))
        except WorkBudgetExceeded:
            raise
        except Exception as error:
            part["errors"] += 1
            rec["disabled"] = True
            rec["errors"].append(dict(where="first_use", type=type(error).__name__, message=str(error)[:300]))
            return
        finally:
            scratch = None
        part["compared"] += 1
        if not same:
            part["mismatches"] += 1
            rec["disabled"] = True
            return
        part["matched"] += 1
        if vkey not in part["soft_verified"]:
            part["soft_verified"].append(vkey)
        if exercised:
            part["barrier_exercised"] += 1
            part["verified"].append(vkey)

    def _ggn_drop(self, dl, reason):
        """GG2: the default diagonal of dl, with the kept locals dropped (the next build is full)."""
        self.ggn_reuse_record["fallbacks"][reason] += 1
        self._ggn_store = None
        return dl.diagonal()

    def _ggn_reuse_launch(self, lin):
        rec = self.ggn_reuse_record
        dl = self.linearize(lin)
        m = dl.model
        f = m._fused if m._fused_diagonal else None
        if rec["disabled"] or f is None:
            if rec["disabled"]:
                self._ggn_store = None
                return _PendingDiagonal(self, dl.diagonal())
            return _PendingDiagonal(self, self._ggn_drop(dl, "unsupported"))
        prepared, u = self._node_prepared, dl._u
        if (prepared is None or prepared.minima is None or prepared.dl is not dl or prepared.u is not u
                or prepared.version != u._version):
            return _PendingDiagonal(self, self._ggn_drop(dl, "no_minima"))
        problem = m.problem
        act = float(problem.barrier_activation)
        key = (problem.grid.grid_id, float(problem.weights.mu), float(problem.weights.nu), float(problem.barrier_weight),
               float(problem.certificate_config.delta), act)
        vkey = list(key)                                       # first-use verification per grid and weights (JSON-safe)
        fused = getattr(prepared, "fused", None)
        near = prepared.minima < self._ggn_theta*act
        store, part = self._ggn_store, rec["first_use"]
        n = 0 if store is None or store["key"] != key else store["count"]
        reason = cells = None
        if store is None:
            reason = "first"
        elif store["key"] != key:
            reason = "key_change"
        elif n % self._ggn_period == 0:
            reason = "period"
        elif vkey not in part["soft_verified"]:
            reason = "first_use"
        if reason is None:
            failure = base = out = None
            try:
                cells = (near | store["near"]).nonzero().view(-1).to(torch.int32)
                # until the barrier part is verified, refreshed barrier-active cells (or an unknown answer) -> full + check
                if vkey not in part["verified"] and self._ggn_barrier_cells(fused, u, cells, act) is not False:
                    reason = "first_use_barrier"
                else:
                    base = dl.diagonal_base()
                    f.diagonal_locals_indexed(u, cells, store["local"])
                    out = m.restrict(f.gather_nodes(base, store["local"]))
            except WorkBudgetExceeded:
                raise
            except Exception as error:
                failure = dict(where="partial", type=type(error).__name__, message=str(error)[:300])
            if failure is not None:
                # review GG2 M3/N1: an error here -> the default from now on. The kept locals (1.3 GB at native) are
                # released outside the except block (whose traceback keeps them alive) before the default build.
                store = cells = base = out = None
                self._ggn_store = None
                rec["disabled"] = True
                rec["errors"].append(failure)
                return _PendingDiagonal(self, self._ggn_drop(dl, "partial_error"))
            if reason is None:
                store["near"], store["count"] = near, n+1
                rec["builds"] += 1
                rec["cells"] = f.ncell
                rec["partial_builds"] += 1
                rec["refresh_cells"].append(int(cells.numel()))
                return _PendingReuseDiagonal(self, out, lambda: self._ggn_drop(dl, "invalid_diagonal"))
        base = dl.diagonal_base()
        rec["builds"] += 1
        rec["cells"] = f.ncell
        self._ggn_store = store = None                         # the old locals go before the full launch
        local = f.diagonal_locals(u)
        if vkey not in part["verified"]:
            self._ggn_first_use(f, fused, u, local, near, act, vkey)
        if not rec["disabled"]:
            self._ggn_store = dict(key=key, local=local, near=near, count=n+1)
        rec["full_builds"] += 1
        rec["full_reasons"][reason] += 1
        return _PendingDiagonal(self, m.restrict(f.gather_nodes(base, local)))

    # ---------------------------------------------------------------- E103 (throughput-v21, M1)

    def metric_prestart_start(self, problem, base, lin, alpha_fn, alpha_key, node):
        """E103: at the inner step budget (base is the outer candidate) the next metric build's host parts start --
        the estimator's device first part now (node_min_prepare, outside the E100 slot), then in one background thread
        the estimator over its rows (same tiles, chunks and pool) and alpha_fn() (the T59 shift). No operation scope
        or work checkpoint. node = (activation, compiled) or None; alpha_key = (shift_factor, cached) or None."""
        from types import SimpleNamespace
        self.metric_prestart_drain()
        record = self.metric_prestart_record
        if not self.metric_prestart or record["disabled"] or (alpha_fn is None and node is None):
            return None
        def job():
            if prepared is not None:
                prepared.estimate()
            return None if alpha_fn is None else alpha_fn()
        try:
            prepared = None if node is None else self.node_min_prepare(lin, node[0], node[1], slot=False)
            if self._prestart_pool is None:
                from concurrent.futures import ThreadPoolExecutor
                # (the prefix differs from thread_chunks' scan prefix: chunks submitted from it go to the shared pool)
                self._prestart_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jd-curl-metric-prestart")
            future = self._prestart_pool.submit(job)
        except Exception as error:      # (no checkpoint is entered there) the next build runs in the v20 order
            record["errors"] += 1
            if len(record["error_types"]) < 8:
                record["error_types"].append(type(error).__name__)
            return None
        self._metric_prestart = SimpleNamespace(
            grid=problem.grid, u=base.u, b=base.image_grad, coefficients=problem.barrier_ops.coefficients,
            node=node, alpha_key=alpha_key, prepared=prepared, alpha=None, future=future)
        record["prestarts"] += 1
        return self._metric_prestart

    def _prestart_join(self, entry):
        record = self.metric_prestart_record
        tick = perf_counter()
        try:
            entry.alpha = entry.future.result()
        except Exception as error:
            record["errors"] += 1
            if len(record["error_types"]) < 8:
                record["error_types"].append(type(error).__name__)
            return False
        finally:
            record["wait_seconds"] += perf_counter()-tick
        return True

    def metric_prestart_take(self, problem, base, alpha_key, node):
        """E103: the prestart for this metric build after its thread ended, when everything it read is the same: the
        grid object, the frozen u and image gradient objects (E76 rule), the barrier coefficients, the node
        (activation, compiled) and alpha (shift_factor, cached) keys. Otherwise None (dropped; the v20 order)."""
        entry, self._metric_prestart = self._metric_prestart, None
        self._prestart_taken = None
        if entry is None:
            return None
        record = self.metric_prestart_record
        if not self._prestart_join(entry):
            record["dropped"] += 1
            return None
        same = (not record["disabled"] and entry.grid is problem.grid and entry.u is base.u
                and entry.b is base.image_grad and _identity._immutable(entry.u) and _identity._immutable(entry.b)
                and entry.coefficients is problem.barrier_ops.coefficients
                and entry.node == node and entry.alpha_key == alpha_key)
        if not same:
            record["dropped"] += 1
            return None
        entry.used = False
        self._prestart_taken = entry
        return entry

    def metric_prestart_drain(self):
        """E103: an unused prestart is dropped after its thread ended (no thread outlives the decision)."""
        entry, self._metric_prestart = self._metric_prestart, None
        if entry is not None:
            self._prestart_join(entry)
            self.metric_prestart_record["dropped"] += 1

    @staticmethod
    def _prestart_key(problem, part):
        return (tuple(problem.grid.shape), tuple(float(h) for h in problem.grid.spacing), part)

    def metric_prestart_ready(self, problem, entry, part):
        """E103: the prestarted part ("alpha" or "node") may be used: it exists and its first use on this grid passed."""
        if entry is None or self.metric_prestart_record["disabled"]:
            return False
        present = entry.alpha_key is not None if part == "alpha" else entry.prepared is not None
        return present and self._prestart_key(problem, part) in self._prestart_verified

    def metric_prestart_adopt(self, lin, entry):
        """E103: a verified prestarted first part takes the E100 slot for this build's node_min_jd call (rebound to its
        device state); False when the device u is not the same tensor (then the v20 order runs)."""
        if not entry.prepared.rebind(self.linearize(lin)):
            return False
        self._node_prepared = entry.prepared
        self.estimator_overlap_record["prepared"] += 1       # (E100's count: one first part per build, as in v20)
        entry.used = True
        self.metric_prestart_record["estimates_used"] += 1
        return True

    def metric_prestart_alpha(self, entry):
        entry.used = True
        self.metric_prestart_record["alpha_used"] += 1
        return entry.alpha

    def metric_prestart_check(self, problem, entry, part, fresh):
        """E103 first use per (grid, part): the prestarted value against this build's own (v20) one, bit for bit --
        alpha by its float bits; the first part by kind, node minima (int64 bits), candidate mask, and every tile's
        candidate cells and estimator values. Equal: the part is verified for this grid. Different: the prestart is
        switched off for the rest of the run, with both sides' summary kept as failure evidence."""
        record = self.metric_prestart_record
        key = self._prestart_key(problem, part)
        if key in self._prestart_verified or record["disabled"]:
            return key in self._prestart_verified
        early = entry.alpha if part == "alpha" else entry.prepared
        if part == "alpha":
            same = isinstance(early, float) and isinstance(fresh, float) and _float_bits(early) == _float_bits(fresh)
            informative = True
        else:
            same = early is not None and fresh is not None and _same_prepared(early, fresh)
            # review E103: a comparison without any estimator value (both failed or nonfinite, no estimates, no
            # candidates) counts, but verifies nothing (as E100's first use)
            informative = same and fresh.failure is None and not fresh.nonfinite and fresh.estimates is not None \
                and any(tile[4] is not None for tile in fresh.estimates.tiles)
        record["compared"] += 1
        entry.used = True
        if same and not informative:
            record["uninformative_comparisons"] += 1
            return False
        if same:
            self._prestart_verified.add(key)
            record["verified_grids"].append([list(key[0]), list(key[1]), part])
            return True
        record["mismatches"] += 1
        record["disabled"] = True
        if len(record["mismatch_details"]) < 8:
            describe = (lambda v: float(v).hex() if isinstance(v, float) else repr(v)) if part == "alpha" else \
                (lambda p: None if p is None else dict(failure=repr(p.failure), nonfinite=p.nonfinite,
                                                       candidates=None if p.estimates is None else p.estimates.candidates()))
            record["mismatch_details"].append(dict(part=part, grid=[list(key[0]), list(key[1])],
                                                   prestart=describe(early), fresh=describe(fresh)))
        return False

    def metric_prestart_finish(self):
        """E103: the prestart this build took counts as used (a part used or compared) or dropped."""
        entry, self._prestart_taken = self._prestart_taken, None
        if entry is not None:
            self.metric_prestart_record["used" if entry.used else "dropped"] += 1

    def _node_with_estimates(self, dl, use, legacy, minima, cand, key, snapshot):
        """E100: node_min_closed27_jd with the prepared per-tile values. The first use per (level grid, extension)
        returns the v17 computation (fused values, j27 and estimator at their places, its own checkpoints) and
        compares the prepared path with it (no checkpoints, a copy of the stats): node array as int64, stats by
        float bits, every tile's candidate cells and rows. A mismatch (or an exception only the prepared path
        raises) switches the overlap off for the rest of the run; candidates on some tile make the check count."""
        record = self.estimator_overlap_record
        j27 = self._guarded_j27(use.fused, dl._u)
        if key in self._overlap_verified:
            return dl.node_min_closed27_jd(**legacy, fused_values=(minima, cand, j27), estimates=use.estimates)
        # minima, cand: v17's own cell_values call at this place (review E100: the prepared pass is compared too)
        trace = []
        out = dl.node_min_closed27_jd(**legacy, fused_values=(minima, cand, j27), trace=trace)
        check_stats = _copy_module.deepcopy(snapshot)
        try:
            same = (use.minima.shape == minima.shape                    # (both are values: checked by the caller)
                    and torch.equal(use.minima.view(torch.int64), minima.view(torch.int64))
                    and use.cand is not None and torch.equal(use.cand, cand))
            if same:
                check = dl.node_min_closed27_jd(**dict(legacy, stats=check_stats), checkpoints=False,
                                                fused_values=(use.minima, use.cand, None), estimates=use.estimates)
                same = (torch.equal(out.view(torch.int64), check.view(torch.int64))
                        and _float_bits(legacy["stats"]) == _float_bits(check_stats)
                        and use.estimates.same_as(trace))
        except Exception:
            same = False
        record["compared"] += 1
        if not same:
            record["mismatches"] += 1
            record["disabled"] = True
        elif use.estimates.informative():
            self._overlap_verified.add(key)
            record["verified"].append([list(key[0]), list(key[1]), use.estimates.candidates()])
        return out

    def _fused_node_first_use(self, dl, fused, key, legacy):
        """E94 first use per (level grid, extension on/off): on a deterministic synthetic field of this grid
        (smooth with sparse rough patches, so the extension has candidates; review E94: the run's own first state
        is often u = 0, J == 1, or has no candidates) and on the current state, the fused result must equal the
        torch path bit for bit (node array as int64, stats compared with float bits). None of these runs passes a
        work checkpoint; every return walks this call's tile checkpoints once, so the checkpoint sequence stays
        v12's. A mismatch or a kernel failure switches the fused pass off; the torch result and stats of this call
        are returned."""
        from ..work_control import WorkBudgetExceeded
        record = self.fused_node_jd_record
        stats = legacy["stats"]
        snapshot = _copy_module.deepcopy(stats)
        reference_stats = _copy_module.deepcopy(stats)
        # the torch result of this call (its errors propagate exactly as in v12)
        reference = dl.node_min_closed27_jd(**dict(legacy, stats=reference_stats), checkpoints=False)
        try:
            same, synthetic_candidates = self._fused_node_agrees_on_synthetic(dl, fused, legacy)
            if same:
                # the synthetic pass reused the kernel's output buffers: this state's values are computed again
                minima, cand, nonfinite = fused.cell_values(dl._u, legacy["extension_activation"])
                fused_stats = _copy_module.deepcopy(snapshot)
                result = dl.node_min_closed27_jd(**dict(legacy, stats=fused_stats), checkpoints=False,
                                                 fused_values=(minima, cand, self._guarded_j27(fused, dl._u)))
                same = (torch.equal(reference.view(torch.int64), result.view(torch.int64))
                        and _float_bits(reference_stats) == _float_bits(fused_stats))
        except WorkBudgetExceeded:
            raise
        except _FusedNodeFailure as failure:
            self._restore_stats(stats, reference_stats)
            self._fused_node_failed(failure.error)
            return self._walk_checkpoints(dl, reference)
        except Exception as error:
            # anything else raised by the extra checking work (not part of v12) only switches the fused pass off
            self._restore_stats(stats, reference_stats)
            self._fused_node_failed(error)
            return self._walk_checkpoints(dl, reference)
        if not same:
            self._restore_stats(stats, reference_stats)
            record["mismatches"] += 1
            record["disabled"] = True
            return self._walk_checkpoints(dl, reference)
        self._restore_stats(stats, fused_stats)
        self._fused_node_verified.add(key)
        record["verified"].append([list(key[0]), list(key[1]), bool(key[2]), synthetic_candidates])
        record["calls"] += 1
        return self._walk_checkpoints(dl, result)

    @staticmethod
    def _synthetic_node_field(u, spacing):
        """E94: deterministic test field on this grid: smooth N(0, (.05h)^2) plus N(0, (.5h)^2) on the distinct nodes
        of max(8, nodes/20000) 3x3x3 patches (independent noise there puts cell minima inside the cell, i.e. T69
        candidates). Patch nodes are made distinct before adding, so no atomic accumulation order is involved."""
        generator = torch.Generator(device=u.device)
        generator.manual_seed(94)
        h = torch.tensor(spacing, dtype=torch.float64, device=u.device)
        field = (torch.randn(u.shape, generator=generator, dtype=torch.float64, device=u.device)*(.05*h)).contiguous()
        nz, ny, nx = field.shape[:3]
        count = max(8, field.numel()//3//20000)
        centre = torch.stack([torch.randint(1, n-1, (count,), generator=generator, device=u.device) for n in (nz, ny, nx)], -1)
        steps = torch.tensor([(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)], device=u.device)
        zyx = (centre[:, None, :]+steps[None]).reshape(-1, 3)
        flat = torch.unique((zyx[:, 0]*ny+zyx[:, 1])*nx+zyx[:, 2])
        noise = torch.randn((flat.numel(), 3), generator=generator, dtype=torch.float64, device=u.device)*(.5*h)
        field.view(-1, 3).index_add_(0, flat, noise)
        return field

    def _fused_node_agrees_on_synthetic(self, dl, fused, legacy):
        """(same, candidates): node array, stats, and per tile the cell values, candidate cells and their rows of the
        fused and torch passes on the synthetic field; plus cell_j27 on a list spanning several kernel blocks against
        the torch path's own values of the first cells (review E94: the node minimum hides most single-cell differences)."""
        from .node_jd_fused import BLOCK
        field = self._synthetic_node_field(dl._u, dl.model.grid.spacing)
        probe = copy(dl)
        probe._u = field
        args = dict(legacy, stats={} if legacy["stats"] is not None else None)
        reference_trace, fused_trace = [], []
        reference = probe.node_min_closed27_jd(**args, checkpoints=False, trace=reference_trace)
        minima, cand, nonfinite = fused.cell_values(field, legacy["extension_activation"])
        if nonfinite:
            return False, None
        j27 = self._guarded_j27(fused, field)
        fused_args = dict(legacy, stats={} if legacy["stats"] is not None else None)
        result = probe.node_min_closed27_jd(**fused_args, checkpoints=False, fused_values=(minima, cand, j27),
                                            trace=fused_trace)
        bits = lambda t: None if t is None else t.contiguous().view(torch.int64)
        same = (torch.equal(reference.view(torch.int64), result.view(torch.int64))
                and _float_bits(args["stats"]) == _float_bits(fused_args["stats"])
                and len(reference_trace) == len(fused_trace))
        for a, b in zip(reference_trace, fused_trace) if same else ():
            same = (same and a["start"] == b["start"] and torch.equal(bits(a["minima"]), bits(b["minima"]))
                    and ((a["cand"] is None) == (b["cand"] is None)) and (a["cand"] is None or torch.equal(a["cand"], b["cand"]))
                    and ((a["rows"] is None) == (b["rows"] is None)) and (a["rows"] is None or torch.equal(bits(a["rows"]), bits(b["rows"]))))
        if same:
            first = torch.cat([e["j"] for e in reference_trace if "j" in e])
            cells = torch.arange(min(first.shape[0], 3*BLOCK+5), device=field.device)
            same = torch.equal(bits(j27(cells)), bits(first[:cells.numel()]))
        return same, (None if cand is None else int(cand.sum()))

    def _walk_checkpoints(self, dl, value):
        """E94: a returned first-use reference passes the same tile checkpoints as v12's call would have."""
        size, n = dl.model._tile_size, dl.model.grid.n_cells
        for _ in dl.model.tiles(np.zeros(-(-n//size), dtype=bool)):
            pass
        return value

    @staticmethod
    def _all_finite(theta):
        """E89: np.all(np.isfinite(theta)) (the same predicate, compiled for large arrays)."""
        from .. import fast_checks
        if fast_checks.enabled(theta.size):
            return fast_checks.all_finite(theta)
        return bool(np.all(np.isfinite(theta)))

    def _submit_trial_digest(self, u):
        """v41 (H2): queue the SHA-256 of this frozen, owned u (problem.array_hash's function and memo) on a small
        dedicated pool; the certificate of this same u takes it (_take_trial_digest)."""
        if self._digest_pool is None:
            from concurrent.futures import ThreadPoolExecutor
            self._digest_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="jd-curl-uhash")
        if self._trial_digest is not None:                         # the previous trial never reached its certificate
            self._drop_trial_digest()                              # (an exception between inject and certificate)
        hasher = _identity._array_hash if _identity._IMMUTABLE_DIGESTS is None else _identity._array_hash_memo
        # a weak reference (review round 3): once the hash has run, nothing here keeps this u alive, so an exception
        # between inject and certificate (e.g. a time box) cannot keep an extra native u through the export
        self._trial_digest = (weakref.ref(u), self._digest_pool.submit(hasher, u))
        self.early_trial_hash_record["submitted"] += 1

    def _take_trial_digest(self, u):
        """v41 (H2): the queued digest future of exactly this u (identity), or None; cleared either way."""
        held, self._trial_digest = self._trial_digest, None
        if held is None or held[0]() is not u:                       # a dead reference never matches
            if held is not None:
                held[1].cancel()
                self.early_trial_hash_record["unused"] += 1
            return None
        self.early_trial_hash_record["used"] += 1
        return held[1]

    def _drop_trial_digest(self):
        """v41 (H2; review round 2): a trial that ends before its certificate (domain or witness rejection) lets go of
        its u at once, so no extra native u outlives it; a hash not yet started is cancelled, a running one finishes and
        its value is discarded. Counted unused."""
        held, self._trial_digest = self._trial_digest, None
        if held is not None:
            held[1].cancel()
            self.early_trial_hash_record["unused"] += 1

    def _owned_pair(self, problem, theta):
        """E89: the values of readonly_copy(theta) and readonly_copy(inject_free(theta)) without copies.
        An immutable C-contiguous theta cannot change (E76 premise) and is used as is; u is injected into a
        fresh owned array and frozen. The theta/u binding and theta's finite fact (checked above) are
        recorded here, where they hold by construction."""
        from .. import host_kernels
        if not (_identity._immutable(theta) and theta.flags.c_contiguous):
            theta = readonly_copy(theta)
        u = host_kernels.inject(theta, problem.grid) if host_kernels.enabled(theta.size) else inject_free(theta, problem.grid)
        u.flags.writeable = False
        _identity.store_fact(theta, _identity.FACT_FINITE)
        bind_theta_u(theta, u)
        return theta, u

    def _device_trial_values(self, model, device_u):
        """Device sampling + energy of one candidate (no state is published)."""
        with self._stage("trial_sample"):
            warped, gradient = self._sample(model, device_u)
            if self.early_warped:
                # E97: the download the trial_download stage makes, moved before the energy kernels (the host is
                # idle either way); its fingerprint hashes during the energy wait. Published only after the proof.
                host = self.device.download_owned(warped)
                self._staged_warped = (warped, host, _PendingFingerprint((host,), single=True)
                                       if self.device_batch is not None else None)
                self.early_warped_record["staged"] += 1
        with self._stage("trial_energy"):
            base = self._lin if (self.device_batch is not None and self._lin is not None
                and self._lin.model is model and self._lin.sample_fingerprint is not None
                and getattr(self._lin.state, 'evaluation_backend_id', None) == self.backend_id) else None
            screen = self._screening(model)
            if base is None:
                energy, differences = self._energy(model, device_u, warped, screen=screen), None
            else:
                energy, differences = self._energy(model, device_u, warped, base=base, screen=screen)
                self.fusion_counts['energy_differences'] += 1
            band = self._energy_band
            if screen:
                energy, differences, band = self._screen_check(model, device_u, warped, base, energy, differences, band)
            self._trial_band = band
        return warped, gradient, energy, base, differences

    def _screening(self, model):
        """T90: whether this trial's energy is screened in float32 (the switch is on, not disabled, the fused kernels
        with the samples barrier are in use)."""
        if not self.fp32_screen or self.fp32_screen_record["disabled"]:
            return False
        if model._fused is None or model._fused.bernstein:
            self.fp32_screen_record["unsupported"] += 1
            return False
        return True

    def _screen_check(self, model, device_u, warped, base, energy, differences, band):
        """T90 first-use and sampled checks: the float64 evaluation of the same trial; every screened term and the
        differences must lie within the band. A violation (or a failure) switches the screening off for the run and
        returns the float64 values for this trial."""
        record = self.fp32_screen_record
        grid = model.grid.grid_id
        n = self._screen_uses.get(grid, 0)
        self._screen_uses[grid] = n+1
        record["screened"] += 1
        if not (n < FP32_SCREEN_CHECKS or (n+1-FP32_SCREEN_CHECKS) % FP32_SCREEN_RECHECK == 0):
            return energy, differences, band
        try:
            if base is None:
                exact, exact_diff = self._energy(model, device_u, warped), None
            else:
                exact, exact_diff = self._energy(model, device_u, warped, base=base)
        except (FloatingPointError, torch.OutOfMemoryError):
            record["errors"] += 1
            record["disabled"] = True
            raise
        record["checks"] += 1
        # per term against its own band (the exact term's magnitude), the differences against the comparison band
        gaps = [abs(energy.jd-exact.jd)/max(FP32_EPS*FP32_BAND_SOFT*exact.jd, 1e-300),
                abs(energy.curl-exact.curl)/max(FP32_EPS*FP32_BAND_SOFT*exact.curl, 1e-300),
                abs(energy.barrier-exact.barrier)/max(FP32_EPS*FP32_BAND_BARRIER*exact.barrier, 1e-300)]
        if differences is not None:
            gaps += [abs(differences[0]-exact_diff[0])/max(band, 1e-300), abs(differences[1]-exact_diff[1])/max(band, 1e-300)]
        gaps = [0. if (g == 0 or g != g) and g == 0 else g for g in gaps]
        rel = max(gaps)
        record["max_rel"] = max(record["max_rel"], min(rel, 1e300))
        if not rel <= 1.:                                                      # NaN-safe
            record["mismatches"] += 1
            record["disabled"] = True
            if len(record["mismatch_details"]) < 8:
                record["mismatch_details"].append(dict(grid=grid, use=n+1, band=band, gaps=[float(g) for g in gaps]))
            return exact, exact_diff, None
        if n+1 == FP32_SCREEN_CHECKS:
            record["verified"].append(grid)
        return energy, differences, band

    def exact_energy(self, problem, state):
        """T90: the float64 energy terms of a device state (its own values when the screening is off). Used by the
        outer loop for every outer candidate's recorded energies, so the records and the audit keep the float64 meaning."""
        if not self.fp32_screen or getattr(state, "evaluation_backend_id", None) is None:
            return state.energy
        data = self._device_state(problem, state)
        self.fp32_screen_record["exact_energies"] += 1
        return self._energy(data.model, data._u, data._warped)

    def _evaluate_overlapped(self, problem, theta, u, model, device_u, certificate_observer):
        """Same decisions as the sequential path; only the wall-clock overlaps.

        The CPU proof is authoritative and awaited before anything is returned.
        A failed proof returns the identical InvalidTrial and discards device
        values. Device numerical failure after a passed proof uses the same CPU
        fallback as before. Device work of an uncertified map is never exposed.
        """
        from concurrent.futures import ThreadPoolExecutor
        if self._certificate_pool is None:
            self._certificate_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jd-curl-proof")
        if self.lean_certificate:
            # E98: the same certificate without the copy and the repeated checks; evaluate_trial has proved this
            # frozen u finite (injected from a finite theta) and checked its domain on this grid.
            from ..feasibility import certify_map_owned
            margins = self._gpu_margins() if self.gpu_edge_hull and self._gpu_scope else None    # E101 (T84)
            digest = self._take_trial_digest(u) if self.early_trial_hash else None              # v41 (H2)
            future = self._certificate_pool.submit(certify_map_owned, u, problem.grid, config=problem.certificate_config,
                                                   finite=True, domain_ok=True,
                                                   **({} if margins is None else dict(margins=margins)),
                                                   **({} if digest is None else dict(digest=digest,
                                                       digest_record=self.early_trial_hash_record)))
        else:
            future = self._certificate_pool.submit(certify_map, u, problem.grid, config=problem.certificate_config)
        values = device_error = None
        try:
            try:
                if model is None or device_u is None:
                    with self._stage("trial_upload"):
                        if model is None:
                            model = self._model_for(problem)
                        if device_u is None:
                            device_u = self.device.upload(u)
                values = self._device_trial_values(model, device_u)
            except (FloatingPointError, torch.OutOfMemoryError) as error:
                device_error = error
            with self._stage("trial_full_certificate"):
                certificate = future.result()
        except BaseException:
            # e.g. a cooperative deadline in a device tile: never leave the proof running unobserved.
            try:
                future.result()
            except Exception:
                pass
            raise
        if certificate_observer is not None:
            certificate_observer(certificate)
        if certificate.status != "certified_feasible":
            if self.early_warped:
                self._staged_warped = None                  # E97: never published
            reason = "FEASIBILITY_UNCERTAIN" if certificate.status == "uncertain" else certificate.reason
            return InvalidTrial(reason, "trial not certified: "+certificate.reason,
                                certificate.n_uncertain+certificate.n_infeasible)
        if device_error is not None:
            if self.early_warped:
                self._staged_warped = None                  # E97: never published
            self._candidate = None
            self.evaluation_counts["cpu_fallback"] += 1
            self.evaluation_counts["full_certifications"] += 1
            return cpu_evaluate(problem, theta)
        return self._finish_trial(problem, theta, u, model, device_u, certificate, values=values)

    def _finish_trial(self, problem, theta, u, model, device_u, certificate, *, values=None):
        try:
            if values is None:
                if model is None or device_u is None:
                    with self._stage("trial_upload"):
                        if model is None:
                            model = self._model_for(problem)
                        if device_u is None:
                            device_u = self.device.upload(u)
                values = self._device_trial_values(model, device_u)
            warped, gradient, energy, base, differences = values
        except (FloatingPointError, torch.OutOfMemoryError):
            if self.early_warped:
                self._staged_warped = None                  # E97: never published
            self._candidate = None
            self.evaluation_counts["cpu_fallback"] += 1
            self.evaluation_counts["full_certifications"] += 1
            return cpu_evaluate(problem, theta)
        identity = hashlib.sha256((problem.problem_id+certificate.state_hash+self.backend_id).encode()).hexdigest()
        if self.lean_identity:
            _identity.seed_certified_map(u, certificate)    # E89: what certify_map proved on u's exact bytes
        staged = None
        if self.early_warped:
            staged, self._staged_warped = self._staged_warped, None
            if staged is not None and staged[0] is not warped:
                staged = None
        with self._stage("trial_download"):
            if staged is not None:
                warped_host = staged[1]                      # E97: downloaded right after the sampling
                self.early_warped_record["published"] += 1
            else:
                warped_host = (self.device.download_owned(warped) if self.lean_identity
                               else readonly_copy(self.device.download(warped)))
            state = AcceleratedTrialState3D(theta, u, warped_host,
                energy, certificate, problem.problem_id, identity, self.backend_id)
        self._candidate = _DeviceState(model, state, device_u, warped, gradient,
                                      None if self.device_batch is None else
                                      staged[2] if (staged is not None and staged[2] is not None
                                                    and state.warped is staged[1]) else
                                      _PendingFingerprint((state.warped,), single=True) if self.lean_identity else
                                      _sample_fingerprint(state.warped))
        if base is not None:
            self._fused_comparison = (problem, base.state, state, base.sample_fingerprint,
                                      self._candidate.sample_fingerprint, differences)
            # T90: the screening band of this comparison, bound to this very tuple (any reset or replacement drops it)
            self._fused_band = (self._fused_comparison, getattr(self, "_trial_band", None))
        self.evaluation_counts["gpu"] += 1
        return state

    @timed("promote_trial")
    def promote_trial(self, problem, trial):
        self._fused_comparison = None
        self._validate(problem, trial)
        if self.reuse_prepared and self._prepared is not None and trial is self._prepared[1]:
            prepared = self._prepared_match(problem, trial.theta)
            if prepared is not None:
                self._lin = self._device_state(problem, prepared)
                self.reuse_counts["promotion_hits"] += 1
                return prepared
        if getattr(trial, "evaluation_backend_id", None) is None:
            self._prepared = None
            return cpu_promote(problem, trial)
        data = self._device_state(problem, trial)
        gradient = self.device.download_owned(data._b) if self.lean_identity else readonly_copy(self.device.download(data._b))
        require_finite_result(gradient, "device promoted image gradient")
        state = AcceleratedAcceptedState3D(trial.theta, trial.u, trial.warped, trial.energy,
            trial.certificate, trial.problem_id, trial.eval_id, gradient, self.backend_id)
        self._lin = _DeviceState(data.model, state, data._u, data._warped, data._b, data.sample_fingerprint)
        self._candidate = None
        self.evaluation_counts["promotions"] += 1
        if self.reuse_prepared:
            self._prepared = (problem, state, self._pending_prepared_fingerprint(problem, state) if self.lean_identity
                              else self._prepared_fingerprint(problem, state))
        return state

    def device_forcing_norms(self, problem, values, target, cpu_values):
        """E109 (B14): the outer norms of a recheck triggered by the forcing target alone. Inside the band, after a
        violation, or on a checked use: the CPU recheck (cpu_values(), v26's values); otherwise the device norms."""
        record = self.device_forcing_record
        record["triggers"] += 1
        self.device_forcing_last = None
        if record["disabled"]:
            record["disabled_cpu"] += 1
            self.device_forcing_last = "cpu_disabled"
            return cpu_values()
        if abs(values[2]-target) <= DEVICE_FORCING_BAND*target:
            record["band_cpu"] += 1
            self.device_forcing_last = "cpu_band"
            return cpu_values()
        grid = problem.grid.grid_id
        n = self._forcing_uses.get(grid, 0)
        self._forcing_uses[grid] = n+1
        if n < DEVICE_FORCING_CHECKS or (n+1-DEVICE_FORCING_CHECKS) % DEVICE_FORCING_RECHECK == 0:
            cpu = cpu_values()
            self.device_forcing_last = "cpu_checked"
            rel = [abs(d-c)/abs(c) if c != 0 else (0. if d == 0 else float("inf")) for d, c in zip(values, cpu)]
            record["checks"] += 1
            record["checked_grids"][grid] = record["checked_grids"].get(grid, 0)+1
            record["max_rel"] = [max(a, b if b <= 1e300 else 1e300) for a, b in zip(record["max_rel"], rel)]
            if any(not r <= DEVICE_FORCING_LIMIT for r in rel):     # NaN-safe (review round 2, M-2)
                record["violations"] += 1
                record["disabled"] = True
                if len(record["violation_details"]) < 8:
                    record["violation_details"].append(dict(grid=grid, use=n+1, device=list(map(float, values)),
                                                            cpu=list(map(float, cpu)), rel=[min(r, 1e300) for r in rel]))
            return cpu
        record["device_used"] += 1
        self.device_forcing_last = "device_forcing"
        return values

    @timed("cpu_outer_recheck")
    def cpu_outer_recheck(self, problem, state):
        """E108 (fused_recheck): gF, gD, gE of state on the CPU -- the vectors of
        outer_gradient_vectors(build_linearization(problem, self.cpu_state(problem, state))) bit for bit, with the same
        failures (FloatingPointError); counted as one cpu_state (a fresh CPU state with its full certificate)."""
        self._validate(problem, state)
        self.evaluation_counts["cpu_state"] += 1
        self.evaluation_counts["full_certifications"] += 1
        vectors = cpu_outer_recheck(problem, state.theta)
        self.fused_recheck_record["rechecks"] += 1      # completed rechecks (review E108, M1)
        return vectors

    @timed("cpu_state")
    def cpu_state(self, problem, state, *, promote=True):
        """Fresh CPU sampling/energy (and derivatives); never relabel a GPU state."""
        self._validate(problem, state)
        if not isinstance(promote, bool):
            raise InputValidationError("promote must be boolean")
        self.evaluation_counts["cpu_state"] += 1
        self.evaluation_counts["full_certifications"] += 1
        result = cpu_evaluate(problem, state.theta)
        if isinstance(result, InvalidTrial):
            raise FloatingPointError("CPU re-evaluation of certified state failed: "+result.reason)
        if promote:
            result = cpu_promote(problem, result)
            if isinstance(result, InvalidTrial):
                raise FloatingPointError("CPU derivative re-evaluation failed: "+result.reason)
        return result

    def _comparison_metadata(self, problem, base, trial):
        result = dict(policy=self.comparison_policy_id)
        if not self.near_boundary_cpu:
            margin = min(s.certificate.min_margin_lower_bound for s in (base, trial))
            result.update(near_boundary_cpu=False, certificate_margin_lower_bound=margin,
                          certificate_near_boundary=margin <= NEAR_BOUNDARY_RELATIVE*(1-problem.certificate_config.delta))
        return result

    def _fallback(self, problem, base, trial, *, outer_only, joint, required, reason):
        if self.fused_cpu:
            result = fused_cpu_reductions(problem, base, trial, outer_only=outer_only, joint=joint)
        else:
            a = self.cpu_state(problem, base, promote=False)
            b = self.cpu_state(problem, trial, promote=False)
            result = (cpu_reductions(problem, a, b) if joint else
                      cpu_reduction(problem, a, b, outer_only=outer_only))
        self.comparison_counts["cpu"] += 1
        key = "fallback_"+reason
        self.comparison_counts[key] = self.comparison_counts.get(key, 0)+1
        self.last_comparison = dict(path="cpu", reason=reason, required_reduction=required,
                                    **self._comparison_metadata(problem, base, trial))
        if self.fused_cpu:
            self.last_comparison["cpu_policy"] = CPU_FUSED_POLICY
        return result

    def _differences(self, problem, base, trial, *, outer_only):
        cached = self._fused_comparison
        if cached is not None and all(a is b for a,b in zip(cached[:3], (problem,base,trial))):
            with self._stage('comparison_cache_binding'):
                checkpoint('device_comparison_binding')
                _validate_bindings(problem, (base,trial))
                for state, fingerprint in zip((base,trial), cached[3:5]):
                    # E89: join a pending fingerprint first, so the check below is a memo lookup.
                    if state.warped.flags.writeable or _resolved(fingerprint) != _sample_fingerprint(state.warped):
                        raise StateMismatchError('fused comparison sample identity changed')
            values = cached[5]
            require_finite_result(values, 'fused device stable differences')
            if values[2] <= 0:
                raise FloatingPointError('fused device comparison barrier point is not positive')
            self.fusion_counts['comparison_hits'] += 1
            fb = getattr(self, "_fused_band", None)                             # T90: the screening band, else None
            self._last_band = fb[1] if (fb is not None and fb[0] is cached) else None
            return values[:2]
        self._last_band = None
        a, b = self._device_state(problem, base), self._device_state(problem, trial)
        m = a.model
        # Canonical residual differences, NOT differences of total energies.
        image_a = m._qd.sqrt()*(a._warped-m._fixed)   # T80
        image_b = m._qd.sqrt()*(b._warped-m._fixed)
        total = .5*((image_b-image_a)*(image_b+image_a)).sum()
        if m._alpha:
            x, y = m.curvature_residual(a._u), m.curvature_residual(b._u)
            total = total+.5*((y-x)*(y+x)).sum()   # T76: an outer (F) block
        outer = total.clone()
        if not outer_only:
            weight = (m.w.lam*m._q).sqrt()[..., None]
            x, y = weight*(a._u-m._ref), weight*(b._u-m._ref)
            total += .5*((y-x)*(y+x)).sum()
        min_gap = torch.full((), float("inf"), dtype=torch.float64, device=a._u.device)
        if m._fused is not None:
            bar = m._fused.barrier_energy(b._u, a._u)
            if not outer_only:
                total = total+m._fused.soft_energy(b._u, m._ref, a._u)[2]
            change = problem.barrier_weight*bar[2]
            total = total+change
            outer = outer+change
            min_gap = torch.minimum(bar[1], bar[3])-problem.certificate_config.delta
        for _, indices in (() if m._fused is not None else m.tiles()):
            al, bl = a._u.reshape(-1, 3)[indices], b._u.reshape(-1, 3)[indices]
            if not outer_only:
                _, _, Ja, Ca = _value_geometry(m, al, m._D, with_curl=True)
                _, _, Jb, Cb = _value_geometry(m, bl, m._D, with_curl=True)
                _, _, Jr, Cr = _value_geometry(m, m._ref.reshape(-1, 3)[indices], m._D, with_curl=True)
                x, y = (m.w.mu*m._qw).sqrt()*(Ja-Jr), (m.w.mu*m._qw).sqrt()*(Jb-Jr)
                total += .5*((y-x)*(y+x)).sum()
                weight = (m.w.nu*m._qw).sqrt()[None, :, None]
                x, y = weight*(Ca-Cr), weight*(Cb-Cr)
                total += .5*((y-x)*(y+x)).sum()
            _, _, Ja, _ = _value_geometry(m, al, m._Dc, with_curl=False)
            _, _, Jb, _ = _value_geometry(m, bl, m._Dc, with_curl=False)
            Ja, Jb = m.barrier_vals(Ja), m.barrier_vals(Jb)   # T68 (identity for the sample basis)
            delta = problem.certificate_config.delta
            min_gap = torch.minimum(min_gap, torch.minimum(Ja.min(), Jb.min())-delta)
            change = problem.barrier_weight*_barrier_difference(Ja, Jb, delta, problem.barrier_activation).sum()
            total += change
            outer += change
        values = self.device.scalar(total), self.device.scalar(outer), self.device.scalar(min_gap)
        require_finite_result(values, "device stable differences")
        if values[2] <= (BERNSTEIN_GUARD if m._TB is not None else 0.):
            raise FloatingPointError("device comparison barrier point is not positive (or within the T68 guard)")
        return values[:2]

    def known_gradient_norm(self, gradient, omega):
        """E89 (P03): gradient_norm_mass(gradient, omega) when gradient() just computed it on this very
        array object (returned unchanged and frozen); otherwise None and the caller computes it."""
        memo = getattr(self, "_gradient_norm_memo", None)
        if memo is not None and memo[0] is gradient and memo[1] == omega and not gradient.flags.writeable:
            return memo[2]
        return None

    def _monitor_margin(self, device_norm, cpu_norm):
        monitor = self.gradient_margin_monitor
        monitor['rechecks'] += 1
        difference = abs(cpu_norm/device_norm-1.) if device_norm > 0 else float('inf')
        monitor['max_relative_difference'] = max(monitor['max_relative_difference'], difference)
        if not monitor['abandoned'] and not difference <= self.cpu_gradient_margin/100:
            monitor['abandoned'] = True

    def _microstep(self, base, trial):
        if self.lean_host:
            from .. import fast_checks
            if fast_checks.enabled(base.theta.size):
                known = self.known_step_extrema(trial.theta, base.theta) if self.step_extrema else None
                if known is not None:
                    # E97: the same two maxima of these same frozen arrays, from the fused axpy or step ratio
                    self.step_extrema_record["microstep_hits"] += 1
                    return known[0] <= MICROSTEP_RELATIVE*max(1., known[1])
                # E76: the same exact max|.| values without three full-size temporaries.
                return (fast_checks.max_abs_diff(trial.theta, base.theta)
                        <= MICROSTEP_RELATIVE*max(1., fast_checks.max_abs(base.theta)))
        return np.max(np.abs(trial.theta-base.theta)) <= MICROSTEP_RELATIVE*max(1., float(np.max(np.abs(base.theta))))

    def _compare(self, problem, base, trial, *, outer_only, joint, required_reduction):
        required = finite_real(required_reduction, "required_reduction")
        if required < 0:
            raise InputValidationError("required_reduction must be nonnegative")
        for state in (base, trial):
            self._validate(problem, state)
        reason = None
        if any(getattr(s, "evaluation_backend_id", None) is None for s in (base, trial)):
            reason = "cpu_state"
        elif self.near_boundary_cpu and min(s.certificate.min_margin_lower_bound for s in (base, trial)) <= NEAR_BOUNDARY_RELATIVE*(1-problem.certificate_config.delta):
            reason = "near_boundary"
        elif self._microstep(base, trial):
            reason = "microstep"
        if reason is not None:
            return self._fallback(problem, base, trial, outer_only=outer_only, joint=joint,
                                  required=required, reason=reason)
        try:
            total, outer = self._differences(problem, base, trial, outer_only=outer_only)
            band = self._last_band
            def reduction(difference, is_outer):
                energy = max(s.energy.outer if is_outer else s.energy.total for s in (base, trial))
                guard = GUARD_FACTOR*np.finfo(float).eps*max(1., energy)
                return EnergyReduction(float(-difference), max(guard, band) if band is not None else guard)
            reductions = ((reduction(total, False), reduction(outer, True)) if joint else
                          (reduction(outer if outer_only else total, outer_only),))
            if band is not None and any(abs(r.actual) <= CRITICAL_GUARDS*r.resolution or
                                        abs(r.certified-required) <= CRITICAL_GUARDS*r.resolution for r in reductions):
                # T90: inside the float32 band the comparison is redone from the float64 device evaluation (the
                # non-cached path); the float64 guard then applies as before
                self.fp32_screen_record["band_hits"] += 1
                self.fp32_screen_record["fp64_rechecks"] += 1
                self._fused_comparison = None
                total, outer = self._differences(problem, base, trial, outer_only=outer_only)
                band = None
                reductions = ((reduction(total, False), reduction(outer, True)) if joint else
                              (reduction(outer if outer_only else total, outer_only),))
            for r in reductions:
                require_finite_result([r.actual, r.resolution], "device guarded reduction")
            if any(abs(r.actual) <= CRITICAL_GUARDS*r.resolution or
                   abs(r.certified-required) <= CRITICAL_GUARDS*r.resolution for r in reductions):
                reason = "critical_margin"
        except (FloatingPointError, torch.OutOfMemoryError):
            reason = "nonfinite_or_oom"
        if reason is not None:
            return self._fallback(problem, base, trial, outer_only=outer_only, joint=joint,
                                  required=required, reason=reason)
        self.comparison_counts["gpu"] += 1
        self.last_comparison = dict(path="gpu", reason="resolved", required_reduction=required,
            guard=[r.resolution for r in reductions], **self._comparison_metadata(problem, base, trial))
        return reductions if joint else reductions[0]

    @timed("stable_energy_reduction")
    def stable_energy_reduction(self, problem, base, trial, *, outer_only=False, required_reduction=0.):
        if not isinstance(outer_only, bool):
            raise InputValidationError("outer_only must be boolean")
        return self._compare(problem, base, trial, outer_only=outer_only, joint=False,
                             required_reduction=required_reduction)

    @timed("stable_energy_reductions")
    def stable_energy_reductions(self, problem, base, trial, *, required_reduction=0.):
        return self._compare(problem, base, trial, outer_only=False, joint=True,
                             required_reduction=required_reduction)

    @timed("outer_gradient_vectors")
    def outer_gradient_vectors(self, lin):
        dl = self.linearize(lin)
        m = dl.model
        stash = None
        if self.boundary_reuse:
            stash, self._joint_stash = self._joint_stash, None
        proven, device_outer = False, None
        if stash is not None and stash[0] is lin.state and stash[1] is lin.problem and stash[2] is m:
            # E89 (P05b): computed by final_gradient. The (now empty) stage keeps v10's work checkpoint and
            # operation events at this place; the pass itself was timed into the same timings key.
            with self._stage('joint_outer_geometry'):
                result = stash[3]
            device_outer = stash[4] if not self.outer_reuse_record["disabled"] else None    # E100 (None unless on)
            self.fusion_counts['joint_gradients'] += 1     # one joint pass per outer call, as in v10
            # E95: a stashed triple was proven finite by final_gradient (total by its decision, the two others
            # by the scan before stashing); its unchanged frozen arrays need no rescan
            proven = self.lean_checks
        else:
            if self.device_batch is not None:
                with self._stage('joint_outer_geometry'):
                    vectors = dl.joint_outer_gradients()
                self.fusion_counts['joint_gradients'] += 1
            else:
                image = m.restrict(m._qd[..., None]*(dl._warped-m._fixed)[..., None]*dl._b)   # T80
                vectors = (dl.gradient("outer"), image, dl.gradient("total"))
            result = tuple(self.device.download(value) for value in vectors)
            device_outer = vectors[0] if self.outer_reuse and not self.outer_reuse_record["disabled"] else None   # E100
        if not proven:
            for value in result:
                require_finite_result(value, "device outer gradient")
        if self.boundary_reuse:
            result[0].flags.writeable = False
            # E100: result[0] is proven finite here (checked above, or by final_gradient before stashing)
            self._outer_stash = (lin.state, m, dl._u, dl._warped, dl._b, result[0], device_outer)
        if self.lean_checks and getattr(self, "_collect_outer", False):
            self._outer_proven = result     # E96: only for outer_norms, which takes it back within the same call
        return result

    def outer_norms(self, lin, omega):
        """E96 (lean_checks): tuple(gradient_norm_mass(g, omega) for g in self.outer_gradient_vectors(lin)) -- the
        same calls in the same order as E95; the proven triple passes from outer_gradient_vectors to
        outer_vector_norms within this call and is not kept afterwards."""
        self._collect_outer = True
        try:
            vectors = self.outer_gradient_vectors(lin)
        finally:
            self._collect_outer = False
        return self.outer_vector_norms(vectors, omega)

    def outer_vector_norms(self, vectors, omega):
        """E95: tuple(gradient_norm_mass(g, omega) for g in vectors) for the triple this backend just returned and
        proved finite: the known norm of the decided total vector, the others without a rescan (same nrm2)."""
        proven, self._outer_proven = getattr(self, "_outer_proven", None), None
        if not (self.lean_checks and proven is not None and len(vectors) == len(proven)
                and all(a is b for a, b in zip(vectors, proven))):
            return tuple(gradient_norm_mass(g, omega) for g in vectors)
        out = []
        for g in vectors:
            known = self.known_gradient_norm(g, omega)
            out.append(known if known is not None else _mass_norm_checked(g, omega))
        if self.outer_reuse:
            # E100: the outer vector's norm by this same function, for the next inner solve's first gradient (that
            # same frozen array); kept until used or replaced
            self._outer_norm = (vectors[0], omega, out[0])
        return tuple(out)
