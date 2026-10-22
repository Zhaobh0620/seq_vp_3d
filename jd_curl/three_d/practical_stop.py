"""H17 (a CHANGE OF THE STOPPING RULE, explicit option only): end the native level
early once the projected MSE-ratio gain over the remaining solve budget is small.

Never a convergence criterion: the success definition and the original GF threshold are unchanged; a practical stop
ends the run with status native_practical_stop and success = false (a budget-type end). Only the native level, only
after a formal native commit, and only from online, label-free quantities: the committed MSE ratio (fixed native-zero
denominator), the native-level clock and the remaining native solve time.

Criterion (c), policy projected-ratio-gain-v1. After the n-th formal native commit at native-level time t_n with ratio
r_n, let the reference commit be the last formal native commit with time <= t_n - W (W = window_seconds). Eligible when
t_n >= min_seconds, n >= min_commits, a reference commit exists and the window (t_n - W, t_n] holds at least 2 commits.
Then rate = (r_ref - r_n)/(t_n - t_ref) and the projected gain is rate * T_rem, T_rem the remaining native solve time
measured right after the commit; stop when projected gain <= epsilon. The offline estimate on TA3 (execution session,
2026-10-12) found this equivalent, within .0002 on average, to a fixed budget ending at the same mean time."""
from dataclasses import asdict, dataclass
import numbers

import numpy as np

from ..validation import InputValidationError

PRACTICAL_STOP_POLICY = "projected-ratio-gain-v1"


@dataclass(frozen=True)
class PracticalStopConfig3D:
    epsilon: float
    window_seconds: float = 30.
    min_seconds: float = 30.
    min_commits: int = 5
    policy: str = PRACTICAL_STOP_POLICY

    def __post_init__(self):
        if self.policy != PRACTICAL_STOP_POLICY:
            raise InputValidationError("practical stop: unknown policy")
        for name in ("epsilon", "window_seconds", "min_seconds"):
            value = getattr(self, name)
            if not isinstance(value, numbers.Real) or isinstance(value, (bool, np.bool_)) or not np.isfinite(value):
                raise InputValidationError(f"practical stop {name} must be a finite real number")
            object.__setattr__(self, name, float(value))
        if self.epsilon < 0 or self.window_seconds <= 0 or self.min_seconds < 0:
            raise InputValidationError("practical stop: epsilon >= 0, window_seconds > 0, min_seconds >= 0")
        if not isinstance(self.min_commits, numbers.Integral) or isinstance(self.min_commits, (bool, np.bool_)) \
                or self.min_commits < 2:
            raise InputValidationError("practical stop min_commits must be an integer >= 2")
        object.__setattr__(self, "min_commits", int(self.min_commits))


def practical_stop_config(spec):
    """dict(epsilon, [window_seconds, min_seconds, min_commits]) -> PracticalStopConfig3D (epsilon is required)."""
    if isinstance(spec, PracticalStopConfig3D):
        return spec
    if not isinstance(spec, dict) or "epsilon" not in spec \
            or set(spec) - {"epsilon", "window_seconds", "min_seconds", "min_commits", "policy"}:
        raise InputValidationError("practical stop: dict(epsilon, window_seconds, min_seconds, min_commits) required")
    return PracticalStopConfig3D(**spec)


def decide(config, commits, remaining_seconds):
    """commits: the formal commits of the native level so far, oldest first, as (time_seconds, ratio) with the
    current commit last; remaining_seconds: T_rem measured after it. Returns the record (pure function; the audit
    replays it from the recorded values)."""
    t_n, r_n = commits[-1]
    n = len(commits)
    record = dict(policy=config.policy, commit_index=n, native_seconds=float(t_n), ratio=float(r_n),
                  remaining_seconds=float(remaining_seconds), epsilon=config.epsilon, eligible=False, triggered=False,
                  reference_seconds=None, reference_ratio=None, rate_per_second=None, projected_gain=None)
    if t_n < config.min_seconds or n < config.min_commits:
        return record
    reference = [(t, r) for t, r in commits[:-1] if t <= t_n-config.window_seconds]
    inside = [t for t, _ in commits if t > t_n-config.window_seconds]
    if not reference or len(inside) < 2:
        return record
    t_ref, r_ref = reference[-1]
    rate = (r_ref-r_n)/(t_n-t_ref)
    projected = rate*max(0., remaining_seconds)
    record.update(eligible=True, reference_seconds=float(t_ref), reference_ratio=float(r_ref),
                  rate_per_second=float(rate), projected_gain=float(projected), triggered=bool(projected <= config.epsilon))
    return record


class PracticalStop:
    """The native level's stopping rule: remaining_seconds() gives T_rem (wall clock) when called."""

    def __init__(self, config, remaining_seconds):
        self.config = practical_stop_config(config)
        if not callable(remaining_seconds):
            raise InputValidationError("practical stop needs a callable remaining_seconds")
        self.remaining_seconds = remaining_seconds
        self.commits, self.triggered_at = [], None

    def after_commit(self, seconds, ratio):
        from time import perf_counter
        self.commits.append((float(seconds), float(ratio)))
        record = decide(self.config, self.commits, float(self.remaining_seconds()))
        if record["triggered"] and self.triggered_at is None:
            self.triggered_at = (perf_counter(), record)
        return record

    def record(self):
        return dict(asdict(self.config))
