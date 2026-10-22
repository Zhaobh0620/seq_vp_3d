"""T86 (S30-G2, explicit option, path only): the geometry weights mu = nu of the inner objective E by outer index.

mu and nu weigh the JD/curl change penalties of the Sequential inner objective E only; F = D + R + B, the E/F
acceptance, the certificate, forcing and GF do not contain them. A schedule gives, for the k-th outer (k = number of
formal commits so far on this level), the weights of that outer's reference problem: 'MUxN,MUxN,...,MU' means MU for
the next N outers, ..., and the last MU for the rest. lam and kappa stay those of the level. The Sequential driver
refreshes the reference with the scheduled weights (reference certification, no reference reuse on a weight change)
and records the weights of every outer in its history rows.
"""
from dataclasses import replace

import numpy as np

from ..validation import InputValidationError
from .problem import Weights3D

MU_SCHEDULE_POLICY = "native-mu-schedule-by-outer-v1"


def parse_mu_schedule(text):
    """'0.3x3,0.1x3,0.03' -> ((.3, 3), (.1, 3), (.03, None)); the last entry has no count."""
    if isinstance(text, (list, tuple)):
        steps = [tuple(s) for s in text]
    else:
        if not isinstance(text, str) or not text.strip():
            raise InputValidationError("mu_schedule: 'MUxN,...,MU' required")
        steps = []
        for item in text.split(","):
            mu, sep, count = item.strip().partition("x")
            try:
                steps.append((float(mu), int(count) if sep else None))
            except ValueError:
                raise InputValidationError("mu_schedule: entries are MUxN with a positive integer N, the last MU alone") from None
    if not steps:
        raise InputValidationError("mu_schedule: at least one entry")
    for i, (mu, count) in enumerate(steps):
        if isinstance(mu, bool) or not isinstance(mu, (int, float)) or not np.isfinite(mu) or mu <= 0:
            raise InputValidationError("mu_schedule: every mu must be a finite positive number")
        last = i == len(steps)-1
        if last and count is not None:
            raise InputValidationError("mu_schedule: the last entry holds for the rest and has no count")
        if not last and (isinstance(count, bool) or not isinstance(count, int) or count < 1):
            raise InputValidationError("mu_schedule: every entry but the last needs a positive integer count")
    return tuple((float(mu), count) for mu, count in steps)


class MuSchedule:
    """Callable k -> Weights3D(lam, mu_k, mu_k, kappa) with lam, kappa of the base weights."""

    def __init__(self, base, steps):
        if not isinstance(base, Weights3D):
            raise InputValidationError("mu_schedule: base Weights3D required")
        self.base, self.steps = base, parse_mu_schedule(steps)

    def mu(self, k):
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 0:
            raise InputValidationError("mu_schedule: the outer index is a nonnegative integer")
        start = 0
        for mu, count in self.steps:
            if count is None or k < start+count:
                return mu
            start += count
        return self.steps[-1][0]        # unreachable: the last entry has no count

    def __call__(self, k):
        mu = self.mu(k)
        return replace(self.base, mu=mu, nu=mu)

    def record(self):
        return dict(policy=MU_SCHEDULE_POLICY, steps=[dict(mu=mu, outers=count) for mu, count in self.steps],
                    lam=self.base.lam, kappa=self.base.kappa, base_mu=self.base.mu, base_nu=self.base.nu)
