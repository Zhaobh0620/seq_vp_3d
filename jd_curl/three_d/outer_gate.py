"""FX1 F4 (throughput-v35, path only; plan results/verification/3d_stage_FX1_v1/plan/protocol.md): an outer iteration of
the native level is not started when the time left before the solve cutoff is shorter than the shortest of the last
`window` complete outer iterations of this solve (top of one outer to the top of the next).

Such an outer would be cut by the wall budget (v34 then ends with its last formal state); not starting it ends the run
earlier with the same formal state. Only an outer faster than every one of the last `window` could have finished in
v34; then this run has one formal commit fewer, and the decision record shows it. Never convergence: GF, its threshold,
E/F, forcing and the certificate are unchanged; the solve ends with status outer_time_budget."""
from time import perf_counter

from ..validation import InputValidationError

OUTER_GATE_POLICY = "no-doomed-outer-v1"


class OuterGate:
    """stop() is called at the top of every outer iteration, after the convergence, stall and iteration checks;
    remaining() is the time left before the solve cutoff (seconds, same clock)."""

    def __init__(self, remaining, window=3, clock=perf_counter):
        if not callable(remaining) or not callable(clock):
            raise InputValidationError("outer gate: remaining and clock must be callables")
        if isinstance(window, bool) or not isinstance(window, int) or window < 1:
            raise InputValidationError("outer gate: window must be a positive integer")
        self.remaining, self.window, self.clock = remaining, window, clock
        self.tops, self.decision = [], None

    def stop(self):
        self.tops.append(self.clock())
        if len(self.tops) <= self.window:
            return False
        recent = [b-a for a, b in zip(self.tops[-self.window-1:-1], self.tops[-self.window:])]
        left = float(self.remaining())
        if left < min(recent):
            self.decision = dict(remaining_seconds=left, predicted_seconds=min(recent), recent_seconds=recent,
                                 outer_tops=len(self.tops))
            return True
        return False

    def record(self):
        return dict(policy=OUTER_GATE_POLICY, window=self.window, outer_tops=len(self.tops),
                    triggered=self.decision is not None, decision=self.decision,
                    role="native level only: an outer that the shortest of the last `window` outers says cannot finish "
                         "before the solve cutoff is not started (F4, path only); never convergence")
