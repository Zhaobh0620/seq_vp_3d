"""E44 same-metric execution. No new objective, direction or stopping policy."""
from contextlib import contextmanager
from time import perf_counter
import numpy as np
import torch

from ...validation import StateMismatchError, require_finite_result
from ..poisson import build_poisson_inverse
from ..residuals import validate_state
from ..backends.torch_ops import TorchPoisson
from ..work_control import checkpoint


class DeviceQPInverse:
    """Stream one sparse rhs; download only A M^-1 row, not its full field."""
    def __init__(self, backend, metric):
        self.device, self.timings, self.metric = backend.device, backend.timings, metric
        self.inverse = TorchPoisson(self.device, metric)
        self.calls = 0
        self._rows = None

    @contextmanager
    def timed(self, name):
        self.device.synchronize(); started = perf_counter()
        try:
            yield
        finally:
            self.device.synchronize()
            self.timings[name] = self.timings.get(name, 0.)+perf_counter()-started

    def apply(self, rhs):
        checkpoint('qp_device_poisson'); self.calls += 1
        with self.timed('qp_full_poisson'):
            result = self.device.download(self.inverse.apply(self.device.upload(rhs)))
        require_finite_result(result, 'device QP response')
        return result

    def schur_column(self, A, column):
        checkpoint('qp_device_schur'); self.calls += 1
        with self.timed('qp_schur_response'):
            dev = self.device.device
            rhs = torch.zeros(A.shape[1], dtype=torch.float64, device=dev)
            row = A.getrow(column)
            indices = torch.tensor(row.indices.astype(np.int64), device=dev)
            rhs[indices] = self.device.upload(row.data)
            response = self.inverse.apply(rhs)
            if not bool(torch.isfinite(response).all()):
                raise FloatingPointError('nonfinite full Schur response')
            # At most 24 entries per physical JD row. Cache only sparse rows.
            key=(A.shape,A.indptr.tobytes(),A.indices.tobytes(),A.data.tobytes())
            if self._rows != key:
                width=max(1,int(np.diff(A.indptr).max()))
                ids=np.zeros((A.shape[0],width),dtype=np.int64)
                values=np.zeros(ids.shape)
                for i in range(A.shape[0]):
                    lo,hi=A.indptr[i:i+2];ids[i,:hi-lo]=A.indices[lo:hi];values[i,:hi-lo]=A.data[lo:hi]
                self._indices=torch.tensor(ids,device=dev)
                self._values=self.device.upload(values);self._rows=key
            answer = self.device.download((response[self._indices]*self._values).sum(dim=1))
        require_finite_result(answer, 'device Schur column')
        return answer

    def matvec(self, vector):
        """Same independently discretized CPU Q1 stencil, not inverse identity."""
        checkpoint('qp_device_metric')
        with self.timed('qp_metric_action'):
            grid = self.metric.grid
            fields = self.device.upload(vector).reshape((3,)+grid.interior_shape)
            def tri(field, axis, diagonal, off):
                out = diagonal*field
                left, right = [slice(None)]*4, [slice(None)]*4
                left[axis], right[axis] = slice(1,None), slice(None,-1)
                out[tuple(left)] += off*field[tuple(right)]
                out[tuple(right)] += off*field[tuple(left)]
                return out
            stiffness = torch.zeros_like(fields)
            for axis, h in enumerate(grid.spacing[::-1], 1):
                factor = (1/np.float64(h))**2
                term = tri(fields, axis, 2*factor, -factor)
                for other in (1,2,3):
                    if other != axis: term = tri(term, other, 2/3, 1/6)
                stiffness += term
            answer = self.device.download((grid.omega*(self.metric.a*fields+self.metric.b*stiffness)).reshape(-1))
        require_finite_result(answer, 'device Q1 metric')
        return answer


class QPContext:
    """One frozen reference; cached state is factory-owned, never cross-reference."""
    def __init__(self, problem, state, gradient, target, backend):
        validate_state(problem, state)
        self.problem, self.state = problem, state
        self.gradient, self.target, self.backend = gradient, target, backend
        self.metric = build_poisson_inverse(problem.grid, a=problem.weights.lam,
            b=(problem.weights.mu+problem.weights.nu)/2, metric='q1')
        self.inverse = self.metric if backend is None else DeviceQPInverse(backend, self.metric)

    def base(self, problem, theta, target):
        if problem is not self.problem or theta is not self.state.theta or target != self.target:
            raise StateMismatchError('QP context requires identical state, reference and target')
        validate_state(problem, self.state)
        return self.state, self.gradient

    def advance(self, state, gradient):
        validate_state(self.problem, state)
        self.state, self.gradient = state, gradient
