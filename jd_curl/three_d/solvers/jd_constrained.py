"""T49 fixed-base Poisson-metric inequality/trust-region direction prototype.

No local displacement scaling, accepted-state mutation or stopping policy.
Small Schur matrices only; inverse constraint columns are streamed, not cached.
"""
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix

from ...validation import (InputValidationError, StateMismatchError, as_float64_array,
                           finite_real, require_finite_result)
from ..cell_q1 import cell_node_indices
from ..residuals import validate_state
from ..stationarity import point_jd_gradient
from ..validation import integer_at_least
from ..work_control import checkpoint


class WorkingSetLimit(RuntimeError):
    """Stop the bounded prototype, never silently discard a blocking constraint."""


class JDWorkingSet:
    """Only locations survive a reference update; all values/rows are rebuilt."""
    def __init__(self, limit=32):
        self.limit = integer_at_least(limit, 'constraint limit')
        self.grid_id = None
        self.points = []

    def bind(self, grid):
        if self.grid_id != grid.grid_id:
            self.grid_id = grid.grid_id
            self.points.clear()
        self.grid = grid

    def add(self, cell, point):
        if self.grid_id is None:
            raise InputValidationError('working set must bind a grid first')
        cell = integer_at_least(cell, 'cell', 0)
        point = as_float64_array(point, 'physical reference point', (3,))
        if cell >= self.grid.n_cells or np.any(point < 0) or np.any(point > 1):
            raise InputValidationError('physical point outside grid/cell')
        key = (cell, tuple(float(x) for x in point))
        if key in self.points:
            return False
        if len(self.points) >= self.limit:
            raise WorkingSetLimit('physical JD working-set capacity reached')
        self.points.append(key)
        return True


@dataclass(frozen=True)
class JDConstraints:
    problem_id: str
    eval_id: str
    A: csr_matrix
    c: np.ndarray
    points: tuple

    def validate(self, problem, base):
        validate_state(problem, base)
        if self.problem_id != problem.problem_id or self.eval_id != base.eval_id:
            raise StateMismatchError('JD rows belong to a different base/reference')


def linearize_points(problem, base, points):
    """Physical cofactor rows, free component-major dofs, no quadrature weights."""
    validate_state(problem, base)
    memory = JDWorkingSet(max(1, len(points)))
    memory.bind(problem.grid)
    for cell, point in points:
        memory.add(cell, point)
    grid = problem.grid
    n = grid.n_free_scalar
    rows, cols, values, margins = [], [], [], []
    for i, (cell, point) in enumerate(memory.points):
        checkpoint('jd_constraint_rows', completed=i, total=len(memory.points))
        nodes = cell_node_indices(grid, slice(cell, cell+1))[0]
        J, grad = point_jd_gradient(base.u.reshape(-1, 3)[nodes], grid.spacing, point)
        margin = J-problem.certificate_config.delta
        if not np.isfinite(margin) or margin <= 0:
            raise FloatingPointError('current physical JD slack contradicts strict feasibility')
        margins.append(margin)
        for node, derivative in zip(nodes, grad):
            z, y, x = np.unravel_index(node, grid.shape)
            if (z in (0, grid.shape[0]-1) or y in (0, grid.shape[1]-1)
                    or x in (0, grid.shape[2]-1)):
                continue
            free = np.ravel_multi_index((z-1, y-1, x-1), grid.interior_shape)
            for r in range(3):
                rows.append(i); cols.append(r*n+free); values.append(derivative[r])
    A = csr_matrix((values, (rows, cols)), shape=(len(margins), 3*n))
    c = np.asarray(margins, dtype=float)
    for a in (A.data, A.indices, A.indptr, c):
        a.flags.writeable = False
    return JDConstraints(problem.problem_id, base.eval_id, A, c, tuple(memory.points))


@dataclass(frozen=True)
class DirectionResult:
    valid: bool
    reason: str
    d: np.ndarray | None
    diagnostics: dict


class PoissonConstraintModel:
    """At ONE fixed base: H=W=M, c+Ad >= theta*c, ||d||_M <= radius.

    inverse exposes apply=M^-1 and matvec=M. E42 may explicitly replace both
    with one frozen SPD H; legacy norm_M labels then refer to that metric.
    Sparse A has at most 24 entries
    per physical JD row. Neither a dense global A nor n-by-m inverse is stored.
    """
    def __init__(self, gradient, constraints, inverse, *, theta=.1, dual_solver=None):
        self.dual_solver = dual_solver
        self._dual_warm = None
        self.g = as_float64_array(gradient, 'gradient')
        if self.g.ndim != 1 or constraints.A.shape[1] != self.g.size:
            raise InputValidationError('gradient/constraint layout mismatch')
        self.theta = finite_real(theta, 'fraction-to-boundary', positive=True)
        if self.theta >= 1:
            raise InputValidationError('fraction-to-boundary must be <1')
        self.support, self.inverse = constraints, inverse
        self.A, self.c = constraints.A, constraints.c
        if self.A.shape[0] != len(self.c) or np.any(self.c <= 0):
            raise InputValidationError('strictly positive current slacks required')
        self.poisson_calls = 0
        started = perf_counter()
        self.d0 = -self._apply(self.g)
        self.d0_norm2 = -float(self.g@self.d0)
        if not np.isfinite(self.d0_norm2) or self.d0_norm2 < 0:
            raise FloatingPointError('Poisson metric is not positive on gradient')
        m = len(self.c)
        self.S = np.empty((m, m))
        for j in range(m):
            checkpoint('jd_schur_column', completed=j, total=m)
            self.S[:, j] = self._schur_column(self.A,j)
        require_finite_result(self.S, 'Schur complement')
        self.S = .5*(self.S+self.S.T)
        self.ad0 = np.asarray(self.A@self.d0)
        self.b = (1-self.theta)*self.c
        diag = self.S.diagonal()
        if np.any(diag < 0):
            raise FloatingPointError('negative Schur diagonal')
        self.scale = np.sqrt(np.where(diag > 0, diag, 1.))
        self.Sn = self.S/self.scale[:, None]/self.scale[None, :]
        self.setup_seconds = perf_counter()-started
        self.setup_poisson_calls = self.poisson_calls

    def extend(self, constraints):
        """Append rows at exactly the same base/metric; stream only NEW columns.

        Old sparse rows and slacks must be byte-value identical. The metric is
        owned by this model and cannot be replaced by an extension. Cross terms
        use self-adjointness of the same Poisson inverse, not cached full fields.
        """
        old = len(self.c)
        A, c = constraints.A, constraints.c
        if (constraints.problem_id != self.support.problem_id
                or constraints.eval_id != self.support.eval_id
                or A.shape[1] != self.g.size or len(c) < old
                or A.shape[0] != len(c)
                or constraints.points[:len(self.support.points)] != self.support.points
                or not np.array_equal(c[:old], self.c)
                or (A[:old] != self.A).nnz):
            raise StateMismatchError('Schur extension must preserve base, metric and old rows')
        if np.any(~np.isfinite(c)) or np.any(c <= 0):
            raise InputValidationError('strictly positive finite appended slacks required')
        if len(c) == old:
            return dict(poisson_calls=0, seconds=0., added_rows=0)
        started = perf_counter(); before = self.poisson_calls
        S = np.empty((len(c), len(c))); S[:old, :old] = self.S
        for j in range(old, len(c)):
            checkpoint('jd_schur_append', completed=j-old, total=len(c)-old)
            column = self._schur_column(A,j)
            S[:, j] = column
            S[j, :old] = column[:old]
        require_finite_result(S, 'extended Schur complement')
        S = .5*(S+S.T)
        if np.any(S.diagonal() < 0):
            raise FloatingPointError('negative extended Schur diagonal')
        self.support, self.A, self.c, self.S = constraints, A, c, S
        self.ad0 = np.asarray(A@self.d0)
        self.b = (1-self.theta)*c
        self.scale = np.sqrt(np.where(S.diagonal() > 0, S.diagonal(), 1.))
        self.Sn = S/self.scale[:, None]/self.scale[None, :]
        self._dual_warm = None
        elapsed = perf_counter()-started; calls = self.poisson_calls-before
        self.setup_seconds += elapsed; self.setup_poisson_calls += calls
        return dict(poisson_calls=calls, seconds=elapsed, added_rows=len(c)-old)

    def _apply(self, rhs):
        checkpoint('jd_direction_poisson')
        self.poisson_calls += 1
        answer = self.inverse.apply(rhs)
        require_finite_result(answer, 'Poisson response')
        return answer

    def _schur_column(self, A, column):
        if hasattr(self.inverse,'schur_column'):
            self.poisson_calls+=1
            return self.inverse.schur_column(A,column)
        rhs=np.zeros(self.g.size);row=A.getrow(column);rhs[row.indices]=row.data
        return A@self._apply(rhs)

    def solve(self, radius, *, dual_maxiter=500, tolerance=1e-7):
        radius = finite_real(radius, 'metric trust radius', positive=True)
        dual_maxiter = integer_at_least(dual_maxiter, 'dual iteration limit')
        tolerance = finite_real(tolerance, 'direction tolerance', positive=True)
        if tolerance >= 1e-2:
            raise InputValidationError('direction tolerance must be <1e-2')
        started = perf_counter(); calls_before = self.poisson_calls
        total_iterations, dual_solves = 0, 0
        last = {}

        def fixed_shift(t):
            nonlocal total_iterations, dual_solves, last
            checkpoint('jd_dual_qp', shift=t-1, supports=len(self.c))
            h = (self.ad0+t*self.b)/self.scale
            if len(h):
                def fun(y):
                    checkpoint('jd_dual_iteration')
                    Sy = self.Sn@y
                    return float(.5*y@Sy+h@y), Sy+h
                if self.dual_solver is None:
                    fit = minimize(fun, np.zeros(len(h)), jac=True, method='L-BFGS-B',
                        bounds=[(0., None)]*len(h), options=dict(maxiter=dual_maxiter,
                        ftol=0., gtol=1e-11, maxls=40))
                else:
                    fit = self.dual_solver(self.Sn, h, start=self._dual_warm, maxiter=dual_maxiter)
                y = np.maximum(fit.x, 0.)
                total_iterations += fit.nit
                residual = self.Sn@y+h
                error = float(np.max(np.abs(np.minimum(y, residual))))
                allowed = tolerance*max(1., float(np.max(np.abs(h))))
                last = dict(dual_solver_success=bool(fit.success), dual_residual=error,
                            dual_tolerance=allowed, dual_message=str(fit.message))
                if not np.all(np.isfinite(y)) or not error <= allowed:
                    raise FloatingPointError('dual QP failed independent KKT check')
                if self.dual_solver is not None:
                    self._dual_warm = y.copy()
                lam = y/self.scale
            else:
                lam = np.empty(0)
                last = dict(dual_solver_success=True, dual_residual=0., dual_tolerance=tolerance)
            dual_solves += 1
            square = (self.d0_norm2+2*lam@self.ad0+lam@self.S@lam)/(t*t)
            if not np.isfinite(square) or square < -tolerance*max(1., self.d0_norm2):
                raise FloatingPointError('invalid reduced trust norm')
            return lam, np.sqrt(max(0., square))

        try:
            t = 1.
            lam, reduced_norm = fixed_shift(t)
            if reduced_norm > radius:
                lo, hi = 1., 2.
                for _ in range(60):
                    lam, reduced_norm = fixed_shift(hi)
                    if reduced_norm <= radius:
                        break
                    lo, hi = hi, hi*2
                else:
                    # 0 belongs to C={d: Ad+b>=0}.  The fixed-shift solution
                    # is the M-projection of d0/t onto C, hence ||d||_M <=
                    # ||d0||_M/t.  Do not confuse an arbitrary doubling cap
                    # with a failed QP.  Retain the old path below that cap.
                    lo = hi/2
                    hi = max(hi, np.nextafter(np.sqrt(self.d0_norm2)/radius, np.inf))
                    if not np.isfinite(hi):
                        raise FloatingPointError('trust shift outside floating-point range')
                    lam, reduced_norm = fixed_shift(hi)
                    if reduced_norm > radius*(1+1e-12):
                        raise FloatingPointError('analytic trust bound failed numerical check')
                for _ in range(50):
                    if reduced_norm >= (1-1e-5)*radius:
                        break
                    mid = .5*(lo+hi)
                    candidate, value = fixed_shift(mid)
                    if value > radius:
                        lo = mid
                    else:
                        hi, lam, reduced_norm = mid, candidate, value
                t = hi
            # One final all-field solve, not one cached all-field vector per row.
            reduced_seconds=perf_counter()-started
            tick=perf_counter()
            rhs = -self.g+np.asarray(self.A.T@lam)
            d = self._apply(rhs)/t
            reconstruct_seconds=perf_counter()-tick;tick=perf_counter()
            Md = self.inverse.matvec(d)
            metric_seconds=perf_counter()-tick;tick=perf_counter()
            require_finite_result(Md, 'metric direction')
            norm2 = float(d@Md)
            slope = float(self.g@d)
            pred = -slope-.5*norm2
            slack = self.A@d+self.b
            primal_error = float(np.max(np.maximum(-slack/self.scale, 0.))) if len(slack) else 0.
            stationarity = self.g+t*Md-self.A.T@lam
            stat_rel = float(np.linalg.norm(stationarity)/max(1., np.linalg.norm(self.g), np.linalg.norm(rhs)))
            complementarity = float(np.max(np.abs(lam*slack))) if len(slack) else 0.
            duality_scale = max(1., abs(slope), norm2)
            trust_error = (t-1)*abs(norm2-radius*radius)/duality_scale
            # Bisection's last evaluated midpoint need not be the selected
            # feasible endpoint. Report residuals of the selected multipliers.
            dual_gradient = (self.S@lam+self.ad0+t*self.b)/self.scale
            dual_residual = float(np.max(np.abs(np.minimum(lam*self.scale, dual_gradient)))) if len(lam) else 0.
            dual_tolerance = tolerance*max(1., float(np.max(np.abs((self.ad0+t*self.b)/self.scale)))) if len(lam) else tolerance
            info = dict(last_internal_dual_solve=last, dual_residual=dual_residual,
                dual_tolerance=dual_tolerance, radius=radius, norm_M=float(np.sqrt(max(0., norm2))),
                trust_multiplier=t-1, gradient_dot_step=slope, predicted_reduction=pred,
                linear_primal_error=primal_error, stationarity_relative=stat_rel,
                complementarity=complementarity, trust_complementarity_relative=trust_error,
                multipliers=lam.tolist(), support_count=len(self.c), dual_iterations=total_iterations,
                dual_solves=dual_solves, setup_poisson_calls=self.setup_poisson_calls,
                solve_poisson_calls=self.poisson_calls-calls_before,
                setup_seconds=self.setup_seconds, solve_seconds=perf_counter()-started,
                reduced_search_seconds=reduced_seconds,reconstruction_seconds=reconstruct_seconds,
                metric_seconds=metric_seconds,verification_seconds=perf_counter()-tick)
            finite = np.all(np.isfinite([norm2, slope, pred, primal_error, stat_rel, complementarity, trust_error]))
            valid = bool(finite and norm2 >= 0 and slope < 0 and pred > 0
                and np.sqrt(norm2) <= radius*(1+1e-5)
                and primal_error <= tolerance*max(1., np.linalg.norm(self.b/self.scale))
                and stat_rel <= tolerance and dual_residual <= dual_tolerance
                and complementarity <= tolerance*duality_scale
                and trust_error <= 1e-4)
            return DirectionResult(valid, 'MODEL_VERIFIED' if valid else 'MODEL_CHECK_FAILED',
                                   d if valid else None, info)
        except FloatingPointError as error:
            return DirectionResult(False, str(error), None, dict(last,
                support_count=len(self.c), dual_iterations=total_iterations, dual_solves=dual_solves,
                setup_poisson_calls=self.setup_poisson_calls, solve_poisson_calls=self.poisson_calls-calls_before,
                setup_seconds=self.setup_seconds, solve_seconds=perf_counter()-started))
