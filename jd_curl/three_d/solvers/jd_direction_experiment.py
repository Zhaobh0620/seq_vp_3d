"""E42 explicit economical supports, bounded SOC and selected barrier metric.

None of these mechanisms changes the true objective or replaces full-cell JD
certification. Selected barrier curvature is NOT the full GGN Hessian.
"""
from time import perf_counter
import numpy as np
from scipy.linalg import cho_factor, cho_solve

from ...validation import InputValidationError, require_finite_result
from ..barrier import barrier_terms
from ..boundary import inject_free
from ..cell_q1 import cell_node_indices
from ..stationarity import point_jd_gradient
from ..work_control import checkpoint
from .jd_constrained import linearize_points
from .small_qp import nonnegative_qp


class CountingInverse:
    def __init__(self, inverse):self.inverse=inverse;self.calls=0
    def apply(self, rhs):self.calls+=1;return self.inverse.apply(rhs)
    def matvec(self, vector):return self.inverse.matvec(vector)
    def schur_column(self, A, column):
        self.calls+=1
        if hasattr(self.inverse,'schur_column'):return self.inverse.schur_column(A,column)
        return A@self.inverse.apply(np.asarray(A.getrow(column).toarray()).ravel())


def filter_seed(problem, base, memory, gradient, inverse):
    """Recompute every old row at actual base. No stale derivative reuse."""
    started=perf_counter();support=linearize_points(problem,base,memory.points)
    if not len(support.c):return dict(removed=[],seconds=perf_counter()-started)
    raw=-inverse.apply(gradient);variation=support.A@raw
    keep=variation<0
    removed=[pt for pt,k in zip(memory.points,keep) if not k]
    memory.points[:]=[pt for pt,k in zip(memory.points,keep) if k]
    return dict(removed=removed,seconds=perf_counter()-started,base_eval_id=base.eval_id,
                rule='remove noncompressing raw-direction seeds; full-candidate proof remains')


def filter_accepted(model, proposed, memory, *, actual_step=None):
    """Persist active/near-linear-bound locations only, no values across bases."""
    lam=np.asarray(proposed.diagnostics['multipliers'])
    slack=model.A@(proposed.d if actual_step is None else actual_step)+model.b
    keep=(lam*model.scale>1e-9)|(slack<=.05*model.c)
    removed=[pt for pt,k in zip(model.support.points,keep) if not k]
    selected={pt for pt,k in zip(model.support.points,keep) if k}
    memory.points[:]=[pt for pt in memory.points if pt in selected]
    return dict(removed=removed,retained=len(memory.points),rule='scaled multiplier>1e-9 or linear slack<=.05*c')


class SelectedBarrierInverse:
    """Woodbury inverse of H=M+A.T*diag(weights)*A, no global inverse columns.

    The Poisson engine owns physical-call accounting. Only m-by-m matrices and
    sparse rows are retained. Curvature frozen at base/initial support.
    """
    def __init__(self, poisson, A, weights, schur):
        self.poisson=poisson;self.A=A;self.weights=np.asarray(weights)
        if self.weights.shape!=(A.shape[0],) or np.any(self.weights<0) or not np.all(np.isfinite(self.weights)):
            raise InputValidationError('nonnegative finite curvature weights required')
        self.sqrt=np.sqrt(self.weights)
        Q=np.eye(len(weights))+self.sqrt[:,None]*schur*self.sqrt[None,:]
        self.factor=cho_factor(Q,lower=True,check_finite=True) if len(weights) else None
        self.condition=float(np.linalg.cond(Q)) if len(weights) else 1.
    def small_action(self, vector):
        return self.sqrt*cho_solve(self.factor,self.sqrt*vector) if self.factor is not None else np.zeros_like(vector)
    def correct(self, poisson_solution):
        if not np.any(self.weights):return poisson_solution.copy()
        rhs=np.asarray(self.A.T@self.small_action(self.A@poisson_solution))
        return poisson_solution-self.poisson.apply(rhs)
    def apply(self, rhs):return self.correct(self.poisson.apply(rhs))
    def matvec(self, vector):return self.poisson.matvec(vector)+np.asarray(self.A.T@(self.weights*(self.A@vector)))


def augment_curvature(problem, model, poisson):
    """Transform an existing Poisson Schur model without re-solving m columns."""
    started=perf_counter();support=model.support
    # Only exact members of the actual closed27 quadrature have these weights.
    points=set(map(tuple,problem.barrier_ops.points));w=np.zeros(len(support.c))
    eligible=np.array([tuple(pt) in points for _,pt in support.points],dtype=bool)
    if np.any(eligible):
        w[eligible]=barrier_terms(support.c[eligible]+problem.certificate_config.delta,
            delta=problem.certificate_config.delta,weight=problem.barrier_weight,
            activation=problem.barrier_activation)[2]
    H=SelectedBarrierInverse(poisson,model.A,w,model.S)
    old=model.S
    transformed=old-np.column_stack([old@H.small_action(old[:,j]) for j in range(len(w))]) if len(w) else old.copy()
    model.d0=H.correct(model.d0);model.d0_norm2=-float(model.g@model.d0)
    model.S=.5*(transformed+transformed.T)
    if np.any(model.S.diagonal()<0):raise FloatingPointError('curvature Schur cancellation: negative diagonal')
    model.scale=np.sqrt(np.where(model.S.diagonal()>0,model.S.diagonal(),1.))
    model.Sn=model.S/model.scale[:,None]/model.scale[None,:]
    model.ad0=np.asarray(model.A@model.d0);model.inverse=H;model._dual_warm=None
    require_finite_result(model.Sn,'curvature normalized Schur')
    return dict(seconds=perf_counter()-started,positive_weights=int(np.sum(w>0)),
        selected_points=len(w),woodbury_condition=H.condition,
        scope='initial selected closed27 barrier-GGN plus M, NOT full GGN')


def minimum_correction(A, rhs, inverse):
    """min .5*s.T*M*s s.t. A*s>=rhs; streamed Schur and independent checks."""
    m=A.shape[0]
    if not m:return np.zeros(A.shape[1]),dict(status='empty_correction',dual_residual=0.)
    S=np.empty((m,m))
    for j in range(m):
        checkpoint('soc_schur',completed=j,total=m)
        S[:,j]=A@inverse.apply(np.asarray(A.getrow(j).toarray()).ravel())
    S=.5*(S+S.T);scale=np.sqrt(np.where(S.diagonal()>0,S.diagonal(),1.))
    fit=nonnegative_qp(S/scale[:,None]/scale[None,:],-rhs/scale)
    residual=float(np.max(np.abs(np.minimum(fit.x,S/scale[:,None]/scale[None,:]@fit.x-rhs/scale)))) if m else 0.
    if not fit.success or residual>1e-7*max(1.,np.linalg.norm(rhs/scale,np.inf)):
        return None,dict(status='correction_qp_failed',dual_residual=residual)
    lam=fit.x/scale;s=inverse.apply(np.asarray(A.T@lam))
    slack=A@s-rhs
    if np.any(slack < -1e-7*scale*max(1.,np.linalg.norm(rhs/scale))):
        return None,dict(status='correction_primal_failed')
    return s,dict(status='correction_solved',dual_residual=residual)


def conditional_soc(problem, base, step, found, inverse, *, theta):
    started=perf_counter();points=list(dict.fromkeys(found))[:8]
    info=dict(eligible=False,attempted=False,points=points)
    if not points:return None,dict(info,status='no_actual_witness')
    support=linearize_points(problem,base,points);linear=support.c+support.A@step
    if np.any(linear < theta*support.c-64*np.finfo(float).eps*np.maximum(1.,support.c)):
        return None,dict(info,status='linear_blocker_not_SOC')
    trial=base.u+inject_free(step,problem.grid);actual=[]
    for cell,pt in points:
        nodes=cell_node_indices(problem.grid,slice(cell,cell+1))[0]
        actual.append(point_jd_gradient(trial.reshape(-1,3)[nodes],problem.grid.spacing,pt)[0]-problem.certificate_config.delta)
    del trial
    actual=np.asarray(actual);rhs=theta*support.c-actual
    info.update(eligible=True,attempted=True,linear_slack=linear.tolist(),actual_slack=actual.tolist(),nonlinear_remainder=(actual-linear).tolist())
    correction,diag=minimum_correction(support.A,rhs,inverse);info.update(diag)
    if correction is None:return None,dict(info,seconds=perf_counter()-started)
    norm2=float(step@inverse.matvec(step));corr2=float(correction@inverse.matvec(correction))
    ratio=np.sqrt(max(0.,corr2)/max(norm2,np.finfo(float).tiny))
    info.update(correction_relative_norm=ratio,seconds=perf_counter()-started)
    if corr2<0 or ratio>.25:return None,dict(info,status='correction_too_large')
    return step+correction,dict(info,status='bounded_correction_ready')
