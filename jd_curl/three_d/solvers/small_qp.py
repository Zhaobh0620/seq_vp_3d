"""E42 bounded active-set nonnegative convex QP. No function-change stopping.

Minimize .5*y.T*S*y+h.T*y, y>=0. Singular passive systems are solved by
rank-revealing least squares; incompatible singular blocks first pivot along
a null-space descent to the nonnegative boundary. The final bounded coordinate
fallback never regularizes the mathematical problem.
"""
from types import SimpleNamespace
import numpy as np

from ...validation import as_float64_array, finite_real, InputValidationError
from ..validation import integer_at_least
from ..work_control import checkpoint


def nonnegative_qp(S, h, *, start=None, maxiter=500, tolerance=1e-11):
    h=as_float64_array(h,'dual linear term');S=as_float64_array(S,'dual Hessian',(len(h),len(h)))
    maxiter=integer_at_least(maxiter,'dual maxiter');tolerance=finite_real(tolerance,'dual tolerance',positive=True)
    if h.ndim!=1 or not np.allclose(S,S.T,rtol=1e-13,atol=1e-14):
        raise InputValidationError('symmetric dual Hessian and vector required')
    n=len(h);scale=max(1.,float(np.max(np.abs(h))) if n else 1.)
    tol=tolerance*scale
    if n and np.linalg.eigvalsh(S).min() < -1e-12*max(1.,np.linalg.norm(S,2)):
        raise InputValidationError('dual Hessian is not positive semidefinite')
    y=np.zeros(n) if start is None else np.maximum(as_float64_array(start,'dual start',h.shape),0.).copy()
    passive=y>0;iterations=0;fallback=False
    def residual():return S@y+h
    def done(r):return not n or max(float(np.max(np.abs(r[y>0]))) if np.any(y>0) else 0.,float(np.max(np.maximum(-r,0.))))<=tol
    def result(ok,message):
        r=residual();error=float(np.max(np.abs(np.minimum(y,r)))) if n else 0.
        return SimpleNamespace(x=y.copy(),nit=iterations,success=ok,message=message,
            projected_residual=error,coordinate_fallback=fallback)
    for _ in range(maxiter):
        iterations+=1;checkpoint('active_dual_qp',iteration=iterations)
        if np.any(passive):
            ids=np.flatnonzero(passive);block=S[np.ix_(ids,ids)]
            z=np.zeros(n)
            z[ids],_,rank,_=np.linalg.lstsq(block,-h[ids],rcond=1e-13)
            negative=passive & (z<=0)
            if np.any(negative):
                denominators=y[negative]-z[negative]
                ratios=np.divide(y[negative],denominators,out=np.zeros_like(denominators),where=denominators>0)
                step=float(np.min(ratios));y+=step*(z-y)
                hit=np.flatnonzero(negative)[np.flatnonzero(ratios<=step+1e-14)]
                y[hit]=0.;passive[hit]=False
                continue
            # Only a feasible passive minimizer needs the stationary solve
            # check.  An ill-conditioned solve may have huge negative
            # multipliers: first pivot those OUT, rather than abandoning the
            # active set for slow coordinate descent before a boundary step.
            if np.max(np.abs(block@z[ids]+h[ids]))>tol:
                if rank<len(ids):
                    # Dependent constraints can have unequal offsets. Their
                    # passive equations then have no interior stationary
                    # solution: transfer mass along the null space and drop
                    # a multiplier, instead of stalling in coordinate descent.
                    values,vectors=np.linalg.eigh(block)
                    null=vectors[:,values<=1e-13*max(0.,values[-1])]
                    direction=-null@(null.T@h[ids])
                    slope=float((block@y[ids]+h[ids])@direction)
                    curvature=float(direction@block@direction)
                    leaving=np.flatnonzero(direction<0)
                    if slope<0 and len(leaving):
                        ratios=-y[ids[leaving]]/direction[leaving]
                        hit=int(leaving[np.argmin(ratios)]);bound=float(np.min(ratios))
                        # Near-null floating-point curvature is not dropped.
                        step=min(bound,-slope/curvature) if curvature>0 else bound
                        if np.isfinite(step) and step>=0:
                            y[ids]=np.maximum(0.,y[ids]+step*direction)
                            if step==bound:y[ids[hit]]=0.;passive[ids[hit]]=False
                            continue
                fallback=True;break
            y=z
        r=residual()
        if done(r):return result(True,'independent gradient/complementarity target')
        candidates=np.where(~passive,r,np.inf);j=int(np.argmin(candidates))
        if candidates[j]>=-tol:
            fallback=True;break
        passive[j]=True
    # Deterministic coordinate minimization is also valid for semidefinite S.
    # It may be slow on dependent rows: exhaustion is a reported failure.
    fallback=True
    for _ in range(maxiter):
        iterations+=1;checkpoint('coordinate_dual_qp',iteration=iterations)
        for j in range(n):
            rj=float(S[j]@y+h[j])
            if S[j,j]>0:y[j]=max(0.,y[j]-rj/S[j,j])
            elif rj < -tol:return result(False,'zero curvature with negative derivative')
        if done(residual()):return result(True,'coordinate fallback gradient target')
    return result(False,'bounded dual iteration limit')
