"""E8 numerical approximate KKT diagnostics ONLY, on a supplied formal state.

True local physical constraints, no mask equations or self-reference penalties.
Values/slack uppers are current-state evaluations; analytic gradients are
floating point, so this is not a rigorous interval stationarity certificate.
"""
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import nnls
from scipy.sparse import csr_matrix

from ..validation import InputValidationError, StateMismatchError, as_float64_array, finite_real, require_finite_result
from . import interval as iv
from .cell_q1 import VERTICES,cell_node_indices
from .feasibility import POINTS,certify_cell
from .geometry import cofactor,determinant
from .linearization import build_linearization,energy_gradient
from .residuals import validate_state
from .risk import scan_risk_cells


def point_jd_gradient(vertices, spacing, point):
    """J and (8,3) cofactor nodal derivative in physical coordinates."""
    u=as_float64_array(vertices,'vertices',(8,3));h=as_float64_array(spacing,'spacing',(3,))
    t=as_float64_array(point,'reference point',(3,))
    if np.any(h<=0) or np.any(t<0) or np.any(t>1):
        raise InputValidationError('positive physical spacing and closed-cell point required')
    D=np.empty((8,3))
    for a,bits in enumerate(VERTICES):
        factors=np.where(bits,t,1-t)
        for j in range(3):
            D[a,j]=(2*bits[j]-1)*np.prod(factors[[k for k in range(3) if k!=j]])/h[j]
    F=np.eye(3)+u.T@D
    return float(determinant(F)),D@cofactor(F).T


def point_jd_interval(vertices,spacing,point):
    # Validate through the public local derivative API; no rounded weights
    # are passed as exact: every (1-t), product and division is enclosed.
    point_jd_gradient(vertices,spacing,point)
    values=iv.Interval.point(vertices);t=iv.Interval.point(point);F=[[None]*3 for _ in range(3)]
    for j in range(3):
        d=iv.Interval.point(np.zeros(3))
        for a,bits in enumerate(VERTICES):
            if bits[j]:continue
            other=list(bits);other[j]=1;b=VERTICES.index(tuple(other))
            weight=iv.Interval.point(1.)
            for k in range(3):
                if k!=j:weight=weight*(t[k] if bits[k] else iv.Interval.point(1.)-t[k])
            d=d+(values[b]-values[a])/spacing[j]*weight
        for r in range(3):F[r][j]=d[r]+float(r==j)
    return (F[0][0]*(F[1][1]*F[2][2]-F[1][2]*F[2][1])
            -F[0][1]*(F[1][0]*F[2][2]-F[1][2]*F[2][0])
            +F[0][2]*(F[1][0]*F[2][1]-F[1][1]*F[2][0]))


@dataclass(frozen=True)
class PhysicalSupport3D:
    problem_id: str
    eval_id: str
    A: csr_matrix
    slack_upper: np.ndarray
    records: tuple
    candidate_count: int
    truncated: bool

    def validate(self,problem,state):
        validate_state(problem,state)
        if self.problem_id!=problem.problem_id or self.eval_id!=state.eval_id:
            raise StateMismatchError('stale physical constraint support')
        if any(r['kind'] not in ('physical_jd','sampling_domain') for r in self.records):
            raise InputValidationError('mask equations/Bernstein coefficients are not physical constraints')
        if self.A.shape!=(len(self.records),3*problem.grid.n_free_scalar):
            raise InputValidationError('physical support layout mismatch')


def domain_support_rows(grid,u,*,band=1e-6):
    """Original normalized node-domain inequalities; local algebra only.

    Calling this helper is NOT a feasibility or stationarity assertion.
    """
    u=as_float64_array(u,'displacement',grid.shape+(3,))
    band=finite_real(band,'domain band',positive=True)
    n=grid.n_free_scalar;candidates=[]
    identity=grid.identity()[1:-1,1:-1,1:-1].reshape(-1,3)
    displacements=u[1:-1,1:-1,1:-1].reshape(-1,3)
    for r,axis in enumerate(grid.axes):
        length=axis[-1]-axis[0]
        coord=identity[:,r]+displacements[:,r]
        coord_iv=iv.Interval.point(identity[:,r])+iv.Interval.point(displacements[:,r])
        length_iv=iv.Interval.point(axis[-1])-axis[0]
        for side in ('lower','upper'):
            ci=((coord_iv-axis[0]) if side=='lower' else (iv.Interval.point(axis[-1])-coord_iv))/length_iv
            c=(coord-axis[0])/length if side=='lower' else (axis[-1]-coord)/length
            for k in np.flatnonzero(ci.lo<=band):
                if ci.hi[k]<0:raise FloatingPointError('domain slack contradicts feasible map')
                rec=dict(kind='sampling_domain',component=r,free_node=int(k),side=side,
                         slack=float(c[k]),slack_lower=float(ci.lo[k]),slack_upper=float(ci.hi[k]))
                candidates.append((float(c[k]),rec,[int(r*n+k)],[(1. if side=='lower' else -1.)/length]))
    return candidates


def build_physical_support(problem,state,*,historical_points=(),max_support=512):
    """Finite deterministic support, same certificate caps, no global optimum claim.

    Historical entries are (cell, reference_xyz) COORDINATES only. All values
    and derivatives are recomputed here, never reused from a previous state.
    """
    from .validation import integer_at_least
    max_support=integer_at_least(max_support,'max_support')
    if max_support>512:raise InputValidationError('E8 diagnostic support cap is 512')
    validate_state(problem,state);grid=problem.grid;delta=problem.certificate_config.delta
    scan=scan_risk_cells(problem,state,band=1e-4);points=set()
    for c in scan['cells']:
        nodes=cell_node_indices(grid,slice(c,c+1))[0];local=state.u.reshape(-1,3)[nodes]
        points.update((c,tuple(t)) for t in POINTS)
        def observe(t,lo,hi):
            if lo<=1e-4:points.add((c,tuple(t)))
        cert=certify_cell(local,grid.spacing,config=problem.certificate_config,point_observer=observe)
        if cert.status!='certified_feasible':raise FloatingPointError('same-state local certificate not reproduced')
    for c,t in historical_points:
        c=integer_at_least(c,'support cell',0)
        if c>=grid.n_cells:raise InputValidationError('support cell outside grid')
        t=as_float64_array(t,'support point',(3,))
        if np.any(t<0) or np.any(t>1):raise InputValidationError('support point outside cell')
        points.add((c,tuple(t)))
    candidates=[];n=grid.n_free_scalar
    for c,t in sorted(points):
        nodes=cell_node_indices(grid,slice(c,c+1))[0];local=state.u.reshape(-1,3)[nodes]
        J,grad=point_jd_gradient(local,grid.spacing,t)
        if J-delta>1e-4:continue
        interval=point_jd_interval(local,grid.spacing,t)-delta
        if float(interval.hi)<0:raise FloatingPointError('point slack contradicts feasible map')
        indices=[];values=[]
        for node,a in zip(nodes,grad):
            z,y,x=np.unravel_index(node,grid.shape)
            if min(z,y,x)==0 or z==grid.shape[0]-1 or y==grid.shape[1]-1 or x==grid.shape[2]-1:continue
            free=np.ravel_multi_index((z-1,y-1,x-1),grid.interior_shape)
            for r in range(3):indices.append(int(r*n+free));values.append(float(a[r]))
        record=dict(kind='physical_jd',cell=c,reference_xyz=list(t),slack=J-delta,
                    slack_lower=float(interval.lo),slack_upper=float(interval.hi))
        candidates.append((J-delta,record,indices,values))
    candidates.extend(domain_support_rows(grid,state.u))
    # Stable generation order breaks equal slack ties deterministically.
    candidates.sort(key=lambda row:row[0]);selected=candidates[:max_support]
    rows=[];cols=[];values=[];records=[]
    for i,(_,rec,indices,entries) in enumerate(selected):
        rows.extend([i]*len(indices));cols.extend(indices);values.extend(entries);records.append(rec)
    A=csr_matrix((values,(rows,cols)),shape=(len(selected),3*n))
    support=PhysicalSupport3D(problem.problem_id,state.eval_id,A,np.array([r['slack_upper'] for r in records]),
                              tuple(records),len(candidates),len(candidates)>len(selected))
    support.validate(problem,state)
    return support


def assess_multipliers(g,A,slack_upper,y,*,omega,epsilon_g,epsilon_comp):
    """Independent residual re-evaluation, not the optimizer's NNLS residual."""
    g=as_float64_array(g,'gF');m=A.shape[0]
    y=as_float64_array(y,'multipliers',(m,));upper=as_float64_array(slack_upper,'slack upper',(m,))
    if A.shape[1:]!=g.shape or np.any(y<0) or np.any(upper<0):
        raise InputValidationError('invalid multiplier signs, slack bounds or layout')
    omega=finite_real(omega,'omega',positive=True)
    epsilon_g=finite_real(epsilon_g,'epsilon_g',positive=True);epsilon_comp=finite_real(epsilon_comp,'epsilon_comp',positive=True)
    r=np.asarray(g-A.T@y);rstat=float(np.linalg.norm(r)/np.sqrt(omega))
    # Outward accumulation for nonnegative multiplier/slack products.
    total=iv.Interval.point(0.)
    for value,bound in zip(y,upper):total=total+iv.Interval.point(value)*bound
    comp=float(total.hi)
    require_finite_result([rstat,comp],'KKT residuals')
    return dict(r_stat=rstat,r_comp_upper=comp,passed=bool(rstat<=epsilon_g and comp<=epsilon_comp),
                multipliers_nonnegative=True,multiplier_norm=float(np.linalg.norm(y)),multipliers=y.tolist())


def fit_multipliers(g,A,slack_upper,*,omega,epsilon_g,epsilon_comp):
    m,n=A.shape
    if not m:return dict(status='no_certificate',reason='EMPTY_SUPPORT',r_stat=float(np.linalg.norm(g)/np.sqrt(omega)),
                         r_comp_upper=0.,passed=False,multipliers=[])
    if n*m*8>64*1024**2:return dict(status='no_certificate',reason='DENSE_FIT_MEMORY_CAP')
    # Only this bounded fitting matrix is dense; geometry supports stay sparse.
    B=A.T.toarray()/np.sqrt(omega);rhs=g/np.sqrt(omega)
    scales=np.linalg.norm(B,axis=0)
    nonzero=scales>0
    if not np.any(nonzero):return dict(status='no_certificate',reason='ZERO_CONSTRAINT_GRADIENTS')
    try:
        fitted,_=nnls(B[:,nonzero]/scales[nonzero],rhs,maxiter=max(1,3*m))
        y=np.zeros(m);y[nonzero]=fitted/scales[nonzero]
        result=assess_multipliers(g,A,slack_upper,y,omega=omega,epsilon_g=epsilon_g,epsilon_comp=epsilon_comp)
        return dict(result,status='numerical_approx_kkt' if result['passed'] else 'no_certificate',
                    reason='RESIDUAL_AND_COMPLEMENTARITY_PASS' if result['passed'] else 'RESIDUAL_OR_COMPLEMENTARITY_FAILED')
    except (RuntimeError,ValueError,FloatingPointError) as exc:
        return dict(status='no_certificate',reason='MULTIPLIER_FIT_FAILED',detail=str(exc))


def diagnose_stationarity(problem,formal_state,*,epsilon_g,F0,historical_points=()):
    started=perf_counter();validate_state(problem,formal_state)
    epsilon_g=finite_real(epsilon_g,'epsilon_g',positive=True)
    F0=finite_real(F0,'F0');epsilon_comp=1e-8*max(1.,abs(F0))
    g=energy_gradient(build_linearization(problem,formal_state),term='outer')
    full=float(np.linalg.norm(g)/np.sqrt(problem.grid.omega))
    support=build_physical_support(problem,formal_state,historical_points=historical_points)
    support.validate(problem,formal_state)
    result=fit_multipliers(g,support.A,support.slack_upper,omega=problem.grid.omega,
                           epsilon_g=epsilon_g,epsilon_comp=epsilon_comp)
    result.update(diagnostic_only=True,termination_policy='original_full_GF_only',hard_feasible=True,
                  GF=full,GF_converged=full<=epsilon_g,epsilon_g=epsilon_g,epsilon_comp=epsilon_comp,
                  problem_id=problem.problem_id,eval_id=formal_state.eval_id,controls_id=problem.controls.controls_id,
                  support=list(support.records),support_count=len(support.records),candidate_count=support.candidate_count,
                  support_truncated=support.truncated,elapsed_seconds=perf_counter()-started)
    return result
