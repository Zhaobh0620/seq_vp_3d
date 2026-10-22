"""T49 bounded one-step A/B gate; NOT a new Sequential stopping policy."""
from dataclasses import dataclass
import heapq
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from ...types import InvalidTrial
from ...validation import finite_real, InputValidationError, StateMismatchError
from ..boundary import inject_free
from ..cell_q1 import CellQ1Operators3D, cell_node_indices, cell_tiles
from ..geometry import _compute_cell_geometry
from ..linearization import build_linearization, energy_gradient
from ..poisson import build_poisson_inverse
from ..residuals import evaluate_trial, promote_trial, validate_state
from ..energy_comparison import stable_energy_reduction
from ..stationarity import point_jd_gradient
from ..validation import integer_at_least
from ..work_control import checkpoint
from .common import SolverCounters3D, finish_inner, gradient_norm_mass, prepare_base, initial_failure_status
from .poisson_gradient import sampling_step_bound
from .jd_constrained import (JDWorkingSet, WorkingSetLimit, linearize_points,
                             PoissonConstraintModel)
from .jd_step import JDStepMemory
from .jd_witness_batch import collect_witnesses
from .jd_positive_pressure import collect_pressure


def scan_monitors(problem, base, direction, *, limit=6):
    """Global closed27 point scan: near JD and earliest LINEAR hitting cells.

    Selected sample points are real physical positions, not enclosure bounds.
    This diagnostic pass costs time and is not a full-cell positivity proof.
    """
    validate_state(problem, base);limit=integer_at_least(limit,'monitor limit')
    grid=problem.grid;ops=CellQ1Operators3D(grid,location='closed27_uniform')
    v=inject_free(direction,grid)
    near, hitting=[],[];max_D=0.;started=perf_counter()
    def keep(heap, score, cell, point):
        item=(-float(score),int(cell),int(point))
        if len(heap)<limit:heapq.heappush(heap,item)
        elif item>heap[0]:heapq.heapreplace(heap,item)
    for tile in cell_tiles(grid,problem.certificate_config.tile_size):
        checkpoint('constraint_monitor_scan',completed=tile.start,total=grid.n_cells)
        geometry=_compute_cell_geometry(base.u,ops,tile=tile)
        nodes=cell_node_indices(grid,tile);local=v.reshape(-1,3)[nodes]
        V=np.empty_like(geometry.F)
        for r in range(3):
            for j,coefficients in enumerate(ops.coefficients):
                V[...,r,j]=local[...,r]@coefficients.T
        max_D=max(max_D,float(np.max(np.linalg.norm(V,axis=(-2,-1)))))
        c=geometry.J-problem.certificate_config.delta
        if np.any(c<=0):raise FloatingPointError('monitored base slack contradicts certification')
        change=np.sum(geometry.A*V,axis=(-2,-1))
        hit=np.full(c.shape,np.inf);np.divide(c,-change,out=hit,where=change<0)
        for field,heap in ((c,near),(hit,hitting)):
            q=field.argmin(axis=1);scores=field[np.arange(len(q)),q]
            # Tile top-k is sufficient for global top-k; avoid a Python loop per cell.
            chosen=np.argsort(scores,kind='stable')[:limit]
            for k in chosen:
                if np.isfinite(scores[k]):keep(heap,scores[k],tile.start+k,q[k])
    positions=[];records=[]
    for kind,heap in [('near_current_jd',near),('earliest_linear_hit',hitting)]:
        for score,cell,q in sorted(heap,reverse=True):
            point=tuple(float(t) for t in ops.points[q]);key=(cell,point)
            if key not in positions:positions.append(key)
            records.append(dict(kind=kind,cell=cell,point=list(point),score=-score))
    return dict(points=positions,records=records,direction_D_frobenius_max=max_D,
                seconds=perf_counter()-started,scope='closed27 samples, NOT full-cell proof')


def witness_diagnostic(problem, base, step, point, *, theta=.1):
    """Same actual base and true violating location, not last formal state."""
    support=linearize_points(problem,base,[point]);c=float(support.c[0])
    dc=float((support.A@step)[0]);grid=problem.grid
    cell,xyz=point;nodes=cell_node_indices(grid,slice(cell,cell+1))[0]
    local=base.u.reshape(-1,3)[nodes].copy()
    z,y,x=np.unravel_index(nodes,grid.shape)
    free=(z>0)&(z<grid.shape[0]-1)&(y>0)&(y<grid.shape[1]-1)&(x>0)&(x<grid.shape[2]-1)
    indices=np.ravel_multi_index((z[free]-1,y[free]-1,x[free]-1),grid.interior_shape)
    local[free]+=step.reshape(3,-1)[:,indices].T
    J,_=point_jd_gradient(local,grid.spacing,xyz)
    actual=J-problem.certificate_config.delta
    return dict(cell=int(cell),point=list(xyz),base_eval_id=base.eval_id,
        c=c,delta_J=dc,linear_slack=c+dc,actual_slack=actual,
        nonlinear_remainder=actual-c-dc,
        classification='linear_blocker' if c+dc<theta*c else 'nonlinear_model_error')


def rejection_diagnostic(problem, base, step, proposed, point, support_points, *, theta):
    """Four disjoint explanations at identical base/location, with actual rounding."""
    row=witness_diagnostic(problem,base,step,point,theta=theta)
    support=linearize_points(problem,base,[point])
    original=float(support.c[0]+(support.A@proposed)[0])
    row.update(proposed_linear_slack=original,in_working_set=point in support_points)
    row['rejection_class']=classify_rejection(row['c'],row['linear_slack'],original,
                                            row['in_working_set'],theta)
    return row


def classify_rejection(c, actual_linear, proposed_linear, member, theta):
    tol=128*np.finfo(float).eps*max(1.,abs(c),abs(actual_linear),abs(proposed_linear))
    if member and proposed_linear<theta*c-tol:return 'subproblem_error'
    if actual_linear<theta*c-tol:
        return 'execution_mismatch' if member else 'omitted_constraint'
    return 'nonlinear_model_error'


@dataclass(frozen=True)
class ConstrainedProbeConfig:
    radius_fraction: float = .1
    theta: float = .1
    max_trials: int = 16
    max_support: int = 32
    dual_maxiter: int = 500
    c1: float = 1e-4
    gain_min: float = .1
    policy: str = 'legacy-e38'

    def __post_init__(self):
        for name in ('radius_fraction','theta','c1','gain_min'):
            value=finite_real(getattr(self,name),name,positive=True)
            if value>=1:raise InputValidationError(name+' must be <1')
            object.__setattr__(self,name,value)
        for name in ('max_trials','max_support','dual_maxiter'):
            object.__setattr__(self,name,integer_at_least(getattr(self,name),name))
        if self.max_support>32:raise InputValidationError('prototype support limit must be <=32')
        if self.policy not in ('legacy-e38','jd-calibrated-v1','jd-calibrated-v2',
                               'positive-pressure-v1','positive-pressure-v2'):
            raise InputValidationError('unknown constrained direction policy')
        if self.policy in ('positive-pressure-v1','positive-pressure-v2') and self.theta!=.75:
            raise InputValidationError('positive-pressure policy requires linear retention theta=.75')


def solve_constrained_probe(problem, theta, *, target=0., config=None, backend=None,
                            budget=None, iteration_observer=None, working_set=None, experiment=None, context=None):
    """One accepted inner step at most; no outer update/BB2/local scaling.

    Work-set persists only geometric locations. Uncertain proofs shrink trust
    without generating constraints. Gain and Armijo use the final domain-scaled
    actual increment. This prototype cannot report constrained convergence.
    SOC is available only in the explicit E42 soc/curvature experiments.
    """
    config=ConstrainedProbeConfig() if config is None else config
    if experiment not in (None,'qp','working_set','soc','curvature'):
        raise InputValidationError('unknown E42 direction experiment')
    if experiment is not None:
        from .small_qp import nonnegative_qp
        from .jd_direction_experiment import (CountingInverse, filter_seed, filter_accepted,
            augment_curvature, conditional_soc)
    economical=experiment in ('working_set','soc','curvature')
    use_soc=experiment in ('soc','curvature')
    physical_inverse=None
    if not isinstance(config,ConstrainedProbeConfig):raise InputValidationError('expected ConstrainedProbeConfig')
    target=finite_real(target,'gradient target')
    if target<0:raise InputValidationError('gradient target must be nonnegative')
    started=perf_counter();counters=SolverCounters3D();history=[];G=None
    if context is not None and context.backend is not backend:
        raise StateMismatchError('QP context backend mismatch')
    base=prepare_base(problem,theta,counters,backend=backend) if context is None else context.base(problem,theta,target)[0]
    def finish(status,reason,accepted=0):
        if physical_inverse is not None:counters.poisson=physical_inverse.calls
        return finish_inner(base,status,reason,G,target,accepted,history,counters,started)
    if isinstance(base,InvalidTrial):
        invalid=base;base=None
        return finish(initial_failure_status(invalid),invalid.reason)
    lin=build_linearization(problem,base)
    gradient=(energy_gradient(lin) if backend is None else backend.gradient(lin,target)) if context is None else context.base(problem,theta,target)[1]
    counters.gradients+=int(context is None);G=gradient_norm_mass(gradient,problem.grid.omega)
    if G<=target:return finish('converged_gradient','GRADIENT_TARGET')
    metric=(build_poisson_inverse(problem.grid,a=problem.weights.lam,
        b=(problem.weights.mu+problem.weights.nu)/2,metric='q1') if context is None else context.metric)
    host=(metric if backend is None else backend.poisson(metric)) if context is None else context.inverse
    inverse=SimpleNamespace(apply=host.apply,matvec=metric.matvec) if context is None else host
    if experiment is not None:
        physical_inverse=CountingInverse(inverse);inverse=physical_inverse
    memory=JDWorkingSet(config.max_support) if working_set is None else working_set
    if not isinstance(memory,JDWorkingSet) or memory.limit>config.max_support:
        raise InputValidationError('working set exceeds prototype configured limit')
    memory.bind(problem.grid)
    from .sequential_qp import CoverageQPMemory, SafeguardedQPConfig3D
    from .qp_safeguard import SafeguardedQPMemory, historical_pressure
    coverage=isinstance(memory,CoverageQPMemory)
    safeguarded=isinstance(config,SafeguardedQPConfig3D)
    if safeguarded != isinstance(memory,SafeguardedQPMemory):
        raise InputValidationError('safeguarded QP requires matching geometric memory')
    seed_filter=(memory.select(problem,base,gradient,inverse) if coverage else
                 filter_seed(problem,base,memory,gradient,inverse) if economical else None)
    evaluator=evaluate_trial if backend is None else backend.evaluate_trial
    model=None;radius=None;extend=False;enrichments=0;radius_seed=None
    calibrated=config.policy!='legacy-e38'
    pressure=config.policy in ('positive-pressure-v1','positive-pressure-v2')
    curvature_info=None
    for attempt in range(config.max_trials):
        checkpoint('constraint_probe_attempt',attempt=attempt+1)
        if budget is not None and budget():return finish('work_budget_exhausted','WALL_BUDGET')
        update=dict(poisson_calls=0,seconds=0.,added_rows=0)
        if model is None:
            support=linearize_points(problem,base,memory.points);support.validate(problem,base)
            model=PoissonConstraintModel(gradient,support,inverse,theta=config.theta,
                **({} if experiment is None else dict(dual_solver=nonnegative_qp)))
            if experiment=='curvature':
                try:curvature_info=augment_curvature(problem,model,inverse)
                except (FloatingPointError,np.linalg.LinAlgError) as error:
                    failure=dict(accepted=False,reason='CURVATURE_MODEL_FAILED',error=str(error),experiment=experiment)
                    history.append(failure)
                    if iteration_observer:iteration_observer(dict(failure))
                    return finish('direction_subproblem_failed',str(error))
            counters.poisson+=model.poisson_calls
            update=dict(poisson_calls=model.setup_poisson_calls,seconds=model.setup_seconds,
                        added_rows=len(support.c))
            if radius is None:
                fraction=config.radius_fraction
                if calibrated and not safeguarded:
                    predictor=JDStepMemory();predictor.bind(problem.grid)
                    for point in memory.points:predictor.witness(*point)
                    bound=sampling_step_bound(problem,base,model.d0)
                    tau,info=predictor.predict(problem,base,model.d0,min(1.,.9*bound))
                    factor=2. if config.policy=='jd-calibrated-v2' or pressure else 8.
                    fraction=min(fraction,factor*tau)
                    radius_seed=dict(prediction=info,fraction=fraction,expansion_factor=factor)
                elif safeguarded:
                    radius_seed=dict(fraction=fraction,rule='independent_QP_metric_radius_NOT_PG_root')
                radius=fraction*np.sqrt(model.d0_norm2)
                if radius<=0:return finish('domain_blocked','NO_POSITIVE_INITIAL_RADIUS')
        elif extend:
            support=linearize_points(problem,base,memory.points);support.validate(problem,base)
            update=model.extend(support);counters.poisson+=update['poisson_calls'];extend=False
        proposed=model.solve(radius,dual_maxiter=config.dual_maxiter)
        counters.poisson+=proposed.diagnostics['solve_poisson_calls']
        row=dict(iteration=0,attempt=attempt+1,accepted=False,base_eval_id=base.eval_id,
            controls_id=problem.controls.controls_id,direction_model=proposed.diagnostics,
            radius_M=radius,reason=proposed.reason,step_policy='JD_CONSTRAINED_POISSON',
            actual_reduction=None,certified_reduction=None,energy_resolution=None,
            classification=None,witnesses=[],support_points=list(memory.points))
        if experiment is not None:
            row.update(experiment=experiment,seed_filter=seed_filter,curvature=curvature_info,
                metric_matrix='H=M+selected_barrier_GGN' if experiment=='curvature' else 'M',
                physical_poisson_calls=physical_inverse.calls)
        if calibrated:
            row.update(radius_seed=radius_seed,model_update=update,policy=config.policy)
        if not proposed.valid:
            history.append(row)
            if iteration_observer:iteration_observer(dict(row))
            return finish('direction_subproblem_failed',proposed.reason)
        bound=sampling_step_bound(problem,base,proposed.d)
        alpha=min(1.,.9*bound)
        if alpha<=0:return finish('domain_blocked','NO_POSITIVE_DOMAIN_STEP')
        candidate=base.theta+alpha*proposed.d;step=candidate-base.theta
        search_metric=model.inverse if experiment=='curvature' else (metric if context is None else context.inverse)
        slope=float(gradient@step);pred=-slope-.5*float(step@search_metric.matvec(step))
        if slope>=0 or pred<=0:return finish('non_descent_direction','ACTUAL_STEP_MODEL_NOT_DESCENT')
        found=[]
        def certificate(proof):
            if proof.status=='infeasible' and proof.witness_cell is not None and proof.witness_reference_point is not None:
                found.append((proof.witness_cell,tuple(proof.witness_reference_point)))
        def witness(cell,point):found.append((int(cell),tuple(point)))
        extra={} if backend is None else dict(witness_observer=witness)
        if pressure:
            candidate_u=inject_free(candidate,problem.grid)
            if safeguarded:
                found,scan=historical_pressure(problem,base,candidate_u,memory)
                row['historical_pressure']=scan
            if not found:
                found,scan=collect_pressure(problem,base,candidate_u,backend=backend)
                row['positive_pressure']=scan
        if pressure and found:
            # This is NOT an infeasible-map certificate, nor an energy trial.
            trial=InvalidTrial('JD_COMPRESSION_PRESSURE','relative slack trust policy',1)
        else:
            counters.trials+=1
            trial=evaluator(problem,candidate,certificate_observer=certificate,**extra)
        if use_soc and isinstance(trial,InvalidTrial) and trial.reason in ('JACOBIAN_VIOLATION','JD_COMPRESSION_PRESSURE') and found:
            corrected,soc=conditional_soc(problem,base,step,found,inverse,theta=config.theta)
            row['soc']=soc
            if corrected is not None:
                new_candidate=base.theta+corrected
                corrected=new_candidate-base.theta  # actual rounded increment
                new_slope=float(gradient@corrected);new_norm2=float(corrected@search_metric.matvec(corrected))
                new_pred=-new_slope-.5*new_norm2
                if new_slope<0 and new_pred>0 and new_norm2<=(1.25*radius)**2:
                    old_found=found
                    # Domain scaling would change this correction: only use it
                    # directly if the unchanged sampling bound admits unit step.
                    if sampling_step_bound(problem,base,corrected)>=1.:
                        step=corrected;candidate=new_candidate;slope=new_slope;pred=new_pred;found=[]
                        soc.update(applied=True,original_witnesses=old_found,
                            role='corrected_actual_increment_NOT_original_QP_solution')
                        if pressure:
                            found,scan=collect_pressure(problem,base,inject_free(candidate,problem.grid),backend=backend)
                            soc['pressure_after']=scan
                        if found:
                            trial=InvalidTrial('JD_COMPRESSION_PRESSURE','corrected relative slack trust policy',1)
                        else:
                            counters.trials+=1
                            trial=evaluator(problem,candidate,certificate_observer=certificate,**extra)
                    else:soc['application_rejected']='sampling_domain'
                else:soc['application_rejected']='slope_model_or_correction_radius'
            row['physical_poisson_calls']=physical_inverse.calls
        row.update(domain_alpha=alpha,gradient_dot_increment=slope,predicted_reduction=pred,
            increment_max=float(np.max(np.abs(step))),increment_l2=float(np.linalg.norm(step)),
            required_reduction=max(-config.c1*slope,config.gain_min*pred))
        rebuild=False
        if isinstance(trial,InvalidTrial):
            row['reason']=trial.reason
            if trial.reason in ('JACOBIAN_VIOLATION','JD_COMPRESSION_PRESSURE') and found:
                if calibrated and trial.reason=='JACOBIAN_VIOLATION':
                    remaining=config.max_support-len(memory.points)
                    new_found=set(found)-set(memory.points)
                    limit=min(8-len(new_found),remaining-len(new_found))
                    if limit>0:
                        additional,scan=collect_witnesses(problem,inject_free(candidate,problem.grid),
                            exclude=list(memory.points)+found,limit=limit,backend=backend)
                        found.extend(additional);row['witness_batch']=scan
                row['witnesses']=[(rejection_diagnostic(problem,base,step,proposed.d,point,model.support.points,theta=config.theta)
                    if context is not None else witness_diagnostic(problem,base,step,point,theta=config.theta)) for point in dict.fromkeys(found)]
                if coverage:
                    for point in dict.fromkeys(found):memory.remember(point,blocking=True)
                    row['history_library']=dict(size=len(memory.library),observations=memory.observations,repeated=memory.repeated)
                row['classification']=('linear_blocker' if any(x['classification']=='linear_blocker'
                    for x in row['witnesses']) else 'nonlinear_model_error')
                try:
                    for point,diagnostic in zip(dict.fromkeys(found),row['witnesses']):
                        if not coverage or diagnostic['rejection_class']=='omitted_constraint':
                            rebuild=memory.add(*point) or rebuild
                except WorkingSetLimit:
                    if calibrated:row['workset_capacity_reached']=True
                    else:row['reason']='WORKING_SET_LIMIT'
            elif trial.reason=='FEASIBILITY_UNCERTAIN':row['classification']='uncertain_not_a_witness'
        else:
            reduction=(stable_energy_reduction(problem,base,trial) if backend is None else
                backend.stable_energy_reduction(problem,base,trial,required_reduction=row['required_reduction']))
            row.update(actual_reduction=reduction.actual,certified_reduction=reduction.certified,
                energy_resolution=reduction.resolution,gain=reduction.actual/pred,
                certificate_min_jd=trial.certificate.min_jd_lower_bound,
                trial_eval_id=trial.eval_id,reason='MODEL_GAIN_OR_ARMIJO_REJECTED')
            if reduction.certified>0 and reduction.certified>=row['required_reduction']:
                promoted=promote_trial(problem,trial) if backend is None else backend.promote_trial(problem,trial)
                counters.promotions+=1
                if not isinstance(promoted,InvalidTrial):
                    if budget is not None and budget():return finish('work_budget_exhausted','WALL_BUDGET')
                    base=promoted;row.update(accepted=True,reason='ACCEPTED')
                    # Never report the OLD-base gradient as the new forcing residual.
                    new_lin=build_linearization(problem,base)
                    grad=energy_gradient(new_lin) if backend is None else backend.gradient(new_lin,target)
                    counters.gradients+=1;G=gradient_norm_mass(grad,problem.grid.omega)
                    if context is not None:context.advance(base,grad)
                    if economical:row['accepted_support_filter']=filter_accepted(model,proposed,memory,actual_step=step)
                else:row['reason']=promoted.reason
        if calibrated and not row['accepted']:
            enrichments+=1
            pressure_v2=config.policy=='positive-pressure-v2'
            keep=(row['classification']=='linear_blocker' and rebuild
                  and enrichments<(4 if pressure_v2 else 2)
                  and (pressure_v2 or len(memory.points)<config.max_support))
            row.update(radius_action=('enrich_bounded' if pressure_v2 else 'enrich_once') if keep else 'shrink',
                       next_radius_M=radius if keep else radius*.25,
                       new_support_count=len(memory.points))
            if pressure_v2:
                row.update(consecutive_rejections=enrichments,same_radius_retry_limit=3)
        row['elapsed_seconds']=perf_counter()-started;history.append(row)
        if iteration_observer:iteration_observer(dict(row))
        if row['accepted']:return finish('max_iterations','ONE_STEP_PROBE_ONLY',1)
        if row['reason']=='WORKING_SET_LIMIT':return finish('working_set_limit',row['reason'])
        if row['reason'].startswith('NONFINITE'):return finish('nonfinite_evaluation',row['reason'])
        if calibrated:
            radius=row['next_radius_M'];extend=rebuild
            if row['radius_action']=='shrink':enrichments=0
        elif row['classification']=='linear_blocker' and rebuild:
            model=None  # same base, extra true inequality, keep radius
        else:
            radius*=.25
            if rebuild:model=None
    return finish('trial_budget_exhausted','BOUNDED_PROBE; NOT_CONVERGENCE')
