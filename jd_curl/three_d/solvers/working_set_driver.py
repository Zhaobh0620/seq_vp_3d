"""E8 opt-in CPU working-set lifecycle around the original one-step solvers.

No residual masking, alternate energy gate, or reference refresh occurs here.
A failed release is transactional. All calls/scans/probes enter elapsed cost.
"""
from dataclasses import dataclass, field, fields, replace
from time import perf_counter

import numpy as np

from ...types import InvalidTrial
from ...validation import InputValidationError, finite_real
from ..linearization import build_linearization, energy_gradient
from ..risk import scan_risk_cells
from ..validation import integer_at_least
from ..working_set import make_working_set
from .common import InnerResult3D, SolverCounters3D, finish_inner, prepare_base, gradient_norm_mass, step_ratio
from .poisson_gradient import solve_poisson_gradient
from .lm import solve_lm


@dataclass(frozen=True)
class MaskConfig3D:
    enabled: bool = True
    risk_band: float = 1e-4
    max_growth: int = 4
    max_frozen_fraction: float = .5
    release_every: int = 5
    cooldown_steps: int = 3
    solver_seconds: float = 90.

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise InputValidationError('enabled must be explicit boolean')
        for name in ('max_growth','release_every','cooldown_steps'):
            object.__setattr__(self,name,integer_at_least(getattr(self,name),name))
        for name in ('risk_band','solver_seconds','max_frozen_fraction'):
            object.__setattr__(self,name,finite_real(getattr(self,name),name,positive=True))
        if self.max_frozen_fraction >= 1:
            raise InputValidationError('frozen fraction cap must be < 1')


@dataclass(frozen=True)
class MaskedInnerResult3D(InnerResult3D):
    diagnostics: dict = field(default_factory=dict)


_FATAL = {'nonfinite_evaluation','invalid_initial_state','invalid_initial_domain',
          'initial_feasibility_uncertain','initial_jacobian_violation'}


def solve_working_set(problem, theta, *, target, config, solver, mask_config, deadline):
    started=perf_counter();counts=SolverCounters3D();history=[];events=[]
    base=prepare_base(problem,theta,counts);accepted=0;G=None;Gfree=None;space=None
    restart={};ages={};cooldown={};scanned=None;growth=0;small=0;release_cost=0.;scan_cost=0.
    release_trials=0;release_calls=0;release_accepts=0;last_sweep=[];complete=False
    probe_cache={};periodic_done=None
    def expired():return perf_counter()>=deadline
    def finish(status,reason):
        raw=finish_inner(None if isinstance(base,InvalidTrial) else base,status,reason,G,target,accepted,history,counts,started)
        diagnostics=dict(events=events,mask_id=None if space is None else space.mask_id,
            cells=[] if space is None else list(space.cells),frozen_fraction=0. if space is None else space.frozen_fraction,
            free_dimension=None if space is None else space.n_free,restricted_gradient=Gfree,full_gradient=G,
            inner_success=status=='converged_gradient',partial_candidate=status=='inner_subspace_stationary' and accepted>0,
            release_sweep_complete=complete,release_sweep=last_sweep,
            sweep_base_eval_id=base.eval_id if complete else None,
            scan_seconds=scan_cost,release_seconds=release_cost,release_trials=release_trials,
            release_calls=release_calls,release_accepts=release_accepts,
            max_frozen_fraction=mask_config.max_frozen_fraction,
            max_observed_frozen_fraction=max([e.get('frozen_fraction',0.) for e in events]+[0.]))
        return MaskedInnerResult3D(**{f.name:getattr(raw,f.name) for f in fields(InnerResult3D)},diagnostics=diagnostics)
    if isinstance(base,InvalidTrial):return finish('invalid_initial_state',base.reason)
    space=make_working_set(problem,base,[])
    inner=solve_poisson_gradient if solver=='pg' else solve_lm
    one=replace(config,max_iterations=1)
    def norms():
        nonlocal G,Gfree
        counts.gradients+=1
        gradient=energy_gradient(build_linearization(problem,base))
        G=gradient_norm_mass(gradient,problem.grid.omega)
        Gfree=gradient_norm_mass(space.restrict(gradient),problem.grid.omega)
    def event(kind,**kw):
        events.append(dict(kind=kind,base_eval_id=base.eval_id,controls_id=problem.controls.controls_id,
                           mask_id=space.mask_id,cells=list(space.cells),frozen_fraction=space.frozen_fraction,
                           accepted_steps=accepted,elapsed_seconds=perf_counter()-started,**kw))
    def attempt(proposed,release=False):
        nonlocal release_cost,release_trials,release_calls
        # A probe's restart and mask are private until a true accepted step.
        trial_restart=dict(restart);certificates=[];tick=perf_counter()
        result=inner(problem,base.theta,target=target,config=one,subspace=proposed,
                     restart=trial_restart,budget=expired,certificate_observer=certificates.append)
        duration=perf_counter()-tick
        numeric_errors=[r['reason'] for r in result.history if r['reason'].startswith('NONFINITE')]
        if numeric_errors:
            result=replace(result,status='nonfinite_evaluation',reason=numeric_errors[-1])
        for k,v in vars(result.counters).items():setattr(counts,k,getattr(counts,k)+v)
        for row in result.history:
            history.append(dict(row,iteration=accepted,working_mask_id=proposed.mask_id,
                                working_cells=list(proposed.cells),frozen_fraction=proposed.frozen_fraction,
                                release_probe=release,elapsed_seconds=perf_counter()-started))
        if release:
            release_calls+=1;release_trials+=result.counters.trials;release_cost+=duration
        record=dict(base_eval_id=base.eval_id,controls_id=problem.controls.controls_id,mask_id=proposed.mask_id,cells=list(proposed.cells),
                    accepted=result.accepted_steps>0,status=result.status,reason=result.reason,
                    accepted_steps=result.accepted_steps,seconds=duration,trials=result.counters.trials,
                    moved_dofs=0 if result.state is None else int(np.count_nonzero(result.state.theta!=base.theta)))
        if release:event('release_probe',probe=record)
        return result,trial_restart,certificates,record
    def adopt(result,new_restart,proposed,release=False):
        nonlocal base,restart,space,accepted,small,scanned,growth,complete,last_sweep,probe_cache,release_accepts,G,Gfree
        removed=set(space.cells)-set(proposed.cells)
        delta=result.state.theta-base.theta
        small=small+1 if step_ratio(delta,base.theta,problem.grid)<=config.step_tol else 0
        base=result.state;restart=new_restart;space=proposed;space.validate(problem,base);accepted+=1
        G=Gfree=None
        if release:
            release_accepts+=1
            for c in removed:cooldown[c]=accepted+mask_config.cooldown_steps;ages.pop(c,None)
        scanned=None;growth=0;complete=False;last_sweep=[];probe_cache={}
        event('accepted_release' if release else 'accepted_step')
    def release_probe(cells):
        key=tuple(cells)
        if key in probe_cache:return probe_cache[key]
        proposed=make_working_set(problem,base,cells,previous=space,max_frozen_fraction=mask_config.max_frozen_fraction)
        result,rs,cs,record=attempt(proposed,True)
        probe_cache[key]=(result,rs,proposed,record)
        return probe_cache[key]
    def sweep():
        nonlocal complete,last_sweep
        last_sweep=[];complete=False
        blocks=sorted(space.components,key=lambda b:(min(ages.get(c,accepted) for c in b),min(b)))
        proposals=[tuple(c for c in space.cells if c not in block) for block in blocks]+[()]
        # For a single block the final all-release probe is the identical attempt;
        # reuse its same-base failure rather than spend twice or change its seed.
        for cells in dict.fromkeys(proposals):
            if expired():return 'budget'
            result,rs,proposed,record=release_probe(cells);last_sweep.append(record)
            if result.accepted_steps:
                adopt(result,rs,proposed,True)
                return 'fatal' if result.status in _FATAL else 'accepted'
            if result.status in _FATAL:return 'fatal'
            if result.status=='work_budget_exhausted':return 'budget'
        complete=True;event('complete_release_sweep',probes=last_sweep)
        return 'complete'
    while True:
        try:norms()
        except FloatingPointError:return finish('nonfinite_evaluation','NONFINITE_GRADIENT')
        if G<=target:return finish('converged_gradient','FULL_GRADIENT_TARGET')
        if expired():return finish('work_budget_exhausted','WALL_BUDGET')
        if accepted>=config.max_iterations:return finish('max_iterations','ACCEPTED_STEP_BUDGET')
        if scanned!=base.eval_id:
            tick=perf_counter()
            try:scan=scan_risk_cells(problem,base,band=mask_config.risk_band)
            except FloatingPointError:return finish('nonfinite_evaluation','NONFINITE_RISK_SCAN')
            scan_cost+=perf_counter()-tick;scanned=base.eval_id
            new=set(scan['cells'])-{c for c,until in cooldown.items() if accepted<until}
            proposed_cells=sorted(set(space.cells)|new)
            if proposed_cells!=list(space.cells):
                try:proposed=make_working_set(problem,base,proposed_cells,previous=space,max_frozen_fraction=mask_config.max_frozen_fraction)
                except InputValidationError:
                    event('mask_cap_recovery',proposed_cells=proposed_cells)
                    outcome=sweep() if space.cells else 'cap'
                    if outcome=='accepted':continue
                    return finish('mask_limit','MASK_CAP_RECOVERY_'+outcome.upper())
                for c in new:ages.setdefault(c,accepted)
                space=proposed;growth+=1;event('freeze_conservative',risk=scan)
                Gfree=None;continue
        if space.cells and (Gfree<=target or small>=config.stall_steps or space.frozen_fraction>=mask_config.max_frozen_fraction):
            outcome=sweep()
            if outcome=='accepted':continue
            if outcome=='budget':return finish('work_budget_exhausted','INCOMPLETE_RELEASE_SWEEP')
            if outcome=='fatal':return finish('nonfinite_evaluation','RELEASE_NUMERIC_FAILURE')
            if Gfree<=target:return finish('inner_subspace_stationary','COMPLETE_SAME_BASE_RELEASE_NO_PROGRESS')
            return finish('mask_blocked','RELEASE_SWEEP_NO_PROGRESS')
        if small>=config.stall_steps:return finish('stagnated','SMALL_ACCEPTED_STEPS')
        if space.cells and accepted and accepted%mask_config.release_every==0 and periodic_done!=base.eval_id:
            periodic_done=base.eval_id
            block=min(space.components,key=lambda b:(min(ages.get(c,accepted) for c in b),min(b)))
            result,rs,proposed,_=release_probe(tuple(c for c in space.cells if c not in block))
            if result.accepted_steps:
                adopt(result,rs,proposed,True)
                if result.status in _FATAL:return finish(result.status,result.reason)
                continue
            if result.status in _FATAL:return finish(result.status,result.reason)
            if result.status=='work_budget_exhausted':return finish(result.status,result.reason)
        result,rs,certs,_=attempt(space)
        if result.accepted_steps:
            adopt(result,rs,space)
            if result.status in _FATAL:return finish(result.status,result.reason)
            continue
        if result.status in _FATAL:return finish(result.status,result.reason)
        if result.status=='work_budget_exhausted':return finish(result.status,result.reason)
        restart=rs # LM damping survives a failed MAIN search and mask switch.
        violations={c.witness_cell for c in certs if c.status=='infeasible' and c.reason=='JACOBIAN_VIOLATION' and c.witness_cell is not None}
        new=violations-set(space.cells)
        if new and growth<mask_config.max_growth:
            try:proposed=make_working_set(problem,base,sorted(set(space.cells)|new),previous=space,max_frozen_fraction=mask_config.max_frozen_fraction)
            except InputValidationError:proposed=None
            if proposed is not None:
                for c in new:ages.setdefault(c,accepted);cooldown.pop(c,None)
                space=proposed;growth+=1;event('freeze_rejected_violation',cells_added=sorted(new));continue
        if space.cells:
            outcome=sweep()
            if outcome=='accepted':continue
            if outcome=='budget':return finish('work_budget_exhausted','INCOMPLETE_RELEASE_SWEEP')
            if outcome=='fatal':return finish('nonfinite_evaluation','RELEASE_NUMERIC_FAILURE')
        # A large restricted gradient with blocked search is NOT subspace stationarity.
        return finish(result.status,result.reason)


def partial_candidate_valid(problem, inner, *, target=None):
    """Separate outer audit. No eligibility from a bare status/boolean claim."""
    if not isinstance(inner,MaskedInnerResult3D) or inner.status!='inner_subspace_stationary':return False
    if target is not None and inner.target!=target:return False
    d=inner.diagnostics
    if inner.state is None or inner.accepted_steps<=0 or not d.get('release_sweep_complete'):return False
    if d.get('sweep_base_eval_id')!=inner.state.eval_id:return False
    space=make_working_set(problem,inner.state,d['cells'],max_frozen_fraction=d['max_frozen_fraction'])
    if not space.cells or space.n_free<=0:return False
    expected={tuple(c for c in space.cells if c not in block) for block in space.components}|{()}
    probes=d['release_sweep']
    if {tuple(r['cells']) for r in probes}!=expected or len(probes)!=len(expected):return False
    if any(r['base_eval_id']!=inner.state.eval_id or r['accepted'] or r['accepted_steps'] or
           r['status'] in _FATAL|{'work_budget_exhausted'} for r in probes):return False
    g=energy_gradient(build_linearization(problem,inner.state))
    return (gradient_norm_mass(g,problem.grid.omega)>inner.target and
            gradient_norm_mass(space.restrict(g),problem.grid.omega)<=inner.target)
