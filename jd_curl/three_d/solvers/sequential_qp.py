"""E43 bounded E42 A+B inner solve, owned by the original Sequential driver.

Only geometric locations survive reference changes. No new stopping policy,
SOC, selected curvature or recovery of a failed inner trajectory.
"""
from dataclasses import dataclass
from collections import OrderedDict
import numpy as np
from time import perf_counter

from ...types import InvalidTrial
from ...validation import InputValidationError
from ..linearization import build_linearization, energy_gradient
from ..poisson import build_poisson_inverse
from ..validation import integer_at_least
from ..work_control import checkpoint, emit
from .common import (SolverCounters3D, prepare_base, initial_failure_status,
                     gradient_norm_mass, finish_inner, validate_target)
from .jd_constrained import JDWorkingSet
from .jd_constraint_probe import ConstrainedProbeConfig, scan_monitors, solve_constrained_probe


@dataclass(frozen=True)
class SequentialQPConfig3D(ConstrainedProbeConfig):
    theta: float = .75
    policy: str = 'positive-pressure-v1'
    max_iterations: int = 3

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self,'max_iterations',integer_at_least(self.max_iterations,'QP inner iterations',0))
        if self.policy!='positive-pressure-v1':
            raise InputValidationError('Sequential QP requires validated E42 A+B pressure policy')


@dataclass(frozen=True)
class EfficientQPConfig3D(SequentialQPConfig3D):
    """Explicit E44 equivalent execution, not a default replacement."""
    pass


class SequentialQPMemory(JDWorkingSet):
    def __init__(self, limit=32):
        super().__init__(limit);self.seeded=False

    def bind(self, grid):
        if self.grid_id!=grid.grid_id:self.seeded=False
        super().bind(grid)


@dataclass(frozen=True)
class CoverageQPConfig3D(EfficientQPConfig3D):
    """E45 bounded location history; unchanged mathematical subproblem."""
    pass


@dataclass(frozen=True)
class SafeguardedQPConfig3D(CoverageQPConfig3D):
    """E47 independent QP radius and physical historical-edge pressure guards."""
    pass


class CoverageQPMemory(SequentialQPMemory):
    """Location-only LRU library, distinct from <=32 current rows."""
    def __init__(self, limit=32):
        super().__init__(limit);self.library=OrderedDict();self.observations=0;self.repeated=0

    def bind(self, grid):
        if self.grid_id!=grid.grid_id:
            self.library.clear();self.observations=0;self.repeated=0
        super().bind(grid)

    def remember(self, point, *, blocking=False):
        # Validate without touching current membership/capacity.
        check=JDWorkingSet(1);check.bind(self.grid);check.add(*point);point=check.points[0]
        old=self.library.pop(point,False)
        if blocking:
            self.observations+=1;self.repeated+=int(old)
        self.library[point]=old or blocking
        if len(self.library)>128:self.library.popitem(last=False)

    def add(self, cell, point):
        self.remember((cell,point))
        return super().add(cell,point)

    def select(self, problem, base, gradient, inverse):
        from .jd_constrained import linearize_points
        tick=perf_counter();points=list(self.library)
        support=linearize_points(problem,base,points)
        raw=-inverse.apply(gradient)
        compression=np.asarray(support.A@raw)
        # Recent actual blockers survive direction coupling even if raw PG expands them.
        cap=min(16,self.limit);recent=[p for p in reversed(points) if self.library[p]][:cap//2]
        risk=np.maximum(0.,-compression)/support.c if len(points) else np.empty(0)
        chosen=list(recent)
        for i in np.argsort(-risk,kind='stable'):
            if len(chosen)>=cap:break
            if risk[i]>0 and points[i] not in chosen:chosen.append(points[i])
        self.points[:]=chosen
        return dict(rule='location_history_128_recent8_risk8_seed16',library_size=len(points),
            retained=len(chosen),recent_retained=len(recent),base_eval_id=base.eval_id,
            controls_id=problem.controls.controls_id,seconds=perf_counter()-tick,
            observations=self.observations,repeated_observations=self.repeated)


def qp_memory(config):
    if isinstance(config,SafeguardedQPConfig3D):
        from .qp_safeguard import SafeguardedQPMemory
        return SafeguardedQPMemory(config.max_support)
    return (CoverageQPMemory if isinstance(config,CoverageQPConfig3D) else SequentialQPMemory)(config.max_support)


def solve_sequential_qp(problem, theta, *, target=0., config=None, backend=None,
                        budget=None, iteration_observer=None, working_set=None):
    config=SequentialQPConfig3D() if config is None else config
    if not isinstance(config,SequentialQPConfig3D):raise InputValidationError('expected SequentialQPConfig3D')
    target=validate_target(target);started=perf_counter();counts=SolverCounters3D();history=[];accepted=0;G=None
    memory=qp_memory(config) if working_set is None else working_set
    if not isinstance(memory,SequentialQPMemory) or memory.limit>config.max_support:
        raise InputValidationError('expected bounded SequentialQPMemory')
    if isinstance(config,CoverageQPConfig3D) != isinstance(memory,CoverageQPMemory):
        raise InputValidationError('coverage policy requires matching history memory')
    memory.bind(problem.grid)
    current=prepare_base(problem,theta,counts,backend=backend)
    def finish(status,reason):
        return finish_inner(current,status,reason,G,target,accepted,history,counts,started)
    if isinstance(current,InvalidTrial):
        invalid=current;current=None
        return finish(initial_failure_status(invalid),invalid.reason)
    lin=build_linearization(problem,current)
    gradient=energy_gradient(lin) if backend is None else backend.gradient(lin,target)
    counts.gradients+=1;G=gradient_norm_mass(gradient,problem.grid.omega)
    if budget is not None and budget():return finish('work_budget_exhausted','WALL_BUDGET')
    if G<=target:return finish('converged_gradient','GRADIENT_TARGET')
    if config.max_iterations==0:return finish('max_iterations','ZERO_INNER_ITERATION_BUDGET')
    context=None
    if isinstance(config,EfficientQPConfig3D):
        from .qp_execution import QPContext
        context=QPContext(problem,current,gradient,target,backend)
    if not memory.seeded:
        metric=(build_poisson_inverse(problem.grid,a=problem.weights.lam,
            b=(problem.weights.mu+problem.weights.nu)/2,metric='q1') if context is None else context.metric)
        inverse=(metric if backend is None else backend.poisson(metric)) if context is None else context.inverse
        tick=perf_counter();direction=-inverse.apply(gradient);counts.poisson+=1
        seeds=scan_monitors(problem,current,direction,limit=max(1,min(6,config.max_support//2)))
        del direction
        for point in seeds['points'][:config.max_support]:memory.add(*point)
        memory.seeded=True
        emit('qp_monitor_seeded',seconds=perf_counter()-tick,points=len(memory.points),
             base_eval_id=current.eval_id,controls_id=problem.controls.controls_id)
    del gradient,lin
    for iteration in range(config.max_iterations):
        checkpoint('sequential_qp_iteration',iteration=iteration)
        if budget is not None and budget():return finish('work_budget_exhausted','WALL_BUDGET')
        def observe(row):
            row=dict(row,iteration=iteration);history.append(row)
            if iteration_observer is not None:iteration_observer(dict(row))
        result=solve_constrained_probe(problem,current.theta,target=target,config=config,
            backend=backend,budget=budget,iteration_observer=observe,working_set=memory,experiment='working_set',
            **({} if context is None else dict(context=context)))
        for key,value in vars(result.counters).items():setattr(counts,key,getattr(counts,key)+value)
        current=result.state;G=result.gradient_norm;accepted+=result.accepted_steps
        if budget is not None and budget():return finish('work_budget_exhausted','WALL_BUDGET')
        if result.status=='converged_gradient':return finish(result.status,result.reason)
        if result.status!='max_iterations' or result.accepted_steps!=1:
            return finish(result.status,result.reason)  # never relabel failure after earlier progress
        if G<=target:return finish('converged_gradient','GRADIENT_TARGET')
    return finish('max_iterations','INNER_ITERATION_BUDGET; not outer convergence')
