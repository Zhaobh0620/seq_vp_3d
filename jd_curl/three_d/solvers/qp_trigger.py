"""E46 bounded PG-first scheduling. This heuristic is not a convergence test."""
from collections import deque
import numpy as np
from .jd_step import JDStepMemory
from .sequential_qp import CoverageQPMemory,CoverageQPConfig3D
from .jd_constraint_probe import solve_constrained_probe
from .qp_execution import QPContext


class TriggerMemory(JDStepMemory):
    def __init__(self):
        super().__init__();self.qp=CoverageQPMemory();self.recent=deque(maxlen=3);self.pg_since_qp=0

    def bind(self,grid):
        if self.grid_id!=grid.grid_id:
            self.recent.clear();self.pg_since_qp=0
        super().bind(grid);self.qp.bind(grid)

    def witness(self,cell,point):
        super().witness(cell,point)
        self.qp.remember((cell,tuple(point)),blocking=True)

    def record_pg(self,low):
        self.recent.append(bool(low));self.pg_since_qp+=1


def geometry_progress(prediction,tau,proposal,bound,slope):
    """Compare model DESCENT, not bare tau, against admissible unit metric step."""
    unit=min(1.,.9*bound)
    expected=-slope*(tau-.5*tau*tau)
    reference=-slope*(unit-.5*unit*unit)
    fraction=expected/reference if reference>0 else float('inf')
    low=bool(prediction is not None and prediction.get('root') is not None
             and prediction.get('fallback') is None and 0<tau<=unit
             and expected>0 and fraction<=.1)
    return dict(low_geometry_progress=low,model_reduction=expected,
                unit_model_reduction=reference,model_fraction=fraction if np.isfinite(fraction) else None,
                model_metric_step_norm=tau*np.sqrt(max(0.,-slope)),unit_tau=unit,
                witness_root=None if prediction is None else prediction.get('root'))


def run_triggered_qp(problem,base,gradient,target,backend,budget,memory,observer):
    # No global monitoring; only actual positions already discovered by this run.
    ctx=QPContext(problem,base,gradient,target,backend)
    memory.pg_since_qp=0;memory.recent.clear()
    result=solve_constrained_probe(problem,base.theta,target=target,
        config=CoverageQPConfig3D(max_trials=4,max_iterations=1),backend=backend,budget=budget,
        working_set=memory.qp,iteration_observer=observer,experiment='working_set',context=ctx)
    for point in memory.qp.library:JDStepMemory.witness(memory,*point)
    return result,ctx
