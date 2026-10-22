"""E47 real edge-minimum nominations, NOT barrier quadrature or JD proof.

On a Q1 cell edge J is quadratic. Its floating-point stationary point is
only a physical-location nomination: relative compression is checked with
outward-rounded point intervals; the original full-cell certificate remains.
"""
from itertools import product
from time import perf_counter

import numpy as np

from ..cell_q1 import CellQ1Operators3D, cell_node_indices
from ..geometry import determinant
from ..stationarity import point_jd_gradient, point_jd_interval
from ..work_control import checkpoint
from .sequential_qp import CoverageQPMemory


def edge_minima(grid, u, cells, activation):
    """Interior minima of each historical cell's 12 edges below activation."""
    cells=sorted(set(cells))
    if not cells:return []
    ops=CellQ1Operators3D(grid,location='closed27_uniform')
    nodes=np.concatenate([cell_node_indices(grid,slice(c,c+1)) for c in cells])
    local=u.reshape(-1,3)[nodes]
    F=np.empty((len(cells),27,3,3))
    for r in range(3):
        for j,coefficients in enumerate(ops.coefficients):
            F[...,r,j]=local[...,r]@coefficients.T+float(r==j)
    J=determinant(F);lookup={tuple(p):i for i,p in enumerate(ops.points)}
    records=[]
    for axis in range(3):
        for ends in product((0.,1.),repeat=2):
            p=np.zeros(3);p[[j for j in range(3) if j!=axis]]=ends
            q=[]
            for t in (0.,.5,1.):p[axis]=t;q.append(lookup[tuple(p)])
            f0,fm,f1=J[:,q].T
            a=2*(f1+f0-2*fm);b=f1-f0-a
            t=np.full(len(cells),np.inf)
            # Avoid manufacturing interior minima from flat-edge roundoff.
            # This is a nomination filter, never a feasibility tolerance.
            noise=64*np.finfo(float).eps*np.max(np.abs(J[:,q]),axis=1)
            np.divide(-b,2*a,out=t,where=a>noise)
            for i in np.flatnonzero((t>0)&(t<1)):
                value=f0[i]+t[i]*(b[i]+t[i]*a[i])
                if value<activation:
                    p[axis]=t[i];records.append((float(value),cells[i],tuple(p)))
    return [(c,p) for _,c,p in sorted(records)]


class SafeguardedQPMemory(CoverageQPMemory):
    def select(self,problem,base,gradient,inverse):
        points=edge_minima(problem.grid,base.u,[c for c,_ in self.library],problem.barrier_activation)
        for point in reversed(points):self.remember(point)
        info=super().select(problem,base,gradient,inverse)
        info.update(edge_minima_nominated=len(points),rule='history128_with_current_edge_minima')
        return info


def historical_pressure(problem,base,u,memory,limit=8):
    """Same-point 50% guard at history and candidate edge minima, interval proof."""
    tick=perf_counter();grid=problem.grid;delta=problem.certificate_config.delta
    activation=problem.barrier_activation
    extra=edge_minima(grid,u,[c for c,_ in memory.library],activation)
    points=list(dict.fromkeys(list(memory.library)+extra));pool=[]
    for cell,point in points:
        nodes=cell_node_indices(grid,slice(cell,cell+1))[0]
        current=base.u.reshape(-1,3)[nodes];trial=u.reshape(-1,3)[nodes]
        j0,_=point_jd_gradient(current,grid.spacing,point)
        j1,_=point_jd_gradient(trial,grid.spacing,point)
        if j0>delta and j1<activation and j1-delta<.5*(j0-delta):
            pool.append(((j1-delta)/(j0-delta),cell,point,current,trial))
    found=[];records=[]
    for _,cell,point,current,trial in sorted(pool,key=lambda r:r[:3]):
        checkpoint('historical_edge_pressure',checked=len(records))
        j0=point_jd_interval(current,grid.spacing,point)
        j1=point_jd_interval(trial,grid.spacing,point)
        if j0.lo>delta and j1.hi<activation and (j1-delta).hi<((j0-delta)*.5).lo:
            found.append((cell,point));records.append(dict(cell=cell,point=list(point),
                base_jd_interval=[float(j0.lo),float(j0.hi)],
                trial_jd_interval=[float(j1.lo),float(j1.hi)]))
            if len(found)==limit:break
    return found,dict(seconds=perf_counter()-tick,edge_minima=len(extra),checked_points=len(points),
        records=records,scope='historical cells only, NOT full-cell positivity proof')
