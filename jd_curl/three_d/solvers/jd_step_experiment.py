"""E41 opt-in coefficient cache and bounded proactive nominations, NOT proof.

Old JDStepMemory/default solvers are unchanged. Cache keys include actual
base, reference/problem, grid, delta and a value hash of the direction.
"""
import hashlib
from time import perf_counter

import numpy as np

from ...validation import InputValidationError, finite_real
from ..boundary import inject_free
from ..cell_q1 import VERTICES, cell_node_indices, cell_tiles
from ..residuals import validate_state
from ..work_control import checkpoint
from .jd_step import JDStepMemory, determinant_coefficients, first_root_before


def local_coefficients(problem, base, direction, cell, points):
    """Same physical Q1 evaluation order as E26; no nodal-JD interpolation."""
    grid=problem.grid;points=np.asarray(points,dtype=float)
    D=np.empty((3,len(points),8))
    for a,vertex in enumerate(VERTICES):
        factors=[points[:,j] if vertex[j] else 1-points[:,j] for j in range(3)]
        for j in range(3):
            other=[k for k in range(3) if k!=j]
            D[j,:,a]=((1 if vertex[j] else -1)/grid.spacing[j]*factors[other[0]]*factors[other[1]])
    nodes=cell_node_indices(grid,slice(cell,cell+1))[0]
    z,y,x=np.unravel_index(nodes,grid.shape);nz,ny,nx=grid.shape
    free=(z>0)&(z<nz-1)&(y>0)&(y<ny-1)&(x>0)&(x<nx-1)
    v=direction.reshape((3,)+grid.interior_shape);local=np.zeros((8,3))
    local[free]=v[:,z[free]-1,y[free]-1,x[free]-1].T
    A=np.einsum('jqa,ar->qrj',D,base.u.reshape(-1,3)[nodes])+np.eye(3)
    B=np.einsum('jqa,ar->qrj',D,local)
    return determinant_coefficients(A,B,problem.certificate_config.delta)


def nominate(problem, base, direction, horizon, *, backend=None, capacity=256):
    """Full closed27 scan, bounded pool; NOT the minimum root over all samples.

    Rank by c0/(|c1|+h|c2|+h²|c3|), a pointwise prefix root lower
    estimate in exact arithmetic. CPU recomputes all retained polynomials.
    Omitted points may limit earlier. CUDA only nominates, never certifies.
    """
    started=perf_counter();grid=problem.grid;memory=JDStepMemory();memory.bind(grid)
    full=inject_free(direction,grid);pool=[]
    def retain(values,cells,qs):
        nonlocal pool
        pool.extend((float(v),int(c),int(q)) for v,c,q in zip(values,cells,qs) if np.isfinite(v))
        pool=sorted(pool)[:capacity]
    if backend is None:
        D=np.asarray(memory.ops.coefficients)
        for tile in cell_tiles(grid,problem.certificate_config.tile_size):
            checkpoint('jd_proactive_nomination',completed=tile.start,total=grid.n_cells)
            nodes=cell_node_indices(grid,tile)
            A=np.einsum('jqa,tar->tqrj',D,base.u.reshape(-1,3)[nodes])+np.eye(3)
            B=np.einsum('jqa,tar->tqrj',D,full.reshape(-1,3)[nodes])
            c=determinant_coefficients(A,B,problem.certificate_config.delta)
            denom=np.abs(c[...,1])+horizon*np.abs(c[...,2])+horizon*horizon*np.abs(c[...,3])
            scores=np.full(denom.shape,np.inf);np.divide(c[...,0],denom,out=scores,where=denom>0)
            q=scores.argmin(axis=1);v=scores[np.arange(len(q)),q]
            chosen=np.argsort(v,kind='stable')[:capacity];retain(v[chosen],chosen+tile.start,q[chosen])
    else:
        import torch
        model=backend._model_for(problem);device=model.device
        current=device.upload(base.u);delta=device.upload(full)
        D=device.upload(memory.ops.coefficients);pending=[]
        for (start,_),nodes in model.tiles():
            checkpoint('jd_proactive_nomination',completed=start,total=grid.n_cells)
            A,coA,J,_=model.geometry(current.reshape(-1,3)[nodes],D,with_curl=False)
            B=torch.einsum('jqa,tar->tqrj',D,delta.reshape(-1,3)[nodes])
            coB=torch.stack((torch.linalg.cross(B[...,1,:],B[...,2,:]),
                torch.linalg.cross(B[...,2,:],B[...,0,:]),torch.linalg.cross(B[...,0,:],B[...,1,:])),dim=-2)
            c1=(coA*B).sum(dim=(-2,-1));c2=(A*coB).sum(dim=(-2,-1));c3=(B[...,0,:]*coB[...,0,:]).sum(dim=-1)
            denom=c1.abs()+horizon*c2.abs()+horizon*horizon*c3.abs()
            score=torch.where(denom>0,(J-problem.certificate_config.delta)/denom,torch.full_like(J,torch.inf))
            values,q=score.min(dim=1);values,cells=torch.topk(values,min(capacity,len(values)),largest=False)
            pending.append(torch.stack((values,(cells+start).to(torch.float64),q[cells].to(torch.float64)),dim=1))
            if len(pending)==8:
                rows=device.download(torch.cat(pending));retain(*rows.T);pending.clear()
        if pending:
            rows=device.download(torch.cat(pending));retain(*rows.T)
    locations=[(cell,tuple(memory.ops.points[q])) for _,cell,q in pool]
    return locations,dict(seconds=perf_counter()-started,pool_size=len(pool),capacity=capacity,
        cells_scanned=grid.n_cells,horizon=horizon,scope='bounded nomination, NOT global sampled root or certificate')


class CachedJDStepMemory(JDStepMemory):
    """Explicit E41 predictor compatible with existing PG injection interface."""
    def __init__(self, *, proactive=False, backend=None):
        super().__init__();self.proactive=proactive;self.backend=backend
        self.key=None;self.coefficients={};self.scanned=False;self.records=[]

    def bind(self, grid):
        if self.grid_id!=grid.grid_id:
            self.key=None;self.coefficients={};self.scanned=False
        super().bind(grid)

    def predict(self, problem, base, direction, limit, safety=.9):
        started=perf_counter();self.bind(problem.grid);validate_state(problem,base)
        limit=finite_real(limit,'prediction limit',positive=True)
        safety=finite_real(safety,'prediction safety',positive=True)
        if safety>=1:raise InputValidationError('prediction safety must be <1')
        direction=np.asarray(direction,dtype=float)
        if direction.shape!=base.theta.shape or not np.all(np.isfinite(direction)):
            raise InputValidationError('invalid direction')
        digest=hashlib.sha256(np.ascontiguousarray(direction).view(np.uint8)).hexdigest()
        key=(problem.problem_id,base.eval_id,problem.grid.grid_id,problem.controls.controls_id,
             problem.certificate_config.delta.hex(),digest)
        reused=key==self.key
        if not reused:
            self.key=key;self.coefficients={};self.scanned=False
        builds=hits=0;minimum=None;limiter=None;fallback=None;scan=None
        def consider(cell,points):
            nonlocal builds,hits,minimum,limiter
            missing=[pt for pt in points if (cell,pt) not in self.coefficients]
            if missing:
                cs=local_coefficients(problem,base,direction,cell,missing);builds+=len(missing)
                for pt,c in zip(missing,cs):self.coefficients[cell,pt]=c
            hits+=len(points)-len(missing)
            for pt in points:
                root=first_root_before(self.coefficients[cell,pt],limit)
                if root is not None and (minimum is None or root<minimum):
                    minimum=root;limiter=(int(cell),pt)
        try:
            # Drop evicted cells. At most 128 historical cells + 256 nominations
            # live during a scan; no full-volume coefficient materialization.
            cells={cell for cell,_ in self.points}
            self.coefficients={k:v for k,v in self.coefficients.items() if k[0] in cells}
            for cell in sorted(cells):
                points=list(map(tuple,self.ops.points))
                points += [pt for c,pt in self.points if c==cell and pt not in points]
                consider(cell,points)
            if self.proactive and not self.scanned:
                locations,scan=nominate(problem,base,direction,minimum or limit,backend=self.backend)
                ranked=[]
                for cell,pt in locations:
                    consider(cell,[pt])
                    root=first_root_before(self.coefficients[cell,pt],limit)
                    if root is not None:ranked.append((root,cell,pt))
                for _,cell,pt in sorted(ranked)[:16]:self.witness(cell,pt)
                self.scanned=True
        except (FloatingPointError,OverflowError) as error:
            fallback=str(error);minimum=None;limiter=None
        tau=limit if minimum is None else min(limit,safety*minimum)
        info=dict(policy='e41-cached-proactive' if self.proactive else 'e41-cached-seed',
            witnesses=len(self.points),base_eval_id=base.eval_id,controls_id=problem.controls.controls_id,
            grid_id=problem.grid.grid_id,direction_id=digest,initial_tau=limit,predicted_tau=tau,
            root=minimum,limiter=limiter,safety=safety,fallback=fallback,cache_reused=reused,
            coefficients_built=builds,coefficient_hits=hits,scan=scan,seconds=perf_counter()-started)
        self.records.append(info)
        return tau,info
