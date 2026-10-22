"""E41 CPU outward-interval proof of JD>delta on space x [0,alpha].

Experimental geometry-only certificate. Does not replace map-domain, energy,
endpoint rounding or stopping checks. A budget/loose bound is UNCERTAIN.
"""
from itertools import product
from time import perf_counter

import numpy as np

from ...validation import as_float64_array, finite_real
from .. import interval as iv
from ..cell_q1 import VERTICES, cell_node_indices, cell_tiles
from ..feasibility import POINTS, _spacing
from ..validation import integer_at_least
from ..work_control import checkpoint


def _matrix(vertices, spacing, identity):
    values=iv.Interval.point(vertices);F=[[None]*3 for _ in range(3)]
    for j in range(3):
        derivative=iv.Interval.point(np.zeros((27,3)))
        for a,bits in enumerate(VERTICES):
            if bits[j]:continue
            other=list(bits);other[j]=1;b=VERTICES.index(tuple(other));weight=np.ones(27)
            for k in range(3):
                if k!=j:weight*=POINTS[:,k] if bits[k] else 1-POINTS[:,k]
            edge=(values[b]-values[a])/spacing[j]
            derivative=derivative+edge[None,:].scale_nonnegative(weight[:,None])
        for r in range(3):F[r][j]=derivative[:,r]+float(identity and r==j)
    return F


def _det(F):
    return (F[0][0]*(F[1][1]*F[2][2]-F[1][2]*F[2][1])
        +F[0][1]*(F[1][2]*F[2][0]-F[1][0]*F[2][2])
        +F[0][2]*(F[1][0]*F[2][1]-F[1][1]*F[2][0]))


def coefficient_intervals(vertices, increment, spacing, delta=0.):
    """4 power-of-alpha coefficients, each degree-(2,2,2) spatial Bernstein.

    Coefficients formed by determinant multilinearity, not cancellation-prone
    interpolation in alpha. Reusable for the SAME actual base and direction.
    """
    u=as_float64_array(vertices,'vertices',(8,3));v=as_float64_array(increment,'increment',(8,3))
    spacing=_spacing(spacing);delta=finite_real(delta,'delta')
    A=_matrix(u,spacing,True);B=_matrix(v,spacing,False)
    coefficients=[iv.Interval.point(np.zeros(27)) for _ in range(4)]
    for choice in product((0,1),repeat=3):
        F=[[(B if choice[j] else A)[r][j] for j in range(3)] for r in range(3)]
        degree=sum(choice);coefficients[degree]=coefficients[degree]+_det(F)
    coefficients[0]=coefficients[0]-delta
    value=iv.stack(coefficients,0)
    return iv.lagrange_to_bernstein(iv.Interval(value.lo.reshape(4,3,3,3),value.hi.reshape(4,3,3,3)))


def time_bernstein(coefficients, alpha):
    alpha=finite_real(alpha,'alpha',positive=True)
    t=iv.Interval.point(alpha);c0,c1,c2,c3=(coefficients[i] for i in range(4))
    a=c1*t;b=c2*t*t;c=c3*t*t*t
    return iv.stack([c0,c0+a/3.,c0+(a*2.)/3.+b/3.,c0+a+b+c],0)


def _split_time(coefficients):
    a,b,c,d=(coefficients[i] for i in range(4))
    ab=(a+b)*.5;bc=(b+c)*.5;cd=(c+d)*.5
    abc=(ab+bc)*.5;bcd=(bc+cd)*.5;mid=(abc+bcd)*.5
    return iv.stack([a,ab,abc,mid],0),iv.stack([mid,bcd,cd,d],0)


def certify_cell_prefix(coefficients, alpha, *, max_boxes=255, max_depth=24):
    """Strict positive Bernstein cover in all four dimensions, no root solver."""
    max_boxes=integer_at_least(max_boxes,'max_boxes');max_depth=integer_at_least(max_depth,'max_depth',0)
    if not iv.arithmetic_supported():return dict(status='uncertain',reason='unsupported_arithmetic',boxes=0)
    try:
        pending=[(time_bernstein(coefficients,alpha),0)];boxes=0;lower=np.inf
        while pending:
            checkpoint('jd_space_time_subdivision',boxes=boxes)
            if boxes>=max_boxes:return dict(status='uncertain',reason='box_budget',boxes=boxes)
            values,depth=pending.pop();boxes+=1;m=float(values.lo.min())
            if m>0:lower=min(lower,m);continue
            # Only actual tensor corners, never negative interior coefficients.
            if np.any(values.hi[[0,3]][:,::2,::2,::2]<=0):
                return dict(status='not_positive_prefix',reason='actual_space_time_corner',boxes=boxes)
            if depth>=max_depth:return dict(status='uncertain',reason='depth_budget',boxes=boxes)
            axis=depth%4
            children=_split_time(values) if axis==3 else iv.split_bernstein(values,axis)
            # Worst lower bound first; both children must be certified.
            pending.extend((child,depth+1) for child in sorted(children,key=lambda c:float(c.lo.min()),reverse=True))
        return dict(status='certified_positive_prefix',lower_bound=lower,boxes=boxes)
    except FloatingPointError:
        return dict(status='uncertain',reason='arithmetic_uncertain',boxes=0)


def swept_edge_enclosure(vertices, increment, spacing, alpha):
    """All spatial derivatives and alpha∈[0,alpha], avoiding nodal hull loss."""
    spacing=_spacing(spacing);alpha=finite_real(alpha,'alpha',positive=True)
    u=iv.Interval.point(vertices);v=iv.Interval.point(increment);t=iv.Interval(0.,alpha)
    F=[[None]*3 for _ in range(3)]
    for j in range(3):
        edges=[]
        for a,bits in enumerate(VERTICES):
            if bits[j]:continue
            other=list(bits);other[j]=1;b=VERTICES.index(tuple(other))
            edges.append((u[:,b]-u[:,a])/spacing[j]+t*((v[:,b]-v[:,a])/spacing[j]))
        hull=iv.Interval(np.minimum.reduce([e.lo for e in edges]),np.maximum.reduce([e.hi for e in edges]))
        for r in range(3):F[r][j]=hull[:,r]+float(r==j)
    return _det(F)


def certify_prefix(u, increment, grid, alpha, *, delta=0., tile_size=8192, max_boxes=255):
    """Full-field geometry prefix only if every cell succeeds. Streamed memory.

    Full-volume power coefficients are intentionally NOT materialized. This
    prototype recomputes the inexpensive sweep per requested alpha and builds
    space-time polynomials only for its inconclusive cells.
    """
    u=as_float64_array(u,'u',grid.shape+(3,));v=as_float64_array(increment,'increment',u.shape)
    alpha=finite_real(alpha,'alpha',positive=True);delta=finite_real(delta,'delta')
    tile_size=integer_at_least(tile_size,'tile_size');started=perf_counter()
    fast=fallback=checked=boxes=0;lower=np.inf;edge_seconds=polynomial_seconds=0.
    def result(status,**kw):
        return dict(status=status,alpha=alpha,delta=delta,cells_checked=checked,total_cells=grid.n_cells,
            fast_cells=fast,fallback_cells=fallback,boxes=boxes,seconds=perf_counter()-started,
            edge_seconds=edge_seconds,polynomial_seconds=polynomial_seconds,
            lower_bound=None if not np.isfinite(lower) else lower,
            scope='JD of exact real interpolation of supplied float64 u and p on [0,alpha]; NOT rounded endpoint/map-domain/energy',**kw)
    if not iv.arithmetic_supported():return result('uncertain',reason='unsupported_arithmetic')
    for tile in cell_tiles(grid,tile_size):
        checkpoint('jd_full_prefix',completed=tile.start,total=grid.n_cells)
        nodes=cell_node_indices(grid,tile);local=u.reshape(-1,3)[nodes];step=v.reshape(-1,3)[nodes]
        tick=perf_counter()
        try:bounds=(swept_edge_enclosure(local,step,grid.spacing,alpha)-delta).lo
        except FloatingPointError:bounds=np.full(len(nodes),-np.inf)
        edge_seconds+=perf_counter()-tick;ok=bounds>0;n=int(ok.sum())
        fast+=n;checked+=n
        if n:lower=min(lower,float(bounds[ok].min()))
        for k in np.flatnonzero(~ok):
            checkpoint('jd_prefix_fallback',cell=tile.start+int(k))
            tick=perf_counter();fallback+=1;checked+=1
            try:
                c=coefficient_intervals(local[k],step[k],grid.spacing,delta)
                proof=certify_cell_prefix(c,alpha,max_boxes=max_boxes)
            except FloatingPointError:proof=dict(status='uncertain',reason='arithmetic_uncertain',boxes=0)
            polynomial_seconds+=perf_counter()-tick;boxes+=proof['boxes']
            if proof['status']!='certified_positive_prefix':
                return result(proof['status'],cell=tile.start+int(k),local=proof)
            lower=min(lower,proof['lower_bound'])
    return result('certified_positive_prefix')
