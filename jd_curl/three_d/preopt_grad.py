"""S30 P-6 (2026-10-16; contract docs/contract_draft_P6_P12_preopt.md §3): the
analytic gradient of the T88 pre-objective, as a ROUNDING-LEVEL replacement of torch.autograd in the float32
pre-optimizer (an uncertified initializer; F, the certificate, E/F and GF are untouched).

P(u) = 1/2 sum_x r(x)^2 + mu/2 sum_cells [(det F - 1)^2 + |curl|^2] + kappa sum_cells relu(j0 - det F)^2
       + alpha/2 sum_interior sum_c lap_c^2,
with the conventions of preopt._pre_energy_tensors: u in voxel units on (3, D, H, W) (components x, y, z along W, H, D),
zero faces, the optimisation variable u_int = u[:, 1:-1, 1:-1, 1:-1]; r = T(x + u) - F with T the trilinear
interpolation of M (grid_sample, align_corners, border); cell-centre derivatives = mean of the 4 edge differences;
F_cd = delta_cd + (h_c/h_d) du_c/dx_d; curl = (F21 - F12, F02 - F20, F10 - F01); lap_c = h_c * Delta_h u_c.

Gradient with respect to u_int (contract §3.3):
  data      r(x) dT/di_a(x + u) chi_a   (chi_a = 0 where the unclamped coordinate is <= 0 or >= n_a - 1, as PyTorch's
            clip_coordinates_set_grad; the chain factor (n-1)/2 * 2/(n-1) is 1)
  geometry  per cell G_cd = (h_c/h_d) [ (mu (J - 1) - 2 kappa relu(j0 - J)) cof_cd + mu curl-adjoint_cd ], scattered
            to the 8 corners with +-1/4 per axis (cof = cofactor = d det / dF_cd, not the transposed adjugate)
  curvature alpha h_c Delta_h(lap_c)    (Delta_h symmetric on the interior with zero Dirichlet faces)

Two implementations:
  pre_energy_grad_reference  plain torch ops (CPU or CUDA, float32 or float64): the formula reference and the fallback;
                             the sample positions and the warped image are computed with the same operations as the
                             autograd path, so the floor cells (hence the trilinear derivative's cell) are the same.
  TritonPreGrad              CUDA float32: a voxel kernel (data residual, value and derivative of the trilinear
                             interpolation), a cell kernel (det, cofactors, curl, energies and the 9 G values) and an
                             interior-node kernel (data gradient + the 8 cells' G); block partial sums reduced by torch;
                             no atomics, so the same input gives the same output (deterministic). FP contraction off.
"""
from time import perf_counter

import numpy as np

POLICY = "preopt-analytic-grad-v1"
GRAD_REL_L2_BOUND = 1e-4          # contract §3.5: ||g_P6 - g_autograd||_2 <= 1e-4 ||g_autograd||_2 (first use per level)
ENERGY_REL_BOUND = 1e-5           # |P_P6 - P_autograd| <= 1e-5 |P_autograd|
BLOCK_VOXEL, BLOCK_CELL, BLOCK_NODE = 256, 128, 256

try:
    import triton
    import triton.language as tl
except ImportError:                # pragma: no cover
    triton = tl = None


# ------------------------------------------------------------------------------------------- torch reference

def _identity(D, H, W, device, dtype):
    import torch
    return torch.meshgrid(torch.arange(D, device=device, dtype=dtype), torch.arange(H, device=device, dtype=dtype),
                          torch.arange(W, device=device, dtype=dtype), indexing="ij")


def sample_coordinates(u, D, H, W):
    """Unclamped voxel coordinates (ix, iy, iz) of the samples x + u, with the operations of the autograd path:
    g = 2*(x + u)/max(n - 1, 1) - 1 (eager torch ops) and grid_sample's align_corners unnormalisation ((g + 1)/2)*(n - 1)."""
    zz, yy, xx = _identity(D, H, W, u.device, u.dtype)
    gx = 2*(xx+u[0])/max(W-1, 1)-1; gy = 2*(yy+u[1])/max(H-1, 1)-1; gz = 2*(zz+u[2])/max(D-1, 1)-1
    return ((gx+1)/2)*(W-1), ((gy+1)/2)*(H-1), ((gz+1)/2)*(D-1), (gx, gy, gz)


def _trilinear_derivatives(moving, ix, iy, iz):
    """d T / d(ix, iy, iz) of the trilinear interpolation at the clamped coordinates (PyTorch's cell = floor; corners
    outside the image contribute 0)."""
    import torch
    D, H, W = moving.shape
    x0, y0, z0 = torch.floor(ix), torch.floor(iy), torch.floor(iz)
    wx1, wy1, wz1 = ix-x0, iy-y0, iz-z0
    wx0, wy0, wz0 = (x0+1)-ix, (y0+1)-iy, (z0+1)-iz
    x0, y0, z0 = x0.long(), y0.long(), z0.long()

    def corner(dz, dy, dx):
        z, y, x = z0+dz, y0+dy, x0+dx
        inside = (z <= D-1) & (y <= H-1) & (x <= W-1)
        flat = (z.clamp(max=D-1)*H+y.clamp(max=H-1))*W+x.clamp(max=W-1)
        return torch.where(inside, moving.reshape(-1)[flat], torch.zeros((), dtype=moving.dtype, device=moving.device))
    c = {(dz, dy, dx): corner(dz, dy, dx) for dz in (0, 1) for dy in (0, 1) for dx in (0, 1)}
    wy, wz, wx = (wy0, wy1), (wz0, wz1), (wx0, wx1)
    dtx = sum((c[(dz, dy, 1)]-c[(dz, dy, 0)])*wy[dy]*wz[dz] for dz in (0, 1) for dy in (0, 1))
    dty = sum((c[(dz, 1, dx)]-c[(dz, 0, dx)])*wx[dx]*wz[dz] for dz in (0, 1) for dx in (0, 1))
    dtz = sum((c[(1, dy, dx)]-c[(0, dy, dx)])*wx[dx]*wy[dy] for dy in (0, 1) for dx in (0, 1))
    return dtx, dty, dtz


def _cell_derivatives(u):
    """(gz_, gy_, gx_): voxel derivatives along z, y, x at the cell centres, as preopt._pre_energy_tensors."""
    import torch

    def d(axis):
        g = torch.diff(u, dim=1+axis)
        for ax in range(3):
            if ax != axis:
                g = .5*(g.narrow(1+ax, 0, g.shape[1+ax]-1)+g.narrow(1+ax, 1, g.shape[1+ax]-1))
        return g
    return d(0), d(1), d(2)


def cell_geometry(u, spacing, mu, kappa_pre, j0):
    """Per cell: det, curl (3,), and G (3 components c, 3 axes d) = dE_cell/d(voxel derivative of u_c along d)."""
    import torch
    gz_, gy_, gx_ = _cell_derivatives(u)
    hx, hy, hz = spacing[0], spacing[1], spacing[2]
    h = (hx, hy, hz)
    der = (gx_, gy_, gz_)                                        # d = 0, 1, 2 -> x, y, z

    def J(c, dd):
        return der[dd][c]*(h[c]/h[dd])
    F00 = 1+J(0, 0); F01 = J(0, 1); F02 = J(0, 2)
    F10 = J(1, 0); F11 = 1+J(1, 1); F12 = J(1, 2)
    F20 = J(2, 0); F21 = J(2, 1); F22 = 1+J(2, 2)
    det = F00*(F11*F22-F12*F21)-F01*(F10*F22-F12*F20)+F02*(F10*F21-F11*F20)
    curl = torch.stack((F21-F12, F02-F20, F10-F01))
    cof = ((F11*F22-F12*F21, -(F10*F22-F12*F20), F10*F21-F11*F20),
           (-(F01*F22-F02*F21), F00*F22-F02*F20, -(F00*F21-F01*F20)),
           (F01*F12-F02*F11, -(F00*F12-F02*F10), F00*F11-F01*F10))
    A = mu*(det-1)-2*kappa_pre*torch.relu(j0-det)
    zero = torch.zeros_like(det)
    adj = ((zero, -curl[2], curl[1]),                            # d(1/2|curl|^2)/dF_cd
           (curl[2], zero, -curl[0]),
           (-curl[1], curl[0], zero))
    G = [[(A*cof[c][dd]+mu*adj[c][dd])*(h[c]/h[dd]) for dd in range(3)] for c in range(3)]
    return det, curl, G


def scatter_cells_to_nodes(G, shape):
    """Node gradient (3, D, H, W) of sum over cells of E_cell given G[c][d] = dE_cell/d(cell derivative of u_c along
    d), the cell derivative being 1/4 sum of the 4 edge differences along d."""
    import torch
    D, H, W = shape
    out = torch.zeros((3, D, H, W), dtype=G[0][0].dtype, device=G[0][0].device)
    for oz in (0, 1):
        for oy in (0, 1):
            for ox in (0, 1):
                sx, sy, sz = (1. if ox else -1.), (1. if oy else -1.), (1. if oz else -1.)
                for c in range(3):
                    out[c, oz:oz+D-1, oy:oy+H-1, ox:ox+W-1] += .25*(sx*G[c][0]+sy*G[c][1]+sz*G[c][2])
    return out


def _laplacian_interior(f, spacing):
    """Seven-point Laplacian (physical) of the interior nodes of a padded (3, D, H, W) field with zero faces."""
    hx, hy, hz = spacing[0], spacing[1], spacing[2]
    ui = f[:, 1:-1, 1:-1, 1:-1]
    return ((f[:, 2:, 1:-1, 1:-1]-2*ui+f[:, :-2, 1:-1, 1:-1])/(hz*hz)
            + (f[:, 1:-1, 2:, 1:-1]-2*ui+f[:, 1:-1, :-2, 1:-1])/(hy*hy)
            + (f[:, 1:-1, 1:-1, 2:]-2*ui+f[:, 1:-1, 1:-1, :-2])/(hx*hx))


def pre_energy_grad_reference(u_int, fixed, moving, spacing, mu, alpha, kappa_pre, j0):
    """(total, det, data, geometry, floor, curvature, grad) with plain torch ops; grad = dP/du_int (3, D-2, H-2, W-2).
    The energies use the expressions of preopt._pre_energy_tensors (same forward values); the gradient is analytic."""
    import torch
    import torch.nn.functional as Fn
    u = Fn.pad(u_int.detach(), (1, 1, 1, 1, 1, 1))
    if not bool(torch.isfinite(u).all()):                           # review M6: no floor(NaN) indices
        raise FloatingPointError("preopt: nonfinite displacement")
    D, H, W = fixed.shape
    ix, iy, iz, (gx, gy, gz) = sample_coordinates(u, D, H, W)
    grid = torch.stack((gx, gy, gz), dim=-1)[None]
    warped = Fn.grid_sample(moving[None, None], grid, mode="bilinear", padding_mode="border", align_corners=True)[0, 0]
    r = warped-fixed
    data = .5*(r**2).sum()
    chi = ((ix > 0) & (ix < W-1), (iy > 0) & (iy < H-1), (iz > 0) & (iz < D-1))
    ixc, iyc, izc = ix.clamp(0, W-1), iy.clamp(0, H-1), iz.clamp(0, D-1)
    dts = _trilinear_derivatives(moving, ixc, iyc, izc)
    grad = torch.stack([torch.where(chi[a], r*dts[a], torch.zeros_like(r)) for a in range(3)])
    det, curl, G = cell_geometry(u, spacing, mu, kappa_pre, j0)
    geometry = .5*mu*(((det-1)**2).sum()+(curl**2).sum())
    floor = kappa_pre*(torch.relu(j0-det)**2).sum()
    grad = grad+scatter_cells_to_nodes(G, (D, H, W))
    curvature = torch.zeros((), device=u.device, dtype=u.dtype)
    if alpha > 0:
        hvec = torch.tensor([spacing[0], spacing[1], spacing[2]], device=u.device, dtype=u.dtype)[:, None, None, None]
        lap = _laplacian_interior(u, spacing)*hvec
        curvature = .5*alpha*(lap**2).sum()
        lap_pad = Fn.pad(lap, (1, 1, 1, 1, 1, 1))
        grad[:, 1:-1, 1:-1, 1:-1] += alpha*hvec*_laplacian_interior(lap_pad, spacing)
    total = data+geometry+floor+curvature
    return total, det, data, geometry, floor, curvature, grad[:, 1:-1, 1:-1, 1:-1].contiguous()


# ------------------------------------------------------------------------------------------------ Triton (CUDA)

if triton is not None:
    @triton.jit(do_not_specialize=["n", "nint", "D", "H", "W"])
    def _data_kernel(U, FIX, MOV, GD, RSUM, n, nint, D, H, W, RW, RH, RD, WM1, HM1, DM1, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        v = pid.to(tl.int64)*BLOCK+tl.arange(0, BLOCK).to(tl.int64)
        m = v < n
        x = v % W
        y = (v//W) % H
        z = v//(W*H)
        u0 = tl.load(U+v, mask=m, other=0.)
        u1 = tl.load(U+n+v, mask=m, other=0.)
        u2 = tl.load(U+2*n+v, mask=m, other=0.)
        # sample coordinates with the autograd path's operations (eager torch on CUDA: x/s == x*(1/s) in fp32)
        gx = ((x.to(tl.float32)+u0)*2.)*RW-1.
        gy = ((y.to(tl.float32)+u1)*2.)*RH-1.
        gz = ((z.to(tl.float32)+u2)*2.)*RD-1.
        ix = ((gx+1.)*.5)*WM1
        iy = ((gy+1.)*.5)*HM1
        iz = ((gz+1.)*.5)*DM1
        cx = (ix > 0.) & (ix < WM1)
        cy = (iy > 0.) & (iy < HM1)
        cz = (iz > 0.) & (iz < DM1)
        ix = tl.minimum(tl.maximum(ix, 0.), WM1)
        iy = tl.minimum(tl.maximum(iy, 0.), HM1)
        iz = tl.minimum(tl.maximum(iz, 0.), DM1)
        fx0 = tl.floor(ix)
        fy0 = tl.floor(iy)
        fz0 = tl.floor(iz)
        x0 = fx0.to(tl.int64)
        y0 = fy0.to(tl.int64)
        z0 = fz0.to(tl.int64)
        wx1 = ix-fx0
        wy1 = iy-fy0
        wz1 = iz-fz0
        wx0 = (fx0+1.)-ix
        wy0 = (fy0+1.)-iy
        wz0 = (fz0+1.)-iz
        inx = (x0+1) <= (W-1)
        iny = (y0+1) <= (H-1)
        inz = (z0+1) <= (D-1)
        b = (z0*H+y0)*W+x0
        c000 = tl.load(MOV+b, mask=m, other=0.)
        c001 = tl.load(MOV+b+1, mask=m & inx, other=0.)
        c010 = tl.load(MOV+b+W, mask=m & iny, other=0.)
        c011 = tl.load(MOV+b+W+1, mask=m & iny & inx, other=0.)
        c100 = tl.load(MOV+b+H*W, mask=m & inz, other=0.)
        c101 = tl.load(MOV+b+H*W+1, mask=m & inz & inx, other=0.)
        c110 = tl.load(MOV+b+H*W+W, mask=m & inz & iny, other=0.)
        c111 = tl.load(MOV+b+H*W+W+1, mask=m & inz & iny & inx, other=0.)
        # value, in grid_sampler_3d's corner order tnw, tne, tsw, tse, bnw, bne, bsw, bse
        val = c000*((wx0*wy0)*wz0)
        val = val+c001*((wx1*wy0)*wz0)
        val = val+c010*((wx0*wy1)*wz0)
        val = val+c011*((wx1*wy1)*wz0)
        val = val+c100*((wx0*wy0)*wz1)
        val = val+c101*((wx1*wy0)*wz1)
        val = val+c110*((wx0*wy1)*wz1)
        val = val+c111*((wx1*wy1)*wz1)
        f = tl.load(FIX+v, mask=m, other=0.)
        r = val-f
        tl.store(RSUM+pid, tl.sum(tl.where(m, r*r, 0.), axis=0))
        dtx = (c001-c000)*(wy0*wz0)+(c011-c010)*(wy1*wz0)+(c101-c100)*(wy0*wz1)+(c111-c110)*(wy1*wz1)
        dty = (c010-c000)*(wx0*wz0)+(c011-c001)*(wx1*wz0)+(c110-c100)*(wx0*wz1)+(c111-c101)*(wx1*wz1)
        dtz = (c100-c000)*(wx0*wy0)+(c101-c001)*(wx1*wy0)+(c110-c010)*(wx0*wy1)+(c111-c011)*(wx1*wy1)
        inner = m & (x > 0) & (x < W-1) & (y > 0) & (y < H-1) & (z > 0) & (z < D-1)
        k = ((z-1)*(H-2)+(y-1))*(W-2)+(x-1)
        zero = tl.zeros_like(r)
        tl.store(GD+k, tl.where(cx, r*dtx, zero), mask=inner)
        tl.store(GD+nint+k, tl.where(cy, r*dty, zero), mask=inner)
        tl.store(GD+2*nint+k, tl.where(cz, r*dtz, zero), mask=inner)

    @triton.jit
    def _cell_derivatives_kernel(base, m, W, sz):
        """Voxel derivatives (x, y, z) of one component at the cell centre, as preopt's d(): the edge differences
        along the axis, then the average over the first remaining axis, then over the second."""
        a000 = tl.load(base, mask=m, other=0.)
        a001 = tl.load(base+1, mask=m, other=0.)
        a010 = tl.load(base+W, mask=m, other=0.)
        a011 = tl.load(base+W+1, mask=m, other=0.)
        a100 = tl.load(base+sz, mask=m, other=0.)
        a101 = tl.load(base+sz+1, mask=m, other=0.)
        a110 = tl.load(base+sz+W, mask=m, other=0.)
        a111 = tl.load(base+sz+W+1, mask=m, other=0.)
        ex = .5*(.5*((a001-a000)+(a101-a100))+.5*((a011-a010)+(a111-a110)))     # along x: avg over z, then y
        ey = .5*(.5*((a010-a000)+(a110-a100))+.5*((a011-a001)+(a111-a101)))     # along y: avg over z, then x
        ez = .5*(.5*((a100-a000)+(a110-a010))+.5*((a101-a001)+(a111-a011)))     # along z: avg over y, then x
        return ex, ey, ez

    @triton.jit(do_not_specialize=["n", "ncell", "H", "W"])
    def _cell_kernel(U, GBUF, SUMS, n, ncell, H, W, R01, R02, R10, R12, R20, R21, MU, KAPPA, J0,
                     BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        cidx = pid.to(tl.int64)*BLOCK+tl.arange(0, BLOCK).to(tl.int64)
        m = cidx < ncell
        cx = cidx % (W-1)
        cy = (cidx//(W-1)) % (H-1)
        cz = cidx//((W-1)*(H-1))
        b = (cz*H+cy)*W+cx
        sz = H*W
        gx0, gy0, gz0 = _cell_derivatives_kernel(U+b, m, W, sz)
        gx1, gy1, gz1 = _cell_derivatives_kernel(U+n+b, m, W, sz)
        gx2, gy2, gz2 = _cell_derivatives_kernel(U+2*n+b, m, W, sz)
        F00 = 1.+gx0
        F01 = gy0*R01
        F02 = gz0*R02
        F10 = gx1*R10
        F11 = 1.+gy1
        F12 = gz1*R12
        F20 = gx2*R20
        F21 = gy2*R21
        F22 = 1.+gz2
        C00 = F11*F22-F12*F21
        C01 = -(F10*F22-F12*F20)
        C02 = F10*F21-F11*F20
        C10 = -(F01*F22-F02*F21)
        C11 = F00*F22-F02*F20
        C12 = -(F00*F21-F01*F20)
        C20 = F01*F12-F02*F11
        C21 = -(F00*F12-F02*F10)
        C22 = F00*F11-F01*F10
        det = F00*C00+F01*C01+F02*C02
        k0 = F21-F12
        k1 = F02-F20
        k2 = F10-F01
        rl = tl.maximum(J0-det, 0.)
        A = MU*(det-1.)-2.*KAPPA*rl
        zero = tl.zeros_like(det)
        dm1 = det-1.
        tl.store(SUMS+pid, tl.sum(tl.where(m, dm1*dm1, zero), axis=0))
        tl.store(SUMS+tl.num_programs(0)+pid, tl.sum(tl.where(m, (k0*k0+k1*k1)+k2*k2, zero), axis=0))
        tl.store(SUMS+2*tl.num_programs(0)+pid, tl.sum(tl.where(m, rl*rl, zero), axis=0))
        tl.store(SUMS+3*tl.num_programs(0)+pid, tl.min(tl.where(m, det, 3.0e38), axis=0))
        # G_cd = (A cof_cd + mu adj_cd) * (h_c/h_d); adj = d(1/2|curl|^2)/dF
        tl.store(GBUF+0*ncell+cidx, A*C00, mask=m)
        tl.store(GBUF+1*ncell+cidx, (A*C01+MU*(-k2))*R01, mask=m)
        tl.store(GBUF+2*ncell+cidx, (A*C02+MU*k1)*R02, mask=m)
        tl.store(GBUF+3*ncell+cidx, (A*C10+MU*k2)*R10, mask=m)
        tl.store(GBUF+4*ncell+cidx, A*C11, mask=m)
        tl.store(GBUF+5*ncell+cidx, (A*C12+MU*(-k0))*R12, mask=m)
        tl.store(GBUF+6*ncell+cidx, (A*C20+MU*(-k1))*R20, mask=m)
        tl.store(GBUF+7*ncell+cidx, (A*C21+MU*k0)*R21, mask=m)
        tl.store(GBUF+8*ncell+cidx, A*C22, mask=m)

    @triton.jit(do_not_specialize=["nint", "ncell", "H", "W"])
    def _node_kernel(GBUF, GD, OUT, nint, ncell, H, W, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        i = pid.to(tl.int64)*BLOCK+tl.arange(0, BLOCK).to(tl.int64)
        m = i < nint
        x = i % (W-2)+1
        y = (i//(W-2)) % (H-2)+1
        z = i//((W-2)*(H-2))+1
        acc0 = tl.load(GD+i, mask=m, other=0.)
        acc1 = tl.load(GD+nint+i, mask=m, other=0.)
        acc2 = tl.load(GD+2*nint+i, mask=m, other=0.)
        for a in tl.static_range(2):
            for bb in tl.static_range(2):
                for cc in tl.static_range(2):
                    cell = ((z-1+a)*(H-1)+(y-1+bb))*(W-1)+(x-1+cc)
                    sx = 0.25 if cc == 0 else -0.25          # node is the cell's high-x corner when cc == 0
                    sy = 0.25 if bb == 0 else -0.25
                    sz = 0.25 if a == 0 else -0.25
                    acc0 += (sx*tl.load(GBUF+0*ncell+cell, mask=m, other=0.)+sy*tl.load(GBUF+1*ncell+cell, mask=m, other=0.)
                             + sz*tl.load(GBUF+2*ncell+cell, mask=m, other=0.))
                    acc1 += (sx*tl.load(GBUF+3*ncell+cell, mask=m, other=0.)+sy*tl.load(GBUF+4*ncell+cell, mask=m, other=0.)
                             + sz*tl.load(GBUF+5*ncell+cell, mask=m, other=0.))
                    acc2 += (sx*tl.load(GBUF+6*ncell+cell, mask=m, other=0.)+sy*tl.load(GBUF+7*ncell+cell, mask=m, other=0.)
                             + sz*tl.load(GBUF+8*ncell+cell, mask=m, other=0.))
        tl.store(OUT+i, acc0, mask=m)
        tl.store(OUT+nint+i, acc1, mask=m)
        tl.store(OUT+2*nint+i, acc2, mask=m)


def triton_supported(device):
    return triton is not None and getattr(device, "type", str(device)) == "cuda"


class TritonPreGrad:
    """The fused float32 CUDA path; buffers are kept per level grid."""

    def __init__(self):
        self._buffers = {}

    def _buf(self, key, numel, device, dtype):
        b = self._buffers.get(key)
        if b is None or b.numel() != numel or b.device != device:
            import torch
            b = torch.empty(numel, device=device, dtype=dtype)
            self._buffers[key] = b
        return b

    def __call__(self, u_int, fixed, moving, spacing, mu, alpha, kappa_pre, j0):
        import torch
        import torch.nn.functional as Fn
        if u_int.dtype != torch.float32 or fixed.dtype != torch.float32 or moving.dtype != torch.float32:
            raise TypeError("TritonPreGrad: float32 tensors required")
        D, H, W = fixed.shape
        dev = u_int.device
        u = Fn.pad(u_int.detach(), (1, 1, 1, 1, 1, 1)).contiguous()
        fixed = fixed.contiguous(); moving = moving.contiguous()
        n, nint, ncell = D*H*W, (D-2)*(H-2)*(W-2), (D-1)*(H-1)*(W-1)
        f32 = lambda v: float(np.float32(v))
        rcp = lambda s: f32(np.float32(1.)/np.float32(max(s-1, 1)))        # eager torch: x / s == x * float32(1/s)
        hx, hy, hz = spacing[0], spacing[1], spacing[2]
        h = (hx, hy, hz)
        ratio = lambda c, d: f32(h[c]/h[d])                                  # tensor * python float: fp32 scalar
        gd = self._buf(("gd", D, H, W), 3*nint, dev, torch.float32)
        gbuf = self._buf(("g", D, H, W), 9*ncell, dev, torch.float32)
        out = torch.empty((3, D-2, H-2, W-2), device=dev, dtype=torch.float32)
        nb_v, nb_c, nb_n = triton.cdiv(n, BLOCK_VOXEL), triton.cdiv(ncell, BLOCK_CELL), triton.cdiv(nint, BLOCK_NODE)
        rsum = self._buf(("rsum", D, H, W), nb_v, dev, torch.float32)
        sums = self._buf(("sums", D, H, W), 4*nb_c, dev, torch.float32)
        _data_kernel[(nb_v,)](u, fixed, moving, gd, rsum, n, nint, D, H, W, rcp(W), rcp(H), rcp(D),
                              f32(W-1), f32(H-1), f32(D-1), BLOCK=BLOCK_VOXEL, num_warps=4, enable_fp_fusion=False)
        _cell_kernel[(nb_c,)](u, gbuf, sums, n, ncell, H, W, ratio(0, 1), ratio(0, 2), ratio(1, 0), ratio(1, 2),
                              ratio(2, 0), ratio(2, 1), f32(mu), f32(kappa_pre), f32(j0), BLOCK=BLOCK_CELL,
                              num_warps=4, enable_fp_fusion=False)
        _node_kernel[(nb_n,)](gbuf, gd, out, nint, ncell, H, W, BLOCK=BLOCK_NODE, num_warps=4, enable_fp_fusion=False)
        s = sums.view(4, nb_c)
        data = .5*rsum.sum()
        geometry = .5*mu*(s[0].sum()+s[1].sum())
        floor = kappa_pre*s[2].sum()
        det_min = s[3].min()
        curvature = torch.zeros((), device=dev, dtype=torch.float32)
        if alpha > 0:
            hvec = torch.tensor([hx, hy, hz], device=dev, dtype=torch.float32)[:, None, None, None]
            lap = _laplacian_interior(u, spacing)*hvec
            curvature = .5*alpha*(lap**2).sum()
            out += alpha*hvec*_laplacian_interior(Fn.pad(lap, (1, 1, 1, 1, 1, 1)), spacing)
        total = data+geometry+floor+curvature
        return total, det_min, data, geometry, floor, curvature, out


# ------------------------------------------------------------------------------------- per-run driver with checks

def new_record(impl):
    return dict(policy=POLICY, impl=impl, grad_rel_l2_bound=GRAD_REL_L2_BOUND, energy_rel_bound=ENERGY_REL_BOUND,
                checks=[], disabled=False, autograd_iterations=0)


class AnalyticPreGradient:
    """Per pre-optimization run: the analytic path (Triton on CUDA float32, else the torch reference), checked against
    autograd at the first iteration of every level (contract §3.5) and, when that iteration ran at u = 0 (level 0: only
    the data term is non-zero there), once more at the level's first non-zero state; a failed check switches the run
    to autograd."""

    def __init__(self, device):
        import torch
        self.device = torch.device(device)
        self.impl = "triton" if (triton_supported(self.device)) else "torch_reference"
        self.record = new_record(self.impl)
        self._triton = TritonPreGrad() if self.impl == "triton" else None
        self._checked_levels = set()
        self._recheck = set()          # review M2: a level first checked at u = 0 (geometry and curvature are 0 there)

    def _analytic(self, u_int, fixed, moving, spacing, mu, alpha, kappa_pre, j0):
        if self._triton is not None:
            return self._triton(u_int, fixed, moving, spacing, mu, alpha, kappa_pre, j0)
        total, det, data, geometry, floor, curvature, grad = pre_energy_grad_reference(
            u_int, fixed, moving, spacing, mu, alpha, kappa_pre, j0)
        return total, det.min(), data, geometry, floor, curvature, grad

    def __call__(self, level, u_int, fixed, moving, spacing, mu, alpha, kappa_pre, j0, autograd_step):
        """Returns (total, det_min, data, geometry, floor, curvature, grad) as tensors; autograd_step() gives the same
        tuple from the autograd path (used for the check and after a failed one)."""
        import torch
        if self.record["disabled"]:
            self.record["autograd_iterations"] += 1
            return autograd_step()
        if level not in self._checked_levels or level in self._recheck:
            zero_state = float(u_int.detach().abs().max()) == 0.
            if level in self._recheck:
                if zero_state:                                      # still the identity: keep waiting
                    return self._analytic(u_int, fixed, moving, spacing, mu, alpha, kappa_pre, j0)
                self._recheck.discard(level)
            else:
                self._checked_levels.add(level)
                if zero_state:
                    self._recheck.add(level)
            tick = perf_counter()
            ref = autograd_step()
            try:
                out = self._analytic(u_int, fixed, moving, spacing, mu, alpha, kappa_pre, j0)
                g, ga = out[-1].double(), ref[-1].double()
                norm = float(torch.linalg.vector_norm(ga))
                rel = float(torch.linalg.vector_norm(g-ga))/norm if norm > 0 else float(torch.linalg.vector_norm(g))
                e, ea = float(out[0]), float(ref[0])
                erel = abs(e-ea)/abs(ea) if ea != 0 else abs(e)
                ok = bool(np.isfinite(rel) and np.isfinite(erel) and rel <= GRAD_REL_L2_BOUND and erel <= ENERGY_REL_BOUND)
                self.record["checks"].append(dict(level=int(level), zero_state=bool(zero_state), grad_rel_l2=rel,
                                                  energy_rel=erel, ok=ok, seconds=perf_counter()-tick))
            except Exception as exc:                                        # recorded; the run continues on autograd
                ok = False
                self.record["checks"].append(dict(level=int(level), error=f"{type(exc).__name__}: {exc}"[:300], ok=False))
            if not ok:
                self.record["disabled"] = dict(level=int(level), reason="first_use_check")
                self.record["autograd_iterations"] += 1
                return ref
            return out
        return self._analytic(u_int, fixed, moving, spacing, mu, alpha, kappa_pre, j0)


def prewarm():
    """Compile the three Triton kernels outside the task clock (integer sizes are not specialised, so these binaries
    serve every level grid). No-op without Triton or CUDA."""
    import torch
    if not triton_supported(torch.device("cuda")) or not torch.cuda.is_available():
        return False
    u = torch.zeros((3, 3, 4, 5), dtype=torch.float32, device="cuda")
    img = torch.linspace(0., 1., 5*6*7, dtype=torch.float32, device="cuda").reshape(5, 6, 7)
    TritonPreGrad()(u, img, img, (1., 1., 1.), .1, 1e-3, 10., .2)
    torch.cuda.synchronize()
    return True
