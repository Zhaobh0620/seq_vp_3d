"""Float64 Q1/barrier expressions without unrequested intermediate outputs.

The caller owns tensor/device and certified-map validation, as for torch_ops.
Selection does not change quadrature, arithmetic order of a requested result,
or authorize a finite-sample Jacobian as a whole-cell proof.
"""
import torch

from ...validation import InputValidationError
from ..validation import integer_at_least


def geometry_selected(local, D, *, cofactor=True, with_curl=True):
    """Return F, optional cofactor, J, optional raw curl for one existing tile."""
    if not isinstance(cofactor, bool) or not isinstance(with_curl, bool):
        raise InputValidationError("geometry selections must be boolean")
    V = torch.einsum("jpa,tar->tprj", D, local)
    F = V+torch.eye(3, dtype=V.dtype, device=V.device)
    row0 = torch.linalg.cross(F[..., 1, :], F[..., 2, :])
    A = None
    if cofactor:
        A = torch.stack((row0,
                         torch.linalg.cross(F[..., 2, :], F[..., 0, :]),
                         torch.linalg.cross(F[..., 0, :], F[..., 1, :])), dim=-2)
    # Keep the legacy row-zero cross product and reduction, not linalg.det.
    J = (F[..., 0, :]*(A[..., 0, :] if cofactor else row0)).sum(dim=-1)
    C = None
    if with_curl:
        C = torch.stack((V[..., 2, 1]-V[..., 1, 2], V[..., 0, 2]-V[..., 2, 0],
                         V[..., 1, 0]-V[..., 0, 1]), dim=-1)
    return F, A, J, C


def barrier_selected(J, delta, weight, activation=None, *, order=0):
    """Return only value (0), first (1), or second (2) barrier derivative.

    Positive-J and finite-output checks stay with the caller, exactly as for
    the legacy device helper. The selector is an integer, never a boolean.
    """
    order = integer_at_least(order, "barrier derivative order", 0)
    if order > 2:
        raise InputValidationError("barrier derivative order must be 0, 1 or 2")
    if activation is not None:
        scale = activation-delta
        t = (torch.minimum(J, torch.full_like(J, activation))-delta)/scale
        r = torch.clamp(activation-J, min=0)/scale
        logt = torch.where(r < .5, torch.log1p(-r), torch.log(t))
        if order == 0:
            return -weight*r*r*logt
        if order == 1:
            return weight*(2*r*logt-r*r/t)/scale
        return weight*(-2*logt+4*r/t+(r/t)**2)/scale**2
    gap, scale = J-delta, 1-delta
    if order == 2:
        return (weight/gap)/gap
    x = (J-1)/scale
    if order == 1:
        return weight*x/gap
    h = torch.full_like(x, -1/11)
    for k in range(8, -1, -1):
        h = h*x+(-1.)**k/(k+2)
    return weight*torch.where(x.abs() < .01, x*x*h, x-torch.log(gap/scale))
