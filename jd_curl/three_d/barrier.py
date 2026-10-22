"""T04/T28 direct point barrier potential/derivatives, not a GN residual."""
import numpy as np

from ..validation import InputValidationError, as_float64_array, finite_real, require_finite_result


def barrier_terms(J, *, delta, weight, activation=None):
    """Return b_B, b_B', b_B''; weight includes kappa times point weight.

    Near J=1 use the same scalar series principle as 2D, but no signed sqrt
    residual, omega/4 normalization or residual-GN curvature is imported.
    """
    J = as_float64_array(J, "barrier.J")
    delta = finite_real(delta, "delta")
    weight = finite_real(weight, "barrier weight", positive=True)
    if delta < 0 or delta >= 1 or np.any(J <= delta):
        raise InputValidationError("barrier: require 0 <= delta < 1 and J > delta")
    if activation is not None:
        activation = finite_real(activation, "barrier activation")
        if not delta < activation <= 1:
            raise InputValidationError("barrier: require delta < activation <= 1")
        value, first, second = (np.zeros_like(J) for _ in range(3))
        active = J < activation
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            scale = activation-delta
            t = (J[active]-delta)/scale
            r = (activation-J[active])/scale
            logt = np.log(t)
            near = r < .5
            logt[near] = np.log1p(-r[near])
            value[active] = -weight*r*r*logt
            first[active] = weight*(2*r*logt-r*r/t)/scale
            second[active] = weight*(-2*logt+4*r/t+(r/t)**2)/scale**2
        for result in (value, first, second):
            require_finite_result(result, "localized barrier")
        return value, first, second
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        scale, gap = 1-delta, J-delta
        x, t = (J-1)/scale, gap/scale
        near = np.abs(x) < .01
        psi = np.empty_like(J)
        xn = x[near]
        h = np.full_like(xn, -1/11)
        for k in range(8, -1, -1):
            h = h*xn + (-1.)**k/(k+2)
        psi[near] = xn*xn*h
        psi[~near] = x[~near]-np.log(t[~near])
        value = weight*psi
        # Equivalent to weight*(1/scale - 1/gap), avoids cancellation at J=1.
        first = weight*((J-1)/scale)/gap
        second = (weight/gap)/gap
    for name, result in (("value", value), ("first", first), ("second", second)):
        require_finite_result(result, "barrier "+name)
    if np.any(value < 0) or np.any(second <= 0):
        raise FloatingPointError("barrier positivity/curvature lost in floating arithmetic")
    return value, first, second
