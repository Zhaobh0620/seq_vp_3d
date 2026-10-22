"""Dimension-specific checks; reuse existing dimension-independent validators."""
from numbers import Integral

import numpy as np

from ..validation import InputValidationError


def integer_at_least(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise InputValidationError(f"{name}: expected integer >= {minimum}")
    return int(value)


def zero_boundary(u):
    """Input must already be a validated (nz,ny,nx,3) array."""
    return not (np.any(u[[0, -1], :, :, :] != 0)
                or np.any(u[:, [0, -1], :, :] != 0)
                or np.any(u[:, :, [0, -1], :] != 0))
