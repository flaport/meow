""" FDE Implementations & Backends """

from functools import wraps

from .lumerical import compute_modes_lumerical as compute_modes_lumerical
from .tidy3d import compute_modes_tidy3d as compute_modes_tidy3d


@wraps(compute_modes_tidy3d)
def compute_modes(*args, **kwargs):
    return compute_modes_tidy3d(*args, **kwargs)
