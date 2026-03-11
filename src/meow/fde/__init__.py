"""FDE Implementations & Backends."""

from __future__ import annotations

from meow.fde.lumerical import compute_modes_lumerical as compute_modes_lumerical
from meow.fde.tidy3d import compute_modes_tidy3d as compute_modes_tidy3d

compute_modes = compute_modes_tidy3d
