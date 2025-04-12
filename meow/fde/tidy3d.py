""" FDE Tidy3d backend (default backend for MEOW) """

from types import SimpleNamespace
from typing import Literal

import numpy as np
import tidy3d
from packaging import version
from pydantic.v1 import validate_arguments
from pydantic.v1.types import PositiveFloat, PositiveInt
from scipy.constants import c
from tidy3d.components.mode.solver import compute_modes as _compute_modes

from ..cross_section import CrossSection
from ..mode import Mode, Modes, is_pml_mode, normalize_product, zero_phase


# @validate_arguments
def compute_modes_tidy3d(
    cs: CrossSection,
    num_modes: PositiveInt = 10,
    target_neff: PositiveFloat | None = None,
    precision: Literal["single", "double"] = "double",
    pml_mode_threshold: float = 1.0,
) -> Modes:
    """compute ``Modes`` for a given ``CrossSection``

    Args:
        cs: The ``CrossSection`` to calculate the modes for
        num_modes: Number of modes returned by mode solver.
        target_neff: Guess for initial effective index of the mode.
        pml_mode_threshold: If the mode has more than `pml_mode_threshold` part of its
            energy in the PML, it will be removed.
            default: 1.0 = 100% = no fitering.
    """

    if num_modes < 1:
        raise ValueError("You need to request at least 1 mode.")

    od = np.zeros_like(cs.nx)  # off diagonal entry
    new_tidy3d = version.parse(tidy3d.__version__) >= version.parse("2.2.0")
    if new_tidy3d:
        eps_cross = [cs.nx**2, od, od, od, cs.ny**2, od, od, od, cs.nz**2]
    else:
        eps_cross = [cs.nx**2, cs.ny**2, cs.nz**2]

    if np.isinf(cs.mesh.bend_radius) or np.isnan(cs.mesh.bend_radius):
        bend_radius = None
        bend_axis = None
    else:
        bend_radius = cs.mesh.bend_radius
        bend_axis = cs.mesh.bend_axis

    mode_spec = SimpleNamespace(  # tidy3d.ModeSpec alternative (prevents type checking)
        num_modes=num_modes,
        target_neff=target_neff,
        num_pml=cs.mesh.num_pml,
        filter_pol=None,
        angle_theta=cs.mesh.angle_theta,
        angle_phi=cs.mesh.angle_phi,
        bend_radius=bend_radius,
        precision=precision,
        bend_axis=bend_axis,
        track_freq="central",
        group_index_step=False,
    )

    ((Ex, Ey, Ez), (Hx, Hy, Hz)), neffs = (
        x.squeeze()
        for x in _compute_modes(
            eps_cross=eps_cross,
            coords=[cs.mesh.x, cs.mesh.y],
            freq=c / (cs.env.wl * 1e-6),
            mode_spec=mode_spec,
            precision=precision,
        )[:2]
    )

    if num_modes == 1:
        modes = [
            Mode(
                cs=cs,
                Ex=Ex,
                Ey=Ey,
                Ez=Ez,
                Hx=Hx,
                Hy=Hy,
                Hz=Hz,
                neff=float(neffs.real) + 1j * float(neffs.imag),
            )
            for _ in range(num_modes)
        ]
    else:  # num_modes > 1
        modes = [
            Mode(
                cs=cs,
                Ex=Ex[..., i],
                Ey=Ey[..., i],
                Ez=Ez[..., i],
                Hx=Hx[..., i],
                Hy=Hy[..., i],
                Hz=Hz[..., i],
                neff=neffs[i],
            )
            for i in range(num_modes)
        ]

    modes = [zero_phase(normalize_product(mode)) for mode in modes]
    modes = sorted(modes, key=lambda m: float(np.real(m.neff)), reverse=True)
    modes = [m for m in modes if not is_pml_mode(m, pml_mode_threshold)]

    return modes
