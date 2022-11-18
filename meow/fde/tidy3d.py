""" FDE Tidy3d backend (default backend for MEOW) """

from types import SimpleNamespace
from typing import Optional

import numpy as np
from pydantic import validate_arguments
from pydantic.types import PositiveFloat, PositiveInt
from scipy.constants import c
from tidy3d.plugins.mode.solver import compute_modes as _compute_modes

from ..cross_section import CrossSection
from ..mode import Mode, Modes, normalize_energy, zero_phase


@validate_arguments
def compute_modes_tidy3d(
    cs: CrossSection,
    num_modes: PositiveInt = 10,
    target_neff: Optional[PositiveFloat] = None,
) -> Modes:
    """compute ``Modes`` for a given ``FdeSpec`` (Tidy3D backend)

    Args:
        cs: The ``CrossSection`` to calculate the modes for
        num_modes: Number of modes returned by mode solver.
        target_neff: Guess for initial effective index of the mode.
    """

    if num_modes < 1:
        raise ValueError("You need to request at least 1 mode.")

    bend_radius = None if cs.cell.mesh.bend_radius > 1e10 else cs.cell.mesh.bend_radius
    bend_axis = None if bend_radius is None else cs.cell.mesh.bend_axis

    ((Ex, Ey, Ez), (Hx, Hy, Hz)), neffs = (
        x.squeeze()
        for x in _compute_modes(
            eps_cross=[cs.nx**2, cs.ny**2, cs.nz**2],
            coords=[cs.mesh.x, cs.mesh.y],
            freq=c / (cs.env.wl * 1e-6),
            mode_spec=SimpleNamespace(
                num_modes=num_modes,
                angle_theta=cs.cell.mesh.angle_theta,
                angle_phi=cs.cell.mesh.angle_phi,
                bend_radius=bend_radius,
                bend_axis=bend_axis,
                target_neff=target_neff or cs.nx.max(),
                num_pml=cs.cell.mesh.num_pml,
                sort_by="largest_neff",
                filter_pol=None,
                precision="double",
            ),
        )
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

    modes = [zero_phase(normalize_energy(mode)) for mode in modes]
    modes = sorted(modes, key=lambda m: float(np.real(m.neff)), reverse=True)

    return modes
