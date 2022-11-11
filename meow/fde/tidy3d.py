""" FDE Tidy3d backend (default backend for MEOW) """

from types import SimpleNamespace
from typing import Literal, Optional, Tuple

import numpy as np
from pydantic import validate_arguments
from pydantic.types import NonNegativeInt, PositiveFloat, PositiveInt
from scipy.constants import c
from tidy3d.plugins.mode.solver import compute_modes as _compute_modes

from ..cross_section import CrossSection
from ..mode import Mode, Modes, normalize_energy, zero_phase


@validate_arguments
def compute_modes_tidy3d(
    cs: CrossSection,
    num_modes: PositiveInt = 10,
    angle_theta: float = 0.0,
    angle_phi: float = 0.0,
    bend_radius: Optional[float] = None,
    bend_axis: Optional[Literal[0, 1]] = None,
    target_neff: Optional[PositiveFloat] = None,
    sort_by: Literal["largest_neff", "te_fraction", "tm_fraction"] = "largest_neff",
    num_pml: Tuple[NonNegativeInt, NonNegativeInt] = (0, 0),
) -> Modes:
    """compute `Modes` for a given `CrossSection` (Tidy3D backend)

    Args:
        cs: the `CrossSection` to calculate the modes for
        num_modes: Number of modes returned by mode solver.
        target_neff: Guess for effective index of the mode.
        num_pml: Number of standard pml layers to add in the two tangential axes.
        sort_by: The solver will always compute the ``num_modes`` modes closest to the ``target_neff``,
            but they can be reordered by the largest ``te_fraction``, defined as the integral of the
            intensity of the E-field component parallel to the first plane axis normalized to the total
            in-plane E-field intensity. Similarly, ``tm_fraction`` uses the E field component parallel
            to the second plane axis.
        angle_theta: Polar angle of the propagation axis from the injection axis.
        angle_phi: Azimuth angle of the propagation axis in the plane orthogonal to the injection axis.
        bend_radius: A curvature radius for simulation of waveguide bends. Can be negative, in which case
            the mode plane center has a smaller value than the curvature center along the tangential axis
            perpendicular to the bend axis.
        bend_axis: Index into the two tangential axes defining the normal to the plane in which the bend
            lies. This must be provided if ``bend_radius`` is not ``None``. For example, for a ring in
            the global xy-plane, and a mode plane in either the xz or the yz plane, the ``bend_axis``
            is always 1 (the global z axis).
    """

    if num_modes < 1:
        raise ValueError("You need to request at least 1 mode.")

    ((Ex, Ey, Ez), (Hx, Hy, Hz)), neffs = (
        x.squeeze()
        for x in _compute_modes(
            eps_cross=[cs.nx**2, cs.ny**2, cs.nz**2],
            coords=[cs.mesh.x, cs.mesh.y],
            freq=c / (cs.env.wl * 1e-6),
            mode_spec=SimpleNamespace(
                num_modes=num_modes,
                angle_theta=angle_theta,
                angle_phi=angle_phi,
                bend_radius=bend_radius,
                bend_axis=bend_axis,
                target_neff=target_neff or cs.nx.max(),
                sort_by=sort_by,
                num_pml=num_pml,
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
