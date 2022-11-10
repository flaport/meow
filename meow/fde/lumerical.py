""" FDE Lumerical Backend """

import numpy as np
from pydantic.types import PositiveFloat, PositiveInt

from ..cross_section import CrossSection
from ..mode import Mode, normalize_energy, zero_phase


# @validate_arguments
def compute_modes_lumerical(
    cs: CrossSection,
    sim: "lumapi.MODE",  # type: ignore
    num_modes: PositiveInt = 10,
    unit: PositiveFloat = 1e-6,
):
    """compute `Modes` for a given `CrossSection` (Lumerical backend)

    cs: the `CrossSection` to calculate the modes for
    sim: the lumerical simulation object
    num_modes: Number of modes returned by mode solver.
    unit: conversion factor from meow to lumerical distances (normaly um -> m, therefore 1e-6)
    """
    sim.switchtolayout()
    sim.deleteall()
    for s in cs.structures:
        s._lumadd(sim, cs.env, unit, "yzx")

    sim.select("FDE")
    sim.delete()
    sim.addfde(
        background_index=1.0,
        solver_type="2D X normal",
        x=float(cs.cell.z * unit),
        y_min=float(cs.mesh.x.min() * unit),
        y_max=float(cs.mesh.x.max() * unit),
        z_min=float(cs.mesh.y.min() * unit),
        z_max=float(cs.mesh.y.max() * unit),
        define_z_mesh_by="number of mesh cells",
        define_y_mesh_by="number of mesh cells",
        mesh_cells_y=cs.mesh.x_.shape[0],
        mesh_cells_z=cs.mesh.y_.shape[0],
    )
    sim.setanalysis("number of trial modes", int(num_modes))
    sim.setanalysis("search", "near n")
    sim.setanalysis("use max index", True)
    sim.setanalysis("wavelength", float(cs.env.wl * unit))
    sim.findmodes()
    modes = []
    for j in range(1, num_modes + 1):
        mode = Mode(
            neff=sim.getdata(f"mode{j}", "neff").ravel().item(),
            cs=cs,
            Ez=sim.getdata(f"mode{j}", "Ex").squeeze()[:-1, :-1],
            Ex=sim.getdata(f"mode{j}", "Ey").squeeze()[:-1, :-1],
            Ey=sim.getdata(f"mode{j}", "Ez").squeeze()[:-1, :-1],
            Hz=sim.getdata(f"mode{j}", "Hx").squeeze()[:-1, :-1],
            Hx=sim.getdata(f"mode{j}", "Hy").squeeze()[:-1, :-1],
            Hy=sim.getdata(f"mode{j}", "Hz").squeeze()[:-1, :-1],
        )
        mode = normalize_energy(mode)
        mode = zero_phase(mode)
        modes.append(mode)

    return sorted(modes, key=lambda m: np.real(m.neff), reverse=True)
