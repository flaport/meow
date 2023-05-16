""" FDE Lumerical Backend """

from typing import Optional

import numpy as np
from pydantic import validate_arguments
from pydantic.types import PositiveInt

from ..cross_section import CrossSection
from ..mode import Mode, normalize_energy, zero_phase

_global = {"sim": None}


def _assert_default_mesh_setting(condition, param_name):
    if condition:
        raise NotImplementedError(
            f"Setting mesh.{param_name} is currently not supported in the Lumerical Backend. "
            "Please open an issue of submit a PR on GitHub to fix this: ",
            "https://github.com/flaport/meow",
        )


def set_sim(
    sim: "lumapi.MODE",  # type: ignore
):
    _global["sim"] = sim


def get_sim():
    sim = _global["sim"]
    if sim is None:
        raise ValueError(
            "Could not start Lumerical simulation. "
            "Please either pass the `lumapi.MODE` simulation object as an argument to `compute_modes_lumerical` or "
            "use `set_sim(sim)` to globally set the lumapi.MODE simulation object."
        )
    return sim


# @validate_arguments
def compute_modes_lumerical(
    cs: CrossSection,
    num_modes: PositiveInt = 10,
    sim: Optional["lumapi.MODE"] = None,  # type: ignore
    unit: float = 1e-6,
):
    """compute ``Modes` for a given ``FdeSpec` (Lumerical backend)

    Args:
        sim: the lumerical simulation object
        spec: The FDE simulation specification
        unit: Conversion factor between MEOW unit (probably um) and Lumerical unit (probably m).
    """
    _assert_default_mesh_setting(cs.cell.mesh.angle_phi != 0, "angle_phi")
    _assert_default_mesh_setting(cs.cell.mesh.angle_theta != 0, "angle_theta")
    _assert_default_mesh_setting(cs.cell.mesh.bend_radius < 1e10, "bend_radius")

    if sim is None:
        sim = get_sim()

    assert sim is not None

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
