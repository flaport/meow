""" FDE Lumerical Backend """

import numpy as np
from pydantic.v1.types import PositiveInt

from ..cross_section import CrossSection
from ..environment import Environment
from ..mode import Mode, normalize_product, zero_phase
from ..structures import Structure3D

_global = {"sim": None}


def get_sim(**kwargs):
    sim = kwargs.get("sim", None)
    if sim is not None:
        return sim
    sim = _global["sim"]
    if sim is None:
        raise ValueError(
            "Could not start Lumerical simulation. "
            "Please either pass the `lumapi.MODE` simulation object as an argument to `compute_modes_lumerical` or "
            "use `set_sim(sim)` to globally set the lumapi.MODE simulation object."
        )
    return sim


def create_lumerical_geometries(
    sim, structures: list[Structure3D], env: Environment, unit: float
):
    sim = get_sim(sim=sim)
    sim.switchtolayout()
    sim.deleteall()
    for s in structures:
        s._lumadd(sim, env, unit, "yzx")


# @validate_arguments
def compute_modes_lumerical(
    cs: CrossSection,
    num_modes: PositiveInt = 10,
    sim=None,
    unit: float = 1e-6,
):
    """compute ``Modes` for a given ``FdeSpec` (Lumerical backend)

    Args:
        sim: the lumerical simulation object
        spec: The FDE simulation specification
        unit: Conversion factor between MEOW unit (probably um) and Lumerical unit (probably m).
    """
    from lumapi import LumApiError  # fmt: skip # type: ignore

    sim = get_sim(sim=sim)
    cell = cs._cell
    assert cell is not None

    _assert_default_mesh_setting(cell.mesh.angle_phi == 0, "angle_phi")
    _assert_default_mesh_setting(cell.mesh.angle_theta == 0, "angle_theta")
    _assert_default_mesh_setting(cell.mesh.bend_radius is None, "bend_radius")

    assert sim is not None
    create_lumerical_geometries(sim, cell.structures, cs.env, unit)
    sim.select("FDE")
    sim.delete()
    pml_settings = {}
    num_pml_y, num_pml_z = 0, 0
    if cell.mesh.num_pml[0] > 0:
        pml_settings.update(
            {
                "y_min_bc": "PML",
                "y_max_bc": "PML",
            }
        )
        num_pml_y = 22  # TODO: allow adjusting these values
    if cell.mesh.num_pml[1] > 0:
        pml_settings.update(
            {
                "z_min_bc": "PML",
                "z_max_bc": "PML",
            }
        )
        num_pml_z = 22  # TODO: allow adjusting these values
    sim.addfde(
        background_index=1.0,
        solver_type="2D X normal",
        x=float(cell.z * unit),
        y_min=float(cell.mesh.x.min() * unit),
        y_max=float(cell.mesh.x.max() * unit),
        z_min=float(cell.mesh.y.min() * unit),
        z_max=float(cell.mesh.y.max() * unit),
        define_y_mesh_by="number of mesh cells",
        define_z_mesh_by="number of mesh cells",
        mesh_cells_y=cell.mesh.x_.shape[0],
        mesh_cells_z=cell.mesh.y_.shape[0],
        **pml_settings,
    )
    # set mesh size again, because PML messes with it:
    if cell.mesh.num_pml[0] > 0:
        sim.setnamed("FDE", "mesh cells y", cell.mesh.x_.shape[0] - num_pml_y)
    if cell.mesh.num_pml[1] > 0:
        sim.setnamed("FDE", "mesh cells z", cell.mesh.y_.shape[0] - num_pml_z)
    sim.setanalysis("number of trial modes", int(num_modes))
    sim.setanalysis("search", "near n")
    sim.setanalysis("use max index", True)
    sim.setanalysis("wavelength", float(cs.env.wl * unit))
    sim.findmodes()
    modes = []
    for j in range(1, num_modes + 1):
        try:
            mode = _lumerical_fields_to_mode(
                cs=cs,
                lneff=sim.getdata(f"mode{j}", "neff"),
                lEx=sim.getdata(f"mode{j}", "Ex"),
                lEy=sim.getdata(f"mode{j}", "Ey"),
                lEz=sim.getdata(f"mode{j}", "Ez"),
                lHx=sim.getdata(f"mode{j}", "Hx"),
                lHy=sim.getdata(f"mode{j}", "Hy"),
                lHz=sim.getdata(f"mode{j}", "Hz"),
            )
        except LumApiError:
            break
        mode = normalize_product(mode)
        mode = zero_phase(mode)
        modes.append(mode)

    return sorted(modes, key=lambda m: np.real(m.neff), reverse=True)


def _lumerical_fields_to_mode(cs, lneff, lEx, lEy, lEz, lHx, lHy, lHz):
    return Mode(
        cs=cs,
        neff=lneff.ravel().item(),
        Ex=lEy.squeeze()[1:, :-1],
        Ey=lEz.squeeze()[:-1, 1:],
        Ez=lEx.squeeze()[:-1, :-1],
        Hx=lHy.squeeze()[:-1, 1:],
        Hy=lHz.squeeze()[1:, :-1],
        Hz=lHx.squeeze()[1:, 1:],
    )


def _assert_default_mesh_setting(condition, param_name):
    if not condition:
        raise NotImplementedError(
            f"Setting mesh.{param_name} is currently not supported in the Lumerical Backend. "
            "Please open an issue of submit a PR on GitHub to fix this: ",
            "https://github.com/flaport/meow",
        )


def set_sim(sim):
    _global["sim"] = sim
