"""Test propagation of the modes."""

from itertools import pairwise

import gdsfactory as gf
import numpy as np
import sax

import meow as mw
import meow.eme.propagate as prop


def test_propagation():
    l_taper = 20.0
    l_center = 20.0
    w_center = 3.0
    num_modes = 10
    env = mw.Environment(wl=1.55, T=25.0)
    structs = _mmi_structs(l_taper, l_center, w_center)
    cells = _mmi_cells(structs)
    modes = _mmi_modes(cells, env, num_modes)
    z = np.linspace(0, l_taper * 2 + l_center, 800)
    y = 0.2

    ex_l = np.zeros(len(modes[0]))
    ex_l[0] = 1.0
    ex_r = np.zeros(len(modes[-1]))
    Ex, x = prop.propagate_modes(modes, cells, ex_l, ex_r, y, z)
    return Ex


def _example_extrusions(
    t_soi: float = 0.4,
) -> dict:
    extrusions = {
        (1, 0): [
            mw.GdsExtrusionRule(
                material=mw.silicon,
                h_min=0.0,
                h_max=0.0 + t_soi,
                mesh_order=1,
            ),
        ],
    }
    return extrusions


def _mmi_structs(
    l_taper: float = 20.0,
    l_center: float = 20.0,
    w_center: float = 3.0,
) -> list[mw.Structure3D]:
    mmi = gf.components.mmi2x2(
        length_taper=l_taper, length_mmi=l_center, width_mmi=w_center
    )

    c = gf.Component()
    ref = c.add_ref(mmi)
    ref.xmin = 0
    mmi = c

    extrusion_rules = _example_extrusions()
    return mw.extrude_gds(mmi, extrusion_rules)


def _mmi_cells(structs: list[mw.Structure3D], eps: float = 1e-2) -> list[mw.Cell]:
    left_cell_edges = np.linspace(0, 20, 11) + eps
    right_cell_edges = np.linspace(40, 60, 11) - eps
    cell_edges = np.concatenate(
        [left_cell_edges[:1], left_cell_edges, right_cell_edges, right_cell_edges[-1:]]
    )

    mesh = mw.Mesh2D(
        x=np.linspace(-2, 2, 101),
        y=np.linspace(-1, 1, 101),
    )

    cells = []
    for z_min, z_max in pairwise(cell_edges):
        cell = mw.Cell(
            structures=structs,
            mesh=mesh,
            z_min=z_min,
            z_max=z_max,
        )
        cells.append(cell)
    return cells


def _mmi_css(cells: list[mw.Cell], env: mw.Environment) -> list[mw.CrossSection]:
    return [mw.CrossSection.from_cell(cell=cell, env=env) for cell in cells]


def _mmi_modes(
    cells: list[mw.Cell], env: mw.Environment, num_modes: int = 16
) -> list[list[mw.Mode]]:
    css = _mmi_css(cells, env)
    modes = [mw.compute_modes(cs, num_modes=num_modes) for cs in css]
    modes = [[mode for mode in modes_ if mode.neff > 1.45] for modes_ in modes]
    return modes


def _mmi_s_matrix(modes: list[list[mw.Mode]], cells: list[mw.Cell]) -> sax.SDense:
    return mw.compute_s_matrix(modes, cells)


if __name__ == "__main__":
    import warnings

    import matplotlib.pyplot as plt

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Ex = test_propagation()

    plt.imshow(abs(Ex.T) ** 2)
    plt.show()
