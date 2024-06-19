""" an EME Cell """

from __future__ import annotations

import warnings
from typing import Annotated, cast, overload

import numpy as np
from pydantic import Field
from scipy.ndimage import convolve

from meow.array import Dim, DType, NDArray
from meow.base_model import BaseModel, cached_property
from meow.materials import Material
from meow.mesh import Mesh2D
from meow.structures import Structure2D, Structure3D, _sort_structures


class Cell(BaseModel):
    """A Cell defines an interval in z (the direction of propagation) within
    the simulation domain. The intersecting Structure3Ds are discretized by
    a given mesh at the center of the Cell"""

    structures: list[Structure3D] = Field(
        description="the 3D structures which will be sliced by the cell"
    )
    mesh: Mesh2D = Field(description="the mesh to discretize the structures with")
    z_min: float = Field(description="the starting z-coordinate of the cell")
    z_max: float = Field(description="the ending z-coordinate of the cell")

    @property
    def z(self):
        return 0.5 * (self.z_min + self.z_max)

    @property
    def length(self):
        return np.abs(self.z_max - self.z_min)

    @cached_property
    def materials(self):
        materials = {}
        for i, structure in enumerate(_sort_structures(self.structures), start=1):
            if not structure.material in materials:
                materials[structure.material] = i
        return materials

    @cached_property
    def structures_2d(self) -> list[Structure2D]:
        z = 0.5 * (self.z_min + self.z_max)
        list_of_list = [s._project(z) for s in self.structures]
        structures = [s for ss in list_of_list for s in ss]
        return structures

    @cached_property
    def m_full(self):
        return _create_full_material_array(
            mesh=self.mesh,
            structures=self.structures_2d,
            materials=self.materials,
        )

    def _visualize(self, ax=None, cbar=True, show=True, **ignored):
        import matplotlib.pyplot as plt  # fmt: skip
        from matplotlib.colors import ListedColormap, to_rgba  # fmt: skip
        from mpl_toolkits.axes_grid1 import make_axes_locatable  # fmt: skip

        colors = [(0, 0, 0, 0)] + [
            to_rgba(m.meta.get("color", (0, 0, 0, 0))) for m in self.materials
        ]
        cmap = ListedColormap(colors=colors)  # type: ignore
        if ax is not None:
            plt.sca(ax)
        else:
            ax = plt.gca()
        plt.pcolormesh(
            self.mesh.X_full,
            self.mesh.Y_full,
            self.m_full,
            cmap=cmap,
            vmin=0,
            vmax=len(self.materials) + 1,
        )
        plt.axis("scaled")
        plt.grid(True)
        if cbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            _cbar = plt.colorbar(
                ticks=np.concatenate(
                    [np.unique(self.m_full) + 0.5, [len(self.materials) + 1.5]]
                ),
                cax=cax,
            )
            labels = [""] + [m.name for m in self.materials] + [""]
            _cbar.ax.set_yticklabels(labels, rotation=90, va="center")
            plt.sca(ax)
        if show:
            plt.show()


def create_cells(
    structures: list[Structure3D],
    mesh: Mesh2D | list[Mesh2D],
    Ls: Annotated[NDArray, Dim(1), DType("float64")],
    z_min: float = 0.0,
) -> list[Cell]:
    """easily create multiple `Cell` objects given a `Mesh` and a collection of cell lengths"""
    warnings.warn(
        "create_cells will be removed in a future version. Please create your cells in a loop instead.",
        DeprecationWarning,
    )

    Ls = np.asarray(Ls, float)
    if Ls.ndim != 1:
        raise ValueError(f"Ls should be 1D. Got shape: {Ls.shape}.")
    if Ls.shape[0] < 0:
        raise ValueError(f"length of Ls array should be nonzero. Got: {Ls}.")

    meshes = [mesh] * Ls.shape[0] if isinstance(mesh, Mesh2D) else mesh
    if len(Ls) != len(meshes):
        raise ValueError(
            f"Number of meshes should correspond to number of lengths (length of Ls). Got {len(meshes)} != {len(Ls)}."
        )

    z = np.cumsum(np.concatenate([np.asarray([z_min], float), Ls]))
    cells = [
        Cell(
            structures=structures,
            mesh=mesh,
            z_min=z_min,
            z_max=z_max,
        )
        for mesh, (z_min, z_max) in zip(meshes, zip(z[:-1], z[1:]))
    ]

    return cells


def _create_full_material_array(
    mesh: Mesh2D,
    structures: list[Structure2D],
    materials: dict[Material, int],
):
    m_full = np.zeros_like(mesh.X_full, dtype=np.int_)
    structures_dict = _classify_structures_by_mesh_order_and_material(
        structures, materials
    )
    for structures in structures_dict.values():
        _m_full = _create_material_array(mesh, materials, structures)
        mask = _m_full > 0
        m_full[mask] = _m_full[mask]

    m_full = _fill_single_pixel_gaps(m_full)

    return m_full


def _create_material_array(
    mesh: Mesh2D,
    materials: dict[Material, int],
    structures: list[Structure2D],
) -> np.ndarray:
    m_full = np.zeros_like(mesh.X_full, dtype=np.int_)
    for structure in structures:
        mask = structure.geometry._mask(mesh.X_full, mesh.Y_full)
        m_full[mask] = materials[structure.material]

    if mesh.ez_interfaces:
        mask_ez_horizontal = np.zeros_like(m_full, dtype=bool)
        mask_ez_horizontal[:, ::2] = True
        mask_ez_vertical = np.zeros_like(m_full, dtype=bool)
        mask_ez_vertical[::2, :] = True
        mask_boundaries_vertical = _get_boundary_mask_vertical(m_full)
        mask_boundaries_vertical = mask_boundaries_vertical & (~mask_ez_vertical)
        mask_boundaries_horizontal = _get_boundary_mask_horizontal(m_full)
        mask_boundaries_horizontal = mask_boundaries_horizontal & (~mask_ez_horizontal)
        mask_temp = mask_boundaries_vertical | mask_boundaries_horizontal
        mask_corner_left = _fill_corner_left_mask(mask_temp)
        mask_corner_right = _fill_corner_right_mask(mask_temp)
        mask_to_remove = mask_temp | mask_corner_left | mask_corner_right
        mask_to_keep = (m_full > 0) & (~mask_to_remove)

        mask_ez = np.zeros_like(m_full, dtype=bool)
        mask_ez[::2, ::2] = True
        final_mask_to_keep = mask_to_keep & mask_ez

        mask_ex = np.zeros_like(m_full, dtype=bool)
        mask_ex[1::2, ::2] = True
        final_mask_to_keep |= mask_to_keep & mask_ex

        mask_ey = np.zeros_like(m_full, dtype=bool)
        mask_ey[::2, 1::2] = True
        final_mask_to_keep |= mask_to_keep & mask_ey

        mask_hz = np.zeros_like(m_full, dtype=bool)
        mask_hz[1::2, 1::2] = True
        final_mask_to_keep |= mask_to_keep & mask_hz

        m_full[~final_mask_to_keep] = 0
    return m_full


def _fill_corner_left_mask(mask):
    return (
        convolve(np.asarray(mask, dtype=float), np.array([[-1.0, +1.0], [+1.0, -1.0]]))
        > 1.0
    )  # type: ignore


def _fill_corner_right_mask(mask):
    return (
        convolve(
            np.asarray(mask, dtype=float),
            np.array([[0.0, 0.0], [+1.0, -1.0], [-1.0, +1.0]]),
        )
        > 1
    )  # type: ignore


def _get_boundary_mask_horizontal(m_full, negate=False):
    mask = np.zeros((m_full.shape[0] + 2, m_full.shape[1] + 2), dtype=bool)
    mask[1:-1, 1:-1] = m_full > 0
    if negate:
        mask = ~mask
    conv1 = cast(
        np.ndarray, convolve(np.array(mask[:, :], dtype=int), np.array([[-1, 1]]))
    )
    conv2 = cast(
        np.ndarray, convolve(np.array(mask[:, ::-1], dtype=int), np.array([[-1, 1]]))
    )
    conv3 = cast(
        np.ndarray, convolve(np.array(mask[::-1, :], dtype=int), np.array([[-1, 1]]))
    )
    mask1 = (conv1 > 0.0)[:, :]
    mask2 = (conv2 > 0.0)[:, ::-1]
    mask3 = (conv3 > 0.0)[::-1, :]
    mask = (mask1 | mask2 | mask3)[1:-1, 1:-1]
    # don't mask edge of simulation area:
    mask[:, 0] = mask[:, -1] = False
    return mask


def _get_boundary_mask_vertical(m_full, negate=False):
    return _get_boundary_mask_horizontal(m_full.T, negate=negate).T


def _fill_single_pixel_gaps(m_full):
    mat_x = np.maximum(
        np.roll(m_full, shift=1, axis=0),
        np.roll(m_full, shift=-1, axis=0),
    )
    mat_y = np.maximum(
        np.roll(m_full, shift=1, axis=1),
        np.roll(m_full, shift=-1, axis=1),
    )
    mat = mat_x | mat_y

    mask = m_full > 0
    mask_x = np.roll(mask, shift=1, axis=0) & (~mask) & np.roll(mask, shift=-1, axis=0)
    mask_y = np.roll(mask, shift=1, axis=1) & (~mask) & np.roll(mask, shift=-1, axis=1)
    mask = mask_x | mask_y
    mask[:1, :] = False
    mask[-1:, :] = False
    mask[:, :1] = False
    mask[:, -1:] = False
    m_full[mask] = mat[mask]
    return m_full


@overload
def _classify_structures_by_mesh_order_and_material(
    structures: list[Structure3D], materials: dict[Material, int]
) -> dict[tuple[int, int], list[Structure3D]]:
    ...


@overload
def _classify_structures_by_mesh_order_and_material(
    structures: list[Structure2D], materials: dict[Material, int]
) -> dict[tuple[int, int], list[Structure2D]]:
    ...


def _classify_structures_by_mesh_order_and_material(
    structures: list[Structure3D] | list[Structure2D],
    materials: dict[Material, int],
) -> (
    dict[tuple[int, int], list[Structure2D]] | dict[tuple[int, int], list[Structure3D]]
):
    structures = _sort_structures(structures)
    structures_dict = {}
    for structure in structures:
        mo = structure.mesh_order
        mat = materials[structure.material]
        if (mo, mat) not in structures_dict:
            structures_dict[mo, mat] = []
        structures_dict[mo, mat].append(structure)
    return structures_dict
