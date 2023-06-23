""" an EME Cell """

from typing import List, Tuple, Union, cast

import numpy as np
from pydantic import Field
from scipy.ndimage import convolve

from .base_model import BaseModel, _array, cached_property
from .mesh import Mesh2d
from .structures import (
    Structure,
    classify_structures_by_mesh_order_and_material,
    sort_structures,
)


class Cell(BaseModel):
    """A `Cell` defines a location in a `Structure` associated with a `Mesh`"""

    structures: List[Structure] = Field(
        description="the structures which will be sliced by the cell"
    )
    mesh: Mesh2d = Field(description="the mesh to slice the structures with")
    z_min: float = Field(description="the starting z-coordinate of the cell")
    z_max: float = Field(description="the ending z-coordinate of the cell")

    ez_interfaces: bool = Field(
        default=False,
        description=(
            "when enabled, the meshing algorithm will throw away any index values "
            "at the interfaces which are not on even (Ez) half-grid locations. "
            "Enabling this should result in more symmetric modes."
        ),
    )

    @property
    def z(self):
        return 0.5 * (self.z_min + self.z_max)

    @property
    def length(self):
        return np.abs(self.z_max - self.z_min)

    @cached_property
    def materials(self):
        materials = {}
        for i, structure in enumerate(sort_structures(self.structures), start=1):
            if not structure.material in materials:
                materials[structure.material] = i
        return materials

    @cached_property
    def m_full(self):
        m_full = np.zeros_like(self.mesh.X_full, dtype=np.int_)
        structures_dict = classify_structures_by_mesh_order_and_material(
            self.structures, self.materials
        )
        for structures in structures_dict.values():
            _m_full = _create_material_array(self, structures, self.ez_interfaces)
            mask = _m_full > 0
            m_full[mask] = _m_full[mask]

        if self.ez_interfaces:  # fill in 1-pixel gaps
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
            mask_x = (
                np.roll(mask, shift=1, axis=0)
                & (~mask)
                & np.roll(mask, shift=-1, axis=0)
            )
            mask_y = (
                np.roll(mask, shift=1, axis=1)
                & (~mask)
                & np.roll(mask, shift=-1, axis=1)
            )
            mask = mask_x | mask_y
            mask[:1, :] = False
            mask[-1:, :] = False
            mask[:, :1] = False
            mask[:, -1:] = False
            m_full[mask] = mat[mask]

        return m_full.view(_array)

    def _visualize(self, ax=None, cbar=True, show=True):
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
    structures: List[Structure],
    mesh: Union[Mesh2d, List[Mesh2d]],
    Ls: np.ndarray[Tuple[int], np.dtype[np.float_]],
    z_min: float = 0.0,
) -> List[Cell]:
    """easily create multiple `Cell` objects given a `Mesh` and a collection of cell lengths"""

    Ls = np.asarray(Ls, float)
    if Ls.ndim != 1:
        raise ValueError(f"Ls should be 1D. Got shape: {Ls.shape}.")
    if Ls.shape[0] < 0:
        raise ValueError(f"length of Ls array should be nonzero. Got: {Ls}.")

    meshes = [mesh] * Ls.shape[0] if isinstance(mesh, Mesh2d) else mesh
    if len(Ls) != len(meshes):
        raise ValueError(
            f"Number of meshes should correspond to number of lengths (length of Ls). Got {len(meshes)} != {len(Ls)}."
        )

    z = np.cumsum(np.concatenate([np.asarray([z_min], float), Ls]))
    cells = [
        Cell(structures=structures, mesh=mesh, z_min=z_min, z_max=z_max)
        for mesh, (z_min, z_max) in zip(meshes, zip(z[:-1], z[1:]))
    ]

    return cells


def _create_material_array(cell, structures, ez_interfaces):
    m_full = np.zeros_like(cell.mesh.X_full, dtype=np.int_)
    for structure in structures:
        mask = structure.geometry._mask2d(cell.mesh.X_full, cell.mesh.Y_full, cell.z)
        m_full[mask] = cell.materials[structure.material]

    if ez_interfaces:
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
    f = np.ndarray[Tuple[int, int], np.dtype[np.float_]]
    conv1 = cast(f, convolve(np.array(mask[:, :], dtype=int), np.array([[-1, 1]])))
    conv2 = cast(f, convolve(np.array(mask[:, ::-1], dtype=int), np.array([[-1, 1]])))
    conv3 = cast(f, convolve(np.array(mask[::-1, :], dtype=int), np.array([[-1, 1]])))
    mask1 = (conv1 > 0.0)[:, :]
    mask2 = (conv2 > 0.0)[:, ::-1]
    mask3 = (conv3 > 0.0)[::-1, :]
    mask = (mask1 | mask2 | mask3)[1:-1, 1:-1]
    # don't mask edge of simulation area:
    mask[:, 0] = mask[:, -1] = False
    return mask


def _get_boundary_mask_vertical(m_full, negate=False):
    return _get_boundary_mask_horizontal(m_full.T, negate=negate).T
