""" an EME Cell """

from typing import List, Tuple, Union

import numpy as np
from pydantic import Field

from .base_model import BaseModel, cached_property
from .mesh import Mesh2d
from .structures import Structure, sort_structures


class Cell(BaseModel):
    """A `Cell` defines a location in a `Structure` associated with a `Mesh`"""

    structures: List[Structure] = Field(
        description="the structures which will be sliced by the cell"
    )
    mesh: Mesh2d = Field(description="the mesh to slice the structures with")
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
        for i, structure in enumerate(sort_structures(self.structures), start=1):
            if not structure.material in materials:
                materials[structure.material] = i
        return materials

    @cached_property
    def m_full(self):
        m_full = np.zeros_like(self.mesh.X_full, dtype=np.int_)
        for structure in sort_structures(self.structures):
            mask = structure.geometry._mask2d(
                self.mesh.X_full, self.mesh.Y_full, self.z
            )
            m_full[mask] = self.materials[structure.material]
        return m_full

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
