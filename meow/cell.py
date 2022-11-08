""" an EME Cell """

from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, to_rgba

from .base_model import BaseModel
from .materials import Materials
from .mesh import Mesh2d, Meshes2d
from .structures import Structures, _sort_structures


class Cell(BaseModel):
    """A `Cell` defines a location in a `Structure` associated with a `Mesh`

    Attributes:
        structures: the structures which will be sliced by the cell
        mesh: the mesh to slice the structures with
        z_min: the starting z-coordinate of the cell
        z_max: the ending z-coordinate of the cell
        materials: (derived) the materials in the cell
        mx: (derived) the material cross section at the Ex grid (integer y-coords, half-integer x-coords)
        my: (derived) the material cross section at the Ey grid (half-integer y-coords, integer x-coords)
        mz: (derived) the material cross section at the Ez grid (integer y-coords, integer x-coords)
    """

    structures: Structures
    mesh: Mesh2d
    z_min: float
    z_max: float
    materials: Materials
    mx: np.ndarray[Tuple[int, int], np.dtype[np.int_]]
    my: np.ndarray[Tuple[int, int], np.dtype[np.int_]]
    mz: np.ndarray[Tuple[int, int], np.dtype[np.int_]]

    def __init__(
        self, structures: Structures, mesh: Mesh2d, z_min: float, z_max: float
    ):
        """A `Cell` defines a location in a `Structure` associated with a `Mesh`

        Args:
            structures: the structures which will be sliced by the cell
            mesh: the mesh to slice the structures with
            z_min: the starting z-coordinate of the cell
            z_max: the ending z-coordinate of the cell
        """
        structures = _sort_structures(structures)
        mx, my, mz = [np.zeros(mesh.Xx.shape, dtype=int) for _ in range(3)]
        z = 0.5 * (z_min + z_max)

        materials = []
        # TODO: ideally we should downselect the relevant structures here. Structures not at z-location should ideally be ignored.

        for structure in structures:
            mask_x, mask_y, mask_z = structure.geometry._mask2d(mesh, z)
            if (not mask_x.any()) or (not mask_y.any()) or (not mask_z.any()):
                continue

            try:
                material_index = materials.index(structure.material) + 1
            except ValueError:
                materials.append(structure.material)
                material_index = len(materials)

            mx[mask_x] = material_index
            my[mask_y] = material_index
            mz[mask_z] = material_index

        super().__init__(
            structures=structures,
            mesh=mesh,
            z_min=z_min,
            z_max=z_max,
            materials=materials,
            mx=mx,
            my=my,
            mz=mz,
        )

    class Config:
        fields = {
            "materials": {"exclude": True},
            "mx": {"exclude": True},
            "my": {"exclude": True},
            "mz": {"exclude": True},
        }

    @property
    def z(self):
        return 0.5 * (self.z_min + self.z_max)

    @property
    def length(self):
        return np.abs(self.z_max - self.z_min)

    def _visualize(self, c="z", axs=None):
        colors = [(0, 0, 0, 0)] + [
            to_rgba(m.meta.get("color", (0, 0, 0, 0))) for m in self.materials
        ]
        cmap = ListedColormap(colors=colors)  # type: ignore
        if axs is None:
            _, axs = plt.subplots(1, len(c), figsize=(3 * len(c), 3))
        c_list = list(c)
        if not all([(c in "xyz") for c in c_list]):
            raise ValueError(f"Invalid component. Got: {c}. Should be 'x', 'y' or 'z'.")
        axs = np.array(axs, dtype=object).ravel()
        for ax, c in zip(axs, c_list):
            X = getattr(self.mesh, f"X{c}")
            Y = getattr(self.mesh, f"Y{c}")
            m = getattr(self, f"m{c}")
            plt.sca(ax)
            if len(c_list) > 1:
                plt.title(f"m{c}")
            plt.pcolormesh(X, Y, m, cmap=cmap, vmin=0, vmax=len(self.materials))
            plt.axis("scaled")
            plt.grid(True)


class Cells(list):  # List[Cell]
    """a list of `Cell` objects"""

    def __init__(
        self,
        structures: Structures,
        mesh: Union[Mesh2d, Meshes2d],
        Ls: np.ndarray[Tuple[int], np.dtype[np.float_]],
        z_min: float = 0.0,
    ):
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

        cells = []
        z = np.cumsum(np.concatenate([np.asarray([z_min], float), Ls]))
        for mesh, (z_min, z_max) in zip(meshes, zip(z[:-1], z[1:])):
            cells.append(Cell(structures, mesh, z_min, z_max))

        super().__init__(cells)

    @classmethod
    def from_list(cls, lst):
        new_lst = list.__new__(cls)
        new_lst.extend(lst)
        return new_lst
