""" an EME Cell """

from typing import Any, Dict, List, Tuple, Union

import numpy as np
from pydantic import Field

from .base_model import BaseModel
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
    extra: Dict[str, Any] = Field(
        default_factory=lambda: {}, description="extra metadata"
    )

    @property
    def mx(self):
        """(derived) the material cross section at the Ex grid (integer y-coords, half-integer x-coords)"""
        return self.extra["mx"]

    @property
    def my(self):
        """(derived) the material cross section at the Ey grid (half-integer y-coords, integer x-coords)"""
        return self.extra["my"]

    @property
    def mz(self):
        """(derived) the material cross section at the Ez grid (integer y-coords, integer x-coords)"""
        return self.extra["mz"]

    @property
    def materials(self):
        """(derived) the materials in the cell"""
        return self.extra["materials"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        structures = sort_structures(self.structures)
        mx, my, mz = [np.zeros(self.mesh.Xx.shape, dtype=int) for _ in range(3)]
        z = 0.5 * (self.z_min + self.z_max)
        # TODO: ideally we should downselect the relevant structures here...
        # structures not at z-location should ideally be ignored.
        materials = []
        for structure in structures:
            mask_x, mask_y, mask_z = structure.geometry._mask2d(self.mesh, z)
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
        self.extra["mx"] = mx
        self.extra["my"] = my
        self.extra["mz"] = mz
        self.extra["materials"] = materials

    @property
    def z(self):
        return 0.5 * (self.z_min + self.z_max)

    @property
    def length(self):
        return np.abs(self.z_max - self.z_min)

    def _visualize(self, c="z", axs=None):
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap, to_rgba

        colors = [(0, 0, 0, 0)] + [
            to_rgba(m.meta.get("color", (0, 0, 0, 0))) for m in self.materials
        ]
        cmap = ListedColormap(colors=colors)  # type: ignore
        if axs is None:
            _, axs = plt.subplots(1, len(c), figsize=(3 * len(c), 3))
        c_list = list(c)
        if any(c not in "xyz" for c in c_list):
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

    class Config:
        fields = {"extra": {"exclude": True}}


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
