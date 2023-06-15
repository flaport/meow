""" A CrossSection """

import numpy as np
from pydantic import Field

from .base_model import BaseModel, cached_property
from .cell import Cell
from .environment import Environment


class CrossSection(BaseModel):
    """A `CrossSection` is created from the association of a `Cell` with an `Environment`,
    which uniquely defines the refractive index everywhere."""

    cell: Cell = Field(
        description="the cell for which the cross section was calculated"
    )
    env: Environment = Field(
        description="the environment for which the cross sectionw was calculated"
    )

    @cached_property
    def _computed(self):
        nx, ny, nz = [np.ones(self.cell.mx.shape) for _ in range(3)]
        for i, material in enumerate(self.cell.materials, start=1):
            nx = np.where(self.cell.mx == i, material(self.env), nx)
            ny = np.where(self.cell.my == i, material(self.env), ny)
            nz = np.where(self.cell.mz == i, material(self.env), nz)
        return {
            "nx": nx,
            "ny": ny,
            "nz": nz,
        }

    @property
    def nx(self):
        """(derived) the index cross section at the Ex grid (integer y-coords, half-integer x-coords)"""
        return self._computed["nx"]

    @property
    def ny(self):
        """(derived) the index cross section at the Ey grid (integer y-coords, half-integer x-coords)"""
        return self._computed["ny"]

    @property
    def nz(self):
        """(derived) the index cross section at the Ez grid (integer y-coords, half-integer x-coords)"""
        return self._computed["nz"]

    @property
    def mesh(self):
        return self.cell.mesh

    @property
    def structures(self):
        return self.cell.structures

    def _visualize(self, c="z", axs=None):
        import matplotlib.pyplot as plt  # fmt: skip

        if axs is None:
            _, axs = plt.subplots(1, len(c), figsize=(3 * len(c), 3))
        c_list = list(c)
        if any(c not in "xyz" for c in c_list):
            raise ValueError(f"Invalid component. Got: {c}. Should be 'x', 'y' or 'z'.")
        axs = np.array(axs, dtype=object).ravel()
        for ax, c in zip(axs, c_list):
            X = getattr(self.mesh, f"X{c}")
            Y = getattr(self.mesh, f"Y{c}")
            n = getattr(self, f"n{c}")
            plt.sca(ax)
            if len(c_list) > 1:
                plt.title(f"n{c}")
            plt.pcolormesh(X, Y, n)
            plt.axis("scaled")
            plt.grid(True)
