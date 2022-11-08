""" A CrossSection """
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from .base_model import BaseModel
from .cell import Cell, Cells
from .environment import Environment


class CrossSection(BaseModel):
    """A `CrossSection` is created from the association of a `Cell` with an `Environment`,
    which uniquely defines the refractive index everywhere.

    Attributes:
        cell: the cell for which the cross section was calculated
        env: the environment for which the cross sectionw was calculated
        mx: (derived) the index cross section at the Ex grid (integer y-coords, half-integer x-coords)
        my: (derived) the index cross section at the Ey grid (half-integer y-coords, integer x-coords)
        mz: (derived) the index cross section at the Ez grid (integer y-coords, integer x-coords)
    """

    cell: Cell
    env: Environment
    nx: np.ndarray[Tuple[int, int], np.dtype[np.float_]]
    ny: np.ndarray[Tuple[int, int], np.dtype[np.float_]]
    nz: np.ndarray[Tuple[int, int], np.dtype[np.float_]]

    def __init__(self, cell: Cell, env: Environment):
        nx, ny, nz = [np.ones(cell.mx.shape) for _ in range(3)]
        for i, material in enumerate(cell.materials, start=1):
            nx = np.where(cell.mx == i, material(env), nx)
            ny = np.where(cell.my == i, material(env), ny)
            nz = np.where(cell.mz == i, material(env), nz)
        super().__init__(
            cell=cell,
            env=env,
            nx=nx,
            ny=ny,
            nz=nz,
        )

    class Config:
        fields = {
            "nx": {"exclude": True},
            "ny": {"exclude": True},
            "nz": {"exclude": True},
        }

    @property
    def mesh(self):
        return self.cell.mesh

    @property
    def structures(self):
        return self.cell.structures

    def _visualize(self, c="z", axs=None):
        if axs is None:
            _, axs = plt.subplots(1, len(c), figsize=(3 * len(c), 3))
        c_list = list(c)
        if not all([(c in "xyz") for c in c_list]):
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


class CrossSections(list):  # List[CrossSection]
    """as list of `CrossSection` objects"""

    def __init__(self, cells: Cells, env: Environment):
        lst = [CrossSection(cell, env=env) for cell in cells]
        super().__init__(lst)

    @classmethod
    def from_list(cls, lst):
        new_lst = list.__new__(cls)
        new_lst.extend(lst)
        return new_lst
