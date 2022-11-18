""" A CrossSection """
from typing import List, Tuple, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
from pydantic import Field

from .base_model import BaseModel
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
    extra: Dict[str, Any] = Field(
        default_factory=lambda: {}, description="extra metadata"
    )

    def __init__(self, *, cell: Cell, env: Environment, **_):
        cell = Cell.parse_obj(cell)
        env = Environment.parse_obj(env)
        nx, ny, nz = [np.ones(cell.mx.shape) for _ in range(3)]
        for i, material in enumerate(cell.materials, start=1):
            nx = np.where(cell.mx == i, material(env), nx)
            ny = np.where(cell.my == i, material(env), ny)
            nz = np.where(cell.mz == i, material(env), nz)
        super().__init__(
            cell=cell,
            env=env,
        )
        self.extra = {}
        self.extra["nx"] = nx
        self.extra["ny"] = ny
        self.extra["nz"] = nz

    @property
    def nx(self):
        """(derived) the index cross section at the Ex grid (integer y-coords, half-integer x-coords)"""
        return self.extra["nx"]

    @property
    def ny(self):
        """(derived) the index cross section at the Ey grid (integer y-coords, half-integer x-coords)"""
        return self.extra["ny"]

    @property
    def nz(self):
        """(derived) the index cross section at the Ez grid (integer y-coords, half-integer x-coords)"""
        return self.extra["nz"]

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

    class Config:
        fields = {"extra": {"exclude": True}}
