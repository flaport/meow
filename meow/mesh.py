""" a 2D Mesh """

from typing import Tuple

import numpy as np
from pydantic import Field

from .base_model import BaseModel


class Mesh(BaseModel):
    """[BaseClass] a `Mesh` describes how a `Structure` is discretized"""


class Mesh2d(Mesh):
    """a 2D Mesh or `Mesh2D` describes how a `Structure` is discritized into a `Cell` or `CrossSection`"""

    x: np.ndarray[Tuple[int], np.dtype[np.float_]] = Field(
        description="x-coordinates of the mesh (Ez locations, i.e. corners of the 2D cell)"
    )
    y: np.ndarray[Tuple[int], np.dtype[np.float_]] = Field(
        description="y-coordinates of the mesh (Ez locations, i.e. corners of the 2D cell)"
    )

    @property
    def dx(self):
        """x-coordinate mesh step (Hz locations, i.e. center of the 2D cell)"""
        return self.x[1:] - self.x[:-1]

    @property
    def dy(self):
        """y-coordinate mesh step (Hz locations, i.e. center of the 2D cell)"""
        return self.y[1:] - self.y[:-1]

    @property
    def x_(self):
        """x-coordinates of the mesh (Hz locations, i.e. center of the 2D cell)"""
        return 0.5 * np.asarray(self.x[1:] + self.x[:-1], dtype=np.float_)

    @property
    def y_(self):
        """y-coordinates of the mesh (Hz locations, i.e. center of the 2D cell)"""
        return 0.5 * np.asarray(self.y[1:] + self.y[:-1], dtype=np.float_)

    @property
    def X(self):
        return np.meshgrid(self.y_, self.x_)[1]

    @property
    def Y(self):
        return np.meshgrid(self.y_, self.x_)[0]

    @property
    def Xx(self):
        return np.meshgrid(self.y[:-1], self.x_)[1]

    @property
    def Yx(self):
        return np.meshgrid(self.y[:-1], self.x_)[0]

    @property
    def Xy(self):
        return np.meshgrid(self.y_, self.x[:-1])[1]

    @property
    def Yy(self):
        return np.meshgrid(self.y_, self.x[:-1])[0]

    @property
    def Xz(self):
        return np.meshgrid(self.y[:-1], self.x[:-1])[1]

    @property
    def Yz(self):
        return np.meshgrid(self.y[:-1], self.x[:-1])[0]

    def __eq__(self, other):
        try:
            x_eq = ((self.x - other.x) < 1e-6).all()
            y_eq = ((self.y - other.y) < 1e-6).all()
        except Exception:
            return False
        return x_eq and y_eq
