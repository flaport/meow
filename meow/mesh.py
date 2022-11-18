""" a 2D Mesh """

from typing import Literal, Tuple

import numpy as np
from pydantic import Field
from pydantic.types import NonNegativeInt

from .base_model import BaseModel


class Mesh(BaseModel):
    """[BaseClass] a ``Mesh`` describes how a ``Structure`` is discretized"""


class Mesh2d(Mesh):
    """a 2D Mesh or ``Mesh2D`` describes how a ``Structure`` is discritized into a ``Cell`` or ``CrossSection``"""

    x: np.ndarray[Tuple[int], np.dtype[np.float_]] = Field(
        description="x-coordinates of the mesh (Ez locations, i.e. corners of the 2D cell)"
    )
    y: np.ndarray[Tuple[int], np.dtype[np.float_]] = Field(
        description="y-coordinates of the mesh (Ez locations, i.e. corners of the 2D cell)"
    )
    angle_phi: float = Field(
        default=0.0,
        description="Azimuth angle of the propagation axis in the plane orthogonal to the mesh.",
    )
    angle_theta: float = Field(
        default=0.0,
        description="Polar angle of the propagation axis from the injection axis.",
    )
    bend_radius: float = Field(
        default=np.inf,
        description=(
            "A curvature radius for simulation of waveguide bends. "
            "Tidy3D: Can be negative, in which case the mode plane center has a smaller value than "
            "the curvature center along the tangential axis perpendicular to the bend axis."
        ),
    )
    bend_axis: Literal[0, 1] = Field(
        default=1,
        description=(
            "Index into the two tangential axes defining the normal to the plane in which the bend lies. "
            "This must be provided if ``bend_radius`` is not ``None``. For example, for a ring in the "
            "global xy-plane, and a mode plane in either the xz or the yz plane, the ``bend_axis`` is "
            "always 1 (the global z axis)."
        ),
    )
    num_pml: Tuple[NonNegativeInt, NonNegativeInt] = Field(
        default=(0, 0),
        description="Number of standard pml layers to add in the two tangential axes.",
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
        return 0.5 * (self.x[1:] + self.x[:-1])

    @property
    def y_(self):
        """y-coordinates of the mesh (Hz locations, i.e. center of the 2D cell)"""
        return 0.5 * (self.y[1:] + self.y[:-1])

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
