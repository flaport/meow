""" a 2D Mesh """

from typing import Literal, Optional, Tuple

import numpy as np
from pydantic import Field
from pydantic.types import NonNegativeInt

from .base_model import BaseModel, _array, cached_property


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
    bend_radius: Optional[float] = Field(
        default=None,
        description=(
            "A curvature radius for simulation of waveguide bends. "
            "Tidy3D: Can be negative, in which case the mode plane center has a smaller value than "
            "the curvature center along the tangential axis perpendicular to the bend axis."
        ),
    )
    bend_axis: Optional[Literal[0, 1]] = Field(
        default=None,
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

    @cached_property
    def dx(self):
        """dx at Hz locations, i.e. center of the 2D cell"""
        return (self.x[1:] - self.x[:-1]).view(_array)

    @cached_property
    def dy(self):
        """dy at Hz locations, i.e. center of the 2D cell"""
        return (self.y[1:] - self.y[:-1]).view(_array)

    @cached_property
    def x_(self):
        """x at Hz locations, i.e. center of the 2D cell"""
        return 0.5 * (self.x[1:] + self.x[:-1]).view(_array)

    @cached_property
    def y_(self):
        """y at Hz locations, i.e. center of the 2D cell"""
        return 0.5 * (self.y[1:] + self.y[:-1]).view(_array)

    @cached_property
    def x_full(self):
        """x at half-integer locations"""
        return (
            np.stack([self.x[:-1], self.x[:-1] + self.dx / 2], 1).ravel().view(_array)
        )

    @cached_property
    def y_full(self):
        """y at half-integer locations"""
        return (
            np.stack([self.y[:-1], self.y[:-1] + self.dy / 2], 1).ravel().view(_array)
        )

    @cached_property
    def XY_full(self):
        """X and Y at half-integer locations"""
        Y_full, X_full = np.meshgrid(self.y_full, self.x_full)
        return X_full.view(_array), Y_full.view(_array)

    @property
    def X_full(self):
        """X at half-integer locations"""
        return self.XY_full[0].view(_array)

    @property
    def Y_full(self):
        """Y at half-integer locations"""
        return self.XY_full[1].view(_array)

    @property
    def Xx(self):
        """X at Ex locations"""
        return self.X_full[1::2, ::2].view(_array)

    @property
    def Yx(self):
        """Y at Ex locations"""
        return self.Y_full[1::2, ::2].view(_array)

    @property
    def Xy(self):
        """X at Ey locations"""
        return self.X_full[::2, 1::2].view(_array)

    @property
    def Yy(self):
        """Y at Ey locations"""
        return self.Y_full[::2, 1::2].view(_array)

    @property
    def Xz(self):
        """X at Ez locations"""
        return self.X_full[::2, ::2].view(_array)

    @property
    def Yz(self):
        """Y at Ez locations"""
        return self.Y_full[::2, ::2].view(_array)

    @property
    def Xz_(self):
        """X at Hz locations"""
        return self.X_full[1::2, 1::2].view(_array)

    @property
    def Yz_(self):
        """Y at Hz locations"""
        return self.Y_full[1::2, 1::2].view(_array)

    def __eq__(self, other):
        eq = True
        for k, v in self.dict().items():
            if isinstance(v, np.ndarray):
                eq &= bool(((v - getattr(other, k)) < 1e-6).all())
            else:
                eq &= bool(v == getattr(other, k))
        return eq
