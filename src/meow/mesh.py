"""A 2D Mesh."""

from __future__ import annotations

from typing import Annotated, Literal

import numpy as np
from pydantic import BeforeValidator, Field
from pydantic.types import NonNegativeInt

from meow.arrays import FloatArray1D, FloatArray2D
from meow.base_model import BaseModel, cached_property


class Mesh2D(BaseModel):
    """A ``Mesh2D`` describes how a ``Structure3D`` is discritized into a ``Cell``."""

    x: FloatArray1D = Field(
        description="x-coords of the mesh (Ez locations, i.e. corners of the 2D cell)"
    )

    y: FloatArray1D = Field(
        description="y-coords of the mesh (Ez locations, i.e. corners of the 2D cell)"
    )

    angle_phi: float = Field(
        default=0.0,
        description=(
            "Azimuth angle of the propagation axis in the plane orthogonal to the mesh."
        ),
    )
    angle_theta: float = Field(
        default=0.0,
        description="Polar angle of the propagation axis from the injection axis.",
    )
    bend_radius: Annotated[
        float, BeforeValidator(lambda x: (np.nan if x is None else x))
    ] = Field(
        default=np.nan,
        description=(
            "A curvature radius for simulation of waveguide bends. Can be negative, "
            "in which case the mode plane center has a smaller value than the "
            "curvature center along the tangential axis perpendicular to the bend axis."
        ),
    )
    bend_axis: Literal[0, 1] = Field(
        default=0,
        description=(
            "Index into the two tangential axes defining the normal to the plane in "
            "which the bend lies. This must be provided if ``bend_radius`` is not "
            "``None``. For example, for a ring in the global xy-plane, and a mode "
            "plane in either the xz or the yz plane, the ``bend_axis`` is always 1 "
            "(the global z axis)."
        ),
    )
    plane_center: tuple[float, float] = Field(
        default=(0.0, 0.0),
        description=(
            "If ``bend_radius`` is not ``None``, the position of the plane "
            "corresponding along the circonference of the circle"
        ),
    )

    num_pml: tuple[NonNegativeInt, NonNegativeInt] = Field(
        default=(0, 0),
        description="Number of standard pml layers to add in the two tangential axes.",
    )

    ez_interfaces: bool = Field(
        default=False,
        description=(
            "when enabled, the meshing algorithm will throw away any index values "
            "at the interfaces which are not on even (Ez) half-grid locations. "
            "Enabling this should result in more symmetric modes."
        ),
    )

    @cached_property
    def dx(self) -> FloatArray1D:
        """dx at Hz locations, i.e. center of the 2D cell."""
        return self.x[1:] - self.x[:-1]

    @cached_property
    def dy(self) -> FloatArray1D:
        """dy at Hz locations, i.e. center of the 2D cell."""
        return self.y[1:] - self.y[:-1]

    @cached_property
    def x_(self) -> FloatArray1D:
        """x at Hz locations, i.e. center of the 2D cell."""
        return 0.5 * (self.x[1:] + self.x[:-1])

    @cached_property
    def y_(self) -> FloatArray1D:
        """y at Hz locations, i.e. center of the 2D cell."""
        return 0.5 * (self.y[1:] + self.y[:-1])

    @cached_property
    def x_full(self) -> FloatArray1D:
        """x at half-integer locations."""
        return np.stack([self.x[:-1], self.x[:-1] + self.dx / 2], 1).ravel()

    @cached_property
    def y_full(self) -> FloatArray1D:
        """y at half-integer locations."""
        return np.stack([self.y[:-1], self.y[:-1] + self.dy / 2], 1).ravel()

    @cached_property
    def XY_full(self) -> tuple[FloatArray2D, FloatArray2D]:
        """X and Y at half-integer locations."""
        Y_full, X_full = np.meshgrid(self.y_full, self.x_full)
        return X_full, Y_full

    @property
    def X_full(self) -> FloatArray2D:
        """X at half-integer locations."""
        return self.XY_full[0]

    @property
    def Y_full(self) -> FloatArray2D:
        """Y at half-integer locations."""
        return self.XY_full[1]

    @property
    def Xx(self) -> FloatArray2D:
        """X at Ex locations."""
        return self.X_full[1::2, ::2]

    @property
    def Yx(self) -> FloatArray2D:
        """Y at Ex locations."""
        return self.Y_full[1::2, ::2]

    @property
    def Xy(self) -> FloatArray2D:
        """X at Ey locations."""
        return self.X_full[::2, 1::2]

    @property
    def Yy(self) -> FloatArray2D:
        """Y at Ey locations."""
        return self.Y_full[::2, 1::2]

    @property
    def Xz(self) -> FloatArray2D:
        """X at Ez locations."""
        return self.X_full[::2, ::2]

    @property
    def Yz(self) -> FloatArray2D:
        """Y at Ez locations."""
        return self.Y_full[::2, ::2]

    @property
    def Xz_(self) -> FloatArray2D:
        """X at Hz locations."""
        return self.X_full[1::2, 1::2]

    @property
    def Yz_(self) -> FloatArray2D:
        """Y at Hz locations."""
        return self.Y_full[1::2, 1::2]
