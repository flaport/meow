"""Integration of mode data."""

from typing import cast

import numpy as np
from scipy.integrate import dblquad, simpson
from scipy.interpolate import RegularGridInterpolator

from .arrays import FloatArray2D


def integrate_interpolate_2d(
    x: np.ndarray,
    y: np.ndarray,
    data: FloatArray2D,
    extent: tuple[tuple[int, int], tuple[int, int]] | None = None,
) -> np.ndarray:
    """Integrate 2D data on a grid using interpolation.

    First the data on the given grid is interpolated and then integrated
    using `scipy.dblquad`. This procedure is best suited if one wants to
    integrate over a (small) region of interest that can be placed at
    off-grid positions

    """
    interp = RegularGridInterpolator((x, y), data)

    if extent is None:
        extent = ((min(x), max(x)), (min(y), max(y)))

    def _integrable(x: float, y: float) -> float:
        return cast(float, interp((x, y)))

    return dblquad(
        _integrable,
        extent[0][0],
        extent[0][1],
        lambda _: extent[1][0],
        lambda _: extent[1][1],
        epsabs=1,
        epsrel=0.001,
    )[0]


def integrate_2d(x: np.ndarray, y: np.ndarray, data: FloatArray2D) -> float:
    """Simple 2D integration of data on a grid."""
    int1 = simpson(data, x=y)
    int2 = simpson(int1, x=x)
    return float(int2)
