from scipy.integrate import dblquad, simpson
from scipy.interpolate import RegularGridInterpolator


def integrate_interpolate_2d(x, y, data, extent=None):
    """
    First the data on the given grid is interpolated and then integrated using `scipy.dblquad`.
    This procedure is best suited if one wants to integrate over a (small) region of interest that can be placed at off-grid positions
    """
    interp = RegularGridInterpolator((x, y), data)
    if extent is None:
        extent = [[min(x), max(x)], [min(y), max(y)]]

    def integrable(x, y):
        return interp((x, y))

    return dblquad(
        integrable,
        extent[0][0],
        extent[0][1],
        lambda _: extent[1][0],
        lambda _: extent[1][1],
        epsabs=1,
        epsrel=0.001,
    )[0]


def integrate_2d(x, y, data) -> float:
    """much simpler integration over the full grid"""
    int1 = simpson(data, x=y)
    int2 = simpson(int1, x=x)
    return float(int2)
