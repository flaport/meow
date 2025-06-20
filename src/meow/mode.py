"""An EigenMode."""

from __future__ import annotations

import pickle
import warnings
from collections.abc import Callable
from itertools import product
from pathlib import Path
from typing import Any, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pydantic import Field
from scipy.constants import epsilon_0 as eps0
from scipy.constants import mu_0 as mu0
from scipy.linalg import norm

from meow.arrays import Complex, ComplexArray2D, FloatArray2D
from meow.base_model import BaseModel, cached_property
from meow.cross_section import CrossSection
from meow.environment import Environment
from meow.integrate import integrate_2d
from meow.mesh import Mesh2D


def _power(amp: np.ndarray) -> np.ndarray:
    """Calculate the power from the amplitude."""
    return np.abs(amp) ** 2


class Mode(BaseModel):
    """A `Mode` contains the field information for a given `CrossSection`."""

    neff: Complex = Field(description="the effective index of the mode")
    cs: CrossSection = Field(
        description="the index cross section for which the mode was calculated"
    )
    Ex: ComplexArray2D = Field(description="the Ex-fields of the mode")
    Ey: ComplexArray2D = Field(description="the Ey-fields of the mode")
    Ez: ComplexArray2D = Field(description="the Ez-fields of the mode")
    Hx: ComplexArray2D = Field(description="the Hx-fields of the mode")
    Hy: ComplexArray2D = Field(description="the Hy-fields of the mode")
    Hz: ComplexArray2D = Field(description="the Hz-fields of the mode")
    interpolation: Literal["Ex", "Ey", "Ez", "Hz", ""] = Field(
        default="",
        description="To which 2D Yee-location the fields are interpolated to.",
    )

    def interpolate(
        self, location: Literal["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
    ) -> Mode:
        """Interpolate the mode to the given location."""
        if self.interpolation != "":
            msg = "Cannot interpolate from already interpolated mode!"
            raise RuntimeError(msg)
        interpolate_funcs = {
            "EX": _interpolate_Ex,
            "EY": _interpolate_Ey,
            "EZ": _interpolate_Ez,
            "HX": _interpolate_Ey,
            "HY": _interpolate_Ex,
            "HZ": _interpolate_Hz,
        }
        interpolate_func = interpolate_funcs[location.upper()]
        return interpolate_func(self)

    @property
    def te_fraction(self) -> float:
        """The TE polarization fraction of the mode."""
        return te_fraction(self)

    @cached_property
    def _pointing(self) -> tuple[ComplexArray2D, ComplexArray2D, ComplexArray2D]:
        """Calculate and cache the poynting vector."""
        vecE = np.stack([self.Ex, self.Ey, self.Ez], axis=-1)
        vecH = np.stack([self.Hx, self.Hy, self.Hz], axis=-1)
        vecP = np.cross(vecE, vecH)
        Px, Py, Pz = np.rollaxis(vecP, -1)
        return Px, Py, Pz

    @property
    def Px(self) -> ComplexArray2D:
        """The x-component of the Poynting vector."""
        return self._pointing[0]

    @property
    def Py(self) -> ComplexArray2D:
        """The y-component of the Poynting vector."""
        return self._pointing[1]

    @property
    def Pz(self) -> ComplexArray2D:
        """The z-component of the Poynting vector."""
        return self._pointing[2]

    @cached_property
    def A(self) -> float:
        """Mode area."""
        vecE = np.stack([self.Ex, self.Ey, self.Ez], axis=-1)
        E_sq = np.asarray(norm(vecE, axis=-1, ord=2))
        E_qu = E_sq**2
        x = self.cs.mesh.x_
        y = self.cs.mesh.y_
        return integrate_2d(x, y, E_sq) ** 2 / integrate_2d(x, y, E_qu)

    @property
    def env(self) -> Environment:
        """The environment of the mode."""
        return self.cs.env

    @property
    def mesh(self) -> Mesh2D:
        """The mesh of the mode."""
        return self.cs.mesh

    def _visualize(  # noqa: PLR0915
        self,
        *,
        title: str | None = None,
        title_prefix: str = "",
        fields: tuple[str, ...] = ("Ex",),
        ax: Any = None,
        n_cmap: str | colors.LinearSegmentedColormap | None = None,
        mode_cmap: str | None = None,
        num_levels: int = 8,
        operation: Callable = _power,
        show: bool = True,
        **ignored: Any,  # noqa: ARG002
    ) -> None:
        from meow.visualize import _figsize_visualize_mode

        W, H = _figsize_visualize_mode(self.cs, 6.4)

        if not fields:
            fields = ("Ex",)

        if len(fields) > 1:
            if len(fields) > 2:
                max_width = 15
                current_width = len(fields) * W
                W, H = _figsize_visualize_mode(self.cs, 6.4 * max_width / current_width)
            if ax is None:
                _, ax = plt.subplots(1, len(fields), figsize=(len(fields) * W, H))
            if len(ax) < len(fields):
                msg = (
                    f"Not enough axes supplied for the number of fields "
                    f"to plot! {len(ax)} < {len(fields)}."
                )
                raise ValueError(msg)
            for field, ax_ in zip(fields, ax, strict=False):
                self._visualize(
                    title=title,
                    title_prefix=title_prefix,
                    fields=(field,),
                    ax=ax_,
                    n_cmap=n_cmap,
                    mode_cmap=mode_cmap,
                    num_levels=num_levels,
                    operation=operation,
                    show=False,
                )
            if show:
                plt.show()
            return

        field = "Ex" if not fields else fields[0]
        valid_fields = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz", "Px", "Py", "Pz"]
        if field not in valid_fields:
            msg = f"Invalid field {field!r}. Valid fields: {', '.join(valid_fields)}."
            raise ValueError(msg)

        if ax is None:
            ax = plt.gca()
        plt.sca(ax)

        if n_cmap is None:
            # little bit lighter colored than the one in cs._visualize:
            n_cmap = colors.LinearSegmentedColormap.from_list(
                name="c_cmap", colors=["#ffffff", "#c1d9ed"]
            )
        self.cs._visualize(ax=ax, n_cmap=n_cmap, cbar=False, show=False)

        x, y = "x", "y"  # currently only propagation in z supported, see Mesh2D
        c = {
            "Ex": "x",
            "Ey": "y",
            "Ez": "z",
            "Hx": "y",
            "Hy": "x",
            "Hz": "z_",
            "Px": "x",
            "Py": "y",
            "Pz": "z",
        }[field]
        if mode_cmap is None:
            mode_cmap = "inferno"
        X = getattr(self.mesh, f"X{c}")
        Y = getattr(self.mesh, f"Y{c}")
        mode = operation(getattr(self, field))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            levels = np.linspace(mode.min(), mode.max(), num_levels + 1)[1:]
            plt.contour(X, Y, mode, cmap=mode_cmap, levels=levels)
            # plt.pcolormesh(X, Y, mode, cmap=mode_cmap, alpha=0.5) #, levels=levels)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(cax=cax)
        plt.sca(ax)

        plt.xlabel(x)
        plt.ylabel(y)
        plt.grid(visible=True, alpha=0.4)
        if title is None:
            plt.title(f"{title_prefix}{field} [neff={float(np.real(self.neff)):.6f}]")
        else:
            plt.title(f"{title_prefix}{title}")
        plt.xlim(X.min(), X.max())
        plt.ylim(Y.min(), Y.max())
        plt.axis("scaled")
        if show:
            plt.show()

    def save(self, filename: str | Path) -> None:
        """Save the mode to a file."""
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(self))

    @classmethod
    def load(cls, filename: str | Path) -> Mode:
        """Load a mode from a file."""
        return cast(Mode, pickle.loads(Path(filename).read_bytes()))

    def __add__(self, other: Mode) -> Mode:
        if not isinstance(other, Mode):
            msg = (
                "unsupported operand type(s) for +: "
                f"'Mode' and '{type(other).__name__}'"
            )
            raise TypeError(msg)
        new_mode = Mode(
            neff=0.5 * (self.neff + other.neff),
            cs=self.cs,
            Ex=self.Ex + other.Ex,
            Ey=self.Ey + other.Ey,
            Ez=self.Ez + other.Ez,
            Hx=self.Hx + other.Hx,
            Hy=self.Hy + other.Hy,
            Hz=self.Hz + other.Hz,
        )
        return new_mode

    def __mul__(self, other: np.complexfloating | complex) -> Mode:
        if not isinstance(
            other, float | np.complexfloating | np.floating | complex | int | np.integer
        ):
            msg = (
                "unsupported operand type(s) for *: "
                f"'Mode' and '{type(other).__name__}'"
            )
            raise TypeError(msg)
        new_mode = Mode(
            neff=self.neff,
            cs=self.cs,
            Ex=self.Ex * other,
            Ey=self.Ey * other,
            Ez=self.Ez * other,
            Hx=self.Hx * other,
            Hy=self.Hy * other,
            Hz=self.Hz * other,
        )
        return new_mode

    def __rmul__(self, other: np.complexfloating | complex) -> Mode:
        return self.__mul__(other)

    def __truediv__(self, other: np.complexfloating | complex) -> Mode:
        if not isinstance(
            other,
            float | np.complexfloating | np.floating | complex | int | np.integer,
        ):
            msg = (
                "unsupported operand type(s) for /: "
                f"'Mode' and '{type(other).__name__}'"
            )
            raise TypeError(msg)
        return self * (1 / other)

    def __sub__(self, other: Mode) -> Mode:
        if not isinstance(other, Mode):
            msg = (
                "unsupported operand type(s) for -: "
                f"'Mode' and '{type(other).__name__}'"
            )
            raise TypeError(msg)
        return self + other * (-1.0)


Modes = list[Mode]


def zero_phase(mode: Mode) -> Mode:
    """Normalize (zero out) the phase of a `Mode`."""
    e = np.abs(energy_density(mode))
    m, n = np.array(np.where(e == e.max()))[:, 0]
    phase = np.exp(-1j * np.angle(np.array(mode.Hx))[m][n])
    new_mode = Mode(
        neff=mode.neff,
        cs=mode.cs,
        Ex=mode.Ex * phase,
        Ey=mode.Ey * phase,
        Ez=mode.Ez * phase,
        Hx=mode.Hx * phase,
        Hy=mode.Hy * phase,
        Hz=mode.Hz * phase,
    )
    if _sum_around(np.real(new_mode.Hx), int(m), int(n)) < 0:
        new_mode = invert_mode(new_mode)
    return new_mode


# def _centroid_idxs(arr2d: np.ndarray) -> tuple[int, int]:
#    centroid_x = np.average(np.arange(arr2d.shape[1]), weights=arr2d.sum(axis=0))
#    centroid_y = np.average(np.arange(arr2d.shape[0]), weights=arr2d.sum(axis=1))
#    return round(float(centroid_x)), round(float(centroid_y))


def _sum_around(field: FloatArray2D, m: int, n: int, r: int = 2) -> float:
    total = 0
    idxs = range(-r, r + 1)
    idx_tups = product(idxs, idxs)
    M, N = field.shape
    for i, j in idx_tups:
        m_ = min(m + i, M - 1) if i >= 0 else max(m + i, 0)
        n_ = min(n + j, N - 1) if j >= 0 else max(n + j, 0)
        total = total + field[m_, n_]
    return total


def invert_mode(mode: Mode) -> Mode:
    """Invert a `Mode`."""
    return Mode(
        neff=mode.neff,
        cs=mode.cs,
        Ex=-mode.Ex,
        Ey=-mode.Ey,
        Ez=-mode.Ez,
        Hx=-mode.Hx,
        Hy=-mode.Hy,
        Hz=-mode.Hz,
    )


def inner_product(mode1: Mode, mode2: Mode) -> float:
    """The inner product of a `Mode` with another `Mode` is uniquely defined."""
    mesh = mode1.mesh
    cross = mode1.Ex * mode2.Hy - mode1.Ey * mode2.Hx
    return np.trapezoid(np.trapezoid(cross, mesh.y_), mesh.x_)


def inner_product_conj(mode1: Mode, mode2: Mode) -> float:
    """The inner product of a `Mode` with another `Mode` is uniquely defined."""
    mesh = mode1.mesh
    cross = mode1.Ex * mode2.Hy.conj() - mode1.Ey * mode2.Hx.conj()
    return np.trapezoid(np.trapezoid(cross, mesh.y_), mesh.x_)


def normalize_product(mode: Mode) -> Mode:
    """Normalize a `Mode` according to the `inner_product` with itself."""
    factor = np.sqrt(inner_product(mode, mode))
    return Mode(
        neff=mode.neff,
        cs=mode.cs,
        Ex=mode.Ex / factor,
        Ey=mode.Ey / factor,
        Ez=mode.Ez / factor,
        Hx=mode.Hx / factor,
        Hy=mode.Hy / factor,
        Hz=mode.Hz / factor,
    )


def electric_energy_density(
    mode: Mode,
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    """Get the electric energy density contained in a `Mode`."""
    epsx, epsy, epsz = mode.cs.nx**2, mode.cs.ny**2, mode.cs.nz**2
    return (
        0.5
        * eps0
        * (
            epsx * np.abs(mode.Ex) ** 2
            + epsy * np.abs(mode.Ey) ** 2
            + epsz * np.abs(mode.Ez) ** 2
        )
    )


def electric_energy(mode: Mode) -> float:
    """Get the electric energy contained in a `Mode`."""
    return electric_energy_density(mode).sum()


def magnetic_energy_density(
    mode: Mode,
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    """Get the magnetic energy density contained in a `Mode`."""
    return (
        0.5 * mu0 * (np.abs(mode.Hx) ** 2 + np.abs(mode.Hy) ** 2 + np.abs(mode.Hz) ** 2)
    )


def magnetic_energy(mode: Mode) -> float:
    """Get the magnetic energy contained in a `Mode`."""
    return magnetic_energy_density(mode).sum()


def energy_density(mode: Mode) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    """Get the energy density contained in a `Mode`."""
    return electric_energy_density(mode) + magnetic_energy_density(mode)


def energy(mode: Mode) -> float:
    """Get the energy contained in a `Mode`."""
    return energy_density(mode).sum()


def normalize_energy(mode: Mode) -> Mode:
    """Normalize a mode according to the energy it contains."""
    e = np.sqrt(2 * electric_energy(mode))
    h = np.sqrt(2 * magnetic_energy(mode))

    return Mode(
        neff=mode.neff,
        cs=mode.cs,
        Ex=mode.Ex / e,
        Ey=mode.Ey / e,
        Ez=mode.Ez / e,
        Hx=mode.Hx / h,
        Hy=mode.Hy / h,
        Hz=mode.Hz / h,
    )


def is_pml_mode(mode: Mode, threshold: float) -> bool:
    """Check whether a mode can be considered a PML mode."""
    threshold = min(max(threshold, 0.0), 1.0)
    if threshold > 0.999:
        return False
    numx, numy = mode.mesh.num_pml
    ed = energy_density(mode)
    m, n = ed.shape
    lft = ed[:numx, :]
    rgt = ed[m - numx :, :]
    top = ed[numx : m - numx, n:numy]
    btm = ed[numx : m - numx, n - numy :]
    rest = ed[numx : m - numx, numy : n - numy]
    pml_sum = lft.sum() + rgt.sum() + top.sum() + btm.sum()
    rest_sum = rest.sum()
    # probably propper integration considering
    # the size of the mesh cells would be better here
    is_pml = pml_sum > threshold * (rest_sum + pml_sum)
    return is_pml


def te_fraction(mode: Mode) -> float:
    """The TE polarization fraction of the `Mode`."""
    epsx = np.real(mode.cs.nx**2)
    e = np.sum(0.5 * eps0 * epsx * np.abs(mode.Ex) ** 2)
    h = np.sum(0.5 * mu0 * np.abs(mode.Hx) ** 2)
    return float(e / (e + h))


def _average(
    field: ComplexArray2D,
    direction: Literal["forward", "backward"] = "forward",
    axis: int = 0,
) -> ComplexArray2D:
    if axis == 1:
        return _average(field.T, direction=direction, axis=0).T
    if axis != 0:
        msg = "axis should be 0 or 1"
        raise ValueError(msg)

    average = 0.5 * (field[1:] + field[:-1])
    zero = np.zeros_like(average[:1])

    _direction = str(direction).lower()
    if _direction == "forward":
        return np.concatenate([zero, average], axis=0)

    if _direction != "backward":
        msg = "direction should be 'forward' or backward"
        raise ValueError(msg)

    return np.concatenate([average, zero], axis=0)


def _interpolate_Ex(mode: Mode) -> Mode:
    # TODO: take grid spacing into account
    Ey_at_Ez = _average(mode.Ey, direction="backward", axis=1)
    Ey_at_Ex = _average(Ey_at_Ez, direction="forward", axis=0)
    Ez_at_Ex = _average(mode.Ez, direction="forward", axis=0)
    Hx_at_Hz = _average(mode.Hx, direction="forward", axis=0)
    Hx_at_Ex = _average(Hx_at_Hz, direction="backward", axis=1)
    Hz_at_Ex = _average(mode.Hz, direction="backward", axis=1)
    return Mode(
        neff=mode.neff,
        cs=mode.cs,
        Ex=mode.Ex,
        Ey=Ey_at_Ex,
        Ez=Ez_at_Ex,
        Hx=Hx_at_Ex,
        Hy=mode.Hy,
        Hz=Hz_at_Ex,
    )


def _interpolate_Ey(mode: Mode) -> Mode:
    # TODO: take grid spacing into account
    Ex_at_Ez = _average(mode.Ex, direction="backward", axis=0)
    Ex_at_Ey = _average(Ex_at_Ez, direction="forward", axis=1)
    Ez_at_Ey = _average(mode.Ez, direction="forward", axis=1)
    Hy_at_Hz = _average(mode.Hy, direction="forward", axis=1)
    Hy_at_Ey = _average(Hy_at_Hz, direction="backward", axis=0)
    Hz_at_Ey = _average(mode.Hz, direction="backward", axis=0)
    return Mode(
        neff=mode.neff,
        cs=mode.cs,
        Ex=Ex_at_Ey,
        Ey=mode.Ey,
        Ez=Ez_at_Ey,
        Hx=mode.Hx,
        Hy=Hy_at_Ey,
        Hz=Hz_at_Ey,
    )


def _interpolate_Ez(mode: Mode) -> Mode:
    # TODO: take grid spacing into account
    return Mode(
        neff=mode.neff,
        cs=mode.cs,
        Ex=_average(mode.Ex, direction="backward", axis=0),
        Ey=_average(mode.Ey, direction="backward", axis=1),
        Ez=mode.Ez,
        Hx=_average(mode.Hx, direction="backward", axis=1),
        Hy=_average(mode.Hy, direction="backward", axis=0),
        Hz=_average(
            _average(mode.Hz, direction="backward", axis=0),
            direction="backward",
            axis=1,
        ),
    )


def _interpolate_Hz(mode: Mode) -> Mode:
    # TODO: take grid spacing into account
    return Mode(
        neff=mode.neff,
        cs=mode.cs,
        Ex=_average(mode.Ex, direction="forward", axis=1),
        Ey=_average(mode.Ey, direction="forward", axis=0),
        Ez=_average(
            _average(mode.Ez, direction="forward", axis=0), direction="forward", axis=1
        ),
        Hx=_average(mode.Hx, direction="forward", axis=0),
        Hy=_average(mode.Hy, direction="forward", axis=1),
        Hz=mode.Hz,
    )
