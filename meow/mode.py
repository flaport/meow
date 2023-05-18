""" An EigenMode """

import pickle
from itertools import product
from typing import Any, List, Tuple

import numpy as np
from pydantic import Field, PrivateAttr
from scipy.constants import epsilon_0 as eps0
from scipy.constants import mu_0 as mu0
from scipy.linalg import norm

from .base_model import BaseModel
from .cross_section import CrossSection
from .integrate import integrate_2d


class Mode(BaseModel):
    """A `Mode` contains the field information for a given `CrossSection`."""

    neff: complex = Field(description="the effective index of the mode")
    cs: CrossSection = Field(
        description="the index cross section for which the mode was calculated"
    )
    Ex: np.ndarray[Tuple[int, int], np.dtype[np.complex_]] = Field(
        description="the Ex-fields of the mode"
    )
    Ey: np.ndarray[Tuple[int, int], np.dtype[np.complex_]] = Field(
        description="the Ey-fields of the mode"
    )
    Ez: np.ndarray[Tuple[int, int], np.dtype[np.complex_]] = Field(
        description="the Ez-fields of the mode"
    )
    Hx: np.ndarray[Tuple[int, int], np.dtype[np.complex_]] = Field(
        description="the Hx-fields of the mode"
    )
    Hy: np.ndarray[Tuple[int, int], np.dtype[np.complex_]] = Field(
        description="the Hy-fields of the mode"
    )
    Hz: np.ndarray[Tuple[int, int], np.dtype[np.complex_]] = Field(
        description="the Hz-fields of the mode"
    )

    _Px: np.ndarray[Tuple[int, int], np.dtype[np.complex_]] = PrivateAttr()
    _Py: np.ndarray[Tuple[int, int], np.dtype[np.complex_]] = PrivateAttr()
    _Pz: np.ndarray[Tuple[int, int], np.dtype[np.complex_]] = PrivateAttr()
    _A: np.float_ = PrivateAttr()

    def __init__(self, **data: Any):
        super().__init__(**data)
        self._Px = None
        self._Py = None
        self._Pz = None
        self._A = None

    @property
    def te_fraction(self):
        """the TE polarization fraction of the mode."""
        return te_fraction(self)

    def _calc_poynting(self):
        """calculate and cache the poynting vector"""
        vecE = np.stack([self.Ex, self.Ey, self.Ez], axis=-1)
        vecH = np.stack([self.Hx, self.Hy, self.Hz], axis=-1)
        vecP = np.cross(vecE, vecH)
        self._Px, self._Py, self._Pz = np.rollaxis(vecP, -1)

    def _calc_area(self):
        vecE = np.stack([self.Ex, self.Ey, self.Ez], axis=-1)
        E_sq = norm(vecE, axis=-1, ord=2)
        E_qu = E_sq**2
        x = self.cs.mesh.x_
        y = self.cs.mesh.y_
        self._A = integrate_2d(x, y, E_sq) ** 2 / integrate_2d(x, y, E_qu)

    @property
    def Px(self):
        if self._Px is None:
            self._calc_poynting()
        return self._Px

    @property
    def Py(self):
        if self._Py is None:
            self._calc_poynting()
        return self._Py

    @property
    def Pz(self):
        if self._Pz is None:
            self._calc_poynting()
        return self._Pz

    @property
    def A(self):
        """mode area"""
        if self._A is None:
            self._calc_area()
        return self._A

    @property
    def env(self):
        return self.cs.env

    @property
    def mesh(self):
        return self.cs.mesh

    @property
    def cell(self):
        return self.cs.cell

    def _visualize(
        self,
        title=None,
        fields=None,
        ax=None,
        n_cmap=None,
        mode_cmap=None,
        num_levels=8,
        operation=lambda x: np.abs(x) ** 2,
    ):
        import matplotlib.pyplot as plt
        from matplotlib import colors

        if fields is None or len(fields) == 0:
            fields = ["Ex"]

        if len(fields) > 1:
            if ax is None:
                _, ax = plt.subplots(len(fields), 1, figsize=(5, len(fields) * 3))
            if len(ax) < len(fields):
                raise ValueError(
                    f"Not enough axes supplied for the number of fields "
                    f"to plot! {len(ax)} < {len(fields)}."
                )
            for field, ax_ in zip(fields, ax):
                self._visualize(
                    field,
                    ax=ax_,
                    n_cmap=n_cmap,
                    mode_cmap=mode_cmap,
                    num_levels=num_levels,
                )
            return

        field = fields[0]
        valid_fields = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz", "Px", "Py", "Pz"]
        if field not in valid_fields:
            raise ValueError(
                f"Invalid field {field!r}. Valid fields: {', '.join(valid_fields)}."
            )

        if ax is None:
            ax = plt.gca()
            plt.axis("scaled")
        plt.sca(ax)

        x, y = "x", "y"  # currently only propagation in z supported, see Mesh2d
        c = field[-1]
        if n_cmap is None:
            n_cmap = colors.LinearSegmentedColormap.from_list(
                name="c_cmap", colors=["#ffffff", "#c1d9ed"]
            )
        if mode_cmap is None:
            mode_cmap = "inferno"
        X = getattr(self.mesh, f"X{c}")
        Y = getattr(self.mesh, f"Y{c}")
        mode = operation(getattr(self, field))
        plt.contour(X, Y, mode, cmap=mode_cmap, levels=np.linspace(mode.min(), mode.max(), num_levels))  # fmt: skip
        plt.colorbar(label="mode")

        n = np.real(getattr(self.cs, f"n{c}"))
        plt.pcolormesh(X, Y, n, cmap=n_cmap)
        plt.colorbar(label="n")
        plt.xlabel(x)
        plt.ylabel(y)
        plt.grid(True, alpha=0.4)
        if title is None:
            plt.title(f"{field} [neff={float(np.real(self.neff)):.6f}]")
        else:
            plt.title(title)
        plt.xlim(X.min(), X.max())
        plt.ylim(Y.min(), Y.max())
        plt.show()

    def save(self, filename):
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, filename):
        with open(filename, "rb") as file:
            return pickle.load(file)


Modes = List[Mode]


def zero_phase(mode: Mode) -> Mode:
    """normalize (zero out) the phase of a `Mode`"""
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
    if _sum_around(np.real(new_mode.Hx), m, n) < 0:
        new_mode = invert_mode(new_mode)
    return new_mode


def _sum_around(field, m, n, r=2):
    total = 0
    idxs = range(-r, r + 1)
    idx_tups = product(idxs, idxs)
    M, N = field.shape
    for i, j in idx_tups:
        m_ = min(m + i, M - 1) if i >= 0 else max(m + i, 0)
        n_ = min(n + j, N - 1) if i >= 0 else max(n + j, 0)
        total = total + field[m_, n_]
    return total


def invert_mode(mode: Mode) -> Mode:
    """invert a `Mode`"""
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
    """the inner product of a `Mode` with another `Mode` is uniquely defined."""
    mesh = mode1.mesh
    cross = mode1.Ex * mode2.Hy.conj() - mode1.Ey * mode2.Hx.conj()
    return 0.25 * np.trapz(np.trapz(cross, mesh.y_), mesh.x_)


def normalize_product(mode: Mode) -> Mode:
    """normalize a `Mode` according to the `inner_product` with itself"""
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
) -> np.ndarray[Tuple[int, int], np.dtype[np.float_]]:
    """get the electric energy density contained in a `Mode`"""
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
    """get the electric energy contained in a `Mode`"""
    return electric_energy_density(mode).sum()


def magnetic_energy_density(
    mode: Mode,
) -> np.ndarray[Tuple[int, int], np.dtype[np.float_]]:
    """get the magnetic energy density contained in a `Mode`"""
    return (
        0.5 * mu0 * (np.abs(mode.Hx) ** 2 + np.abs(mode.Hy) ** 2 + np.abs(mode.Hz) ** 2)
    )


def magnetic_energy(mode: Mode) -> float:
    """get the magnetic energy contained in a `Mode`"""
    return magnetic_energy_density(mode).sum()


def energy_density(mode: Mode) -> np.ndarray[Tuple[int, int], np.dtype[np.float_]]:
    """get the energy density contained in a `Mode`"""
    return electric_energy_density(mode) + magnetic_energy_density(mode)


def energy(mode: Mode) -> float:
    """get the energy contained in a `Mode`"""
    return energy_density(mode).sum()


def normalize_energy(mode: Mode) -> Mode:
    """normalize a mode according to the energy it contains"""
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


def te_fraction(mode: Mode) -> float:
    """the TE polarization fraction of the `Mode`"""
    epsx = mode.cs.nx**2
    e = np.sum(0.5 * eps0 * epsx * np.abs(mode.Ex) ** 2)
    h = np.sum(0.5 * mu0 * np.abs(mode.Hx) ** 2)
    return float(e / (e + h))
