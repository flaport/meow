"""A CrossSection."""

from __future__ import annotations

from typing import Any, Self

import numpy as np
from pydantic import Field
from pydantic.v1 import PrivateAttr

from meow.arrays import FloatArray2D
from meow.base_model import BaseModel, cached_property
from meow.cell import Cell, _create_full_material_array, _sort_structures
from meow.environment import Environment
from meow.mesh import Mesh2D
from meow.structures import Structure2D


class CrossSection(BaseModel):
    """A `CrossSection` is built from a `Cell` with an `Environment`.

    This uniquely defines the refractive index everywhere.
    """

    structures: list[Structure2D] = Field(
        description="the 2D structures in the CrossSection"
    )
    mesh: Mesh2D = Field(description="the mesh to discretize the structures with")
    env: Environment = Field(
        description="the environment for which the cross section was calculated"
    )
    _cell: Cell | None = PrivateAttr(default=None)

    @classmethod
    def from_cell(cls, *, cell: Cell, env: Environment) -> Self:
        """Create a CrossSection from a Cell and Environment."""
        return cls(structures=cell.structures_2d, mesh=cell.mesh, env=env, _cell=cell)

    @cached_property
    def materials(self) -> dict[Structure2D, int]:
        """Return a dictionary mapping materials to their indices."""
        materials = {}
        for i, structure in enumerate(_sort_structures(self.structures), start=1):
            if structure.material not in materials:
                materials[structure.material] = i
        return materials

    @cached_property
    def n_full(self) -> FloatArray2D:
        """Return the refractive index array for the full mesh."""
        m_full = _create_full_material_array(self.mesh, self.structures, self.materials)
        n_full = np.ones_like(self.mesh.X_full)
        for material, idx in self.materials.items():
            n_full = np.where(m_full == idx, material(self.env), n_full)
        return n_full

    @property
    def nx(self) -> FloatArray2D:
        """Return the refractive index on the Ex positions."""
        return self.n_full[1::2, ::2]

    @property
    def ny(self) -> FloatArray2D:
        """Return the refractive index on the Ey positions."""
        return self.n_full[::2, 1::2]

    @property
    def nz(self) -> FloatArray2D:
        """Return the refractive index on the Ez positions."""
        return self.n_full[::2, ::2]

    def _visualize(
        self,
        *,
        ax: Any = None,
        n_cmap: Any = None,
        cbar: bool = True,
        show: bool = True,
        **ignored: Any,
    ) -> None:
        import matplotlib.pyplot as plt  # fmt: skip
        from matplotlib import colors  # fmt: skip
        from mpl_toolkits.axes_grid1 import make_axes_locatable  # fmt: skip

        debug_grid = ignored.pop("debug_grid", False)
        if n_cmap is None:
            n_cmap = colors.LinearSegmentedColormap.from_list(
                name="c_cmap", colors=["#ffffff", "#86b5dc"]
            )
        if ax is not None:
            plt.sca(ax)
        else:
            ax = plt.gca()
        n_full = np.real(self.n_full).copy()
        n_full[0, 0] = 1.0
        plt.pcolormesh(self.mesh.X_full, self.mesh.Y_full, n_full, cmap=n_cmap)
        plt.axis("scaled")
        if not debug_grid:
            plt.grid(visible=True)
        else:
            dx = self.mesh.dx
            dy = self.mesh.dy
            x_ticks = np.sort(np.unique(self.mesh.X_full.ravel()))[::2]
            y_ticks = np.sort(np.unique(self.mesh.Y_full.ravel()))[::2]
            plt.xticks(x_ticks - 0.25 * dx, ["" for _ in x_ticks - 0.25 * dx])
            plt.yticks(y_ticks - 0.25 * dy, ["" for _ in y_ticks - 0.25 * dy])
            plt.xticks(
                x_ticks + 0.25 * dx, ["" for _ in x_ticks + 0.25 * dx], minor=True
            )
            plt.yticks(
                y_ticks + 0.25 * dy, ["" for _ in y_ticks + 0.25 * dy], minor=True
            )
            plt.grid(visible=True, which="major", ls="-")
            plt.grid(visible=True, which="minor", ls=":")
        if cbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            values = np.unique(self.n_full)
            _cbar = plt.colorbar(ticks=values, cax=cax)
            # material_names = ['air'] + [mat.name for mat in self.cell.materials]
            # labels = [f"\n{n}\n{v:.3f}" for n, v in zip(material_names, values)]
            labels = [f"{v:.3f}" for v in values]
            _cbar.ax.set_yticklabels(labels, rotation=90, va="center", ha="center")
            plt.sca(ax)
        if show:
            plt.show()
