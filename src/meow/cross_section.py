"""A CrossSection."""

from __future__ import annotations

from typing import Any, Literal, Self

import numpy as np
import shapely
from pydantic import Field
from pydantic.v1 import PrivateAttr
from shapely.ops import unary_union

from meow.arrays import ComplexArray2D, IntArray2D
from meow.base_model import BaseModel, cached_property
from meow.cell import Cell, _create_full_material_array, sort_structures
from meow.environment import Environment
from meow.materials import Material
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
    def materials(self) -> dict[Material, int]:
        """Return a dictionary mapping materials to their indices."""
        materials: dict[Material, int] = {}
        for i, structure in enumerate(sort_structures(self.structures), start=1):
            if structure.material not in materials:
                materials[structure.material] = i
        return materials

    @cached_property
    def _m_full(self) -> IntArray2D:
        """Return the material index array for the full mesh."""
        return _create_full_material_array(self.mesh, self.structures, self.materials)

    @cached_property
    def n_full(self) -> ComplexArray2D:
        """Return the refractive index array for the full mesh."""
        n_full = np.ones_like(self.mesh.X_full, dtype=np.complex128)
        for material, idx in self.materials.items():
            n_full = np.where(self._m_full == idx, material(self.env), n_full)
        return n_full

    @cached_property
    def nx(self) -> ComplexArray2D:
        """Return the smoothed refractive index on the Ex positions."""
        return _compute_smoothed_n(
            self.mesh, self._m_full, self.materials, self.env, self.structures, "x"
        )

    @cached_property
    def ny(self) -> ComplexArray2D:
        """Return the smoothed refractive index on the Ey positions."""
        return _compute_smoothed_n(
            self.mesh, self._m_full, self.materials, self.env, self.structures, "y"
        )

    @cached_property
    def nz(self) -> ComplexArray2D:
        """Return the smoothed refractive index on the Ez positions."""
        return _compute_smoothed_n(
            self.mesh, self._m_full, self.materials, self.env, self.structures, "z"
        )

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
            values = np.unique(np.real(self.n_full))
            _cbar = plt.colorbar(ticks=values, cax=cax)
            # material_names = ['air'] + [mat.name for mat in self.cell.materials]
            # labels = [f"\n{n}\n{v:.3f}" for n, v in zip(material_names, values)]
            labels = [f"{v:.3f}" for v in values]
            _cbar.ax.set_yticklabels(labels, rotation=90, va="center", ha="center")
            plt.sca(ax)
        if show:
            plt.show()


# --- Subpixel permittivity smoothing ---

_COMPONENT_SLICES = {
    "x": (slice(1, None, 2), slice(None, None, 2)),  # Ex: [1::2, ::2]
    "y": (slice(None, None, 2), slice(1, None, 2)),  # Ey: [::2, 1::2]
    "z": (slice(None, None, 2), slice(None, None, 2)),  # Ez: [::2, ::2]
}


def _dual_cell_bounds(
    mesh: Mesh2D,
    component: Literal["x", "y", "z"],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute (x_lo, x_hi, y_lo, y_hi) for the dual cells of a field component.

    Each field component lives at a specific position on the Yee grid. Its dual
    cell is the area over which the effective permittivity should be averaged.

    Full-grid indices (in x_full/y_full):
      Ex at (2i+1, 2j): dual cell x=[x_full[2i], x_full[2i+2]], y=[y_full[2j-1], y_full[2j+1]]
      Ey at (2i, 2j+1): dual cell x=[x_full[2i-1], x_full[2i+1]], y=[y_full[2j], y_full[2j+2]]
      Ez at (2i, 2j):   dual cell x=[x_full[2i-1], x_full[2i+1]], y=[y_full[2j-1], y_full[2j+1]]
    """
    xf = mesh.x_full
    yf = mesh.y_full

    si, sj = _COMPONENT_SLICES[component]
    # Number of component grid points
    ni = len(range(*si.indices(len(xf))))
    nj = len(range(*sj.indices(len(yf))))

    # Full-grid indices for this component
    fi = np.arange(len(xf))[si]  # shape (ni,)
    fj = np.arange(len(yf))[sj]  # shape (nj,)

    def _bounds(vals: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lo = np.empty(len(indices))
        hi = np.empty(len(indices))
        for k, idx in enumerate(indices):
            if idx > 0:
                lo[k] = vals[idx - 1]
            else:
                lo[k] = vals[0] - (vals[1] - vals[0])
            if idx < len(vals) - 1:
                hi[k] = vals[idx + 1]
            else:
                hi[k] = vals[-1] + (vals[-1] - vals[-2])
        return lo, hi

    x_lo, x_hi = _bounds(xf, fi)
    y_lo, y_hi = _bounds(yf, fj)

    return x_lo, x_hi, y_lo, y_hi


def _compute_smoothed_n(
    mesh: Mesh2D,
    m_full: IntArray2D,
    materials: dict[Material, int],
    env: Environment,
    structures: list[Structure2D],
    component: Literal["x", "y", "z"],
) -> ComplexArray2D:
    """Compute subpixel-smoothed refractive index for a field component.

    At interface pixels, the effective permittivity is computed using
    area-fraction-weighted averaging:
      - E parallel to interface:      arithmetic avg  eps_eff = sum(f_i * eps_i)
      - E perpendicular to interface: harmonic avg    1/eps_eff = sum(f_i / eps_i)
    """
    si, sj = _COMPONENT_SLICES[component]
    m_comp = m_full[si, sj]

    # Build eps array from material indices
    env_eps = np.complex128(1.0) ** 2  # background: air (n=1)
    eps = np.full_like(m_comp, env_eps, dtype=np.complex128)
    mat_eps: dict[int, complex] = {0: env_eps}
    for material, idx in materials.items():
        n_val = material(env)
        e = np.complex128(n_val) ** 2
        eps[m_comp == idx] = e
        mat_eps[idx] = e

    # Identify interface pixels (where material differs from any 4-neighbor)
    padded = np.pad(m_comp, 1, mode="edge")
    is_interface = (
        (m_comp != padded[:-2, 1:-1])  # top
        | (m_comp != padded[2:, 1:-1])  # bottom
        | (m_comp != padded[1:-1, :-2])  # left
        | (m_comp != padded[1:-1, 2:])  # right
    )

    if not np.any(is_interface):
        return np.sqrt(eps)

    # Compute dual cell bounds for this component
    x_lo, x_hi, y_lo, y_hi = _dual_cell_bounds(mesh, component)

    # Get interface pixel indices
    ii, jj = np.where(is_interface)

    # Build dual cell boxes (vectorized)
    dual_boxes = shapely.box(x_lo[ii], y_lo[jj], x_hi[ii], y_hi[jj])
    dual_areas = shapely.area(dual_boxes)

    # Compute area fractions per material for each interface pixel
    # Group structures by material index
    mat_to_polys: dict[int, list] = {}
    for structure in structures:
        idx = materials[structure.material]
        if idx not in mat_to_polys:
            mat_to_polys[idx] = []
        mat_to_polys[idx].append(structure.geometry._shapely_polygon())

    # Area fractions: shape (n_interface_pixels,) per material
    fractions: dict[int, np.ndarray] = {}
    total_struct_fraction = np.zeros(len(ii))
    for idx, polys in mat_to_polys.items():
        poly = unary_union(polys)
        intersections = shapely.intersection(poly, dual_boxes)
        areas = shapely.area(intersections)
        frac = areas / dual_areas
        fractions[idx] = frac
        total_struct_fraction += frac

    # Background (air) gets the remainder
    fractions[0] = np.maximum(1.0 - total_struct_fraction, 0.0)

    # Determine interface orientation
    # Use padded array to check material changes in x and y directions
    diff_x = padded[:-2, 1:-1][ii, jj] != padded[2:, 1:-1][ii, jj]
    diff_y = padded[1:-1, :-2][ii, jj] != padded[1:-1, 2:][ii, jj]

    normal_x = diff_x & ~diff_y  # vertical interface, normal along x
    normal_y = ~diff_x & diff_y  # horizontal interface, normal along y
    corner = ~normal_x & ~normal_y  # corner or ambiguous -> arithmetic avg

    # Determine which averaging to use for this component at each pixel
    # harmonic when E is perpendicular to interface (E component == normal direction)
    # arithmetic when E is parallel to interface
    use_harmonic = np.zeros(len(ii), dtype=bool)
    if component == "x":
        use_harmonic[normal_x] = True  # Ex perpendicular to vertical interface
    elif component == "y":
        use_harmonic[normal_y] = True  # Ey perpendicular to horizontal interface
    # Ez is always parallel to interface -> always arithmetic
    # Corners -> arithmetic (use_harmonic stays False)

    # Compute effective eps for each interface pixel
    eps_eff = np.zeros(len(ii), dtype=np.complex128)

    # Arithmetic average: eps_eff = sum(f_i * eps_i)
    arith_mask = ~use_harmonic
    if np.any(arith_mask):
        eps_arith = np.zeros(arith_mask.sum(), dtype=np.complex128)
        for idx, frac in fractions.items():
            eps_arith += frac[arith_mask] * mat_eps[idx]
        eps_eff[arith_mask] = eps_arith

    # Harmonic average: 1/eps_eff = sum(f_i / eps_i)
    if np.any(use_harmonic):
        inv_eps_harm = np.zeros(use_harmonic.sum(), dtype=np.complex128)
        for idx, frac in fractions.items():
            e = mat_eps[idx]
            if abs(e) > 0:
                inv_eps_harm += frac[use_harmonic] / e
        # Avoid division by zero
        safe = np.abs(inv_eps_harm) > 1e-30
        eps_harm = np.zeros_like(inv_eps_harm)
        eps_harm[safe] = 1.0 / inv_eps_harm[safe]
        eps_harm[~safe] = eps[ii[use_harmonic], jj[use_harmonic]][~safe]
        eps_eff[use_harmonic] = eps_harm

    # Write back smoothed values
    eps[ii, jj] = eps_eff

    return np.sqrt(eps)
