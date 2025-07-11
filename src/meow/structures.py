"""A Structure is a combination of a Geometry with a material."""

from __future__ import annotations

from typing import Any, overload

from pydantic import Field

from meow.base_model import BaseModel
from meow.environment import Environment
from meow.geometries import Geometry2D, Geometry3D
from meow.materials import Material

DEFAULT_MESH_ORDER = 5


@overload
def Structure(
    *,
    material: Material,
    geometry: Geometry2D,
    mesh_order: int = DEFAULT_MESH_ORDER,
) -> Structure2D: ...


@overload
def Structure(
    *,
    material: Material,
    geometry: Geometry3D,
    mesh_order: int = DEFAULT_MESH_ORDER,
) -> Structure3D: ...


def Structure(  # noqa: N802
    *,
    material: Material,
    geometry: Geometry2D | Geometry3D,
    mesh_order: int = DEFAULT_MESH_ORDER,
) -> Structure2D | Structure3D:
    """Create a Structure from a Material and Geometry."""
    kwargs = {
        "material": material,
        "geometry": geometry,
        "mesh_order": mesh_order,
    }
    if isinstance(geometry, Geometry2D):
        return Structure2D(**kwargs)
    return Structure3D(**kwargs)


class Structure2D(BaseModel):
    """A `Structure2D` is an association between a `Geometry2D` and a `Material`."""

    material: Material = Field(description="the material of the structure")
    geometry: Geometry2D = Field(description="the geometry of the structure")
    mesh_order: int = Field(
        default=DEFAULT_MESH_ORDER, description="the mesh order of the structure"
    )

    def _visualize(self, **ignored: Any) -> None:  # noqa: ARG002
        color = self.material.meta.get("color", None)
        return self.geometry._visualize(color=color)


class Structure3D(BaseModel):
    """A `Structure3D` is an association between a `Geometry3D` and a `Material`."""

    material: Material = Field(description="the material of the structure")
    geometry: Geometry3D = Field(description="the geometry of the structure")
    mesh_order: int = Field(
        default=DEFAULT_MESH_ORDER, description="the mesh order of the structure"
    )

    def _project(self, z: float) -> list[Structure2D]:
        geometry_2d = self.geometry._project(z)
        structs = []
        for geom in geometry_2d:
            struct = Structure2D(
                material=self.material,
                geometry=geom,
                mesh_order=self.mesh_order,
            )
            structs.append(struct)
        return structs

    def _lumadd(
        self, sim: Any, env: Environment, unit: float = 1e-6, xyz: str = "yzx"
    ) -> None:
        material_name = self.material._lumadd(sim, env, unit)
        self.geometry._lumadd(sim, material_name, self.mesh_order, unit, xyz)

    def _trimesh(
        self, color: str | None = None, scale: tuple[float, float, float] | None = None
    ) -> Any:
        return self.geometry._trimesh(
            color=(color or self.material.meta.get("color")),
            scale=scale,
        )

    def _visualize(
        self,
        scale: tuple[float, float, float] | None = None,
        **ignored: Any,  # noqa: ARG002
    ) -> None:
        return self._trimesh(scale=scale).show()


@overload
def _sort_structures(structures: list[Structure3D]) -> list[Structure3D]: ...


@overload
def _sort_structures(structures: list[Structure2D]) -> list[Structure2D]: ...


def _sort_structures(
    structures: list[Structure3D] | list[Structure2D],
) -> list[Structure2D] | list[Structure3D]:
    struct_info = [(s.mesh_order, -i, s) for i, s in enumerate(structures)]
    sorted_struct_info = sorted(struct_info, key=lambda I: (I[0], I[1]), reverse=True)
    return [s for _, _, s in sorted_struct_info]  # type: ignore[reportReturnType]
