""" a Structure2D/Structure3D is a combination of a Geometry2D/Geometry3D with a material (and an optional mesh order) """

from typing import Dict, List, Tuple, Union, overload

import numpy as np
from pydantic import Field

from .base_model import BaseModel
from .geometries import Geometry2D, Geometry3D
from .materials import Material

DEFAULT_MESH_ORDER = 5


def Structure(
    *,
    material: Material,
    geometry: Union[Geometry2D, Geometry3D],
    mesh_order: int = DEFAULT_MESH_ORDER,
):
    kwargs = {
        "material": material,
        "geometry": geometry,
        "mesh_order": mesh_order,
    }
    if isinstance(geometry, Geometry2D):
        return Structure2D(**kwargs)
    else:
        return Structure3D(**kwargs)


class Structure2D(BaseModel):
    """a `Structure2D` is an association between a `Geometry2D` and a `Material`"""

    material: Material = Field(description="the material of the structure")
    geometry: Geometry2D = Field(description="the geometry of the structure")
    mesh_order: int = Field(
        default=DEFAULT_MESH_ORDER, description="the mesh order of the structure"
    )

    def _visualize(self):
        color = self.material.meta.get("color", None)
        return self.geometry._visualize(color=color)


class Structure3D(BaseModel):
    """a `Structure3D` is an association between a `Geometry3D` and a `Material`"""

    material: Material = Field(description="the material of the structure")
    geometry: Geometry3D = Field(description="the geometry of the structure")
    mesh_order: int = Field(
        default=DEFAULT_MESH_ORDER, description="the mesh order of the structure"
    )

    def _project(self, z) -> List[Structure2D]:
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

    def _lumadd(self, sim, env, unit=1e-6, xyz="yzx"):
        material_name = self.material._lumadd(sim, env, unit)
        self.geometry._lumadd(sim, material_name, self.mesh_order, unit, xyz)

    def _trimesh(self, color=None, scale=None):
        return self.geometry._trimesh(
            color=(color or self.material.meta.get("color")),
            scale=scale,
        )

    def _visualize(self, scale=None):
        return self._trimesh(scale=scale).show()


def visualize_structures(structures: List[Structure3D], scale=None):
    """easily visualize a collection (list) of `Structure3D` objects"""
    from trimesh.scene import Scene  # fmt: skip
    from trimesh.transformations import rotation_matrix  # fmt: skip

    scene = Scene(
        geometry=[s._trimesh(scale=scale) for s in sort_structures(structures)]
    )
    scene.apply_transform(rotation_matrix(np.pi - np.pi / 6, (0, 1, 0)))
    return scene.show()


@overload
def sort_structures(structures: List[Structure3D]) -> List[Structure3D]:
    ...


@overload
def sort_structures(structures: List[Structure2D]) -> List[Structure2D]:
    ...


def sort_structures(
    structures: Union[List[Structure3D], List[Structure2D]]
) -> Union[List[Structure2D], List[Structure3D]]:
    struct_info = [(s.mesh_order, -i, s) for i, s in enumerate(structures)]
    sorted_struct_info = sorted(struct_info, key=lambda I: (I[0], I[1]), reverse=True)
    return [s for _, _, s in sorted_struct_info]  # type: ignore


@overload
def classify_structures_by_mesh_order_and_material(
    structures: List[Structure3D], materials: Dict[Material, int]
) -> Dict[Tuple[int, int], List[Structure3D]]:
    ...


@overload
def classify_structures_by_mesh_order_and_material(
    structures: List[Structure2D], materials: Dict[Material, int]
) -> Dict[Tuple[int, int], List[Structure2D]]:
    ...


def classify_structures_by_mesh_order_and_material(
    structures: Union[List[Structure3D], List[Structure2D]],
    materials: Dict[Material, int],
) -> Union[
    Dict[Tuple[int, int], List[Structure2D]], Dict[Tuple[int, int], List[Structure3D]]
]:
    structures = sort_structures(structures)
    structures_dict = {}
    for structure in structures:
        mo = structure.mesh_order
        mat = materials[structure.material]
        if (mo, mat) not in structures_dict:
            structures_dict[mo, mat] = []
        structures_dict[mo, mat].append(structure)
    return structures_dict
