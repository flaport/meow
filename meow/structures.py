""" a Structure is a combination of a geometry with a material (and an optional mesh order) """

from typing import List

import numpy as np
from trimesh.scene import Scene
from trimesh.transformations import rotation_matrix

from .base_model import BaseModel
from .geometries import Geometry
from .materials import Material


class Structure(BaseModel):
    """a `Structure` is an association between a `Geometry` and a `Material`

    Attributes:
        material: the material of the structure
        geometry: the geometry of the structure
        mesh_order: the mesh order of the structure

    """

    material: Material
    geometry: Geometry
    mesh_order: int = 5

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


Structures = List[Structure]


def _visualize_structures(structures, scale=None):
    """easily visualize a collection (list) of `Structure` objects"""
    scene = Scene(
        geometry=[s._trimesh(scale=scale) for s in _sort_structures(structures)]
    )
    scene.apply_transform(rotation_matrix(np.pi - np.pi / 6, (0, 1, 0)))
    return scene.show()


def _sort_structures(structures):
    struct_info = [(s.mesh_order, -i, s) for i, s in enumerate(structures)]
    sorted_struct_info = sorted(struct_info, key=lambda I: (I[0], I[1]), reverse=True)
    return [s for _, _, s in sorted_struct_info]
