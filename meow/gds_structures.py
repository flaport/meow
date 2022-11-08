""" GDS Extrusions """
# TODO: Maybe it makes more sense to use native GDSFactory tooling for this

from typing import Dict, List, Tuple

import numpy as np
import shapely.geometry as sg

from .base_model import BaseModel
from .geometries import Prism
from .materials import Material
from .structures import Structure


class GdsExtrusionRule(BaseModel):
    """a `GdsExtrusionRule` describes a single extrusion rule.
    Multiple of such rules can later be associated with a gds layer tuple.

    Attributes:
        material: the material of the extrusion
        h_min: the extrusion starting height
        h_max: the extrusion ending height
        buffer: an extra buffer (=grow or shrink) operation applied to the polygon
        mesh_order: the mesh order of the resulting `Structure`
    """

    material: Material
    h_min: float
    h_max: float
    buffer: float = 0.0
    mesh_order: int = 5

    def __call__(self, poly) -> Structure:
        if self.buffer > 0:
            poly = np.asarray(sg.Polygon(poly).buffer(self.buffer).boundary.coords)
        return Structure(
            material=self.material,
            geometry=Prism(
                poly=poly,
                h_min=self.h_min,
                h_max=self.h_max,
                axis="y",
            ),
            mesh_order=self.mesh_order,
        )


def extrude_gds(
    cell: "gdspy.Cell",  # type: ignore
    extrusions: Dict[Tuple[int, int], List[GdsExtrusionRule]],
):
    """extrude a gds cell given a dictionary of extruson rules

    Args:
        cell: a gdspy.Cell to extrude
        extrusions: the extrusion rules to use (if not given, the example extrusions will be used.)
    """
    structs = []
    for layer, polys in cell.get_polygons(by_spec=True, depth=None).items():
        for poly in polys:
            if layer not in extrusions:
                continue
            for extrusion in extrusions[layer]:
                structs.append(extrusion(poly))
    return structs
