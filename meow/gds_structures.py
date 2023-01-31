""" GDS Extrusions """
# TODO: Maybe it makes more sense to use native GDSFactory tooling for this

from typing import Dict, List, Tuple

import numpy as np
import shapely.geometry as sg
from pydantic import Field

from .base_model import BaseModel
from .geometries import Prism
from .materials import Material
from .structures import Structure


class GdsExtrusionRule(BaseModel):
    """a `GdsExtrusionRule` describes a single extrusion rule.
    Multiple of such rules can later be associated with a gds layer tuple."""

    material: Material = Field(description="the material of the extrusion")
    h_min: float = Field(description="the extrusion starting height")
    h_max: float = Field(description="the extrusion ending height")
    buffer: float = Field(
        default=0.0,
        description="an extra buffer (=grow or shrink) operation applied to the polygon",
    )
    mesh_order: int = Field(
        default=5.0, description="the mesh order of the resulting `Structure`"
    )

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
    cell,  # type: ignore
    extrusions: Dict[Tuple[int, int], List[GdsExtrusionRule]],
):
    """extrude a gds cell given a dictionary of extruson rules

    Args:
        cell: a gdspy or gdstk Cell to extrude
        extrusions: the extrusion rules to use (if not given, the example extrusions will be used.)
    """
    structs = []
    for layer, polys in cell.get_polygons(by_spec=True, depth=None).items():
        for poly in polys:
            if layer not in extrusions:
                continue
            structs.extend(extrusion(poly) for extrusion in extrusions[layer])
    return structs
