"""GDS Extrusions."""

from __future__ import annotations

from typing import Any

import numpy as np
import shapely.geometry as sg
from pydantic import Field

from meow.arrays import FloatArray2D
from meow.base_model import BaseModel
from meow.geometries import Prism
from meow.materials import Material
from meow.structures import Structure3D

# TODO: Maybe it makes more sense to use native GDSFactory tooling for this


class GdsExtrusionRule(BaseModel):
    """A `GdsExtrusionRule` describes a single extrusion rule.

    Multiple of such rules can later be associated with a gds layer tuple.
    """

    material: Material = Field(description="the material of the extrusion")
    h_min: float = Field(description="the extrusion starting height")
    h_max: float = Field(description="the extrusion ending height")
    buffer: float = Field(
        default=0.0,
        description="an extra buffer (grow / shrink) operation applied to the polygon",
    )
    mesh_order: int = Field(
        default=5, description="the mesh order of the resulting `Structure3D`"
    )

    def __call__(self, poly: FloatArray2D) -> Structure3D:
        """Apply the extrusion rule to a polygon."""
        if self.buffer > 0:
            try:
                poly = np.asarray(sg.Polygon(poly).buffer(self.buffer).boundary.coords)
            except NotImplementedError as e:
                import gdspy

                polygonset = gdspy.offset(gdspy.Polygon(poly), 0.25)
                if polygonset is None:
                    msg = (
                        "The polygon could not be offset."
                        "Please check the input polygon."
                    )
                    raise ValueError(msg) from e
                poly = polygonset.polygons[0]
        return Structure3D(
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
    cell: Any,  # gf.Component | gdspy.Cell | gdstk.Cell
    extrusions: dict[tuple[int, int], list[GdsExtrusionRule]],
) -> list[Structure3D]:
    """Extrude a gds cell given a dictionary of extruson rules.

    Args:
        cell: a gdspy or gdstk Cell to extrude
        extrusions: the extrusion rules to use
            (if not given, the example extrusions will be used.)
    """
    structs = []
    for layer, polys in _get_polygons(cell).items():
        for poly in polys:
            if layer not in extrusions:
                continue
            structs.extend(extrusion(poly) for extrusion in extrusions[layer])
    return structs


def _get_polygons(cell: Any) -> dict[tuple[int, int], list[np.ndarray]]:
    if _get_major_gdsfactory_version() < 8:
        return cell.get_polygons(by_spec=True, depth=None)
    dbu = cell.layout().dbu
    polys = cell.get_polygons()
    layers = cell.layout().layer_infos()
    return {
        (layers[i].layer, layers[i].datatype): [
            np.asarray([(p.x * dbu, p.y * dbu) for p in p.each_point_hull()])
            for p in ps
        ]
        for i, ps in polys.items()
    }


def _get_major_gdsfactory_version() -> int:
    from gdsfactory.config import __version__

    major_version = int(__version__.split(".")[0])
    return major_version
