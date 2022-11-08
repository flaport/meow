""" Geometries """

from secrets import token_hex
from typing import Dict, List, Literal, Tuple, Union, cast

import numpy as np
import shapely.geometry as sg
from matplotlib.colors import to_rgba as _to_rgba_mpl
from pydantic import validator
from trimesh import Trimesh
from trimesh.creation import extrude_polygon
from trimesh.scene import Scene
from trimesh.transformations import rotation_matrix

from .base_model import BaseModel
from .mesh import Mesh2d

GEOMETRIES: Dict[str, type] = {}


class Geometry(BaseModel):
    type: str = ""

    def __init_subclass__(cls):
        GEOMETRIES[cls.__name__] = cls

    @validator("type", pre=True, always=True)
    def validate_type(cls, value):
        if (not value) or value == cls.__name__:  # type: ignore
            return cls.__name__  # type: ignore
        else:
            raise TypeError("Please do not set type variable for Geometry manually.")

    @classmethod
    def from_dict(cls, dic):
        if cls is Geometry:
            if "type" not in dic:
                raise TypeError(
                    "Cannot parse geometry dict. Please specify the geometry type in the dictionary!"
                )
            cls_name = dic["type"]
            if cls_name not in GEOMETRIES:
                raise TypeError(
                    f"Cannot parse geometry dict. Geometry type {cls_name!r} not known!"
                )
            cls = GEOMETRIES[cls_name]
        elif "type" in dic:
            cls_name = dic["type"]
            if cls_name != cls.__name__:
                raise TypeError(
                    f"Cannot parse geometry dict. Geometry type given by dict {cls_name!r} does not match invoking geometry type {cls.__name__!r}!"
                )
        return cls(**{k: v for k, v in dic.items() if k != "type"})

    def _visualize(self, scale=None):
        scene = Scene()
        scene.add_geometry(self._trimesh(scale=scale))
        scene.apply_transform(rotation_matrix(-np.pi / 6, (0, 1, 0)))
        return scene.show()

    def _mask2d_single(self, X, Y, z):
        raise NotImplementedError(f"{self.__class__.__name__!r} cannot be masked.")

    def _mask2d(self, mesh: Mesh2d, z: float):
        mx = self._mask2d_single(mesh.Xx, mesh.Yx, z)
        my = self._mask2d_single(mesh.Xy, mesh.Yy, z)
        mz = self._mask2d_single(mesh.Xz, mesh.Yz, z)
        return mx, my, mz

    def _lumadd(self, sim, material_name, mesh_order, unit=1e-6, xyz="yzx"):
        raise NotImplementedError(
            f"{self.__class__.__name__!r} cannot be added to Lumerical."
        )

    def _trimesh(self, **kwargs):
        raise NotImplementedError(f"{self.__class__.__name__!r} cannot be visualized.")


Geometries = List[Geometry]


class Box(Geometry):
    """A Box is a simple rectangular cuboid

    Attributes:
        x_min: the minimum x-value of the box
        x_max: the maximum x-value of the box
        y_min: the minimum y-value of the box
        y_max: the maximum y-value of the box
        z_min: the minimum z-value of the box
        z_max: the maximum z-value of the box
    """

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    def _mask2d_single(self, X, Y, z):
        if (z < self.z_min) or (self.z_max < z):
            return np.zeros_like(X, dtype=bool)
        mask = (
            (self.x_min <= X)
            & (X <= self.x_max)
            & (self.y_min <= Y)
            & (Y <= self.y_max)
        )
        return mask

    def _lumadd(self, sim, material_name, mesh_order, unit, xyz):
        x, y, z = xyz
        name = token_hex(4)
        kwargs = {
            f"{x}_min": float(self.x_min * unit),
            f"{x}_max": float(self.x_max * unit),
            f"{y}_min": float(self.y_min * unit),
            f"{y}_max": float(self.y_max * unit),
            f"{z}_min": float(self.z_min * unit),
            f"{z}_max": float(self.z_max * unit),
        }
        sim.addrect(
            name=name,
            material=material_name,
            override_mesh_order_from_material_database=True,
            mesh_order=mesh_order,
            **kwargs,
        )

    def _trimesh(self, color=None, scale=None):
        sx, sy, sz = scale or (1, 1, 1)
        poly = sg.Polygon(
            [
                (self.x_min * sx, self.y_min * sy),
                (self.x_min * sx, self.y_max * sy),
                (self.x_max * sx, self.y_max * sy),
                (self.x_max * sx, self.y_min * sy),
            ],
        )
        prism = extrude_polygon(poly, self.z_max * sz - self.z_min * sz)
        prism = cast(Trimesh, prism.apply_translation((0, 0, self.z_min * sz)))
        if color is not None:
            prism.visual.face_colors = _to_rgba(color)  # type: ignore
        return prism


AxisDirection = Union[Literal["x"], Literal["y"], Literal["z"]]


class Prism(Geometry):
    """A prism is a 2D Polygon extruded along a certain axis direction ('x', 'y', or 'z').

    Attributes:
        poly: the 2D array (Nx2) with polygon vertices
        h_min: the start height of the extrusion
        h_max: the end height of the extrusion
        axis: the axis along which the polygon will be extruded ('x', 'y', or 'z').

    Note:
        currently only extrusions along 'y' (perpendicular to the chip)
        are fully supported!
    """

    poly: np.ndarray[Tuple[int, Literal[2]], np.dtype[np.float_]]
    h_min: float
    h_max: float
    axis: AxisDirection = "y"

    def _mask2d_single(self, X, Y, z):
        poly = sg.Polygon(self.poly)
        if self.axis == "x":
            # x, y, z -> y, z, x
            y_min, _ = self.poly.min(0)
            y_max, _ = self.poly.max(0)
            line = sg.LineString([(y_min, z), (y_max, z)])
            intersection = np.asarray(poly.intersection(line).coords)
            if not intersection.shape[0]:
                return np.zeros_like(X, dtype=bool)
            (y_min, _), (y_max, _) = intersection
            y_min, y_max = min(y_min, y_max), max(y_min, y_max)
            x_min, x_max = min(self.h_min, self.h_max), max(self.h_min, self.h_max)
            mask = (x_min <= X) & (X <= x_max) & (y_min <= Y) & (Y <= y_max)
        elif self.axis == "y":
            # x, y, z -> z, x, y
            _, x_min = self.poly.min(0)
            _, x_max = self.poly.max(0)
            line = sg.LineString([(z, x_min), (z, x_max)])
            intersection = np.asarray(poly.intersection(line).coords)
            if not intersection.shape[0]:
                return np.zeros_like(X, dtype=bool)
            (_, x_min), (_, x_max) = intersection
            x_min, x_max = min(x_min, x_max), max(x_min, x_max)
            y_min, y_max = min(self.h_min, self.h_max), max(self.h_min, self.h_max)
            mask = (x_min <= X) & (X <= x_max) & (y_min <= Y) & (Y <= y_max)
        else:
            # x, y, z -> x, y, z
            if (z < self.h_min) or (self.h_max < z):
                return np.zeros_like(X, dtype=bool)
            mask = np.asarray(
                [poly.contains(sg.Point(x, y)) for x, y in zip(X.ravel(), Y.ravel())],
                dtype=bool,
            ).reshape(X.shape)
        return mask

    def _lumadd(self, sim, material_name, mesh_order, unit=1e-6, xyz="yzx"):
        name = token_hex(4)

        if xyz not in ("xyz", "yzx", "zxy"):
            raise ValueError(
                f"Prism axes should be positively oriented when adding to Lumerical. Got: {xyz!r}"
            )

        sim.addpoly(
            name=name,
            material=material_name,
            override_mesh_order_from_material_database=True,
            mesh_order=mesh_order,
            use_relative_coordinates=True,
            vertices=np.asarray(self.poly, float) * float(unit),
            x=0.0,
            y=0.0,
            z_min=float(self.h_min * unit),
            z_max=float(self.h_max * unit),
        )

        x, y, z = xyz
        if self.axis == "x":
            z, x, y = x, y, z
            # raise NotImplementedError(
            #    "Only prisms extruded perpendicular to the 'chip surface' are currently supported in Lumerical."
            # )
        elif self.axis == "y":
            y, z, x = x, y, z
        xyz = f"{x}{y}{z}"

        if xyz == "yzx":
            raise NotImplementedError(
                "Only prisms extruded perpendicular to the 'chip surface' are currently supported in Lumerical."
            )
            sim.setnamed(name, "first axis", "z")
            sim.setnamed(name, "rotation 1", 90.0)
            sim.setnamed(name, "second axis", "y")
            sim.setnamed(name, "rotation 2", -90.0)
        elif xyz == "zxy":
            raise NotImplementedError(
                "Only prisms extruded perpendicular to the 'chip surface' are currently supported in Lumerical."
            )
            sim.setnamed(name, "first axis", "y")
            sim.setnamed(name, "rotation 1", 270.0)
            sim.setnamed(name, "second axis", "z")
            sim.setnamed(name, "rotation 2", 270.0)

    def _trimesh(self, color=None, scale=None):
        poly = sg.Polygon(self.poly)
        prism = extrude_polygon(poly, self.h_max - self.h_min)
        prism = prism.apply_translation((0, 0, self.h_min))
        if self.axis == "x":
            prism.vertices = np.roll(prism.vertices, shift=1, axis=1)  # type: ignore
        if self.axis == "y":
            prism.vertices = np.roll(prism.vertices, shift=-1, axis=1)  # type: ignore
        if scale is not None:
            sx, sy, sz = scale
            prism.vertices *= np.array([[sx, sy, sz]])  # type: ignore
        if color is not None:
            prism.visual.face_colors = _to_rgba(color)  # type: ignore
        return prism

    def _center(self):
        import shapely.geometry as sg

        a, b = np.array(sg.Polygon(self.poly).centroid.xy).ravel()
        c = 0.5 * (self.h_min + self.h_max)
        x, y, z = {
            "x": (c, a, b),
            "y": (b, c, a),
            "z": (a, b, c),
        }[self.axis]
        return x, y, z

    def _axis_tuple(self):
        return {
            "x": (1, 0, 0),
            "y": (0, 1, 0),
            "z": (0, 0, 1),
        }[self.axis]


def _to_rgba(c):
    r, g, b, a = _to_rgba_mpl(c)
    a = min(max(a, 0.1), 0.9)
    return float(r), float(g), float(b), float(a)
