"""MEOW: Modeling of Eigenmodes and Overlaps in Waveguides."""

from __future__ import annotations

__author__ = "Floris Laporte"
__version__ = "0.14.1"

from meow.arrays import Dim as Dim
from meow.arrays import DType as DType
from meow.arrays import NDArray as NDArray
from meow.arrays import Shape as Shape
from meow.base_model import BaseModel as BaseModel
from meow.cell import Cell as Cell
from meow.cell import create_cells as create_cells
from meow.cross_section import CrossSection as CrossSection
from meow.environment import Environment as Environment
from meow.fde import compute_modes as compute_modes
from meow.fde import compute_modes_lumerical as compute_modes_lumerical
from meow.fde import compute_modes_tidy3d as compute_modes_tidy3d
from meow.gds_structures import GdsExtrusionRule as GdsExtrusionRule
from meow.gds_structures import extrude_gds as extrude_gds
from meow.geometries import Box as Box
from meow.geometries import Geometry2D as Geometry2D
from meow.geometries import Geometry2DBase as Geometry2DBase
from meow.geometries import Geometry3D as Geometry3D
from meow.geometries import Geometry3DBase as Geometry3DBase
from meow.geometries import Polygon2D as Polygon2D
from meow.geometries import Prism as Prism
from meow.geometries import Rectangle as Rectangle
from meow.materials import IndexMaterial as IndexMaterial
from meow.materials import Material as Material
from meow.materials import MaterialBase as MaterialBase
from meow.materials import SampledMaterial as SampledMaterial
from meow.materials import TidyMaterial as TidyMaterial
from meow.materials import silicon as silicon
from meow.materials import silicon_nitride as silicon_nitride
from meow.materials import silicon_oxide as silicon_oxide
from meow.mesh import Mesh2D as Mesh2D
from meow.mode import Mode as Mode
from meow.mode import electric_energy as electric_energy
from meow.mode import electric_energy_density as electric_energy_density
from meow.mode import inner_product as inner_product
from meow.mode import invert_mode as invert_mode
from meow.mode import magnetic_energy as magnetic_energy
from meow.mode import magnetic_energy_density as magnetic_energy_density
from meow.mode import normalize as normalize
from meow.mode import zero_phase as zero_phase
from meow.structures import Structure as Structure
from meow.structures import Structure2D as Structure2D
from meow.structures import Structure3D as Structure3D
from meow.visualize import vis as vis
from meow.visualize import visualize as visualize

from . import eme as eme
from . import fde as fde
