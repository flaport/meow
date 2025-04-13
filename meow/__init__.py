""" MEOW: Modeling of Eigenmodes and Overlaps in Waveguides """

__author__ = "Floris Laporte"
__version__ = "0.13.0"

from .array import Dim as Dim
from .array import DType as DType
from .array import NDArray as NDArray
from .array import Shape as Shape
from .base_model import BaseModel as BaseModel
from .cell import Cell as Cell
from .cell import create_cells as create_cells
from .cross_section import CrossSection as CrossSection
from .eme import compute_interface_s_matrices as compute_interface_s_matrices
from .eme import compute_interface_s_matrix as compute_interface_s_matrix
from .eme import compute_propagation_s_matrices as compute_propagation_s_matrices
from .eme import compute_propagation_s_matrix as compute_propagation_s_matrix
from .eme import compute_s_matrix as compute_s_matrix
from .eme import select_ports as select_ports
from .environment import Environment as Environment
from .fde import compute_modes as compute_modes
from .fde import compute_modes_lumerical as compute_modes_lumerical
from .fde import compute_modes_tidy3d as compute_modes_tidy3d
from .gds_structures import GdsExtrusionRule as GdsExtrusionRule
from .gds_structures import extrude_gds as extrude_gds
from .geometries import Box as Box
from .geometries import Geometry2D as Geometry2D
from .geometries import Geometry2DBase as Geometry2DBase
from .geometries import Geometry3D as Geometry3D
from .geometries import Geometry3DBase as Geometry3DBase
from .geometries import Prism as Prism
from .geometries import Rectangle as Rectangle
from .integrate import integrate_2d as integrate_2d
from .integrate import integrate_interpolate_2d as integrate_interpolate_2d
from .materials import IndexMaterial as IndexMaterial
from .materials import Material as Material
from .materials import MaterialBase as MaterialBase
from .materials import SampledMaterial as SampledMaterial
from .materials import TidyMaterial as TidyMaterial
from .materials import silicon as silicon
from .materials import silicon_nitride as silicon_nitride
from .materials import silicon_oxide as silicon_oxide
from .mesh import Mesh2D as Mesh2D
from .mode import Mode as Mode
from .mode import electric_energy as electric_energy
from .mode import electric_energy_density as electric_energy_density
from .mode import inner_product as inner_product
from .mode import inner_product_conj as inner_product_conj
from .mode import invert_mode as invert_mode
from .mode import magnetic_energy as magnetic_energy
from .mode import magnetic_energy_density as magnetic_energy_density
from .mode import normalize_product as normalize_product
from .mode import zero_phase as zero_phase
from .structures import Structure as Structure
from .structures import Structure2D as Structure2D
from .structures import Structure3D as Structure3D
from .visualize import visualize as vis
from .visualize import visualize as visualize
