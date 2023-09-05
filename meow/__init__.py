""" MEOW: Modeling of Eigenmodes and Overlaps in Waveguides """

__author__ = "Floris Laporte"
__version__ = "0.8.0"

import warnings

# Silence Excessive Logging...

try:
    from loguru import logger

    logger.disable("gdsfactory")
except ImportError:
    pass

try:
    from numexpr.utils import log

    log.setLevel("CRITICAL")
except ImportError:
    pass

try:
    from rich import pretty

    old_install = pretty.install
    pretty.install = lambda *_, **__: None
    import tidy3d

    pretty.install = old_install
except ImportError:
    pass


try:
    import sax

    warnings.filterwarnings(action="ignore", module="sax")
except ImportError:
    pass

from . import base_model as base_model
from . import cell as cell
from . import cross_section as cross_section
from . import eme as eme
from . import environment as environment
from . import fde as fde
from . import gds_structures as gds_structures
from . import geometries as geometries
from . import materials as materials
from . import mesh as mesh
from . import mode as mode
from . import structures as structures

# from . import visualize as visualize
from .base_model import BaseModel as BaseModel
from .cell import Cell as Cell
from .cell import create_cells as create_cells
from .cross_section import CrossSection as CrossSection
from .eme import compute_interface_s_matrices as compute_interface_s_matrices
from .eme import compute_interface_s_matrix as compute_interface_s_matrix
from .eme import compute_propagation_s_matrices as compute_propagation_s_matrices
from .eme import compute_propagation_s_matrix as compute_propagation_s_matrix
from .eme import compute_s_matrix as compute_s_matrix
from .eme import compute_s_matrix_sax as compute_s_matrix_sax
from .eme import select_ports as select_ports
from .environment import Environment as Environment
from .fde import compute_modes as compute_modes
from .fde import compute_modes_lumerical as compute_modes_lumerical
from .fde import compute_modes_tidy3d as compute_modes_tidy3d
from .gds_structures import GdsExtrusionRule as GdsExtrusionRule
from .gds_structures import extrude_gds as extrude_gds
from .geometries import Box as Box
from .geometries import Geometry2D as Geometry2D
from .geometries import Geometry3D as Geometry3D
from .geometries import Prism as Prism
from .geometries import Rectangle as Rectangle
from .materials import Material as Material
from .materials import SampledMaterial as SampledMaterial
from .materials import TidyMaterial as TidyMaterial
from .materials import silicon as silicon
from .materials import silicon_nitride as silicon_nitride
from .materials import silicon_oxide as silicon_oxide
from .mesh import Mesh as Mesh
from .mesh import Mesh2D as Mesh2D
from .mesh import Mesh2d as Mesh2d
from .mode import Mode as Mode
from .mode import electric_energy as electric_energy
from .mode import electric_energy_density as electric_energy_density
from .mode import energy as energy
from .mode import energy_density as energy_density
from .mode import inner_product as inner_product
from .mode import inner_product_conj as inner_product_conj
from .mode import invert_mode as invert_mode
from .mode import is_pml_mode as is_pml_mode
from .mode import magnetic_energy as magnetic_energy
from .mode import magnetic_energy_density as magnetic_energy_density
from .mode import normalize_energy as normalize_energy
from .mode import normalize_product as normalize_product
from .mode import te_fraction as te_fraction
from .mode import zero_phase as zero_phase
from .structures import Structure as Structure
from .structures import Structure2D as Structure2D
from .structures import Structure3D as Structure3D
from .visualize import vis as vis
from .visualize import visualize as visualize
