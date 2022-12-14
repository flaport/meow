""" MEOW: Modeling of Eigenmodes and Overlaps in Waveguides """

__author__ = "Floris Laporte"
__version__ = "0.1.3"

# Silence Excessive Logging...

from loguru import logger

logger.disable("gdsfactory")
from numexpr.utils import log

log.setLevel("CRITICAL")


# disable rich pretty printing enabled by tidy3d (if I want it, I'll enable it myself...)
from rich import pretty

old_install = pretty.install
pretty.install = lambda *_, **__: None
import tidy3d

pretty.install = old_install

import warnings

warnings.filterwarnings(action="ignore", module="sax")
import sax

# from . import cell as cell
from . import base_model as base_model
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
from .cell import *
from .cross_section import *
from .eme import *
from .environment import *
from .fde import *
from .gds_structures import *
from .geometries import *
from .materials import *
from .mesh import *
from .mode import *
from .structures import *
from .visualize import *
