from typing import Callable, List, Tuple

import numpy as np
import sax
from pydantic import Field

from meow.eme.propagate import _connect_two

from .base_model import BaseModel, _array, cached_property
from .cell import Cell
from .cross_section import CrossSection
from .fde import compute_modes as fde_compute_modes
from .mode import Modes, inner_product
from .structures import Structure, sort_structures


class Port(BaseModel):
    extend_x: Tuple[float, float] = Field(description="xmin, xmax")
    extend_y: Tuple[float, float] = Field(description="ymin, ymax")
    fg_structures: List[Structure] = Field(
        description="Structures that will be selectively removed outside the port extend"
    )

    num_modes: int = Field(description="Number of modes to be considered for this port")

    def compute_modes(self, cs: CrossSection, engine: Callable = None):
        if engine is None:
            engine = fde_compute_modes
        cell = PortCell(port=self, **cs.cell.dict())
        cs = cs.copy(update={"cell": cell})
        return engine(cs, num_modes=self.num_modes)


Ports = List[Port]


class PortCell(Cell):
    """A port for external s parameter evaluation"""

    port: Port = Field(description="The port that masks the foreground structures")

    @cached_property
    def m_full(self):
        X = self.mesh.X_full
        Y = self.mesh.Y_full
        m_full = np.zeros_like(self.mesh.X_full, dtype=np.int_)
        port = self.port
        for structure in sort_structures(self.structures):
            mask = structure.geometry._mask2d(X, Y, self.z)
            if structure in port.fg_structures:
                mask[X < port.extend_x[0]] = 0
                mask[X > port.extend_x[1]] = 0
                mask[Y < port.extend_y[0]] = 0
                mask[Y > port.extend_y[1]] = 0

            m_full[mask] = self.materials[structure.material]
        return m_full

def ordered_sdense(x):
    S, pm = sax.sdense(x)
    keys = list(pm.keys())
    reorder = np.argsort(keys)
    new_pm_keys = np.sort(keys)
    pm = {k:i for i,k in enumerate(new_pm_keys)}
    S = S[reorder, :][:, reorder]
    return (S, pm)
    
def compute_port_modes(cs: CrossSection, ports: Ports):
    """computes the set of modes for a set of ports on a CrossSection"""
    modes = []
    for port in ports:
        modes += port.compute_modes(cs)
    return modes


def overlap_matrix(modes_l: Modes, modes_r: Modes):
    """compute the overlaps between port and crosssection modes used for deembedding the inner S-matrix"""
    def norm_inner_product(l, r):
        def abs_sqrt(a):
            return np.sqrt(np.abs(inner_product(a,a)))
        return inner_product(l,r)/(abs_sqrt(l)*abs_sqrt(r))
    forward = {
        (f"left@{m}", f"right@{n}"): norm_inner_product(l, r)
        for n,r in enumerate(modes_r)
        for m,l in enumerate(modes_l)
    }
    return sax.reciprocal(forward)


def outer_S_matrix(modes: Modes, ports: Tuple[Ports, Ports], inner_S):
    """Deembed the inner S-matrix with respect to the given ports"""
    port_modes_l = compute_port_modes(modes[0][0].cs, ports[0])
    port_modes_r = compute_port_modes(modes[-1][0].cs, ports[-1])
    O_L = overlap_matrix(port_modes_l, modes[0])
    O_R = overlap_matrix(modes[-1], port_modes_r)
    S, pm = ordered_sdense(_connect_two(O_L, _connect_two(inner_S, O_R)))
    S = np.asarray(S).view(_array)
    return S, pm
