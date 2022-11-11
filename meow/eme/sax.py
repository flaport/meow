""" SAX backend for EME (default backend) """
from functools import partial
from typing import List

import sax

from ..mode import Mode
from .common import compute_interface_s_matrices, compute_propagation_s_matrices
from .common import select_ports as select_ports

try:
    import klujax
except ImportError:
    klujax = None


def _get_netlist(propagations, interfaces):
    """get the netlist of a stack of `Modes`"""

    instances = {
        **{k: _load_constant_model(S) for k, S in propagations.items()},
        **{k: _load_constant_model(S) for k, S in interfaces.items()},
    }

    connections = {}
    for i in range(len(interfaces)):
        connections[f"p_{i},right"] = f"i_{i}_{i+1},left"
        connections[f"i_{i}_{i+1},right"] = f"p_{i+1},left"

    ports = {
        f"left": f"p_0,left",
        f"right": f"p_{len(propagations)-1},right",
    }

    return {"instances": instances, "connections": connections, "ports": ports}


def _load_constant_model(value):
    def model():
        return value

    return model


def _validate_sax_backend(sax_backend):
    if sax_backend is None:
        sax_backend = "klu" if klujax is not None else "default"

    if sax_backend not in ["default", "klu"]:
        raise ValueError(
            f"Invalid SAX Backend. Got: {sax_backend!r}. Should be 'default' or 'klu'."
        )
    return sax_backend


def compute_s_matrix_sax(
    modes: List[List[Mode]],
    sax_backend=None,
    enforce_reciprocity=True,
    enforce_lossy_unitarity=False,
    **kwargs,
):
    """Calculate the S-matrix for given sets of modes, each set belonging to a `Cell`

    Args:
        modes: Each collection of modes for each of the `Cell` objects
        backend: which SAX backend to use to calculate the final S-matrix.
    """
    sax_backend = _validate_sax_backend(sax_backend)
    get_interface_s_matrices_ = kwargs.pop(
        "get_interface_s_matrices", compute_interface_s_matrices
    )
    get_interface_s_matrices = partial(get_interface_s_matrices_, **kwargs)

    num_modes = len(modes[0])
    propagations = compute_propagation_s_matrices(modes)
    interfaces = get_interface_s_matrices(
        modes,
        enforce_reciprocity=enforce_reciprocity,
        enforce_lossy_unitarity=enforce_lossy_unitarity,
    )
    net = _get_netlist(propagations, interfaces)
    mode_names = [f"{i}" for i in range(num_modes)]

    _circuit, _ = sax.circuit(netlist=net, backend=sax_backend, modes=mode_names)

    # maybe in the future we should return the sax model and not the S-matrix?
    sdict = sax.sdict(_circuit())
    sdict = {k: sdict[k] for k in sorted(sdict)}
    return sax.sdense(sdict)
