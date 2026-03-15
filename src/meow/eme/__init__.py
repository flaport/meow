"""SAX EME."""

from meow.eme.cascade import (
    compute_s_matrix_sax,
    downselect_s,
)
from meow.eme.default import (
    compute_s_matrix,
)
from meow.eme.interface import (
    PassivityMethod,
    compute_interface_s_matrices,
    compute_interface_s_matrix,
    enforce_passivity,
    overlap_matrix,
)
from meow.eme.propagation import (
    compute_propagation_s_matrices,
    compute_propagation_s_matrix,
    propagate_modes,
    select_ports,
    track_modes,
)
from meow.eme.solve import (
    tsvd_solve,
)

__all__ = [
    "PassivityMethod",
    "compute_interface_s_matrices",
    "compute_interface_s_matrix",
    "compute_propagation_s_matrices",
    "compute_propagation_s_matrix",
    "compute_s_matrix",
    "compute_s_matrix_sax",
    "downselect_s",
    "enforce_passivity",
    "overlap_matrix",
    "propagate_modes",
    "select_ports",
    "track_modes",
    "tsvd_solve",
]
