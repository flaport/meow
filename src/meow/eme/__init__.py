"""EME Implementations & Backends."""

from __future__ import annotations

from meow.eme.common import compute_interface_s_matrices as compute_interface_s_matrices
from meow.eme.common import compute_interface_s_matrix as compute_interface_s_matrix
from meow.eme.common import (
    compute_propagation_s_matrices as compute_propagation_s_matrices,
)
from meow.eme.common import compute_propagation_s_matrix as compute_propagation_s_matrix
from meow.eme.common import select_ports as select_ports
from meow.eme.saxify import compute_s_matrix_sax as compute_s_matrix_sax

compute_s_matrix = compute_s_matrix_sax
