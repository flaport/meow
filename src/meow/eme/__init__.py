"""EME Implementations & Backends."""

from .common import compute_interface_s_matrices as compute_interface_s_matrices
from .common import compute_interface_s_matrix as compute_interface_s_matrix
from .common import compute_propagation_s_matrices as compute_propagation_s_matrices
from .common import compute_propagation_s_matrix as compute_propagation_s_matrix
from .common import select_ports as select_ports
from .sax import compute_s_matrix_sax as compute_s_matrix_sax

compute_s_matrix = compute_s_matrix_sax
