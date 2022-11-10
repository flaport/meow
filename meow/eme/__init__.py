""" EME Implementations & Backends """

from functools import wraps

from .common import compute_interface_s_matrices as compute_interface_s_matrices
from .common import compute_interface_s_matrix as compute_interface_s_matrix
from .common import compute_propagation_s_matrices as compute_propagation_s_matrices
from .common import compute_propagation_s_matrix as compute_propagation_s_matrix
from .common import select_ports as select_ports
from .sax import compute_s_matrix_sax as compute_s_matrix_sax


@wraps(compute_s_matrix_sax)
def compute_s_matrix(*args, **kwargs):
    return compute_s_matrix_sax(*args, **kwargs)
