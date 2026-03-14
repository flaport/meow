"""Different solvers for Ax=b."""

import numpy as np


def tsvd_solve(
    A: np.ndarray, B: np.ndarray, rcond: float = 1e-3
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """TSVD Solve.

    TSVD-regularized linear solve for better numerical stability.
    We use a relative cutoff: s_cut = rcond * s_max.

    Args:
        A: the matrix to solve
        B: the vector to solve agains
        rcond: the condition to drop a singular value for.

    Returns:
        the solved X and some metadata.
    """
    U, s, Vh = np.linalg.svd(A, full_matrices=False)
    s_cut = rcond * float(s[0]) if s.size else 0.0
    keep = s >= s_cut
    if not np.any(keep):
        msg = "TSVD rejected all singular values; lower rcond."
        raise RuntimeError(msg)
    A_pinv = (Vh.conj().T[:, keep] * (1.0 / s[keep])) @ U.conj().T[keep, :]
    X = A_pinv @ B
    return X, s, s_cut, int(np.count_nonzero(keep))
