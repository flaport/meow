"""Different solvers for Ax=b."""

import numpy as np


def tsvd_solve(
    A: np.ndarray, B: np.ndarray, rcond: float = 1e-3
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Solve ``A X = B`` with a truncated-SVD pseudoinverse.

    This is the regularized solve used by the EME interface construction. The
    smallest singular directions of ``A`` are the first place where truncated
    mode sets become numerically unstable, so the solve is performed as

    ``X = pinv_tsvd(A) @ B``

    with a relative cutoff

    ``s_cut = rcond * s_max``.

    Singular values below ``s_cut`` are discarded entirely.

    High-level behavior:
        - small ``rcond`` keeps more singular directions and is closer to the
          exact inverse, but is more sensitive to ill-conditioning;
        - large ``rcond`` drops more singular directions and is more stable,
          but biases the result towards a lower-rank approximation.

    This routine does not enforce passivity by itself. It only stabilizes the
    linear inversion. Passivity is handled afterwards on the assembled
    interface S-matrix.

    Args:
        A: System matrix to invert approximately.
        B: Right-hand side.
        rcond: Relative singular-value cutoff. Singular values smaller than
            ``rcond * max(s)`` are discarded.

    Returns:
        A tuple ``(X, s, s_cut, rank_kept)`` containing the solved result,
        the singular values of ``A``, the absolute cutoff, and the number of
        retained singular directions.

    Raises:
        RuntimeError: If all singular values are rejected.
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
