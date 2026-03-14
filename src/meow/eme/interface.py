"""Interface S-matrix between two sets of modes."""

from collections.abc import Callable
from typing import Literal, TypeAlias

import numpy as np

from meow.eme.solve import tsvd_solve
from meow.mode import Modes

PassivityMethod: TypeAlias = Literal["none", "clip", "invert", "subtract"]


def overlap_matrix(
    modes1: Modes, modes2: Modes, inner_product: Callable
) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
    """Overlap matrix between two sets of modes in a different basis."""
    M = np.zeros((len(modes1), len(modes2)), dtype=np.complex128)
    for i, ma in enumerate(modes1):
        for j, mb in enumerate(modes2):
            M[i, j] = inner_product(ma, mb)
    return M


def interface_smatrix(
    modes1: Modes,
    modes2: Modes,
    inner_product: Callable,
    *,
    tsvd_rcond: float = 1e-3,
    passivity_method: PassivityMethod = "invert",
) -> np.ndarray:
    """Get the interface S-matrix."""
    # TODO: this should work mL != mR too, no?
    N = min(len(modes1), len(modes2))
    mL = modes1[:N]
    mR = modes2[:N]
    O_LR = overlap_matrix(mL, mR, inner_product)
    O_RL = overlap_matrix(mR, mL, inner_product)

    I_N = np.eye(N)

    A_LR = O_LR + O_RL.T
    T_LR, *_ = tsvd_solve(A_LR, 2.0 * I_N, rcond=tsvd_rcond)

    A_RL = O_RL + O_LR.T
    T_RL, *_ = tsvd_solve(A_RL, 2.0 * I_N, rcond=tsvd_rcond)

    # Compute R from both continuity equations and average; this reduces sensitivity
    # to cancellation relative to directly forming (O_RL^T - O_LR).
    R_LL_e = O_RL.T @ T_LR - I_N
    R_LL_h = I_N - O_LR @ T_LR
    R_LL = 0.5 * (R_LL_e + R_LL_h)

    R_RR_e = O_LR.T @ T_RL - I_N
    R_RR_h = I_N - O_RL @ T_RL
    R_RR = 0.5 * (R_RR_e + R_RR_h)

    # Full S-matrix: [a-; b+] = S [a+; b-]
    S = np.block(
        [
            [R_LL, T_RL],
            [T_LR, R_RR],
        ]
    )
    return enforce_passivity(S, method=passivity_method)


def enforce_passivity(
    singular_values: np.ndarray, *, method: PassivityMethod = "invert"
) -> np.ndarray:
    """Enforce passivity."""
    match method:
        case "none":
            return singular_values
        case "clip":
            return np.where(singular_values > 1.0, 1.0, singular_values)
        case "invert":
            return np.where(singular_values > 1.0, 1 / singular_values, singular_values)
        case "subtract":
            return np.where(
                singular_values > 1.0, 2.0 - singular_values, singular_values
            )
        case _:
            msg = f"Unknown passivity enforcement method {method}."
            raise ValueError(msg)
