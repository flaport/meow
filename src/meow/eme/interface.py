"""Interface S-matrix between two sets of modes."""

import inspect
import warnings
from collections.abc import Callable
from functools import partial
from itertools import pairwise
from typing import Literal, TypeAlias

import jax.numpy as jnp
import numpy as np
import sax

from meow.eme.solve import tsvd_solve
from meow.mode import Modes, inner_product

PassivityMethod: TypeAlias = Literal["none", "clip", "invert", "subtract"]
"""Method for enforcing S-matrix passivity."""


def overlap_matrix(
    modes1: Modes, modes2: Modes, inner_product: Callable
) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
    """Build the modal overlap matrix between two mode sets.

    The entry ``M[i, j]`` is ``inner_product(modes1[i], modes2[j])``. In the
    interface derivation this is used both for self-overlaps (the metric ``G``)
    and cross-overlaps between the left and right waveguide bases.

    Sharp edge:
        The meaning of the overlap matrix is entirely determined by the
        ``inner_product`` callable. If modes were orthonormalized with one
        inner product but overlaps are formed with another, the simplified
        ``G = I`` interface formulas are no longer valid.

    Args:
        modes1: Left/test mode set.
        modes2: Right/expansion mode set.
        inner_product: Modal inner-product callable.

    Returns:
        Complex overlap matrix of shape ``(len(modes1), len(modes2))``.
    """
    M = np.zeros((len(modes1), len(modes2)), dtype=np.complex128)
    for i, ma in enumerate(modes1):
        for j, mb in enumerate(modes2):
            M[i, j] = inner_product(ma, mb)
    return M


def compute_interface_s_matrix(
    modes1: Modes,
    modes2: Modes,
    *,
    inner_product: Callable = inner_product,
    conjugate: bool | None = None,
    tsvd_rcond: float = 1e-3,
    passivity_method: PassivityMethod = "invert",
    enforce_reciprocity: bool = True,
    ignore_warnings: bool = True,
) -> sax.SDenseMM:
    """Compute the interface S-matrix between two modal bases.

    This implements the overlap-based EME interface solve for a step
    discontinuity. The transmission blocks are obtained from a TSVD-regularized
    solve, then the reflection blocks are reconstructed from the two continuity
    equations and averaged:

    - ``R_LL = 0.5 * ((O_RL^adj @ T_LR - I) + (I - O_LR @ T_LR))``
    - ``R_RR = 0.5 * ((O_LR^adj @ T_RL - I) + (I - O_RL @ T_RL))``

    High-level assumptions:
        - the supplied mode sets are already orthonormalized in the same metric
          used by ``inner_product``;
        - the interface formulas are therefore used in their simplified
          ``G = I`` form;
        - any remaining non-passivity is treated as a numerical/truncation
          issue and corrected afterwards via singular-value processing.

    Sharp edges:
        - If modes were orthonormalized with a different inner product than the
          one passed here, the result is not physically meaningful.
        - ``conjugate`` must match the intended overlap convention. If omitted,
          it is inferred from ``inner_product``.
        - ``tsvd_rcond`` controls a stability/accuracy tradeoff. Too small can
          amplify ill-conditioned directions; too large can discard physically
          relevant channels.
        - passivity correction modifies the raw interface solve. This is often
          necessary for truncated bases, but it is still a model correction, not
          a proof that the original solve was complete.

    Args:
        modes1: Modes on the left side of the interface.
        modes2: Modes on the right side of the interface.
        inner_product: Inner-product callable used to form the overlap matrices.
            This must be consistent with how the modes were orthonormalized.
        conjugate: Whether to use the conjugated (power-conserving) formulation.
            If None, inferred from the inner_product callable. Must match the
            conjugate setting used in inner_product for physical consistency:
            - conjugate=True: uses O_RL.conj().T (Hermitian transpose)
            - conjugate=False: uses O_RL.T (transpose)
        tsvd_rcond: Relative singular-value cutoff for the TSVD solve used to
            build the transmission blocks.
        passivity_method: Method for enforcing passivity
            ("none", "clip", "invert", "subtract").
        enforce_reciprocity: Whether to symmetrize the final S-matrix as
            ``0.5 * (S + S.T)``.
        ignore_warnings: Whether to suppress numerical warnings.

    Returns:
        A tuple ``(S, port_map)`` in SAX dense-matrix format.

    Notes:
        The default path is internally consistent because the default
        ``meow.mode.inner_product`` is the asymmetric, unconjugated overlap and
        the default mode post-processing uses that same callable. If you change
        the interface inner product, you should generally use the same callable
        during mode post-processing.
    """
    if ignore_warnings:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*divide by zero.*")
            warnings.filterwarnings("ignore", message=".*overflow encountered.*")
            warnings.filterwarnings("ignore", message=".*invalid value.*")
            return compute_interface_s_matrix(
                modes1=modes1,
                modes2=modes2,
                inner_product=inner_product,
                conjugate=conjugate,
                tsvd_rcond=tsvd_rcond,
                passivity_method=passivity_method,
                enforce_reciprocity=enforce_reciprocity,
                ignore_warnings=False,
            )

    # Infer conjugate from inner_product if not explicitly provided
    if conjugate is None:
        conjugate = _infer_conjugate(inner_product)

    # Supports unequal number of modes on left and right
    N_L, N_R = len(modes1), len(modes2)

    # Overlap matrices: O_LR is (N_L, N_R), O_RL is (N_R, N_L)
    O_LR = overlap_matrix(modes1, modes2, inner_product)
    O_RL = overlap_matrix(modes2, modes1, inner_product)

    I_L = np.eye(N_L)
    I_R = np.eye(N_R)

    # Use Hermitian transpose (.conj().T) for conjugated inner product,
    # plain transpose (.T) for unconjugated inner product.
    # This matches the Lorentz reciprocity relation for each formulation.
    O_RL_adj = O_RL.conj().T if conjugate else O_RL.T
    O_LR_adj = O_LR.conj().T if conjugate else O_LR.T

    # T_LR: (N_R, N_L) — maps N_L left inputs to N_R right outputs
    # Solve: A_LR @ T_LR = 2 * I_L, where A_LR is (N_L, N_R)
    # tsvd_solve(A, B) returns pinv(A) @ B
    # pinv(A_LR) is (N_R, N_L), so result is (N_R, N_L) @ (N_L, N_L) = (N_R, N_L)
    A_LR = O_LR + O_RL_adj  # (N_L, N_R)
    T_LR, *_ = tsvd_solve(A_LR, 2.0 * I_L, rcond=tsvd_rcond)  # (N_R, N_L)

    # T_RL: (N_L, N_R) — maps N_R right inputs to N_L left outputs
    # pinv(A_RL) is (N_L, N_R), so result is (N_L, N_R) @ (N_R, N_R) = (N_L, N_R)
    A_RL = O_RL + O_LR_adj  # (N_R, N_L)
    T_RL, *_ = tsvd_solve(A_RL, 2.0 * I_R, rcond=tsvd_rcond)  # (N_L, N_R)

    # Compute R from both continuity equations and average; this reduces sensitivity
    # to cancellation relative to directly forming (O_RL^adj - O_LR).
    # R_LL: (N_L, N_L)
    R_LL_e = O_RL_adj @ T_LR - I_L  # (N_L, N_R) @ (N_R, N_L) = (N_L, N_L)
    R_LL_h = I_L - O_LR @ T_LR  # (N_L, N_R) @ (N_R, N_L) = (N_L, N_L)
    R_LL = 0.5 * (R_LL_e + R_LL_h)

    # R_RR: (N_R, N_R)
    R_RR_e = O_LR_adj @ T_RL - I_R  # (N_R, N_L) @ (N_L, N_R) = (N_R, N_R)
    R_RR_h = I_R - O_RL @ T_RL  # (N_R, N_L) @ (N_L, N_R) = (N_R, N_R)
    R_RR = 0.5 * (R_RR_e + R_RR_h)

    # Full S-matrix: [a-; b+] = S [a+; b-]
    # Shape: (N_L + N_R, N_L + N_R)
    # Block structure:
    #   [[R_LL (N_L, N_L),  T_RL (N_L, N_R)],
    #    [T_LR (N_R, N_L),  R_RR (N_R, N_R)]]
    S = np.block(
        [
            [R_LL, T_RL],
            [T_LR, R_RR],
        ]
    )

    # Passivity enforcement via SVD
    U, sigma, Vh = np.linalg.svd(S, full_matrices=False)
    sigma_corrected = enforce_passivity(sigma, method=passivity_method)
    S = (U * sigma_corrected) @ Vh

    if enforce_reciprocity:
        S = 0.5 * (S + S.T)

    in_ports = [f"left@{i}" for i in range(N_L)]
    out_ports = [f"right@{i}" for i in range(N_R)]
    port_map = {p: i for i, p in enumerate(in_ports + out_ports)}
    return jnp.asarray(S), port_map


def compute_interface_s_matrices(
    modes: list[Modes],
    *,
    inner_product: Callable = inner_product,
    conjugate: bool | None = None,
    tsvd_rcond: float = 1e-3,
    passivity_method: PassivityMethod = "invert",
    enforce_reciprocity: bool = True,
    ignore_warnings: bool = True,
) -> dict[str, sax.SDenseMM]:
    """Compute interface S-matrices for each adjacent pair of mode sets.

    This is a thin wrapper around :func:`compute_interface_s_matrix` applied to
    every neighboring pair in ``modes``.

    The same sharp edges apply here as for the single-interface solve:
    orthonormalization, inner-product choice, TSVD cutoff, and passivity method
    must all be interpreted consistently across the full stack.
    """
    return {
        f"i_{i}_{i + 1}": compute_interface_s_matrix(
            modes1=modes1,
            modes2=modes2,
            inner_product=inner_product,
            conjugate=conjugate,
            tsvd_rcond=tsvd_rcond,
            passivity_method=passivity_method,
            enforce_reciprocity=enforce_reciprocity,
            ignore_warnings=ignore_warnings,
        )
        for i, (modes1, modes2) in enumerate(pairwise(modes))
    }


def enforce_passivity(
    singular_values: np.ndarray, *, method: PassivityMethod = "invert"
) -> np.ndarray:
    """Project singular values onto a passive interval.

    This function post-processes the singular values of an S-matrix so that the
    resulting matrix has singular values no larger than one.

    Available methods:
        - ``"none"``: leave the singular values unchanged.
        - ``"clip"``: replace values larger than one by exactly one.
        - ``"invert"``: replace ``sigma`` by ``1 / sigma`` when ``sigma > 1``.
        - ``"subtract"``: replace ``sigma`` by ``max(0, 2 - sigma)`` when
          ``sigma > 1``.

    High-level interpretation:
        ``clip`` is the most conservative algebraic projection. ``invert`` and
        ``subtract`` both map modest gain above one to modest loss below one,
        which can be useful when the excess gain is interpreted as missing
        radiation channels in a truncated basis.

    Sharp edge:
        These are heuristic corrections on top of a truncated modal solve. They
        are often useful, but they are not substitutes for a converged basis.

    Args:
        singular_values: Singular values to correct.
        method: Passivity enforcement strategy.

    Returns:
        Corrected singular values.

    Raises:
        ValueError: If ``method`` is unknown.
    """
    match method:
        case "none":
            return singular_values
        case "clip":
            return np.where(singular_values > 1.0, 1.0, singular_values)
        case "invert":
            return np.where(singular_values > 1.0, 1 / singular_values, singular_values)
        case "subtract":
            singular_values = np.where(
                singular_values > 1.0, 2.0 - singular_values, singular_values
            )
            return np.where(singular_values < 0.0, 0.0, singular_values)
        case _:
            msg = f"Unknown passivity enforcement method {method}."
            raise ValueError(msg)


def _infer_conjugate(ip: Callable) -> bool:
    """Infer the ``conjugate`` flag from an inner-product callable.

    Works with functools.partial wrapping meow.mode.inner_product.
    Raises ValueError if the setting cannot be determined.
    """
    # Handle functools.partial wrapping meow.mode.inner_product
    if isinstance(ip, partial):
        is_ip = ip.func is inner_product
        is_ip = is_ip or getattr(ip.func, "__name__", None) == "inner_product"
        if is_ip:
            # Look for 'conjugate' in keywords
            if "conjugate" in ip.keywords:
                return bool(ip.keywords["conjugate"])
            # Not explicitly set, use default (False)
            return False

    # Check if it's the inner_product function itself (not wrapped)
    if ip is inner_product:
        params = inspect.signature(inner_product).parameters
        if "conjugate" in params:
            param = params["conjugate"]
            if param.default is True:
                return True
            if param.default is False:
                return False

    # Cannot determine - raise error
    msg = (
        "Cannot infer 'conjugate' setting from inner_product callable. "
        "Please explicitly pass conjugate=True or conjugate=False to "
        "compute_interface_s_matrix."
    )
    raise ValueError(msg)
