"""Mode post-processing utilities."""

from collections.abc import Callable, Iterable

import numpy as np

from meow.mode import (
    Modes,
    inner_product,
    is_lossy_mode,
    is_pml_mode,
    normalize,
    zero_phase,
)


def post_process_modes(
    modes: Modes,
    inner_product: Callable = inner_product,
    gm_tolerance: float = 0.01,
) -> Modes:
    """Default post-processing pipeline after FDE.

    Args:
        modes: the modes to post process
        inner_product: the inner product with which to post-process the modes
        gm_tolerance: Gramm-Schmidt orthonormalization tolerance

    Returns:
        Filtered and orthonormalized modes.

    Notes:
        This is the default ``post_process`` used by ``compute_modes``. The
        choice of ``inner_product`` here matters downstream: if interface
        overlaps are later built with a different inner product, the resulting
        mode basis is no longer orthonormal in the interface metric.
    """
    return orthonormalize_modes(
        filter_modes(modes),
        inner_product,
        tolerance=gm_tolerance,
    )


def filter_modes(
    modes: Modes,
    conditions: Iterable[Callable] = (is_pml_mode, is_lossy_mode),
) -> Modes:
    """Filter a set of modes according to certain criteria.

    Args:
        modes: the list of modes to filter
        conditions: the conditions to filter the modes with

    Returns:
        the filtered modes
    """
    kept = []
    for mode in modes:
        for condition_fn in conditions:
            if condition_fn(mode):
                break
        else:
            kept.append(mode)
    return kept


def normalize_modes(modes: Modes, inner_product: Callable) -> Modes:
    """Self-normalize a set of modes.

    This only fixes the norm of each mode individually. It does not make the
    mode set mutually orthogonal. For overlap-based interface formulas, use
    :func:`orthonormalize_modes` if you need the simplified ``G = I`` metric.
    """
    return [zero_phase(normalize(m, inner_product)) for m in modes]


def orthonormalize_modes(
    modes: Modes,
    inner_product: Callable,
    *,
    tolerance: float = 0.01,
) -> Modes:
    """Gram-Schmidt orthonormalization with drop tolerance.

    Args:
        modes: the modes to orthonormalize
        inner_product: the inner product to orthonormalize them under
        tolerance: any mode that can't expand the basis beyond this tolerance
            will be dropped.

    Returns:
        Orthonormalized mode basis.

    Notes:
        This routine first self-normalizes each mode and then applies
        Gram-Schmidt. The ``inner_product`` passed here should generally be the
        same one used later to build interface overlaps; otherwise the final
        basis is orthonormal in the wrong metric.
    """
    if not modes:
        return []
    modes = normalize_modes(modes, inner_product)
    basis = []
    n_dropped = 0
    for mode in modes:
        current = mode
        for b in basis:
            current = current - (inner_product(b, current) / inner_product(b, b)) * b
        norm_sq = inner_product(current, current)
        if abs(norm_sq) < tolerance:
            n_dropped += 1
            continue
        basis.append(current / np.sqrt(norm_sq))
    return basis
