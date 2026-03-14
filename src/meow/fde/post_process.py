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

    Note to orthogonalize them too, use 'orthonormalize'.

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
