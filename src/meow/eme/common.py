"""SAX backend for EME (default backend)."""

from __future__ import annotations

import warnings
from itertools import pairwise
from typing import Any, cast

import numpy as np
import sax

from ..cell import Cell
from ..mode import Mode, inner_product_conj
from ..mode import inner_product as inner_product_normal

DEFAULT_CONJUGATE = True
DEFAULT_ENFORCE_RECIPROCITY = False
DEFAULT_ENFORCE_LOSSY_UNITARITY = False


def compute_interface_s_matrix(
    modes1: list[Mode],
    modes2: list[Mode],
    *,
    conjugate: bool = DEFAULT_CONJUGATE,
    enforce_reciprocity: bool = DEFAULT_ENFORCE_RECIPROCITY,
    enforce_lossy_unitarity: bool = DEFAULT_ENFORCE_LOSSY_UNITARITY,
    ignore_warnings: bool = True,
) -> sax.SDenseMM:
    """Get the S-matrix of the interface between two `CrossSection` objects."""
    if ignore_warnings:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*divide by zero.*")
            warnings.filterwarnings("ignore", message=".*overflow encountered.*")
            warnings.filterwarnings("ignore", message=".*invalid value.*")
            return compute_interface_s_matrix(
                modes1=modes1,
                modes2=modes2,
                conjugate=conjugate,
                enforce_reciprocity=enforce_reciprocity,
                enforce_lossy_unitarity=enforce_lossy_unitarity,
                ignore_warnings=False,
            )

    # overlap matrices
    inner_product = inner_product_conj if conjugate else inner_product_normal
    conj = np.conj if conjugate else lambda a: a

    NL, NR = len(modes1), len(modes2)
    O_LL = np.array([inner_product(modes1[m], modes1[m]) for m in range(NL)])
    O_RR = np.array([inner_product(modes2[n], modes2[n]) for n in range(NR)])
    O_LR = np.array(
        [[inner_product(modes1[m], modes2[n]) for n in range(NR)] for m in range(NL)]
    )
    O_RL = np.array(
        [[inner_product(modes2[m], modes1[n]) for n in range(NL)] for m in range(NR)]
    )

    # additional phase correction (disabled?).

    if conjugate:
        O_LL = np.real(O_LL)
        O_RR = np.real(O_RR)

    # ignoring the phase seems to corresponds best with lumerical.

    # alternative phase correction (probably worth testing this out)
    # Question: is this not just an abs ;) ?
    # O_LL = O_LL*np.exp(-1j*np.angle(O_LL))
    # O_RR = O_RR*np.exp(-1j*np.angle(O_RR))

    # yet another alternative phase correction (probably worth testing this out too)
    # O_LR = O_LR*np.diag(np.exp(-1j*np.angle(np.diag(O_LR))))
    # O_RL = O_RL*np.diag(np.exp(-1j*np.angle(np.diag(O_RL))))

    # transmission L->R
    LHS = conj(O_LR) + O_RL.T
    RHS = np.diag(2 * O_LL)

    T_LR, _, _, _ = np.linalg.lstsq(LHS, RHS, rcond=None)

    # HACK: we don't expect gain --> invert singular values that lead to gain
    # see: https://github.com/BYUCamachoLab/emepy/issues/12
    U, t, V = np.linalg.svd(T_LR, full_matrices=False)
    t = np.where(t > 1, 1 / t, t)
    T_LR = U @ np.diag(t) @ V

    # transmission R->L
    LHS = conj(O_RL) + O_LR.T
    RHS = np.diag(2 * O_RR)
    T_RL, _, _, _ = np.linalg.lstsq(LHS, RHS, rcond=None)

    # HACK: we don't expect gain --> invert singular values that lead to gain
    U, t, V = np.linalg.svd(T_RL, full_matrices=False)
    t = np.where(t > 1, 1 / t, t)
    T_RL = U @ np.diag(t) @ V

    # reflection
    R_LR = np.diag(1 / (2 * O_LL)) @ (O_RL.T - conj(O_LR)) @ T_LR
    R_RL = np.diag(1 / (2 * O_RR)) @ (O_LR.T - conj(O_RL)) @ T_RL

    # s-matrix
    S = np.concatenate(
        [
            np.concatenate([R_LR, T_RL], 1),
            np.concatenate([T_LR, R_RL], 1),
        ],
        0,
    )

    # enforce S@S.H is diagonal: HACK!
    if enforce_lossy_unitarity:
        U, s, V = np.linalg.svd(S)
        S = np.diag(s) @ U @ V

    # ensure reciprocity: HACK?
    if enforce_reciprocity:
        S = 0.5 * (S + S.T)

    # create port map
    in_ports = [f"left@{i}" for i in range(len(modes1))]
    out_ports = [f"right@{i}" for i in range(len(modes2))]
    port_map = {p: i for i, p in enumerate(in_ports + out_ports)}

    return cast(sax.SDenseMM, (S, port_map))


def compute_interface_s_matrices(
    modes: list[list[Mode]],
    *,
    conjugate: bool = DEFAULT_CONJUGATE,
    enforce_reciprocity: bool = DEFAULT_ENFORCE_RECIPROCITY,
    enforce_lossy_unitarity: bool = DEFAULT_ENFORCE_LOSSY_UNITARITY,
) -> dict[str, sax.SDenseMM]:
    """Get all the S-matrices of all the interfaces between `CrossSection` objects."""
    return {
        f"i_{i}_{i + 1}": compute_interface_s_matrix(
            modes1=modes1,
            modes2=modes2,
            conjugate=conjugate,
            enforce_reciprocity=enforce_reciprocity,
            enforce_lossy_unitarity=enforce_lossy_unitarity,
        )
        for i, (modes1, modes2) in enumerate(pairwise(modes))
    }


def compute_propagation_s_matrix(modes: list[Mode], cell_length: float) -> sax.SDictMM:
    """Get the propagation S-matrix of each `Mode`."""
    s_dict = {
        (f"left@{i}", f"right@{i}"): np.exp(
            2j * np.pi * mode.neff / mode.env.wl * cell_length
        )
        for i, mode in enumerate(modes)
    }
    s_dict = {**s_dict, **{(p2, p1): v for (p1, p2), v in s_dict.items()}}
    return cast(sax.SDictMM, s_dict)


def compute_propagation_s_matrices(
    modes: list[list[Mode]],
    cells: list[Cell] | None = None,
    cell_lengths: list[float] | None = None,
) -> dict[str, sax.SDictMM]:
    """Get all the propagation S-matrices of all the `Modes`."""
    if cells is None and cell_lengths is None:
        msg = (
            "Please specify one of both when calculating the S-matrix: "
            "`cells` or `cell_lengths`."
        )
        raise ValueError(msg)
    if cells is not None and cell_lengths is not None:
        msg = "Please specify EITHER `cells` OR `cell_lengths` (not both)."
        raise ValueError(msg)

    if cells:
        cell_lengths = [cell.length for cell in cells]

    if cell_lengths is None:
        msg = (
            "The given cells do not have a length attribute: "
            "Please specify `cell_lengths`."
        )
        raise ValueError(msg)

    if len(cell_lengths) != len(modes):
        msg = f"len(cell_lengths) != len(modes): {len(cell_lengths)} != {len(modes)}"
        raise ValueError(msg)
    return {
        f"p_{i}": compute_propagation_s_matrix(modes_, cell_length=cell_length)
        for i, (modes_, cell_length) in enumerate(zip(modes, cell_lengths, strict=True))
    }


def select_ports(
    S: np.ndarray[Any, np.dtype[np.float64]],
    port_map: dict[str, int],
    ports: list[str],
) -> sax.SDenseMM:
    """Keep subset of an S-matrix."""
    idxs = np.array([port_map[port] for port in ports], dtype=np.int_)
    s = S[idxs, :][:, idxs]
    new_port_map = {p: i for i, p in enumerate(ports)}
    return cast(sax.SDenseMM, (s, new_port_map))
