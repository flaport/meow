""" SAX backend for EME (default backend) """

from typing import Any, Dict, List

import numpy as np

from ..mode import Mode
from ..mode import inner_product as inner_product_normal
from ..mode import inner_product_conj

DEFAULT_CONJUGATE_TRANSPOSE = True
DEFAULT_ENFORCE_RECIPROCITY = False
DEFAULT_ENFORCE_LOSSY_UNITARITY = False


def compute_interface_s_matrix(
    modes1: List[Mode],
    modes2: List[Mode],
    conjugate_transpose: bool = DEFAULT_CONJUGATE_TRANSPOSE,
    enforce_reciprocity: bool = DEFAULT_ENFORCE_RECIPROCITY,
    enforce_lossy_unitarity: bool = DEFAULT_ENFORCE_LOSSY_UNITARITY,
):
    """get the S-matrix of the interface between two `CrossSection`s"""
    # overlap matrices
    inner_product = inner_product_conj if conjugate_transpose else inner_product_normal
    conjugate = np.conj if conjugate_transpose else lambda a: a

    NL, NR = len(modes1), len(modes2)
    O_LL = np.array([inner_product(modes1[m], modes1[m]) for m in range(NL)])
    O_RR = np.array([inner_product(modes2[n], modes2[n]) for n in range(NR)])
    O_LR = np.array([[inner_product(modes1[m], modes2[n]) for n in range(NR)] for m in range(NL)])  # fmt: skip
    O_RL = np.array([[inner_product(modes2[m], modes1[n]) for n in range(NL)] for m in range(NR)])  # fmt: skip

    # extra phase correction (disabled?).

    # if conjugate_transpose:
    #    O_LL = np.real(O_LL)
    #    O_RR = np.real(O_RR)

    # ignoring the phase seems to corresponds best with lumerical.

    # alternative phase correction (probably worth testing this out)
    # Question: is this not just a conjugation?
    # O_LL = O_LL*np.exp(-1j*np.angle(O_LL))
    # O_RR = O_RR*np.exp(-1j*np.angle(O_RR))

    # yet another alternative phase correction (probably worth testing this out too)
    # O_LR = O_LR*np.diag(np.exp(-1j*np.angle(np.diag(O_LR))))
    # O_RL = O_RL*np.diag(np.exp(-1j*np.angle(np.diag(O_RL))))

    # transmission L->R
    LHS = conjugate(O_LR) + O_RL.T
    RHS = np.diag(2 * O_LL)

    # print(f"LHS: {LHS}")
    # vis(LHS)

    # print(f"RHS: {RHS}")
    # vis(RHS)

    T_LR, _, _, _ = np.linalg.lstsq(LHS, RHS, rcond=None)

    # HACK: we don't expect gain --> invert singular values that lead to gain
    # see: https://github.com/BYUCamachoLab/emepy/issues/12
    U, t, V = np.linalg.svd(T_LR, full_matrices=False)
    t = np.where(t > 1, 1 / t, t)
    T_LR = U @ np.diag(t) @ V

    # transmission R->L
    LHS = conjugate(O_RL) + O_LR.T
    RHS = np.diag(2 * O_RR)
    T_RL, _, _, _ = np.linalg.lstsq(LHS, RHS, rcond=None)

    # HACK: we don't expect gain --> invert singular values that lead to gain
    U, t, V = np.linalg.svd(T_RL, full_matrices=False)
    t = np.where(t > 1, 1 / t, t)
    T_RL = U @ np.diag(t) @ V

    # reflection
    R_LR = np.diag(1 / (2 * O_LL)) @ (O_RL.T - conjugate(O_LR)) @ T_LR  # type: ignore
    R_RL = np.diag(1 / (2 * O_RR)) @ (O_LR.T - conjugate(O_RL)) @ T_RL  # type: ignore

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

    return S, port_map


def compute_interface_s_matrices(
    modes: List[List[Mode]],
    conjugate_transpose: bool = DEFAULT_CONJUGATE_TRANSPOSE,
    enforce_reciprocity: bool = DEFAULT_ENFORCE_RECIPROCITY,
    enforce_lossy_unitarity: bool = DEFAULT_ENFORCE_LOSSY_UNITARITY,
):
    """get all the S-matrices of all the interfaces in a collection of `CrossSections`"""
    return {
        f"i_{i}_{i + 1}": compute_interface_s_matrix(
            modes1=modes1,
            modes2=modes2,
            conjugate_transpose=conjugate_transpose,
            enforce_reciprocity=enforce_reciprocity,
            enforce_lossy_unitarity=enforce_lossy_unitarity,
        )
        for i, (modes1, modes2) in enumerate(zip(modes[:-1], modes[1:]))
    }


def compute_propagation_s_matrix(modes: List[Mode]):
    """get the propagation S-matrix of each `Mode` belonging to a `CrossSection` in a `Cell` with a certain length."""
    s_dict = {
        (f"left@{i}", f"right@{i}"): np.exp(
            2j * np.pi * mode.neff / mode.env.wl * mode.cell.length
        )
        for i, mode in enumerate(modes)
    }
    s_dict = {**s_dict, **{(p2, p1): v for (p1, p2), v in s_dict.items()}}
    return s_dict


def compute_propagation_s_matrices(modes: List[List[Mode]]):
    """get all the propagation S-matrices of all the `Modes` belonging to each `CrossSection`"""
    return {
        f"p_{i}": compute_propagation_s_matrix(modes_) for i, modes_ in enumerate(modes)
    }


def select_ports(
    S: np.ndarray[Any, np.dtype[np.float_]], port_map: Dict[str, int], ports: List[str]
):
    """Keep subset of an S-matrix

    Args:
        S: the S-matrix to downselect from
        port_map: a port name to s-matrix index mapping
        ports: the port names to keep

    Returns:
        the downselected s-matrix and port map
    """
    idxs = np.array([port_map[port] for port in ports], dtype=np.int_)
    s = S[idxs, :][:, idxs]
    new_port_map = {p: i for i, p in enumerate(ports)}
    return s, new_port_map
