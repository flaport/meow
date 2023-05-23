""" SAX backend for EME (default backend) """
from typing import Any, Dict, List

import numpy as np

from ..mode import Mode, inner_product


def compute_interface_s_matrix(
    modes1: List[Mode],
    modes2: List[Mode],
    enforce_reciprocity: bool = True,
    enforce_lossy_unitarity: bool = False,
):
    """get the S-matrix of the interface between two `CrossSection`s"""
    # overlap matrices
    NL, NR = len(modes1), len(modes2)
    O_LL = np.array([inner_product(modes1[m], modes1[m]) for m in range(NL)])
    O_RR = np.array([inner_product(modes2[n], modes2[n]) for n in range(NR)])
    O_LR = np.array(
        [[inner_product(modes1[m], modes2[n]) for n in range(NR)] for m in range(NL)]
    )
    O_RL = np.array(
        [[inner_product(modes2[m], modes1[n]) for n in range(NL)] for m in range(NR)]
    )

    # extra phase correction.

    # ignoring the phase seems to corresponds best with lumerical.
    O_LL = np.abs(O_LL)
    O_RR = np.abs(O_RR)

    # alternative phase correction (probably worth testing this out)
    # O_LL = O_LL*np.exp(-1j*np.angle(O_LL))
    # O_RR = O_RR*np.exp(-1j*np.angle(O_RR))

    # yet another alternative phase correction (probably worth testing this out too)
    # O_LR = O_LR@np.diag(np.exp(-1j*np.angle(np.diag(O_LR))))
    # O_RL = O_RL@np.diag(np.exp(-1j*np.angle(np.diag(O_RL))))

    # transmission L->R
    LHS = O_LR + O_RL.T
    RHS = np.diag(2 * O_LL)
    T_LR = np.linalg.solve(LHS, RHS)
    U, t, V = np.linalg.svd(T_LR)

    # HACK: we don't expect gain --> invert singular values that lead to gain
    # see: https://github.com/BYUCamachoLab/emepy/issues/12
    t = np.where(t > 1, 1 / t, t)

    T_LR = U @ np.diag(t) @ V

    # transmission R->L
    LHS = O_RL + O_LR.T
    RHS = np.diag(2 * O_RR)
    T_RL = np.linalg.solve(LHS, RHS)
    U, t, V = np.linalg.svd(T_RL)

    # HACK: we don't expect gain --> invert singular values that lead to gain
    t = np.where(t > 1, 1 / t, t)

    T_RL = U @ np.diag(t) @ V

    # reflection
    R_LR = np.diag(1 / (2 * O_LL)) @ (O_RL.T - O_LR) @ T_LR  # type: ignore
    R_RL = np.diag(1 / (2 * O_RR)) @ (O_LR.T - O_RL) @ T_RL  # type: ignore

    # s-matrix
    S = np.concatenate(
        [
            np.concatenate([R_LR, T_RL], 1),
            np.concatenate([T_LR, R_RL], 1),
        ],
        0,
    )

    # enforce S@S.H is diagonal
    if enforce_lossy_unitarity:  # HACK!
        U, s, V = np.linalg.svd(S)
        S = np.diag(s) @ U @ V

    # ensure reciprocity:
    if enforce_reciprocity:
        S = 0.5 * (S + S.T)

    # create port map
    in_ports = [f"left@{i}" for i in range(len(modes1))]
    out_ports = [f"right@{i}" for i in range(len(modes2))]
    port_map = {p: i for i, p in enumerate(in_ports + out_ports)}

    return S, port_map


def compute_interface_s_matrices(
    modes: List[List[Mode]],
    enforce_reciprocity: bool = True,
    enforce_lossy_unitarity: bool = False,
):
    """get all the S-matrices of all the interfaces in a collection of `CrossSections`"""
    return {
        f"i_{i}_{i + 1}": compute_interface_s_matrix(
            modes1=modes1,
            modes2=modes2,
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
