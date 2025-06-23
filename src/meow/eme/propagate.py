"""Propagating fields throug devices."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import sax
from sax.backends import circuit_backends

from meow.arrays import ComplexArray1D, ComplexArray2D, FloatArray1D
from meow.cell import Cell
from meow.eme import (
    compute_interface_s_matrices,
    compute_interface_s_matrix,
    compute_propagation_s_matrices,
)
from meow.mode import Mode


def _connect_two(
    l: sax.STypeMM, r: sax.STypeMM, sax_backend: sax.Backend
) -> sax.STypeMM:
    # TODO: there must be an easier way to do this...
    _s_l, p_l = sax.sdense(l)
    _s_r, p_r = sax.sdense(r)
    instances: dict[str, sax.STypeMM] = {"l": l, "r": r}
    p_lr = [p for p in p_l if "right" in p]  # right ports of left
    p_rl = [p for p in p_r if "left" in p]  # left ports of right

    p_ll = [p for p in p_l if "left" in p]  # left ports of left
    p_rr = [p for p in p_r if "right" in p]  # right ports of right

    p_lr.sort()
    p_rl.sort()
    connections = {f"l,{pl}": f"r,{pr}" for pl, pr in zip(p_lr, p_rl, strict=False)}
    ports = {**{p: f"l,{p}" for p in p_ll}, **{p: f"r,{p}" for p in p_rr}}
    net = {
        "instances": instances,
        "connections": connections,
        "ports": ports,
    }
    _, analyze_circuit, evaluate_circuit = circuit_backends[sax_backend]
    net["instances"] = {k: sax.scoo(v) for k, v in net["instances"].items()}
    analyzed = analyze_circuit(net["instances"], net["connections"], net["ports"])
    return evaluate_circuit(analyzed, net["instances"])


def pi_pairs(
    propagations: dict[str, sax.SDictMM],
    interfaces: dict[str, sax.SDenseMM],
    sax_backend: sax.Backend,
) -> list[sax.STypeMM]:
    """Generates the S-matrices of cells."""
    S = []
    for i in range(len(propagations)):
        p = propagations[f"p_{i}"]
        if i == len(interfaces):
            S.append(p)
        else:
            c = interfaces[f"i_{i}_{i + 1}"]
            S.append(_connect_two(p, c, sax_backend))

    return S


def l2r_matrices(
    pairs: list[sax.STypeMM], identity: sax.SDenseMM, sax_backend: sax.Backend
) -> list[sax.STypeMM]:
    """Left to right S-matrices."""
    Ss: list[sax.STypeMM] = [identity]

    for p in pairs[:-1]:
        Ss.append(_connect_two(Ss[-1], p, sax_backend))

    return Ss


def r2l_matrices(
    pairs: list[sax.STypeMM], sax_backend: sax.Backend
) -> list[sax.STypeMM]:
    """Right to left S-matrices."""
    Ss = [pairs[-1]]

    for p in pairs[-1::-1]:
        Ss.append(_connect_two(p, Ss[-1], sax_backend))

    return Ss[::-1]


def split_square_matrix(
    matrix: ComplexArray2D, idx: int
) -> tuple[
    tuple[ComplexArray2D, ComplexArray2D], tuple[ComplexArray2D, ComplexArray2D]
]:
    """Split a square matrix into its four submatrices."""
    if matrix.shape[0] != matrix.shape[1]:
        msg = "Matrix has to be square"
        raise ValueError(msg)
    return (matrix[:idx, :idx], matrix[:idx, idx:]), (
        matrix[idx:, :idx],
        matrix[idx:, idx:],
    )


def propagate(
    l2rs: list[sax.STypeMM],
    r2ls: list[sax.STypeMM],
    excitation_l: ComplexArray1D,
    excitation_r: ComplexArray1D,
) -> tuple[list[ComplexArray1D], list[ComplexArray1D]]:
    """Propagate the modes through the S-matrices."""
    forwards = []
    backwards = []
    for l2r, r2l in zip(l2rs, r2ls, strict=False):
        s_l2r, p = sax.sdense(l2r)
        s_r2l, _ = sax.sdense(r2l)
        m = len([k for k in p if "right" in k])
        f, b = compute_mode_amplitudes(
            np.asarray(s_l2r), np.asarray(s_r2l), m, excitation_l, excitation_r
        )
        forwards.append(f)
        backwards.append(b)
    return forwards, backwards


def compute_mode_amplitudes(
    u: ComplexArray2D,
    v: ComplexArray2D,
    m: int,
    excitation_l: ComplexArray1D,
    excitation_r: ComplexArray1D,
) -> tuple[ComplexArray1D, ComplexArray1D]:
    """Compute the mode amplitudes for the left and right propagating modes."""
    n = u.shape[0] - m
    _, [u21, u22] = split_square_matrix(u, n)
    [v11, v12], _ = split_square_matrix(v, m)

    RHS = u21 @ excitation_l + u22 @ v12 @ excitation_r
    LHS = np.diag(np.ones(m)) - u22 @ v11
    forward = np.linalg.solve(LHS, RHS)
    backward = v12 @ excitation_r + v11 @ forward  # Attention v21 was v12
    return forward, backward


def plot_fields(
    modes: list[list[Mode]],
    cells: list[Cell],
    forwards: list[ComplexArray1D],
    backwards: list[ComplexArray2D],
    y: float,
    z: FloatArray1D,
) -> tuple[ComplexArray2D, FloatArray1D]:
    """Plot the fields of the propagated modes."""
    mode_set = modes[0]
    mesh_y = cells[0].mesh.y
    mesh_x = cells[0].mesh.x
    mesh_x = mesh_x[:-1] + np.diff(mesh_x) / 2
    i_y = np.argmin(np.abs(mesh_y - y))

    E_tot = np.zeros((len(z), len(mesh_x)), dtype=complex)
    for mode_set, forward, backward, cell in zip(
        modes, forwards, backwards, cells, strict=False
    ):
        Ex = np.array(0 + 0j)
        i_min = np.argmax(z >= cell.z_min)
        i_max = np.argmax(z > cell.z_max)
        z_ = z[i_min:] if i_max == 0 else z[i_min:i_max]
        z_local = z_ - cell.z_min  # [:-1] + np.diff(z_) / 2
        for mode, f, b in zip(mode_set, forward, backward, strict=False):
            E_slice = mode.Ex[:, i_y]

            Ex += jnp.outer(
                f * E_slice.T, jnp.exp(2j * np.pi * mode.neff / mode.env.wl * z_local)
            )

            Ex += jnp.outer(
                b * E_slice.T, jnp.exp(-2j * np.pi * mode.neff / mode.env.wl * z_local)
            )

        if i_max == 0:
            E_tot[i_min:] = Ex.T
        else:
            E_tot[i_min:i_max] = Ex.T

        # X, Y = np.meshgrid(z_, mesh_x)
    # plt.xlabel("z in um")
    # plt.ylabel("x in um")
    return E_tot, mesh_x


def propagate_modes(
    modes: list[list[Mode]],
    cells: list[Cell],
    ex_l: ComplexArray1D,
    ex_r: ComplexArray1D,
    y: float,
    z: FloatArray1D,
    sax_backend: sax.BackendLike = "default",
) -> tuple[ComplexArray2D, FloatArray1D]:
    """Propagate the modes through the cells."""
    actual_sax_backend = sax.into[sax.Backend](sax_backend)
    propagations = compute_propagation_s_matrices(modes, cells)
    interfaces = compute_interface_s_matrices(
        modes,
        enforce_reciprocity=False,
    )
    identity = compute_interface_s_matrix(
        modes[0],
        modes[0],
        enforce_reciprocity=False,
    )

    pairs = pi_pairs(propagations, interfaces, actual_sax_backend)
    l2rs = l2r_matrices(pairs, identity, actual_sax_backend)
    r2ls = r2l_matrices(pairs, actual_sax_backend)

    forwards, backwards = propagate(l2rs, r2ls, ex_l, ex_r)
    return plot_fields(modes, cells, forwards, backwards, y, z)
