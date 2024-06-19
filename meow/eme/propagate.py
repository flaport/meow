"""Propagating fields throug devices"""

from typing import Any

import jax.numpy as np
import numpy as onp
import sax
from sax.backends import circuit_backends

from meow.eme import (
    compute_interface_s_matrices,
    compute_interface_s_matrix,
    compute_propagation_s_matrices,
)
from meow.eme.sax import _validate_sax_backend


def _connect_two(l: sax.SType, r: sax.SType, sax_backend: str):
    """l -> left, r -> right"""
    # TODO there must be an easier way to do this...
    _s_l, p_l = sax.sdense(l)
    _s_r, p_r = sax.sdense(r)
    instances: dict[str, sax.SType] = {"l": l, "r": r}
    p_lr = [p for p in p_l.keys() if "right" in p]  # right ports of left
    p_rl = [p for p in p_r.keys() if "left" in p]  # left ports of right

    p_ll = [p for p in p_l.keys() if "left" in p]  # left ports of left
    p_rr = [p for p in p_r.keys() if "right" in p]  # right ports of right

    p_lr.sort()
    p_rl.sort()
    connections = {f"l,{pl}": f"r,{pr}" for pl, pr in zip(p_lr, p_rl)}
    ports = {**{p: f"l,{p}" for p in p_ll}, **{p: f"r,{p}" for p in p_rr}}
    net: dict[str, dict[str, Any]] = dict(
        instances=instances, connections=connections, ports=ports
    )
    _, analyze_circuit, evaluate_circuit = circuit_backends[sax_backend]
    net["instances"] = {k: sax.scoo(v) for k, v in net["instances"].items()}
    analyzed = analyze_circuit(net["instances"], net["connections"], net["ports"])
    return evaluate_circuit(analyzed, net["instances"])


def pi_pairs(propagations, interfaces, sax_backend):
    """generates the S-matrices of cells: a combination of propagation and interface matrix"""
    S = []
    for i in range(len(propagations)):
        p = propagations[f"p_{i}"]
        if i == len(interfaces):
            S.append(p)
        else:
            c = interfaces[f"i_{i}_{i+1}"]
            S.append(_connect_two(p, c, sax_backend))

    return S


def l2r_matrices(pairs, identity, sax_backend):
    Ss = [identity]

    for p in pairs[:-1]:
        Ss.append(_connect_two(Ss[-1], p, sax_backend))

    return Ss


def r2l_matrices(pairs, sax_backend):
    Ss = [pairs[-1]]

    for p in pairs[-1::-1]:
        Ss.append(_connect_two(p, Ss[-1], sax_backend))

    return Ss[::-1]


def split_square_matrix(matrix, idx):
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix has to be square")
    return [matrix[:idx, :idx], matrix[:idx, idx:]], [
        matrix[idx:, :idx],
        matrix[idx:, idx:],
    ]


def propagate(l2rs, r2ls, excitation_l, excitation_r):
    forwards = []
    backwards = []
    for l2r, r2l in zip(l2rs, r2ls):
        s_l2r, p = sax.sdense(l2r)
        s_r2l, _ = sax.sdense(r2l)
        m = len([k for k in p.keys() if "right" in k])
        f, b = compute_mode_amplitudes(s_l2r, s_r2l, m, excitation_l, excitation_r)
        forwards.append(f)
        backwards.append(b)
    return forwards, backwards


def compute_mode_amplitudes(u, v, m, excitation_l, excitation_r):
    n = u.shape[0] - m
    l = v.shape[0] - m
    [u11, u12], [u21, u22] = split_square_matrix(u, n)
    [v11, v12], [v21, v22] = split_square_matrix(v, m)

    RHS = u21 @ excitation_l + u22 @ v12 @ excitation_r
    LHS = np.diag(np.ones(m)) - u22 @ v11
    forward = np.linalg.solve(LHS, RHS)
    backward = v12 @ excitation_r + v11 @ forward  # Attention v21 was v12
    return forward, backward


def plot_fields(modes, cells, forwards, backwards, y, z, lim=1):
    mode_set = modes[0]
    mesh_y = cells[0].mesh.y
    mesh_x = cells[0].mesh.x
    mesh_x = mesh_x[:-1] + np.diff(mesh_x) / 2
    i_y = np.argmin(np.abs(mesh_y - y))

    lim = None
    E_tot = onp.zeros((len(z), len(mesh_x)), dtype=complex)
    for mode_set, forward, backward, cell in zip(modes, forwards, backwards, cells):
        Ex = np.array(0 + 0j)
        i_min = np.argmax(z >= cell.z_min)
        i_max = np.argmax(z > cell.z_max)
        if i_max == 0:
            z_ = z[i_min:]
        else:
            z_ = z[i_min:i_max]

        z_local = z_ - cell.z_min  # [:-1] + np.diff(z_) / 2
        for mode, f, b in zip(mode_set, forward, backward):
            E_slice = mode.Ex[:, i_y]

            Ex += np.outer(
                f * E_slice.T, np.exp(2j * np.pi * mode.neff / mode.env.wl * z_local)
            )

            Ex += np.outer(
                b * E_slice.T, np.exp(-2j * np.pi * mode.neff / mode.env.wl * z_local)
            )

        if i_max == 0:
            E_tot[i_min:] = Ex.T
        else:
            E_tot[i_min:i_max] = Ex.T

        # X, Y = np.meshgrid(z_, mesh_x)
        # plt.pcolormesh(X, Y, np.abs(Ex), vmin = -lim, vmax = lim)
    # plt.xlabel("z in um")
    # plt.ylabel("x in um")
    return E_tot, mesh_x


def propagate_modes(modes, cells, ex_l, ex_r, y, z, sax_backend=None):
    sax_backend = _validate_sax_backend(sax_backend)
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

    pairs = pi_pairs(propagations, interfaces, sax_backend)
    l2rs = l2r_matrices(pairs, identity, sax_backend)
    r2ls = r2l_matrices(pairs, sax_backend)

    forwards, backwards = propagate(l2rs, r2ls, ex_l, ex_r)
    return plot_fields(modes, cells, forwards, backwards, y, z)
