"""Propagation utilities for EME stacks."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import numpy as np
import sax
from sax.backends import circuit_backends

from meow.arrays import ComplexArray1D, ComplexArray2D, FloatArray1D
from meow.cell import Cell
from meow.eme.interface import compute_interface_s_matrices, compute_interface_s_matrix
from meow.mode import Mode, Modes


def compute_propagation_s_matrix(modes: Modes, cell_length: float) -> sax.SDictMM:
    """Return the diagonal propagation S-matrix for one cell.

    Each mode acquires a phase ``exp(2j * pi * neff / wl * cell_length)`` while
    propagating through the cell. Backward propagation is mirrored by the
    bidirectional port mapping in the returned SAX dictionary.
    """
    s_dict = {
        (f"left@{i}", f"right@{i}"): jnp.exp(
            2j * jnp.pi * mode.neff / mode.env.wl * cell_length
        )
        for i, mode in enumerate(modes)
    }
    return {**s_dict, **{(p2, p1): v for (p1, p2), v in s_dict.items()}}


def compute_propagation_s_matrices(
    modes: list[Modes],
    cells: list[Cell],
    *,
    cell_lengths: list[float] | None = None,
) -> dict[str, sax.SDictMM]:
    """Return the propagation S-matrix for every cell in a stack.

    Args:
        modes: Modal basis for each cell.
        cells: Cells through which the modes propagate.
        cell_lengths: Optional explicit lengths. If omitted, they are derived
            from ``cell.length``.
    """
    if len(cells) != len(modes):
        msg = f"len(cells) != len(modes): {len(cells)} != {len(modes)}"
        raise ValueError(msg)

    if cell_lengths is None:
        cell_lengths = [cell.length for cell in cells]

    if len(cell_lengths) != len(modes):
        msg = f"len(cell_lengths) != len(modes): {len(cell_lengths)} != {len(modes)}"
        raise ValueError(msg)

    return {
        f"p_{i}": compute_propagation_s_matrix(modes_, cell_length=cell_length)
        for i, (modes_, cell_length) in enumerate(zip(modes, cell_lengths, strict=True))
    }


def select_ports(S: sax.SDenseMM, ports: list[str]) -> sax.SDenseMM:
    """Keep a subset of ports from an S-matrix."""
    s, pm = S
    idxs = jnp.array([pm[port] for port in ports], dtype=jnp.int32)
    s = s[idxs, :][:, idxs]
    new_port_map = {p: i for i, p in enumerate(ports)}
    return s, new_port_map


def _connect_two(
    left: sax.STypeMM, right: sax.STypeMM, sax_backend: sax.Backend
) -> sax.STypeMM:
    """Cascade two SAX components by wiring left/right ports."""
    _, p_l = sax.sdense(left)
    _, p_r = sax.sdense(right)
    instances: dict[str, sax.STypeMM] = {"l": left, "r": right}
    p_lr = sorted(p for p in p_l if "right" in p)
    p_rl = sorted(p for p in p_r if "left" in p)
    p_ll = [p for p in p_l if "left" in p]
    p_rr = [p for p in p_r if "right" in p]

    connections = {f"l,{pl}": f"r,{pr}" for pl, pr in zip(p_lr, p_rl, strict=False)}
    ports = {**{p: f"l,{p}" for p in p_ll}, **{p: f"r,{p}" for p in p_rr}}
    nets = [{"p1": p1, "p2": p2} for p1, p2 in connections.items()]
    net: dict = {"instances": instances, "nets": nets, "ports": ports}
    _, analyze_circuit, evaluate_circuit = circuit_backends[sax_backend]
    net["instances"] = {k: sax.scoo(v) for k, v in net["instances"].items()}
    analyzed = analyze_circuit(net["instances"], net["nets"], net["ports"])
    return evaluate_circuit(analyzed, net["instances"])


def pi_pairs(
    propagations: dict[str, sax.SDictMM],
    interfaces: dict[str, sax.SDenseMM],
    sax_backend: sax.Backend,
) -> list[sax.STypeMM]:
    """Return propagation-interface pairs for a full stack."""
    pairs: list[sax.STypeMM] = []
    for i in range(len(propagations)):
        propagation = propagations[f"p_{i}"]
        if i == len(interfaces):
            pairs.append(propagation)
        else:
            pairs.append(
                _connect_two(propagation, interfaces[f"i_{i}_{i + 1}"], sax_backend)
            )
    return pairs


def l2r_matrices(
    pairs: list[sax.STypeMM], identity: sax.SDenseMM, sax_backend: sax.Backend
) -> list[sax.STypeMM]:
    """Return cumulative left-to-right S-matrices."""
    matrices: list[sax.STypeMM] = [identity]
    for pair in pairs[:-1]:
        matrices.append(_connect_two(matrices[-1], pair, sax_backend))
    return matrices


def r2l_matrices(
    pairs: list[sax.STypeMM], sax_backend: sax.Backend
) -> list[sax.STypeMM]:
    """Return cumulative right-to-left S-matrices."""
    matrices = [pairs[-1]]
    for pair in pairs[-2::-1]:
        matrices.append(_connect_two(pair, matrices[-1], sax_backend))
    return matrices[::-1]


def split_square_matrix(
    matrix: ComplexArray2D, idx: int
) -> tuple[
    tuple[ComplexArray2D, ComplexArray2D], tuple[ComplexArray2D, ComplexArray2D]
]:
    """Split a square matrix into its four block submatrices."""
    if matrix.shape[0] != matrix.shape[1]:
        msg = "Matrix has to be square."
        raise ValueError(msg)
    return (matrix[:idx, :idx], matrix[:idx, idx:]), (
        matrix[idx:, :idx],
        matrix[idx:, idx:],
    )


def compute_mode_amplitudes(
    u: ComplexArray2D,
    v: ComplexArray2D,
    m: int,
    excitation_l: ComplexArray1D,
    excitation_r: ComplexArray1D,
) -> tuple[ComplexArray1D, ComplexArray1D]:
    """Solve for the forward and backward modal amplitudes in one cell."""
    n = u.shape[0] - m
    _, [u21, u22] = split_square_matrix(u, n)
    [v11, v12], _ = split_square_matrix(v, m)

    rhs = u21 @ excitation_l + u22 @ v12 @ excitation_r
    lhs = np.eye(m, dtype=complex) - u22 @ v11
    forward = np.linalg.solve(lhs, rhs)
    backward = v12 @ excitation_r + v11 @ forward
    return forward, backward


def propagate(
    l2rs: list[sax.STypeMM],
    r2ls: list[sax.STypeMM],
    excitation_l: ComplexArray1D,
    excitation_r: ComplexArray1D,
) -> tuple[list[ComplexArray1D], list[ComplexArray1D]]:
    """Propagate boundary excitations through cumulative S-matrices."""
    forwards = []
    backwards = []
    for l2r, r2l in zip(l2rs, r2ls, strict=False):
        s_l2r, ports = sax.sdense(l2r)
        s_r2l, _ = sax.sdense(r2l)
        n_right = len([key for key in ports if "right" in key])
        fwd, bwd = compute_mode_amplitudes(
            np.asarray(s_l2r), np.asarray(s_r2l), n_right, excitation_l, excitation_r
        )
        forwards.append(fwd)
        backwards.append(bwd)
    return forwards, backwards


def plot_fields(
    modes: list[list[Mode]],
    cells: list[Cell],
    forwards: list[ComplexArray1D],
    backwards: list[ComplexArray1D],
    y: float,
    z: FloatArray1D,
) -> tuple[ComplexArray2D, FloatArray1D]:
    """Reconstruct an ``Ex(x, z)`` field slice from propagated modal amplitudes."""
    mesh_y = cells[0].mesh.y
    mesh_x = cells[0].mesh.x
    mesh_x = mesh_x[:-1] + np.diff(mesh_x) / 2
    i_y = np.argmin(np.abs(mesh_y - y))

    e_tot = np.zeros((len(z), len(mesh_x)), dtype=complex)
    for mode_set, forward, backward, cell in zip(
        modes, forwards, backwards, cells, strict=False
    ):
        ex = np.array(0 + 0j)
        i_min = np.argmax(z >= cell.z_min)
        i_max = np.argmax(z > cell.z_max)
        z_ = z[i_min:] if i_max == 0 else z[i_min:i_max]
        z_local = z_ - cell.z_min
        for mode, fwd, bwd in zip(mode_set, forward, backward, strict=False):
            e_slice = mode.Ex[:, i_y]
            ex += jnp.outer(
                fwd * e_slice.T, jnp.exp(2j * np.pi * mode.neff / mode.env.wl * z_local)
            )
            ex += jnp.outer(
                bwd * e_slice.T,
                jnp.exp(-2j * np.pi * mode.neff / mode.env.wl * z_local),
            )

        if i_max == 0:
            e_tot[i_min:] = ex.T
        else:
            e_tot[i_min:i_max] = ex.T

    return e_tot, mesh_x


def _default_excitation(n_modes: int, excite_mode: int) -> ComplexArray1D:
    """Build a unit excitation vector."""
    if excite_mode < 0 or excite_mode >= n_modes:
        msg = f"excite_mode out of range: {excite_mode} not in [0, {n_modes})."
        raise ValueError(msg)
    excitation = np.zeros(n_modes, dtype=complex)
    excitation[excite_mode] = 1.0
    return excitation


def _default_z(cells: list[Cell], num_z: int) -> FloatArray1D:
    """Create a global z-grid spanning the full device."""
    return np.linspace(cells[0].z_min, cells[-1].z_max, num_z)


def propagate_modes(
    modes: list[list[Mode]],
    cells: list[Cell],
    *,
    excitation_l: ComplexArray1D | None = None,
    excitation_r: ComplexArray1D | None = None,
    excite_mode_l: int = 0,
    excite_mode_r: int | None = None,
    y: float | None = None,
    z: FloatArray1D | None = None,
    num_z: int = 1000,
    sax_backend: sax.BackendLike = "default",
    interface_kwargs: dict[str, Any] | None = None,
    interfaces_fn: Callable = compute_interface_s_matrices,
    interface_fn: Callable = compute_interface_s_matrix,
) -> tuple[ComplexArray2D, FloatArray1D]:
    """Propagate modal excitations through a stack of cells.

    This is a convenience wrapper around the propagation and interface S-matrix
    machinery. The only required positional inputs are the modal bases and the
    cells. Everything else has defaults that are inferred from the stack:

    - by default the left boundary excites mode 0 with unit amplitude;
    - by default there is no right-side excitation;
    - by default ``y`` is the center of the transverse mesh;
    - by default ``z`` spans the full device with ``num_z`` samples.

    Args:
        modes: Mode set for each cell.
        cells: Cells associated with ``modes``.
        excitation_l: Optional explicit left excitation vector.
        excitation_r: Optional explicit right excitation vector.
        excite_mode_l: Left mode to excite if ``excitation_l`` is omitted.
        excite_mode_r: Right mode to excite if ``excitation_r`` is omitted. If
            omitted, no right-side excitation is applied.
        y: Transverse y-coordinate at which to reconstruct ``Ex``.
        z: Global z-grid on which to reconstruct the field.
        num_z: Number of z samples if ``z`` is omitted.
        sax_backend: SAX backend used for cascading.
        interface_kwargs: Optional keyword arguments forwarded to both
            ``interfaces_fn`` and ``interface_fn``.
        interfaces_fn: Factory for interface S-matrices across the stack.
        interface_fn: Factory for the identity-like same-basis interface used to
            seed the left-to-right accumulation.

    Returns:
        ``(field, x)`` where ``field`` is the reconstructed ``Ex(z, x)`` slice
        and ``x`` is the transverse sampling grid.
    """
    if len(cells) != len(modes):
        msg = f"len(cells) != len(modes): {len(cells)} != {len(modes)}"
        raise ValueError(msg)
    if not cells:
        msg = "At least one cell is required."
        raise ValueError(msg)

    actual_sax_backend = sax.into[sax.Backend](sax_backend)
    interface_kwargs = dict(interface_kwargs or {})
    interface_kwargs.setdefault("enforce_reciprocity", False)

    if excitation_l is None:
        excitation_l = _default_excitation(len(modes[0]), excite_mode_l)
    if excitation_r is None:
        if excite_mode_r is None:
            excitation_r = np.zeros(len(modes[-1]), dtype=complex)
        else:
            excitation_r = _default_excitation(len(modes[-1]), excite_mode_r)

    if y is None:
        y = float(0.5 * (cells[0].mesh.y.min() + cells[0].mesh.y.max()))
    if z is None:
        z = _default_z(cells, num_z)

    propagations = compute_propagation_s_matrices(modes, cells)
    interfaces = interfaces_fn(modes, **interface_kwargs)
    identity = interface_fn(modes[0], modes[0], **interface_kwargs)

    pairs = pi_pairs(propagations, interfaces, actual_sax_backend)
    l2rs = l2r_matrices(pairs, identity, actual_sax_backend)
    r2ls = r2l_matrices(pairs, actual_sax_backend)

    forwards, backwards = propagate(l2rs, r2ls, excitation_l, excitation_r)
    return plot_fields(modes, cells, forwards, backwards, y, z)
