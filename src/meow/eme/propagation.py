"""Propagation S-Matrices."""

from __future__ import annotations

import jax.numpy as jnp
import sax

from meow.cell import Cell
from meow.mode import Mode, Modes


def compute_propagation_s_matrix(modes: Modes, cell_length: float) -> sax.SDictMM:
    """Get the propagation S-matrix of each `Mode`."""
    s_dict = {
        (f"left@{i}", f"right@{i}"): jnp.exp(
            2j * jnp.pi * mode.neff / mode.env.wl * cell_length
        )
        for i, mode in enumerate(modes)
    }
    s_dict = {**s_dict, **{(p2, p1): v for (p1, p2), v in s_dict.items()}}
    return s_dict


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
    S: sax.SDense,
    ports: list[str],
) -> sax.SDenseMM:
    """Keep subset of an S-matrix."""
    s, pm = S
    idxs = jnp.array([pm[port] for port in ports], dtype=jnp.int32)
    s = s[idxs, :][:, idxs]
    new_port_map = {p: i for i, p in enumerate(ports)}
    return s, new_port_map
