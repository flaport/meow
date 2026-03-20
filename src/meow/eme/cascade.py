"""SAX backend for EME (default backend)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import sax
from sax.backends import circuit_backends

from meow.cell import Cell
from meow.eme.interface import (
    compute_interface_s_matrices,
)
from meow.eme.propagation import (
    compute_propagation_s_matrices,
)
from meow.mode import Modes


def compute_s_matrix_sax(
    modes: list[Modes],
    *,
    cells: list[Cell] | None = None,
    cell_lengths: list[float] | None = None,
    sax_backend: sax.Backend = "klu",
    interfaces_fn: Callable = compute_interface_s_matrices,
    propagations_fn: Callable = compute_propagation_s_matrices,
    **_: Any,
) -> sax.SDenseMM:
    """Calculate the S-matrix for given sets of modes.

    Args:
        modes: Modal basis for each cell in the stack.
        cells: Cells from which to derive propagation lengths. Either cells
            or cell_lengths must be provided.
        cell_lengths: Optional explicit propagation lengths per cell.
        sax_backend: SAX backend used for circuit evaluation.
        interfaces_fn: Callable that computes interface S-matrices.
        propagations_fn: Callable that computes propagation S-matrices.

    Returns:
        A tuple ``(S, port_map)`` in SAX dense multimode format.
    """
    propagations = propagations_fn(modes, cells, cell_lengths=cell_lengths)
    interfaces = interfaces_fn(modes)
    net = _get_netlist(propagations, interfaces)
    _, analyze_fn, evaluate_fn = circuit_backends[sax_backend]  # type: ignore[reportArgumentType]
    # TODO: use analyze_instances instead of manually converting to scoo ?
    net["instances"] = {k: sax.scoo(v) for k, v in net["instances"].items()}
    analyzed = analyze_fn(net["instances"], net["nets"], net["ports"])
    S, port_map = sax.sdense(
        evaluate_fn(
            analyzed,
            instances=net["instances"],
        )
    )
    S = np.asarray(S)

    # final sorting of result:
    current_port_map = {
        (p, int(i)): j
        for (p, i), j in {
            tuple(pm.split("@")): idx for pm, idx in port_map.items()
        }.items()
    }
    desired_port_map = {pm: i for i, pm in enumerate(sorted(current_port_map))}
    idxs = [current_port_map[pm] for pm in desired_port_map]
    S = S[idxs, :][:, idxs]
    port_map = {f"{p}@{m}": v for (p, m), v in desired_port_map.items()}

    return cast(sax.SDenseMM, (S, port_map))


def _get_netlist(
    propagations: dict[str, sax.SDictMM],
    interfaces: dict[str, sax.SDenseMM],
) -> dict:
    """Get the netlist of a stack of `Modes`."""
    instances = {**dict(propagations), **dict(interfaces)}
    propagation_keys = list(propagations)
    connections = {}
    for i in range(len(interfaces)):
        for port_mode in sax.get_ports(interfaces[f"i_{i}_{i + 1}"]):
            other_port_mode = _other_port(port_mode)
            if "left" in port_mode:
                connections[f"p_{i},{other_port_mode}"] = f"i_{i}_{i + 1},{port_mode}"
            elif "right" in port_mode:
                connections[f"i_{i}_{i + 1},{port_mode}"] = (
                    f"p_{i + 1},{other_port_mode}"
                )

    ports = {}
    for port_mode in sax.get_ports(propagations[propagation_keys[0]]):
        if "right" in port_mode:
            continue
        ports[port_mode] = f"{propagation_keys[0]},{port_mode}"
    for port_mode in sax.get_ports(propagations[propagation_keys[-1]]):
        if "left" in port_mode:
            continue
        ports[port_mode] = f"{propagation_keys[-1]},{port_mode}"

    nets = [{"p1": p1, "p2": p2} for p1, p2 in connections.items()]
    net = {"instances": instances, "nets": nets, "ports": ports}
    return net


def downselect_s(S: sax.SDenseMM, ports: list[str]) -> sax.SDenseMM:
    """Downselect the S-matrix to the given ports.

    Args:
        S: A tuple ``(S_matrix, port_map)`` in SAX dense multimode format.
        ports: Port names to keep.

    Returns:
        A new ``(S_matrix, port_map)`` tuple containing only the selected ports.
    """
    S_matrix, port_map = S
    idxs = [port_map[port] for port in ports]
    S_matrix = S_matrix[idxs, :][:, idxs]
    port_map = {port: i for i, port in enumerate(ports)}
    return S_matrix, port_map


def _other_port(port_mode: str) -> str:
    if "left" in port_mode:
        return port_mode.replace("left", "right")
    if "right" in port_mode:
        return port_mode.replace("right", "left")
    return port_mode
