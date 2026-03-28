"""MkDocs hooks for meow docs preprocessing."""  # noqa: INP001

from typing import Any

import matplotlib.pyplot as plt

import meow

plt.rcParams.update(
    {
        "figure.figsize": (6, 2.5),
        "axes.grid": True,
        "lines.color": "grey",
        "patch.edgecolor": "grey",
        "text.color": "grey",
        "axes.facecolor": "ffffff00",
        "axes.edgecolor": "grey",
        "axes.labelcolor": "grey",
        "xtick.color": "grey",
        "ytick.color": "grey",
        "grid.color": "grey",
        "figure.facecolor": "ffffff00",
        "figure.edgecolor": "ffffff00",
        "savefig.facecolor": "ffffff00",
        "savefig.edgecolor": "ffffff00",
    }
)


def on_page_markdown(
    markdown: str,
    page: Any,  # noqa: ARG001
    config: Any,  # noqa: ARG001
    files: Any,  # noqa: ARG001
    **kwargs: Any,  # noqa: ARG001
) -> str:
    """Process markdown content before it's converted to HTML."""
    blocks = markdown.split("```")

    for i, block in enumerate(blocks):
        if i % 2:
            if (special := _parse_special(block)) is not None:
                blocks[i] = special
            else:
                blocks[i] = f"```{block}```"
            continue

        lines = block.split("\n")
        _insert_cross_refs(lines)
        blocks[i] = "\n".join(lines)

    return "".join(blocks)


def _parse_special(content: str) -> str | None:
    """Format contents of a special code block differently."""
    lines = content.strip().split("\n")
    first = lines[0].strip()
    rest = lines[1:]
    if not (first.startswith("{") and first.endswith("}")):
        return None
    code_block_type = first[1:-1].strip()
    return _format_admonition(code_block_type, rest)


def _format_admonition(admonition_type: str, lines: list[str]) -> str:
    """Format lines as an admonition."""
    if admonition_type == "hint":
        admonition_type = "info"
    ret = f"!!! {admonition_type}\n\n"
    for line in lines:
        ret += f"    {line.strip()}\n"
    return ret


_ALIASES: dict[str, str | None] = {
    "compute_modes": "meow.compute_modes_tidy3d",
    "compute_s_matrix": "meow.compute_s_matrix_sax",
}


def _resolve_ref(name: str) -> str | None:
    """Resolve a name to its mkdocstrings anchor identifier."""
    *first, short = name.split(".")
    if first and first[0] != "meow":
        return None
    if short in _ALIASES:
        return _ALIASES.get(short)
    if hasattr(meow, short):
        return f"meow.{short}"
    if hasattr(meow.eme, short):
        return f"meow.eme.{short}"
    if hasattr(meow.fde, short):
        return f"meow.fde.{short}"
    return None


def _insert_cross_refs(lines: list[str]) -> None:
    """Insert cross-references in the markdown lines."""
    cross_refs = {}
    for line in lines:
        parts = line.split("`")
        for k, part in enumerate(parts):
            if k % 2 == 0:
                continue
            ref = _resolve_ref(part)
            if ref is not None:
                cross_refs[f"`{part}`"] = f"[`{part}`][{ref}]"
    for j, line in enumerate(lines):
        for k, v in cross_refs.items():
            line = line.replace(k, v)
        lines[j] = line
