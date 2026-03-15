from typing import Any, cast

import numpy as np
import pytest
import sax

import meow.eme.propagation as eme_propagation
from meow.eme.interface import (
    compute_interface_s_matrix,
    enforce_passivity,
)
from meow.mode import Mode


def test_r2l_matrices_does_not_duplicate_last_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    def fake_connect_two(l: Any, r: Any, sax_backend: sax.Backend) -> str:  # noqa: ARG001
        calls.append((l, r))
        return f"({l}>{r})"

    monkeypatch.setattr(eme_propagation, "_connect_two", fake_connect_two)

    pairs = cast(list[sax.STypeMM], ["p0", "p1", "p2"])
    matrices = eme_propagation.r2l_matrices(pairs, sax_backend="klu")

    assert calls == [("p1", "p2"), ("p0", "(p1>p2)")]
    assert matrices == ["(p0>(p1>p2))", "(p1>p2)", "p2"]


def test_enforce_passivity_clips_singular_values() -> None:
    sigma = np.array([0.5, 1.0, 1.5, 2.0])

    result_clip = enforce_passivity(sigma, method="clip")
    assert np.allclose(result_clip, [0.5, 1.0, 1.0, 1.0])

    result_invert = enforce_passivity(sigma, method="invert")
    assert np.allclose(result_invert, [0.5, 1.0, 1 / 1.5, 1 / 2.0])

    result_subtract = enforce_passivity(sigma, method="subtract")
    assert np.allclose(result_subtract, [0.5, 1.0, 0.5, 0.0])

    result_none = enforce_passivity(sigma, method="none")
    assert np.allclose(result_none, sigma)


def test_passivity_enforcement_in_interface_s_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Interface S-matrix with passivity_method='clip' has singular values <= 1."""
    from mode_data import MODE_DATA  # type: ignore[reportMissingImports]

    mode = Mode.model_validate(MODE_DATA)
    modes_l = [mode]
    modes_r = [mode]

    S_clip, _ = compute_interface_s_matrix(
        modes_l,
        modes_r,
        passivity_method="clip",
        enforce_reciprocity=False,
    )
    s_vals = np.linalg.svd(np.asarray(S_clip), compute_uv=False)
    assert float(s_vals.max()) <= 1.0 + 1e-12

    S_none, _ = compute_interface_s_matrix(
        modes_l,
        modes_r,
        passivity_method="none",
        enforce_reciprocity=False,
    )
    # With passivity_method="none", no correction is applied
    # (singular values may or may not exceed 1 depending on modes)
    assert S_none.shape == S_clip.shape
