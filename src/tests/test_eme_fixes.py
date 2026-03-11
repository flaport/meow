import numpy as np

import meow.eme.common as eme_common
import meow.eme.propagate as eme_propagate


def test_r2l_matrices_does_not_duplicate_last_pair(monkeypatch):
    calls: list[tuple[str, str]] = []

    def fake_connect_two(l, r, sax_backend):  # noqa: ANN001,ARG001
        calls.append((l, r))
        return f"({l}>{r})"

    monkeypatch.setattr(eme_propagate, "_connect_two", fake_connect_two)

    pairs = ["p0", "p1", "p2"]
    matrices = eme_propagate.r2l_matrices(pairs, sax_backend="default")

    assert calls == [("p1", "p2"), ("p0", "(p1>p2)")]
    assert matrices == ["(p0>(p1>p2))", "(p1>p2)", "p2"]


def test_enforce_lossy_unitarity_projects_to_contractive_matrix(monkeypatch):
    left = object()
    right = object()

    def fake_inner_product_conj(a, b):  # noqa: ANN001
        if a is left and b is left:
            return 1.0
        if a is right and b is right:
            return 1.0
        if a is left and b is right:
            return 0.01
        if a is right and b is left:
            return 10.0
        msg = "unexpected mode pair"
        raise AssertionError(msg)

    monkeypatch.setattr(eme_common, "inner_product_conj", fake_inner_product_conj)

    S_no, _ = eme_common.compute_interface_s_matrix(
        [left],
        [right],
        enforce_lossy_unitarity=False,
        ignore_warnings=False,
    )
    S_yes, _ = eme_common.compute_interface_s_matrix(
        [left],
        [right],
        enforce_lossy_unitarity=True,
        ignore_warnings=False,
    )

    s_no = np.linalg.svd(S_no, compute_uv=False)
    assert float(s_no.max()) > 1.0

    U, s, Vh = np.linalg.svd(S_no, full_matrices=False)
    expected = U @ np.diag(np.minimum(s, 1.0)) @ Vh
    assert np.allclose(S_yes, expected)

    s_yes = np.linalg.svd(S_yes, compute_uv=False)
    assert float(s_yes.max()) <= 1.0 + 1e-12
