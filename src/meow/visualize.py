""" Visualizations for common meow-datatypes """

from collections.abc import Iterable
from typing import Any, Callable

import numpy as np

from meow.structures import Structure3D, _sort_structures

try:
    import matplotlib.pyplot as plt  # fmt: skip


except ImportError:
    plt = None

try:
    import gdsfactory as gf  # fmt: skip
except ImportError:
    gf = None

try:
    from jaxlib.xla_extension import DeviceArray  # fmt: skip # type: ignore
except ImportError:
    DeviceArray = None


def _visualize_s_matrix(S, fmt=None, title=None, show=True, phase=False, ax=None):
    import matplotlib.pyplot as plt  # fmt: skip
    if phase:
        fmt = ".0f"
    else:
        fmt = ".3f"

    Z = np.abs(S)
    y, x = np.arange(Z.shape[0])[::-1], np.arange(Z.shape[1])
    Y, X = np.meshgrid(y, x)

    if ax:
        plt.sca(ax)
    else:
        plt.figure(figsize=(2 * x.shape[0] / 3, 2 * y.shape[0] / 3))

    plt.pcolormesh(X, Y, Z[::-1].T, cmap="Greys", vmin=0.0, vmax=2.0 * Z.max())

    coords_ = np.concatenate(
        [np.array([x[0] - 1]), x, np.array([x[-1] + 1])], axis=0, dtype=float
    )
    labels = ["" for _ in coords_]
    plt.xticks(coords_ + 0.5, labels)
    plt.xlim(coords_[0] + 0.5, coords_[-1] - 0.5)

    coords_ = np.concatenate(
        [np.array([y[0] + 1]), y, np.array([y[-1] - 1])], axis=0, dtype=float
    )
    coords_ = coords_[::-1]  # reverse
    labels = ["" for _ in coords_]
    plt.yticks(coords_ + 0.5, labels)
    plt.ylim(coords_[-1] - 0.5, coords_[0] + 0.5)
    plt.grid(True)

    for x, y, z in zip(X.ravel(), Y.ravel(), S[::-1].T.ravel()):
        if np.abs(z) > 0.0005:
            if phase:
                z = np.angle(z) * 180 / np.pi
            text = eval(f"f'{{z:{fmt}}}'")  # 😅
            text = text.replace("+", "\n+")
            text = text.replace("-", "\n-")
            if text[0] == "\n":
                text = text[1:]
            plt.text(x, y, text, ha="center", va="center", fontsize=8)

    if title is not None:
        plt.title(title)

    if show:
        plt.show()


def _visualize_s_pm_matrix(Spm, fmt=None, title=None, show=True, phase=False, ax=None):
    import matplotlib.pyplot as plt  # fmt: skip

    S, pm = Spm
    _visualize_s_matrix(S, fmt=fmt, title=title, show=False, phase=phase, ax=ax)
    num_left = len([p for p in pm if "left" in p])
    Z = np.abs(S)
    _, x = np.arange(Z.shape[0])[::-1], np.arange(Z.shape[1])

    plt.axvline(x[num_left] - 0.5, color="red")
    plt.axhline(x[num_left] - 0.5, color="red")

    if show:
        plt.show()


def _visualize_overlap_density(
    two_modes,
    conjugated=True,
    x_symmetry=False,
    y_symmetry=False,
    ax=None,
    n_cmap=None,
    mode_cmap=None,
    num_levels=8,
    show=True,
):
    import matplotlib.pyplot as plt  # fmt: skip

    from .mode import Mode  # fmt: skip

    mode1, mode2 = two_modes
    if conjugated:
        cross = mode1.Ex * mode2.Hy.conj() - mode1.Ey * mode2.Hx.conj()
    else:
        cross = mode1.Ex * mode2.Hy - mode1.Ey * mode2.Hx
    if x_symmetry:
        cross = 0.5 * (cross + cross[::-1])
    if y_symmetry:
        cross = 0.5 * (cross + cross[:, ::-1])
    zeros = np.zeros_like(cross)
    overlap = Mode(
        neff=mode1.neff,
        cs=mode1.cs,
        Ex=cross,
        Ey=zeros,
        Ez=zeros,
        Hx=zeros,
        Hy=zeros,
        Hz=zeros,
    )
    if ax is None:
        W, H = _figsize_visualize_mode(mode1.cs, 5)
        _, ax = plt.subplots(1, 3, figsize=(3 * W, H))

    field = "Ex" if mode1.te_fraction > 0.5 else "Hx"
    mode1._visualize(
        title=f"mode 1: {field}",
        fields=[field],
        ax=ax[0],
        n_cmap=n_cmap,
        mode_cmap=mode_cmap,
        num_levels=num_levels,
        show=False,
    )

    field = "Ex" if mode2.te_fraction > 0.5 else "Hx"
    mode2._visualize(
        title=f"mode 2: {field}",
        fields=[field],
        ax=ax[1],
        n_cmap=n_cmap,
        mode_cmap=mode_cmap,
        num_levels=num_levels,
        show=False,
    )

    title = "overlap density" + ("" if conjugated else " (no conjugations)")
    p = overlap._visualize(
        title=title,
        fields=["Ex"],
        ax=ax[2],
        n_cmap=n_cmap,
        mode_cmap=mode_cmap,
        num_levels=num_levels,
        show=False,
    )

    if show:
        plt.show()

    return p


def _visualize_gf_component(comp):
    import gdsfactory as gf  # fmt: skip

    gf.plot(comp)  # type: ignore


def _is_two_tuple(obj):
    if not isinstance(obj, tuple):
        return False
    try:
        x, y = obj  # type: ignore
        return True
    except Exception:
        return False


def _figsize_visualize_mode(cs, W0):
    x_min, x_max = cs.mesh.x.min(), cs.mesh.x.max()
    y_min, y_max = cs.mesh.y.min(), cs.mesh.y.max()
    delta_x = x_max - x_min
    delta_y = y_max - y_min
    aspect = delta_y / delta_x
    W, H = W0 + 1, W0 * aspect + 1
    return W, H


def _visualize_modes(
    modes,
    n_cmap=None,
    mode_cmap=None,
    num_levels=8,
    operation=lambda x: np.abs(x) ** 2,
    show=True,
    plot_width=6.4,
    fields=("Ex", "Hx"),
    ax=None,
):
    import matplotlib.pyplot as plt  # fmt: skip

    num_modes = len(modes)
    cs = modes[0].cs
    W, H = _figsize_visualize_mode(cs, plot_width)

    if ax is None:
        fig, ax = plt.subplots(
            num_modes,
            2,
            figsize=(2 * W, num_modes * H),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
    else:
        fig = None
    for i, m in enumerate(modes):
        m._visualize(
            title=None,
            title_prefix=f"m{i}: ",
            fields=list(fields),
            ax=ax[i],
            n_cmap=n_cmap,
            mode_cmap=mode_cmap,
            num_levels=num_levels,
            operation=operation,
            show=False,
        )
    if fig is not None:
        fig.subplots_adjust(hspace=0, wspace=2 / (2 * W))
    if show:
        plt.show()


def _visualize_base_model(obj, **kwargs):
    return obj._visualize(**kwargs)


def _is_mode_list(obj: Any) -> bool:
    from .mode import Mode  # fmt: skip
    return isinstance(obj, Iterable) and all(isinstance(o, Mode) for o in obj)


def _is_structure_3d_list(obj: Any) -> bool:
    from .structures import Structure3D  # fmt: skip
    return isinstance(obj, Iterable) and all(isinstance(o, Structure3D) for o in obj)


def _is_base_model(obj: Any) -> bool:
    from .base_model import BaseModel  # fmt: skip
    return isinstance(obj, BaseModel)


def _is_mode_overlap(obj: Any) -> bool:
    from .mode import Mode  # fmt: skip
    return _is_two_tuple(obj) and all(isinstance(o, Mode) for o in obj)


def _is_s_matrix(obj: Any):
    return (
        (
            isinstance(obj, np.ndarray)
            or (DeviceArray is not None and isinstance(obj, DeviceArray))
        )
        and obj.ndim == 2
        and obj.shape[0] > 1
        and obj.shape[1] > 1
    )


def _is_s_pm_matrix(obj):
    return _is_two_tuple(obj) and _is_s_matrix(obj[0]) and isinstance(obj[1], dict)


def _is_gf_component(obj):
    return gf is not None and isinstance(obj, gf.Component)


def _visualize_structures(structures: list[Structure3D], scale=None):
    """easily visualize a collection (list) of `Structure3D` objects"""
    from trimesh.scene import Scene  # fmt: skip
    from trimesh.transformations import rotation_matrix  # fmt: skip

    scene = Scene(
        geometry=[s._trimesh(scale=scale) for s in _sort_structures(structures)]
    )
    scene.apply_transform(rotation_matrix(np.pi - np.pi / 6, (0, 1, 0)))
    return scene.show()


VISUALIZATION_MAPPING: dict[Callable, Callable] = {
    _is_base_model: _visualize_base_model,
    _is_mode_list: _visualize_modes,
    _is_structure_3d_list: _visualize_structures,
    _is_mode_overlap: _visualize_overlap_density,
    _is_s_matrix: _visualize_s_matrix,
    _is_s_pm_matrix: _visualize_s_pm_matrix,
    _is_gf_component: _visualize_gf_component,
}


def visualize(obj: Any, **kwargs: Any):
    """visualize any meow object

    Args:
        obj: the meow object to visualize
        **kwargs: extra configuration to visualize the object

    Note:
        Most meow objects have a `._visualize` method.
        Check out its help to see which kwargs are accepted.
    """
    try:
        is_empty = bool(not obj)
    except ValueError:
        if isinstance(obj, np.ndarray):
            is_empty = obj.size == 0
        else:
            is_empty = False

    if is_empty:
        raise ValueError("Nothing to visualize!")

    if plt is None:
        print(obj)
        return

    for check_func, vis_func in VISUALIZATION_MAPPING.items():
        if check_func(obj):
            return vis_func(obj, **kwargs)
    print(obj)


vis = visualize  # shorthand for visualize
