""" Visualizations for common meow-datatypes """

import warnings
from typing import Any

import numpy as np

try:
    import matplotlib.pyplot as plt


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


def _visualize_s_matrix(S, fmt=".3f", title=None, show=True):
    import matplotlib.pyplot as plt  # fmt: skip

    Z = np.abs(S)
    y, x = np.arange(Z.shape[0])[::-1], np.arange(Z.shape[1])
    Y, X = np.meshgrid(y, x)

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


def _visualize_s_pm_matrix(Spm, fmt=".3f", title=None, show=True):
    import matplotlib.pyplot as plt  # fmt: skip

    S, pm = Spm
    _visualize_s_matrix(S, fmt=fmt, title=title, show=False)
    num_left = len([p for p in pm if "left" in p])
    Z = np.abs(S)
    y, x = np.arange(Z.shape[0])[::-1], np.arange(Z.shape[1])

    plt.axvline(x[num_left] - 0.5, color="red")
    plt.axhline(x[num_left] - 0.5, color="red")

    if show:
        plt.show()


def _visualize_gdsfactory(comp):
    import gdsfactory as gf  # fmt: skip

    gf.plot(comp)  # type: ignore


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


def _is_two_tuple(obj):
    try:
        x, y = obj
        return True
    except Exception:
        return False


def _visualize_modes(modes):
    import matplotlib.pyplot as plt  # fmt: skip
    from matplotlib.colors import LinearSegmentedColormap  # fmt: skip
    from mpl_toolkits.axes_grid1 import make_axes_locatable  # fmt: skip

    num_modes = len(modes)
    cs = modes[0].cs
    X, Y, n = cs.mesh.Xz, cs.mesh.Yz, cs.nz
    x_min, x_max = cs.mesh.x.min(), cs.mesh.x.max()
    y_min, y_max = cs.mesh.y.min(), cs.mesh.y.max()
    delta_x = x_max - x_min
    delta_y = y_max - y_min
    aspect = delta_y / delta_x
    W0 = 6.4
    W, H = W0 + 1, W0 * aspect + 2
    n_cmap = LinearSegmentedColormap.from_list(name="c_cmap", colors=["#ffffff", "#c1d9ed"])  # fmt: skip
    fig, ax = plt.subplots(
        num_modes,
        2,
        figsize=(2 * W, num_modes * H),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    # mx = {
    #    "Ex": max([np.max(np.abs(m.Ex)**2) for m in modes]),
    #    "Hx": max([np.max(np.abs(m.Hx)**2) for m in modes]),
    # }
    for i, m in enumerate(modes):
        for j, field in enumerate(["Ex", "Hx"]):
            plt.sca(ax[i, j])
            if i == num_modes - 1:
                plt.xlabel("x")
            if j == 0:
                plt.ylabel("y")
            plt.axis("equal")
            value = np.abs(getattr(m, field)) ** 2
            plt.title(
                f"M{i}: {field}: n={m.neff.real:.2f}+{m.neff.imag:.1e}j, {100*m.te_fraction:.1f}%TE",
                fontsize=9,
            )
            plt.pcolormesh(X, Y, n, cmap=n_cmap)
            value[value < 1e-2] = np.nan
            # value[0, 0] = 0.0
            # value[-1, -1] = mx[field]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                plt.contour(X, Y, value, cmap="inferno")
            divider = make_axes_locatable(ax[i, j])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(cax=cax)
    fig.subplots_adjust(hspace=0, wspace=2 / (2 * W))


def visualize(obj: Any, **kwargs: Any):
    """visualize any meow object

    Args:
        obj: the meow object to visualize
        **kwargs: extra configuration to visualize the object

    Note:
        Most meow objects have a `._visualize` method.
        Check out its help to see which kwargs are accepted.
    """
    from .base_model import BaseModel  # fmt: skip
    from .mode import Mode  # fmt: skip
    from .structures import Structure, visualize_structures  # fmt: skip

    # if isinstance(obj, Mode):
    #    return _visualize_mode(obj)
    if isinstance(obj, list) and all(isinstance(o, Mode) for o in obj):
        return _visualize_modes(obj)
    elif isinstance(obj, BaseModel):
        return obj._visualize(**kwargs)
    elif _is_s_matrix(obj) and plt is not None:
        _visualize_s_matrix(obj, **kwargs)
    elif (
        _is_two_tuple(obj)
        and _is_s_matrix(obj[0])
        and isinstance(obj[1], dict)
        and plt is not None
    ):
        if kwargs.get("angle", False):
            obj_ = np.angle(obj)
            obj_[np.abs(obj) < 0.0005] = 0
            obj = obj_
            del kwargs["angle"]
        _visualize_s_matrix(obj, **kwargs)
    elif gf is not None and isinstance(obj, gf.Component):
        _visualize_gdsfactory(obj, **kwargs)
    else:
        try:
            (*objs,) = obj  # type: ignore
        except TypeError:
            return obj

        if all(isinstance(obj, Structure) for obj in objs):
            return visualize_structures(objs, **kwargs)
        return objs


vis = visualize  # shorthand for visualize
