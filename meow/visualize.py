""" Visualizations for common meow-datatypes """

from typing import Any

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import gdsfactory as gf
except ImportError:
    gf = None

try:
    from jaxlib.xla_extension import DeviceArray
except ImportError:
    DeviceArray = None


def _visualize_s_matrix(S, fmt=".3f", title=None, show=True):
    import matplotlib.pyplot as plt

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

    for x, y, z in zip(X.ravel(), Y.ravel(), Z[::-1].T.ravel()):
        if z > 0.0005:
            text = eval(f"f'{{z:{fmt}}}'")  # 😅
            plt.text(x, y, text, ha="center", va="center")

    if title is not None:
        plt.title(title)

    if show:
        plt.show()


def _visualize_s_pm_matrix(Spm, fmt=".3f", title=None, show=True):
    import matplotlib.pyplot as plt

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
    import gdsfactory as gf

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


def visualize(obj: Any, **kwargs: Any):
    """visualize any meow object

    Args:
        obj: the meow object to visualize
        **kwargs: extra configuration to visualize the object

    Note:
        Most meow objects have a `._visualize` method.
        Check out its help to see which kwargs are accepted.
    """
    from .base_model import BaseModel
    from .structures import Structure, visualize_structures

    if isinstance(obj, BaseModel):
        return obj._visualize(**kwargs)
    elif _is_s_matrix(obj) and plt is not None:
        _visualize_s_matrix(obj, **kwargs)
    elif (
        _is_two_tuple(obj)
        and _is_s_matrix(obj[0])
        and isinstance(obj[1], dict)
        and plt is not None
    ):
        _visualize_s_pm_matrix(obj, **kwargs)
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
