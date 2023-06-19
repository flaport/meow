""" Materials """

import os
import re
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd
import tidy3d as td
from numpy.typing import NDArray
from pydantic import Field, validator
from scipy.constants import c
from scipy.ndimage import map_coordinates
from tidy3d import material_library

from .base_model import BaseModel, _array
from .environment import Environment

MATERIAL_TYPES: Dict[str, type] = {}


class Material(BaseModel):
    """a `Material` defines the refractive index of a `Structure` within an `Environment`."""

    type: str = ""

    name: str = Field(description="the name of the material")

    meta: Dict[str, Any] = Field(
        default_factory=lambda: {}, description="metadata for the material"
    )

    def __init_subclass__(cls):
        MATERIAL_TYPES[cls.__name__] = cls

    def __new__(cls, **kwargs):
        cls = MATERIAL_TYPES.get(kwargs.get("type", cls.__name__), cls)
        return BaseModel.__new__(cls)  # type: ignore

    @validator("type", pre=True, always=True)
    def validate_type(cls, value):
        if not value:
            value = getattr(cls, "__name__", "Geometry")
        if value not in MATERIAL_TYPES:
            raise ValueError(
                f"Invalid Material type. Got: {value!r}. Valid types: {MATERIAL_TYPES}."
            )
        return value

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        MATERIALS[self.name] = self

    def __call__(self, env: Environment) -> NDArray[np.complex_]:
        """returns an array of the refractive index at the wavelengths specified by the environment"""
        raise NotImplementedError("Please use one of the Material child classes")

    def _lumadd(self, sim, env, unit):
        from matplotlib.cm import get_cmap  # fmt: skip

        n = self(env)
        wl = np.asarray(env.wl * unit, dtype=complex).ravel()
        eps = np.asarray(n, dtype=complex).ravel() ** 2
        data = np.stack([c / wl, eps], 1)  # permittivity, not index

        if not sim.materialexists(self.name):
            sim.setmaterial(sim.addmaterial("Sampled data"), "name", self.name)
            color = np.asarray(
                self.meta.get("color") or get_cmap("jet")(np.abs(eps) / 15.0)
            )
            sim.setmaterial(self.name, "color", color)

        sim.setmaterial(self.name, "sampled data", data)

        return self.name

    class Config:
        fields = {
            "meta": {"exclude": True},
        }


class TidyMaterial(Material):
    name: str = Field(description="The material name as also used by tidy3d")
    variant: str = Field(description="The material name as also used by tidy3d")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # try:
        # # somehow black complains about the except
        # except ModuleNotFoundError:
        #    raise ModuleNotFoundError("Tidy3D has to be installed to use its material database")

        if self.name not in material_library:
            raise ValueError(
                f"Specified material name is invalid. Use one of {material_library.keys()}"
            )
        _material = material_library[self.name]
        _variants = getattr(_material, "variants", {})
        if not _variants:
            raise ValueError(f"Tidy3D material '{self.name}' not supported.")
        if self.variant not in _variants:
            _variant_options = list(_variants.keys())
            raise ValueError(
                f"Specified variant is invalid. Use one of {_variant_options}."
            )

    def __call__(self, env: Environment) -> NDArray[np.complex_]:
        if not isinstance(env, Environment):
            env = Environment(**env)
        _material = material_library[self.name]
        _variants = getattr(_material, "variants", None)
        assert _variants is not None
        mat = _material[self.variant]  # type: ignore
        eps_comp = getattr(mat, "eps_comp", None)
        assert eps_comp is not None
        eps = eps_comp(0, 0, td.C_0 / env.wl)
        return np.sqrt(eps)


class SampledMaterial(Material):
    params: Dict[str, np.ndarray[Tuple[int], np.dtype[np.float_]]] = Field(
        description="the wavelength over which the refractive index is defined."
    )
    n: np.ndarray[Tuple[int], np.dtype[np.complex_]] = Field(
        description="the complex refractive index of the material"
    )

    @staticmethod
    def _validate_array(arr):
        if isinstance(arr, dict):
            r = np.asarray(arr.get("real", 0.0), dtype=np.complex_)
            i = np.asarray(arr.get("imag", 0.0), dtype=np.complex_)
            new = r + 1j * i
        else:
            new = np.asarray(arr)
        return new.view(_array)

    @staticmethod
    def _validate_1d(name, arr):
        if arr.ndim != 1:
            raise ValueError(f"{name} should be 1D. Got a {arr.ndim}D array.")
        return arr

    @validator("n", pre=True)
    def validate_n_pre(cls, n):
        return cls._validate_1d("n", cls._validate_array(n))

    @validator("params", pre=True)
    def validate_params_pre(cls, params):
        if not isinstance(params, Mapping):
            raise TypeError(f"instance of dict expected. Got: {type(params)}.")
        return {
            k: cls._validate_1d(f"params:{k}", cls._validate_array(v))
            for k, v in params.items()
        }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for p, v in self.params.items():
            Lp = v.shape[0]
            Ln = self.n.shape[0]
            if Lp != Ln:
                raise ValueError(
                    f"length of parameter array {p} does not match length of refractive index array n. \n"
                    f"{Lp} != {Ln}"
                )

    @classmethod
    def from_path(cls, path, meta=None):
        path = _validate_path(path)
        name = re.sub(r"\.csv$", "", os.path.split(path)[-1])
        df = pd.read_csv(path)
        return cls.from_df(name, df, meta=meta)

    @classmethod
    def from_df(cls, name, df, meta=None):
        meta = meta or {}

        nr = df["nr"].values
        ni = np.zeros_like(nr) if "ni" not in df else df["ni"].values
        n = nr + 1j * ni

        columns = [c for c in df.columns if c not in ["nr", "ni"]]
        params = {c: np.asarray(df[c].values, dtype=np.float_) for c in columns}

        return cls(name=name, params=params, n=n, meta=meta)

    def __call__(self, env: Environment) -> NDArray[np.complex_]:
        if not isinstance(env, Environment):
            env = Environment(**env)
        # n = interp1d(1 / self.params['wl'], self.n, fill_value="extrapolate")(1 / env.wl)
        df = pd.DataFrame({**self.params, "nr": np.real(self.n), "ni": np.imag(self.n)})
        data, params, strings = _to_ndgrid(df, wl_key="wl")
        result, axs, pos = _evaluate_general_corner_model(
            data,
            params,
            strings,
            **{k: np.atleast_1d(v) for k, v in env.dict().items()},
        )
        nr = result.take(pos["targets"]["nr"], axs["targets"])
        ni = result.take(pos["targets"]["ni"], axs["targets"])
        n = nr + 1j * ni
        # FIXME: ideally we should just be able to return `n` here...
        # return n
        return np.squeeze(np.real(n))  # TODO: allow complex multi-dimensional n


Materials = List[Material]
MATERIALS: Dict[str, Material] = {}


def _to_ndgrid(df, wl_key="wl"):
    """convert stacked data to hypercube data

    Args:
        df: the dataframe to convert to hypercube data
        wl_key: which key to use to sort. Options: 'wl' or 'f'. Using 'f' yields better interpolation results
            as most optical phenomina are more linear in 'f' than in 'wl'.

    Note:
        This only works if the number of other parameters (e.g. # wls) stays consistent for each corner.
    """
    df = df.copy()
    if wl_key not in ["f", "wl"]:
        raise ValueError(
            f"Unsupported wl_key. Valid choices: 'wl', 'f'. Got: {wl_key}."
        )
    df["f"] = c / df["wl"].values
    value_columns = [c for c in ["nr", "ni"] if c in df.columns]
    param_columns = [c for c in df.columns if c not in value_columns + ["wl", "f"]] + [
        wl_key
    ]
    param_columns = [c for c in param_columns if c in df.columns]
    df = _sort_rows(
        df[param_columns + value_columns], not_by=value_columns, wl_key=wl_key
    )
    params = {c: df[c].unique() for c in param_columns}
    data = df[value_columns].values
    data = data.reshape(*(v.shape[0] for v in params.values()), len(value_columns))
    data = np.asarray(data)
    params["targets"] = np.array(value_columns, dtype=object)
    params, strings = _extract_strings(params)
    return data, params, strings


def _evaluate_general_corner_model(data, params, strings, /, **kwargs):
    given_params = {k: kwargs.get(k, np.asarray(v).mean()) for k, v in params.items()}
    given_strings = {
        p: (list(v.values()) if p not in kwargs else [v[vv] for vv in kwargs[p]])
        for p, v in strings.items()
    }
    string_locations = {k: i for i, k in enumerate(given_params) if k in given_strings}
    param_locations = {
        k: i for i, k in enumerate(given_params) if k not in given_strings
    }
    data = np.transpose(
        data, axes=[*param_locations.values(), *string_locations.values()]
    )
    _, string_shape = (
        data.shape[: len(param_locations)],
        data.shape[len(param_locations) :],
    )
    given_params = {k: v for k, v in given_params.items() if k not in given_strings}
    num_params = len(given_params)
    data = _downselect(data, given_strings.values())
    param_shape, string_shape = data.shape[:num_params], data.shape[num_params:]
    data = data.reshape(*param_shape, -1)
    given_param_values = [np.asarray(v) for v in given_params.values()]
    stacked_params = np.stack(np.broadcast_arrays(*given_param_values), 0)
    coords = _get_coordinates([v for k, v in params.items() if k in given_params], stacked_params)  # type: ignore
    result = _map_coordinates(data, coords)
    axs = {k: i for i, k in enumerate(given_strings)}
    rev_strings = {k: {vv: kk for kk, vv in v.items()} for k, v in strings.items()}
    pos = {
        k: {rev_strings[k][vv]: i for i, vv in enumerate(v)}
        for k, v in given_strings.items()
    }
    return result.reshape(*string_shape, *result.shape[1:]), axs, pos


def _downselect(data, idxs_list):
    for i, idxs in enumerate(reversed(idxs_list), start=1):
        data = data.take(np.asarray(idxs, dtype=int), axis=-i)
    return data


# @partial(jax.vmap, in_axes=(-1, None), out_axes=0)
# def _map_coordinates(input, coordinates):
#    return jax.scipy.ndimage.map_coordinates(input, coordinates, 1, mode="nearest")


def _map_coordinates(data: np.ndarray, coordinates):
    result = []
    for i in range(data.shape[-1]):
        current_result = map_coordinates(
            data[..., i], coordinates, order=1, mode="nearest"
        )
        result.append(current_result)
    return np.stack(result, 0)


def _get_coordinate(arr1d: np.ndarray, value: np.ndarray):
    return np.interp(value, arr1d, np.arange(arr1d.shape[0]))


def _get_coordinates(arrs1d: List[np.ndarray], values: np.ndarray):
    # don't use vmap as arrays in arrs1d could have different shapes...
    return np.array([_get_coordinate(a, v) for a, v in zip(arrs1d, values)])


def _extract_strings(params):
    string_map = {}
    new_params = {}
    for k, v in list(params.items()):
        if v.dtype == object:  # probably string array!
            string_map[k] = {s: i for i, s in enumerate(v)}
            new_params[k] = np.array(list(string_map[k].values()), dtype=int)
        else:
            new_params[k] = np.array(v, dtype=float)
    return new_params, string_map


def _sort_rows(df, not_by=("nr", "ni"), wl_key="wl"):
    not_by = ["wl", "f"] + list(not_by)
    by = [c for c in df.columns if c not in not_by] + [wl_key]
    return df.sort_values(by=by).reset_index(drop=True)


def _validate_path(path):
    if not path.endswith(".csv"):
        path = f"{path}.csv"
    path_parts = path.replace("\\", "/").split("/")
    if not os.path.exists(path) and len(path_parts) == 1:
        lib_path = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(lib_path, "assets", path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Material file {path!r} not found.")
    path = os.path.relpath(path, start=os.getcwd())
    return path


silicon = SampledMaterial.from_path(
    path="silicon",
    meta={"color": (0.9, 0, 0, 0.9)},
)
silicon_oxide = SampledMaterial.from_path(
    path="silicon_oxide",
    meta={"color": (0.9, 0.9, 0.9, 0.9)},
)

silicon_nitride = TidyMaterial(name="Si3N4", variant="Luke2015PMLStable")
