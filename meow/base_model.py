""" Provides a custom pydantic BaseModel which handles numpy arrays better """

from typing import Any, Tuple

import numpy as np
from pydantic import validator
from pydantic.main import BaseModel as _BaseModel
from pydantic.main import ModelMetaclass as _ModelMetaclass


class _array(np.ndarray):
    """just an array with a nicer repr"""

    def __repr__(self):
        if self.ndim == 0:
            return f"{self:.3e}"
        cls_str = f"array{{{'x'.join(str(i) for i in self.shape)}}}"
        self = self.ravel()
        num_els = self.shape[0]
        if num_els == 0:
            return f"{cls_str}([])"
        elif num_els == 1:
            return f"{cls_str}([{self[0]:.3e}])"
        elif num_els == 2:
            return f"{cls_str}([{self[0]:.3e}, {self[1]:.3e}])"
        elif num_els == 3:
            return f"{cls_str}([{self[0]:.3e}, {self[1]:.3e}, {self[2]:.3e}])"
        else:
            return f"{cls_str}([{self[0]:.3e}, {self[1]:.3e}, ..., {self[-1]:.3e}])"


def _serialize_array(arr):
    if np.iscomplexobj(arr):
        return {
            "real": _serialize_array(np.real(arr)),
            "imag": _serialize_array(np.imag(arr)),
        }
    else:
        return np.round(arr, 12).tolist()


class ModelMetaclass(_ModelMetaclass):
    """A metaclass (used in our custom BaseModel) to handle numpy array type hints better.

    Generic numpy type hints (will) have the following syntax::

        arr: np.ndarray[Shape, DType]

    for example::

        arr: np.ndarray[Tuple[Literal[1], int], np.dtype[np.float64]] = np.array([[1, 2, 3]])

    When using this ModelMetaclass, the BaseModel will automatically try to cast the arrays according to the given array type-hint.

    Moreover, the following config will be injected (added to the config class that might already be present)::

        Config:
            arbitrary_types_allowed = True
            json_encoders = {
                np.ndarray: lambda arr: np.round(arr, 12).tolist()
            }

    """

    def __new__(cls, name, bases, dct, **kwargs):
        # Inject sensible default Configuration (Config class)
        extra_config = {
            "arbitrary_types_allowed": True,
            "json_encoders": {
                np.ndarray: _serialize_array,
                _array: _serialize_array,
                complex: lambda value: {"real": np.real(value), "imag": np.imag(value)},
            },
        }
        if not "Config" in dct:
            dct["Config"] = type("Config", (), extra_config)
        else:
            config = {**dct["Config"].__dict__}
            config["arbitrary_types_allowed"] = True
            config["json_encoders"] = {
                **config.get("json_encoders", {}),
                **extra_config["json_encoders"],
            }
            dct["Config"] = type("Config", (dct["Config"],), config)

        # Enforce numpy array annotations
        for attr, annot in dct.get("__annotations__", {}).items():
            if not cls._is_array_annot(annot):
                continue

            @validator(attr, pre=True, allow_reuse=True, always=True)
            def validate(cls, x):
                shape, dtype = cls._parse_array_type_info(annot)
                x = np.asarray(x, dtype=(None if dtype is Any else dtype))
                if shape is not Any:
                    try:
                        shape = np.broadcast_shapes(shape, x.shape)
                    except ValueError:
                        raise ValueError(
                            f"Invalid shape for attribute 'x': Expected: {shape}. Got: {x.shape}."
                        )
                    x = np.broadcast_to(x, shape)
                return x.view(_array)  # enables a more concise repr

            dct[f"validate_{attr}"] = validate

        return super().__new__(cls, name, bases, dct, **kwargs)

    @classmethod
    def _parse_array_type_info(cls, annotation) -> Tuple[Any, Any]:
        try:
            type_info = annotation.__args__
        except AttributeError:
            type_info = tuple()
        if len(type_info) == 1:
            (shape,) = type_info
            dtype = Any
        elif len(type_info) == 2:
            shape, dtype = type_info
        else:
            shape, dtype = Any, Any
        try:
            (dtype,) = dtype.__args__  # type: ignore
        except Exception:
            dtype = Any

        try:
            shape = shape.__args__  # type: ignore
            shape = tuple(cls._try_parse_shape_int(i) for i in shape)
        except Exception:
            shape = Any

        return shape, dtype

    @staticmethod
    def _try_parse_shape_int(value):
        from typing import _LiteralGenericAlias  # type: ignore

        if isinstance(value, _LiteralGenericAlias):
            (value,) = value.__args__
        try:
            return int(value)
        except Exception:
            return 1

    @staticmethod
    def _is_array_annot(annot):
        try:
            arr = annot(0)
        except Exception:
            return False
        return isinstance(arr, np.ndarray)


class BaseModel(_BaseModel, metaclass=ModelMetaclass):
    """A customized pydantic base model that handles numpy array type hints"""

    def _repr(self, indent=0, shift=2):
        s = f"{self.__class__.__name__}("
        dic = self.dict()
        if dic:
            s = s + "\n"
        for key in dic:
            if key == "data":
                continue
            attr = getattr(self, key)
            if isinstance(attr, BaseModel):
                attr_str = attr._repr(indent=indent + shift, shift=shift)
            else:
                attr_str = repr(attr)
            s += f"{' '*(indent + shift)}{key}={attr_str}\n"
        s += f"{' '*indent})"
        return s

    def _visualize(self):
        raise NotImplementedError(
            f"visualization for {self.__class__.__name__!r} not (yet) implemented."
        )

    def __repr__(self):
        return self._repr()

    def __str__(self):
        return self._repr()
