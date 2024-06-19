""" the pydantic base model all other models are based on """

from __future__ import annotations

from functools import wraps
from hashlib import md5
from typing import Any

import black
import numpy as np
from pydantic import BaseModel as _BaseModel
from pydantic import ConfigDict, Field, PrivateAttr, field_validator, model_validator

from meow.cache import cache_model

MODELS = {}


class BaseModel(_BaseModel):
    type: str = Field(default="", validate_default=True)
    _cache: dict = PrivateAttr(default_factory=dict)

    model_config = ConfigDict(
        extra="ignore",
        frozen=True,
    )

    def __init_subclass__(cls):
        MODELS[cls.__name__] = cls

    @field_validator("type")
    @classmethod
    def _validate_field_type(cls, field):
        field = cls.__name__
        return field

    @model_validator(mode="before")
    @classmethod
    def _validate_model(cls, obj: Any):
        if isinstance(obj, dict):
            type = obj.get("type", cls.__name__)
            if type != cls.__name__:
                obj["type"] = type
                cls = MODELS.get(type, cls)
                obj = cls.model_validate(obj)
        return obj

    @model_validator(mode="after")
    def _retrieve_cached_model(self):
        return cache_model(self)

    @classmethod
    def model_validate(
        cls,
        obj: Any,
        *,
        strict: bool | None = None,
        from_attributes: bool | None = None,
        context: dict[str, Any] | None = None,
    ):
        """Validate a pydantic model instance.

        Args:
            obj: The object to validate.
            strict: Whether to enforce types strictly.
            from_attributes: Whether to extract data from object attributes.
            context: Additional context to pass to the validator.

        Raises:
            ValidationError: If the object could not be validated.

        Returns:
            The validated model instance.
        """
        __tracebackhide__ = True

        if isinstance(obj, cls):
            return obj

        if isinstance(obj, dict):
            cls = MODELS.get(obj.get("type", cls.__name__), cls)

        return cls.__pydantic_validator__.validate_python(
            obj, strict=strict, from_attributes=from_attributes, context=context
        )

    def __hash__(self):
        to_hash = {}
        for k, v in self:
            if isinstance(v, np.ndarray):
                try:
                    v = v.item()
                except ValueError:
                    pass
            if isinstance(v, np.ndarray):
                to_hash[k] = np.round(v, 9).tobytes()
            elif isinstance(v, str):
                to_hash[k] = v.encode()
            elif isinstance(v, float):
                to_hash[k] = f"{v:.9f}".encode()
            else:
                to_hash[k] = str(v).encode()
        bts = b""
        for k, v in to_hash.items():
            bts += k.encode() + b":" + v + b"|"
        return abs(int.from_bytes(md5(bts[:-1]).digest(), byteorder="big") % 10**18)

    def __eq__(self, other):
        eq = True
        for k, v in self.model_dump().items():
            if isinstance(v, np.ndarray):
                eq &= bool(((v - getattr(other, k)) < 1e-6).all())
            else:
                eq &= bool(v == getattr(other, k))
        return eq

    def _repr(self, indent=0, shift=2):
        s = f"{self.__class__.__name__}("

        dct: dict[str, Any] = {}
        fields = set()
        for k, v in self:
            fields.add(k)
            if isinstance(v, np.ndarray):
                dct[k] = v.view(_compact_repr_array)
            else:
                dct[k] = v
        if dct:
            s = s + "\n"
        for key in dct:
            if key == "data":
                continue
            attr = getattr(self, key)
            if isinstance(attr, BaseModel):
                attr_str = attr._repr(indent=indent + shift, shift=shift)
            else:
                attr_str = repr(attr)
            s += f"{' '*(indent + shift)}{key}={attr_str}\n"
        if dct:
            s += f"{' '*indent})"
        else:
            s += ")"
        return s

    def __repr__(self):
        return self._repr()

    def __str__(self):
        return self._repr()

    def _visualize(self, **kwargs):
        raise NotImplementedError(
            f"visualization for {self.__class__.__name__!r} not (yet) implemented."
        )


def cache(prop):
    prop_name = prop.__name__

    @wraps(prop)
    def getter(self):
        stored_value = self._cache.get(prop_name)

        if stored_value is not None:
            return stored_value

        computed = prop(self)
        self._cache[prop_name] = computed
        return computed

    return getter


def cached_property(method):
    return property(cache(method))


class _compact_repr_array(np.ndarray):
    """just an utility array class with a more compact repr"""

    def __repr__(self):
        if self.ndim == 0:
            return f"{self:.3e}"
        cls_str = (
            f"array___7B{'x'.join(str(i) for i in self.shape)}___2C{self.dtype}___7D"
        )
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
