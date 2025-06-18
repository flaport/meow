"""The pydantic base model all other models are based on."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import suppress
from functools import wraps
from hashlib import md5
from typing import Any, Self

import numpy as np
import orjson
from pydantic import BaseModel as _BaseModel
from pydantic import (
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic._internal._model_construction import ModelMetaclass as _ModelMetaclass

from meow.cache import cache_model

MODELS = {}


class ModelMetaclass(_ModelMetaclass):
    """Metaclass for all models."""

    def __call__(cls, *args: Any, **kwargs: Any) -> type:  # noqa: N805,ANN401,D102
        obj = super().__call__(*args, **kwargs)
        return cache_model(obj)


class BaseModel(_BaseModel, metaclass=ModelMetaclass):
    """Base model class for all models."""

    type: str = Field(default="", validate_default=True)
    _cache: dict = PrivateAttr(default_factory=dict)

    model_config = ConfigDict(
        extra="ignore",
        frozen=True,
    )

    def __init_subclass__(cls) -> None:
        MODELS[cls.__name__] = cls

    @field_validator("type")
    @classmethod
    def _validate_field_type(cls, field: str) -> str:
        field = cls.__name__
        return field

    @model_validator(mode="before")
    @classmethod
    def _validate_model(cls, obj: Any) -> Any:  # noqa: ANN401
        if isinstance(obj, dict):
            tpe = obj.get("type", cls.__name__)
            if tpe != cls.__name__:
                obj["type"] = tpe
                cls = MODELS.get(tpe, cls)  # noqa: PLW0642
                obj = cls.model_validate(obj)
        return obj

    @classmethod
    def model_validate(
        cls,
        obj: Any,  # noqa: ANN401
        *,
        strict: bool | None = None,
        from_attributes: bool | None = None,
        context: dict[str, Any] | None = None,
    ) -> Self:
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
            cls = MODELS.get(obj.get("type", cls.__name__), cls)  # noqa: PLW0642

        return cls.__pydantic_validator__.validate_python(
            obj, strict=strict, from_attributes=from_attributes, context=context
        )

    @classmethod
    def model_validate_json(
        cls,
        json_data: str | bytes | bytearray,
        *,
        strict: bool | None = None,
        context: dict[str, Any] | None = None,
    ) -> Self:
        """Validate a pydantic model instance.

        Args:
            json_data: The object to validate.
            strict: Whether to enforce types strictly.
            from_attributes: Whether to extract data from object attributes.
            context: Additional context to pass to the validator.

        Raises:
            ValidationError: If the object could not be validated.

        Returns:
            The validated model instance.
        """
        if isinstance(json_data, str):
            json_data = json_data.encode()
        dct = orjson.loads(json_data)
        return cls.model_validate(dct, strict=strict, context=context)

    def __hash__(self) -> int:
        to_hash = {}
        for k, v in self:
            if isinstance(v, np.ndarray):
                with suppress(ValueError):
                    v = v.item()
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
        return abs(int.from_bytes(md5(bts[:-1]).digest(), byteorder="big") % 10**18)  # noqa: S324

    def __eq__(self, other: object) -> bool:
        if isinstance(other, dict):
            try:
                other = self.__class__.model_validate(other)
            except Exception:  # noqa: BLE001
                return False
        elif isinstance(other, str):
            try:
                other = self.__class__.model_validate_json(other)
            except Exception:  # noqa: BLE001
                return False
        return _eq(self, other)

    def _repr(self, *, indent: int = 0, shift: int = 2) -> str:
        start = f"{self.__class__.__name__}("
        dct: dict[str, Any] = dict(self)  # don't model_dump() here.
        return _dict_repr(
            dct,
            indent=indent,
            shift=shift,
            start=start,
            end=")",
            eq="=",
        )

    def __repr__(self) -> str:
        return self._repr()

    def __str__(self) -> str:
        return self._repr()

    def _visualize(self, **kwargs: Any) -> None:  # noqa: ANN401
        msg = f"visualization for {self.__class__.__name__!r} not (yet) implemented."
        raise NotImplementedError(msg)


def cache(prop: Callable) -> Callable:
    """Decorator to cache the result of a property method."""
    prop_name = prop.__name__

    @wraps(prop)
    def getter(self):  # noqa: ANN001,ANN202
        stored_value = self._cache.get(prop_name)

        if stored_value is not None:
            return stored_value

        computed = prop(self)
        self._cache[prop_name] = computed
        return computed

    return getter


def cached_property(method: Callable):  # noqa: ANN201
    """Decorator to cache the result of a property method."""
    return property(cache(method))


def _dict_repr(
    dct: dict,
    *,
    indent: int = 0,
    shift: int = 2,
    start: str = "{",
    end: str = "}",
    eq: str = ": ",
) -> str:
    from .array import SerializedArray

    try:
        arr = SerializedArray.model_validate(dct).to_array()
        return _array_repr(arr)
    except ValidationError:
        pass

    s = f"{start}"
    if dct:
        s = s + "\n"
    for key, attr in dct.items():
        if key == "data":
            continue
        if isinstance(attr, BaseModel):
            attr_str = attr._repr(indent=indent + shift, shift=shift)  # noqa: SLF001
        elif isinstance(attr, dict):
            attr_str = _dict_repr(attr, indent=indent + shift, shift=shift)
        elif isinstance(attr, list):
            attr_str = _dict_repr(
                dict(enumerate(attr)), indent=indent + shift, shift=shift
            )
        elif isinstance(attr, np.ndarray):
            attr_str = _array_repr(attr)
        else:
            attr_str = repr(attr)
        s += f"{' ' * (indent + shift)}{key}{eq}{attr_str}\n"
    if dct:
        s += f"{' ' * indent}{end}"
    else:
        s += end
    return s


def _array_repr(arr: np.ndarray) -> str:
    if arr.ndim == 0:
        return f"{arr:.3e}"
    cls_str = f"array{{{'x'.join(str(i) for i in arr.shape)},{arr.dtype}}}"
    arr = arr.ravel()
    num_els = arr.shape[0]
    if num_els == 0:
        return f"{cls_str}([])"
    if num_els == 1:
        return f"{cls_str}([{arr[0]:.3e}])"
    if num_els == 2:
        return f"{cls_str}([{arr[0]:.3e}, {arr[1]:.3e}])"
    if num_els == 3:
        return f"{cls_str}([{arr[0]:.3e}, {arr[1]:.3e}, {arr[2]:.3e}])"
    return f"{cls_str}([{arr[0]:.3e}, {arr[1]:.3e}, ..., {arr[-1]:.3e}])"


def _eq(self: object, other: object) -> bool:
    if isinstance(self, BaseModel):
        if isinstance(other, dict):
            return _eq(self.model_dump(), other)
        if isinstance(other, BaseModel):
            return _eq(self.model_dump(), other.model_dump())
        return False
    if isinstance(self, dict):
        if not isinstance(other, dict):
            return False
        if not _eq(list(self), list(other)):
            return False
        for k, v in self.items():
            if k not in other:
                return False
            w = other[k]
            if not _eq(v, w):
                return False
        return True
    if isinstance(self, np.ndarray):
        try:
            other = np.asarray(other)
        except Exception:  # noqa: BLE001
            return False
        if self.dtype != other.dtype:
            return False
        try:
            self, other = np.broadcast_arrays(self, other)
        except ValueError:
            return False
        if self.ndim == 0:
            return _eq(self.item(), other.item())
        if self.dtype == object:
            return _eq(self.tolist(), other.tolist())
        return bool((abs(self - other) < 1e-6).all())
    if isinstance(self, str):
        if isinstance(other, bytes):
            other = other.decode()
        return self == other
    if isinstance(self, bytes):
        if isinstance(other, str):
            other = other.encode()
        return self == other
    if isinstance(self, Iterable):
        if not isinstance(other, Iterable):
            return False
        if not isinstance(self, list) or not isinstance(other, list):
            return _eq(list(self), list(other))
        return all(_eq(a, b) for a, b in zip(self, other, strict=False))
    if isinstance(self, float) or isinstance(other, float):
        return abs(self - other) < 1e-6  # type: ignore[reportOperatorIssue]
    return self == other
