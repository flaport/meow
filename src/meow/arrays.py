"""MEOW array tools for pydantic models."""

from __future__ import annotations

from functools import partial
from typing import Annotated, Any, Self

import numpy as np
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    GetPydanticSchema,
    InstanceOf,
    PlainSerializer,
)


class SerializedArray(BaseModel):
    """A serialized representation of a numpy array."""

    values: list[Any]
    shape: tuple[int, ...]
    dtype: str

    @classmethod
    def from_array(cls, x: np.ndarray) -> Self:
        """Create a SerializedArray from a numpy array."""
        x = np.asarray(x)
        shape = x.shape
        dtype = str(x.dtype)
        if dtype == "complex64":
            _x = x.ravel().view("float32")
        elif dtype == "complex128":
            _x = x.ravel().view("float64")
        else:
            _x = x.ravel()
        return cls(shape=shape, dtype=dtype, values=_x.tolist())

    def to_array(self) -> np.ndarray:
        """Convert the SerializedArray back to a numpy array."""
        if self.dtype == "complex128":
            arr = np.asarray(self.values, dtype="float64").view("complex128")
        elif self.dtype == "complex64":
            arr = np.asarray(self.values, dtype="float32").view("complex64")
        else:
            arr = np.asarray(self.values, dtype=self.dtype)

        if not self.shape:
            return arr
        return arr.reshape(*self.shape)


def _validate_ndarray(x: Any) -> np.ndarray:
    if isinstance(x, dict):
        return SerializedArray.model_validate(x).to_array()

    if isinstance(x, SerializedArray):
        return x.to_array()

    try:
        return np.asarray(x)
    except Exception as e:
        msg = f"Could not validate {x} as an array"
        raise ValueError(msg) from e


def _serialize_ndarray(x: np.ndarray) -> dict:
    return SerializedArray.from_array(x).model_dump()


def _coerce_immutable(x: np.ndarray) -> np.ndarray:
    x.setflags(write=False)
    return x


def _coerce_shape(arr: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    shape_to_coerce = []
    for i in range(len(shape)):
        n = shape[-i - 1]
        if n < 0 and i < len(arr.shape):
            n = arr.shape[-i - 1]
        shape_to_coerce.insert(0, n)
    return np.broadcast_to(arr, tuple(shape_to_coerce))


def _assert_shape(arr: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    shape_to_assert = []
    for i in range(len(shape)):
        n = shape[-i - 1]
        if n < 0 and i < len(arr.shape):
            n = arr.shape[-i - 1]
        shape_to_assert.insert(0, n)
    shape = tuple(shape_to_assert)
    if arr.shape != shape:
        msg = f"Expected an array of shape {shape}. Got {arr.shape}."
        raise ValueError(msg)
    return arr


def _coerce_dim(arr: np.ndarray, ndim: int) -> np.ndarray:
    if arr.ndim > ndim:
        if arr.shape[0] < 2:
            return _coerce_dim(arr[0], ndim)
        msg = f"Can't coerce arr with shape {arr.shape} into an {ndim}D array."
        raise ValueError(msg)
    if arr.ndim < ndim:
        return _coerce_dim(arr[None], ndim)
    return arr


def _assert_dim(arr: np.ndarray, ndim: int) -> np.ndarray:
    if arr.ndim != ndim:
        msg = f"Expected a {ndim}D array. Got a {arr.ndim}D array."
        raise ValueError(msg)
    return arr


def _coerce_dtype(arr: np.ndarray, dtype: str) -> np.ndarray:
    return np.asarray(arr, dtype=dtype)


def _assert_dtype(arr: np.ndarray, dtype: str) -> np.ndarray:
    if not str(arr.dtype).startswith(dtype):
        msg = (
            f"Expected an array with dtype {dtype!r}. "
            f"Got an array with dtype {str(arr.dtype)!r}."
        )
        raise ValueError(msg)
    return arr


def Dim(ndim: int, *, coerce: bool = True) -> AfterValidator:  # noqa: N802
    """Validator to ensure the array has a specific number of dimensions."""
    f = _coerce_dim if coerce else _assert_dim
    return AfterValidator(partial(f, ndim=ndim))


def DType(dtype: str, *, coerce: bool = True) -> AfterValidator:  # noqa: N802
    """Validator to ensure the array has a specific dtype."""
    f = _coerce_dtype if coerce else _assert_dtype
    return AfterValidator(partial(f, dtype=dtype))


def Shape(*shape: int, coerce: bool = True) -> AfterValidator:  # noqa: N802
    """Validator to ensure the array has a specific shape."""
    f = _coerce_shape if coerce else _assert_shape
    return AfterValidator(partial(f, shape=shape))


def _get_ndarray_core_schema(_t, h):  # noqa: ANN001,ANN202
    return h(InstanceOf[np.ndarray])


def _get_ndarray_json_schema(_t, _h):  # noqa: ANN001,ANN202
    return SerializedArray.model_json_schema()


ArraySchema = GetPydanticSchema(_get_ndarray_core_schema, _get_ndarray_json_schema)

NDArray = Annotated[
    np.ndarray,
    ArraySchema,
    PlainSerializer(_serialize_ndarray),
    BeforeValidator(_validate_ndarray),
    AfterValidator(_coerce_immutable),
]

ComplexArray2D = Annotated[NDArray, Dim(2), DType("complex128")]
FloatArray2D = Annotated[NDArray, Dim(2), DType("float64")]
BoolArray2D = Annotated[NDArray, Dim(2), DType("bool")]
IntArray2D = Annotated[NDArray, Dim(2), DType("int64")]

Complex = Annotated[NDArray, Dim(0), DType("complex128")]
