"""Cache management for models in the Meow framework."""

import os
from collections import deque
from contextlib import suppress
from functools import wraps
from typing import Any, TypeVar, cast

T = TypeVar("T", bound=type)


def _string_is_true(s: str) -> bool:
    s = str(s).lower()
    return s in ("true", "1", "yes")


CACHE_SETTINGS = {
    "size": 10000,
    "enabled": not _string_is_true(os.environ.get("MEOW_DISABLE_CACHE", "")),
}
CACHED_MODELS = {}


def enable_cache() -> bool:
    """Enable caching of models."""
    CACHE_SETTINGS["enabled"] = True
    return CACHE_SETTINGS["enabled"]


def disable_cache() -> bool:
    """Disable caching of models."""
    empty_cache()
    CACHE_SETTINGS["enabled"] = False
    return CACHE_SETTINGS["enabled"]


def empty_cache() -> None:
    """Empty the cache of models."""
    for cache in [CACHED_MODELS]:
        for k in list(cache):
            with suppress(KeyError):
                del cache[k]


def cache_model(obj: T) -> T:
    """Cache a model object if caching is enabled."""
    if not CACHE_SETTINGS["enabled"]:
        return obj
    key = hash(obj)
    if key in CACHED_MODELS:
        obj = CACHED_MODELS[key]
    else:
        CACHED_MODELS[key] = obj
    queue = deque(CACHED_MODELS)
    while len(queue) > CACHE_SETTINGS["size"]:
        key = queue.popleft()
        with suppress(KeyError):
            del CACHED_MODELS[key]
    return obj


def cached_model(cls: type[T]) -> type[T]:
    """Decorator to cache a model class instance."""

    @wraps(cls)  # type: ignore[reportArgumentType]
    def model(*args: Any, **kwargs: Any) -> T:
        return cache_model(cls(*args, **kwargs))

    return cast(type[T], model)
