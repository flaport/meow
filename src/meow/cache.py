import os
from collections import deque
from typing import Type, TypeVar

CACHE_SETTINGS = {
    "size": 10000,
    "enabled": not os.environ.get("MEOW_DISABLE_CACHE", False),
}
CACHED_MODELS = {}


def enable_cache():
    CACHE_SETTINGS["enabled"] = True
    return CACHE_SETTINGS["enabled"]


def disable_cache():
    empty_cache()
    CACHE_SETTINGS["enabled"] = False
    return CACHE_SETTINGS["enabled"]


def empty_cache():
    for cache in [CACHED_MODELS]:
        for k in list(cache):
            try:
                del cache[k]
            except KeyError:
                pass  # sometimes in threaded apps key might already be deleted


def cache_model(obj):
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
        try:
            del CACHED_MODELS[key]
        except KeyError:
            pass  # sometimes in threaded apps key might already be deleted
    return obj


T = TypeVar("T", bound=Type)


def cached_model(cls: T) -> T:
    def model(*args, **kwargs) -> cls:  # type: ignore
        return cache_model(cls(*args, **kwargs))

    return model  # type: ignore
