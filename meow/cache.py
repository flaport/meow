from collections import deque

CACHE_SETTINGS = {"size": 1000, "enabled": True}
CACHED_MODELS = {}
CACHED_ARRAYS = {}


def enable_cache():
    CACHE_SETTINGS["enabled"] = True
    return CACHE_SETTINGS["enabled"]


def disable_cache():
    empty_cache()
    CACHE_SETTINGS["enabled"] = False
    return CACHE_SETTINGS["enabled"]


def empty_cache():
    for cache in [CACHED_MODELS, CACHED_ARRAYS]:
        for k in list(cache):
            try:
                del cache[k]
            except KeyError:
                pass  # sometimes in threaded apps key might already be deleted


def cache_model(model):
    return _cache_obj(CACHED_MODELS, model)


def cache_array(arr):
    return _cache_obj(CACHED_ARRAYS, arr)


def _cache_obj(cache, obj):
    if not CACHE_SETTINGS["enabled"]:
        return obj
    key = hash(obj)
    if key in cache:
        obj = cache[key] = cache.pop(key, obj)
    else:
        cache[key] = obj
    queue = deque(cache)
    while len(queue) > CACHE_SETTINGS["size"]:
        key = queue.popleft()
        try:
            del cache[key]
        except KeyError:
            pass  # sometimes in threaded apps key might already be deleted
    return obj
