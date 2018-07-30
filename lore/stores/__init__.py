import os
import lore
from lore.stores.disk import Disk
from lore.stores.ram import Ram


cache = Ram()
query_cache = Disk(os.path.join(lore.env.DATA_DIR, 'query_cache'))


def cached(func):
    global cache
    return _cached(func, cache)


def query_cached(func):
    global query_cache
    return _cached(func, query_cache)


def _cached(func, store):
    def wrapper(self, *args, **kwargs):
        cache = kwargs.pop('cache', False)
        if not cache:
            return func(self, *args, **kwargs)
        
        key = store.key(instance=self, caller=func, *args, **kwargs)
        if key not in store:
            store[key] = func(self, *args, **kwargs)
        return store[key]
    
    return wrapper
