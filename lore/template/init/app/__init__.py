import lore
import lore.stores
import os

lore.env.APP = __name__
lore.stores.query_cache.limit = int(os.environ.get('LORE_QUERY_CACHE_LIMIT', 10000000000))
