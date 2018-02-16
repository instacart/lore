import lore
import lore.stores
import os

lore.env.project = __name__
lore.stores.query_cache.limit = os.environ.get('LORE_QUERY_CACHE_LIMIT', 10000000000)
