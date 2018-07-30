from lore.stores.base import Base


class Redis(Base):
    def __init__(self, redis_conn):
        self.r = redis_conn

    def __getitem__(self, key):
        return self.r.get(key)

    def __setitem__(self, key, value):
        self.r.set(key, value)

    def __delitem__(self, key):
        self.r.delete(key)

    def __contains__(self, key):
        return self.r.get(key) != None

    def __len__(self):
        raise "Operation can be expensive. Aborting"

    def keys(self):
        raise "Operation can be expensive. Aborting"

    def values(self):
        raise "Operation can be expensive. Aborting"

    def batch_get(self, keys):
        return self.r.mget(keys)

    def batch_set(self, data_dict):
        self.r.mset(data_dict)

