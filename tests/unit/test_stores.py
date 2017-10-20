import datetime
import unittest
import os

import lore
from lore.stores import query_cached
from lore.stores.disk import Disk


class TestDisk(unittest.TestCase):
    def setUp(self):
        self.one_shot_calls = 0
        
    @query_cached
    def one_shot_function(self, **kwargs):
        if self.one_shot_calls > 0:
            raise "Can't call me twice"
        self.one_shot_calls += 1
        return self.one_shot_calls
    
    def test_disk(self):
        cache = Disk(os.path.join(lore.env.data_dir, 'cache'))

        self.assertEqual(len(cache), 0)
        self.assertEqual(cache['a'], None)
        self.assertEqual(cache.keys(), [])

        cache['a'] = 1
        self.assertEqual(len(cache), 1)
        self.assertEqual(cache['a'], 1)
        self.assertEqual(cache.keys(), ['a'])

        cache['b'] = 2
        self.assertEqual(len(cache), 2)
        self.assertEqual(cache.lru(), 'a')
        self.assertEqual(cache['b'], 2)
        self.assertEqual(cache.keys(), ['a', 'b'])

        cache['b'] = 3
        self.assertEqual(len(cache), 2)
        self.assertEqual(cache.lru(), 'a')
        self.assertEqual(cache['b'], 3)
        self.assertEqual(cache.keys(), ['a', 'b'])

        del cache['b']
        self.assertEqual(len(cache), 1)
        self.assertEqual(cache.lru(), 'a')
        self.assertFalse('b' in cache)
        self.assertEqual(cache.keys(), ['a'])

        cache.limit = 0
        cache['a'] = 1
        self.assertEqual(len(cache), 0)
        self.assertEqual(cache.lru(), None)
        self.assertFalse('a' in cache)
        self.assertEqual(cache.keys(), [])

    def test_query_cached(self):
        cache = lore.stores.query_cache
        length = len(cache)
        now = datetime.datetime.now()
        
        # first copy is stored in the cache
        calls = self.one_shot_function(when=now, int=length, str='hi', cache=True)
        self.assertEqual(length + 1, len(cache))
        self.assertEqual(1, calls)

        # second is retrieved
        calls = self.one_shot_function(when=now, int=length, str='hi', cache=True)
        self.assertEqual(length + 1, len(cache))
        self.assertEqual(1, calls)
        self.assertEqual(1, self.one_shot_calls)

