import unittest
from lore.stores.redis import Redis
from mock import patch, Mock

class TestRedis(unittest.TestCase):


    def test_redis(self):
        cache = Redis()
        self.assertTrue(cache)