# coding=utf-8
import unittest
import lore
import lore.env


class TestEnv(unittest.TestCase):
    def test_initialization(self):
        self.assertEqual(lore.env.NAME, lore.env.TEST)
