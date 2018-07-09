# coding=utf-8
from __future__ import unicode_literals

import unittest

import lore.io

import os


class TestRemoteFromLocalPaths(unittest.TestCase):
    def test_works_dir_is_removed(self):
        local = os.path.join(lore.env.WORK_DIR, 'README.rst')
        self.assertIsNotNone(lore.env.WORK_DIR)
        self.assertEqual(lore.io.remote_from_local(local), '/README.rst')

    def test_relative_is_ok(self):
        local = 'README.rst'
        self.assertEqual(lore.io.remote_from_local(local), 'README.rst')

    def test_absolute_is_ok(self):
        local = '/README.rst'
        self.assertEqual(lore.io.remote_from_local(local), '/README.rst')


class TestPrefixRemoteRoot(unittest.TestCase):
    def test_absolute(self):
        path = '/README.rst'
        self.assertEqual(lore.io.prefix_remote_root(path), 'test/README.rst')

    def test_relative(self):
        path = 'README.rst'
        self.assertEqual(lore.io.prefix_remote_root(path), 'test/README.rst')

    def test_is_absolute_pre_prefixed_safe(self):
        path = '/test/README.rst'
        self.assertEqual(lore.io.prefix_remote_root(path), 'test/README.rst')

    def test_is_relative_pre_prefixed_safe(self):
        path = 'test/README.rst'
        self.assertEqual(lore.io.prefix_remote_root(path), 'test/README.rst')

    def test_is_not_short_sighted(self):
        path = 'test_not_env/README.rst'
        self.assertEqual(lore.io.prefix_remote_root(path), 'test/test_not_env/README.rst')

    def test_does_not_double_slash(self):
        path = '/test_not_env/README.rst'
        self.assertEqual(lore.io.prefix_remote_root(path), 'test/test_not_env/README.rst')
