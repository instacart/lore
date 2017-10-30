# coding=utf-8
from __future__ import unicode_literals

import unittest

import lore.io

import os


class TestRemoteFromLocalPaths(unittest.TestCase):
    def test_absolute_is_works_dir_relative(self):
        local = '/README.rst'
        self.assertEqual(lore.io.remote_from_local(local), 'test/README.rst')

    def test_relative_is_environment_specific(self):
        local = 'README.rst'
        self.assertEqual(lore.io.remote_from_local(local), 'test/README.rst')

    def test_work_dir_is_relative_base(self):
        local = os.path.join(lore.env.work_dir, 'README.rst')
        self.assertIsNotNone(lore.env.work_dir)
        self.assertEqual(lore.io.remote_from_local(local), 'test/README.rst')

