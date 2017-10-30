import unittest

class TestConnection(unittest.TestCase):
    def test_connection(self):
        # make sure this at least gets loaded
        import lore.__main__
        self.assertTrue(True)
