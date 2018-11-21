import unittest

class TestConnection(unittest.TestCase):
    def test_connection(self):
        # make sure this at least gets loaded
        import lore.__main__
        self.assertTrue(True)


class TestTask(unittest.TestCase):
    def test_task(self):
        import lore.__main__

        args = ('task', 'tests.mocks.tasks.EchoTask', '--arg1', 'true')

        with self.assertLogs('lore.__main__') as log:
            lore.__main__.main(args)
            self.assertEqual(log.output, [
                "INFO:lore.__main__:starting task: " +
                "tests.mocks.tasks.EchoTask {'arg1': 'true'}"
            ])
