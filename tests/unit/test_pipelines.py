import tests.mocks.pipelines
import unittest


class TestTimedTrainTestSplit(unittest.TestCase):
    def test_timed_train_test_split(self):
        mock = tests.mocks.pipelines.MockData(sort_by=None)
        self.assertEqual(mock.training_data['a'].max(), 8)
        self.assertEqual(mock.validation_data['a'].max(), 9)
        self.assertEqual(mock.test_data['a'].max(), 10)

        mock = tests.mocks.pipelines.MockData(sort_by='a')
        self.assertEqual(mock.training_data['a'].max(), 8)
        self.assertEqual(mock.validation_data['a'].max(), 9)
        self.assertEqual(mock.test_data['a'].max(), 10)

        mock = tests.mocks.pipelines.MockData(sort_by='b')
        self.assertEqual(mock.training_data['b'].max(), 28)
        self.assertEqual(mock.validation_data['b'].max(), 29)
        self.assertEqual(mock.test_data['b'].max(), 30)
