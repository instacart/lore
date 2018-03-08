from __future__ import generators

import unittest

import pandas

import tests.mocks.pipelines


class TestTimedTrainTestSplit(unittest.TestCase):
    
    def test_time_series(self):
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


class TestLowMemory(unittest.TestCase):
    
    def setUp(self):
        self.dataframe = tests.mocks.pipelines.Users.dataframe
        self.pipeline = tests.mocks.pipelines.Users()
    
    # def test_has_a_name(self):
    #     self.assertIsNotNone(self.pipeline.name)
    #
    # def test_columns(self):
    #     self.assertEqual(set(self.pipeline.columns), set(self.dataframe.columns))
    #
    # def test_split(self):
    #     self.pipeline._split_data()
    #     self.assertEqual(self.pipeline.table_length(self.pipeline.table_training), 800)
    #     self.assertEqual(self.pipeline.table_length(self.pipeline.table_validation), 100)
    #     self.assertEqual(self.pipeline.table_length(self.pipeline.table_test), 100)
    #
    # def test_split_stratify(self):
    #     self.pipeline = tests.mocks.pipelines.Users()
    #     self.pipeline.stratify = 'last_name'
    #
    #     data = pandas.concat([chunk for chunk in self.pipeline.training_data])
    #     self.assertTrue(len(data['last_name'].drop_duplicates()), 80)
    #
    #     data = pandas.concat([chunk for chunk in self.pipeline.validation_data])
    #     self.assertTrue(len(data['last_name'].drop_duplicates()), 10)
    #
    #     data = pandas.concat([chunk for chunk in self.pipeline.test_data])
    #     self.assertTrue(len(data['last_name'].drop_duplicates()), 10)
    #
    # def test_split_subsample(self):
    #     self.pipeline = tests.mocks.pipelines.Users()
    #     self.pipeline.subsample = 50
    #
    #     self.pipeline._split_data()
    #     self.assertEqual(self.pipeline.table_length(self.pipeline.table_training), 40)
    #     self.assertEqual(self.pipeline.table_length(self.pipeline.table_validation), 5)
    #     self.assertEqual(self.pipeline.table_length(self.pipeline.table_test), 5)

    def test_encoded_data(self):
        self.pipeline = tests.mocks.pipelines.MockData1()
        self.pipeline.subsample = 50
        # self.pipeline.connection.execute('drop table if exists {name}'.format(name=self.pipeline.table_training + '_random'))
        print(self.pipeline.encoded_training_data)
        # self.assertEqual(len(pandas.concat([chunk.x for chunk in self.pipeline.encoded_training_data])), 40)
        # self.assertEqual(len(pandas.concat([chunk.x for chunk in self.pipeline.encoded_validation_data])), 5)
        # self.assertEqual(len(pandas.concat([chunk.x for chunk in self.pipeline.encoded_test_data])), 5)

    # def test_preserves_types(self):
    #     self.pipeline = tests.mocks.pipelines.Users()
    #     training_data = pandas.concat([chunk for chunk in self.pipeline.training_data])
    #     self.assertTrue(training_data['id'].dtype, 'integer')
    #     self.assertTrue(training_data['first_name'].dtype, 'object')
    #     self.assertTrue(training_data['last_name'].dtype, 'object')
    #     self.assertTrue(training_data['subscriber'].dtype, 'bool')
    #     self.assertTrue(training_data['signup_at'].dtype, 'datetime64[ns]')
    #
    # def test_generator(self):
    #     pipeline = tests.mocks.pipelines.Users()
    #     pipeline.stratify = 'last_name'
    #     chunks = 0
    #     length = 0
    #     for chunk in pipeline.generator(pipeline.table_training, orient='row', encoded=True, stratify=False, chunksize=200):
    #         chunks += 1
    #         length += len(chunk.x)
    #     self.assertEqual(chunks, 4)
    #     self.assertEqual(length, 800)
    #
    #     chunks = 0
    #     length = 0
    #     for chunk in pipeline.generator(pipeline.table_training, orient='row', encoded=True, stratify=True, chunksize=10):
    #         chunks += 1
    #         length += len(chunk.x)
    #     self.assertEqual(chunks, 80)
    #     self.assertEqual(length, 800)
    #
    #     chunks = 0
    #     length = 0
    #     for chunk in pipeline.generator(pipeline.table_training, orient='column', encoded=True):
    #         chunks += 1
    #         length += len(chunk.x)
    #     self.assertEqual(chunks, 5)
    #     self.assertEqual(length, 5 * 800)
