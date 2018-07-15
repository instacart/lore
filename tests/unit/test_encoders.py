# coding=utf-8
from __future__ import unicode_literals

import unittest
from datetime import datetime

import lore.encoders
import numpy
import pandas

import lore


class TestEquals(unittest.TestCase):
    def setUp(self):
        self.encoder = lore.encoders.Equals('left', 'right')

    def test_equality(self):
        data = pandas.DataFrame({
            'left': [None, 1, 2, 2],
            'right': [None, None, 1, 2]
        })
        
        self.encoder.fit(data)
        encoded = self.encoder.transform(data)
        self.assertEqual(list(encoded), [0., 0., 0., 1.])

    def test_cardinality(self):
        self.assertEqual(2, self.encoder.cardinality())


class TestUniform(unittest.TestCase):
    def setUp(self):
        self.encoder = lore.encoders.Uniform('test')
        self.encoder.fit(pandas.DataFrame({'test': [3, 1, 2, 1, 4]}))

    def test_cardinality(self):
        self.assertRaises(ValueError, self.encoder.cardinality)

    def test_reverse_transform(self):
        a = [1, 2, 3]
        b = self.encoder.reverse_transform(
            self.encoder.transform(pandas.DataFrame({'test': a}))
        ).tolist()
        self.assertEqual(a, b)

    def test_high_outliers_are_capped(self):
        a = self.encoder.transform(pandas.DataFrame({'test': [100, 10000, float('inf')]})).tolist()
        self.assertEqual(a, [1, 1, 1])

    def test_low_outliers_are_capped(self):
        a = self.encoder.transform(pandas.DataFrame({'test': [-100, -10000, -float('inf')]})).tolist()
        self.assertEqual(a, [0, 0, 0])

    def test_handles_nans(self):
        self.encoder = lore.encoders.Uniform('test')
        self.encoder.fit(pandas.DataFrame({'test': [3, 1, 2, 1, 4, None]}))
        a = self.encoder.transform(pandas.DataFrame({'test': [None, float('nan')]}))
        self.assertEqual(a.tolist(), [0.0, 0.0])


class TestNorm(unittest.TestCase):
    def setUp(self):
        self.encoder = lore.encoders.Norm('test', dtype=numpy.float64)
        self.x = [3, 1, 2, 1, 4]
        self.mean = numpy.mean(self.x)
        self.std = numpy.std(self.x)
        self.encoder.fit(pandas.DataFrame({'test': self.x}))

    def test_mean_sd(self):
        a = numpy.arange(10)
        enc = lore.encoders.Norm('test', dtype=numpy.float64)
        data = pandas.DataFrame({'test': a})
        enc.fit(data)
        b = enc.transform(data)
        self.assertAlmostEqual(numpy.mean(b), 0)
        self.assertAlmostEqual(numpy.std(b), 1)

    def test_cardinality(self):
        self.assertRaises(ValueError, self.encoder.cardinality)

    def test_reverse_transform(self):
        a = [1, 2, 3]
        b = self.encoder.reverse_transform(
            self.encoder.transform(pandas.DataFrame({'test': a}))
        ).tolist()
        self.assertEqual(a, b)

    def test_high_outliers_are_capped(self):
        a = self.encoder.transform(pandas.DataFrame({'test': [100, 10000, float('inf')]})).tolist()
        b = (([4, 4, 4] - self.mean) / self.std).tolist()
        self.assertEqual(a, b)

    def test_low_outliers_are_capped(self):
        a = self.encoder.transform(pandas.DataFrame({'test': [-100, -10000, -float('inf')]})).tolist()
        b = (([1, 1, 1] - self.mean) / self.std).tolist()
        self.assertEqual(a, b)

    def test_handles_nans(self):
        self.encoder = lore.encoders.Norm('test')
        self.encoder.fit(pandas.DataFrame({'test': [3, 1, 2, 1, 4, None]}))
        a = self.encoder.transform(pandas.DataFrame({'test': [None, float('nan')]}))
        self.assertEqual(a.tolist(), [0.0, 0.0])

    def test_accidental_object_type_array_from_none(self):
        self.encoder = lore.encoders.Norm('test')
        self.encoder.fit(pandas.DataFrame({'test': [3, 1, 2, 1, 4]}))
        a = self.encoder.transform(pandas.DataFrame({'test': [None]}))
        self.assertEqual(a.tolist(), [0.0])


class TestDiscrete(unittest.TestCase):
    def setUp(self):
        self.encoder = lore.encoders.Discrete('test', bins=5)
        self.encoder.fit(pandas.DataFrame({'test': [0, 4]}))

    def test_cardinality(self):
        self.assertEqual(6, self.encoder.cardinality())

    def test_bins(self):
        b = self.encoder.transform(pandas.DataFrame({'test': [0, 1, 2, 3, 4]}))
        self.assertEqual(b.tolist(), [0.0, 1.0, 2.0, 3.0, 4.0])

    def test_bins_halved(self):
        b = self.encoder.transform(pandas.DataFrame({'test': [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5]}))
        self.assertEqual(b.tolist(), [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0])

    def test_limits(self):
        b = self.encoder.transform(pandas.DataFrame({'test': [float('nan'), -1, 0, 3, 7, float('inf')]}))
        self.assertEqual(b.tolist(), [5.0, 0.0, 0.0, 3.0, 4.0, 4.0])

    def test_datetimes(self):
        self.encoder = lore.encoders.Discrete('test', bins=5)
        fit = [datetime(1, 1, 1), datetime(5, 5, 5)]
        self.encoder.fit(pandas.DataFrame({'test': fit}))
        test = [
            datetime(1, 1, 1),
            datetime(2, 2, 2),
            datetime(3, 3, 3),
            datetime(4, 4, 4),
            datetime(5, 5, 5),
        ]
        b = self.encoder.transform(pandas.DataFrame({'test': test}))
        self.assertEqual(b.tolist(), [0.0, 1.0, 1.0, 3.0, 4.0])


class TestBoolean(unittest.TestCase):
    def setUp(self):
        self.encoder = lore.encoders.Boolean('test')
    
    def test_cardinality(self):
        self.assertEqual(self.encoder.cardinality(), 3)
    
    def test_handles_nans(self):
        a = self.encoder.transform(pandas.DataFrame({'test': [
            0, False, 123, True, float('nan'), None, float('inf')
        ]}))
        
        self.assertEqual(a.tolist(), [0, 0, 1, 1, 2, 2, 1])


class TestEnum(unittest.TestCase):
    def setUp(self):
        self.encoder = lore.encoders.Enum('test')
        self.encoder.fit(pandas.DataFrame({'test': [0, 1, 4]}))
        
    def test_cardinality(self):
        self.assertEqual(self.encoder.cardinality(), 7)
        
    def test_outliers_are_unfit(self):
        a = self.encoder.transform(pandas.DataFrame({'test': [100, -1]})).tolist()
        self.assertEqual(a, [self.encoder.unfit_value, self.encoder.unfit_value])

    def test_handles_nans(self):
        self.encoder = lore.encoders.Enum('test')
        self.encoder.fit(pandas.DataFrame({'test': [3, 1, 2, 1, 4, None]}))
        a = self.encoder.transform(pandas.DataFrame({'test': [
            None, float('nan'), 0, 4.0, 5, float('inf')
        ]}))
        self.assertEqual(a.tolist(), [
            self.encoder.missing_value,
            self.encoder.missing_value,
            0,
            4,
            self.encoder.unfit_value,
            self.encoder.unfit_value
        ])


class TestQuantile(unittest.TestCase):
    def setUp(self):
        self.encoder = lore.encoders.Quantile('test', quantiles=5)
        self.encoder.fit(pandas.DataFrame({'test': [1, 2, 3, 4, 5, 6, 10, 20, 30, 40, 50, 100, 1000]}))
    
    def test_cardinality(self):
        self.assertEqual(self.encoder.cardinality(), 8)
    
    def test_unfit_data(self):
        a = self.encoder.transform(pandas.DataFrame({'test': [0, 1, 2, 3, 4, 5, 1110000, float('inf'), None, float('nan')]})).tolist()
        self.assertEqual(a, [5, 0, 0, 0, 1, 1, 6, 6, 7, 7])

    def test_long_ties(self):
        self.encoder.fit(pandas.DataFrame({'test': [1, 1, 1, 1, 1, 1, 1, 1, 9, 10]}))
        self.assertEqual(self.encoder.cardinality(), 5)  # 1, [9, 10] + lower, upper, missing
        a = self.encoder.transform(pandas.DataFrame({'test': range(0, 12)}))
        self.assertEqual(a.tolist(), [2, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 3])

    def test_reverse_transform(self):
        a = self.encoder.transform(pandas.DataFrame({'test': [0, 1, 2, 3, 4, 5, 1110000, None, float('nan')]}))
        self.assertEqual(self.encoder.reverse_transform(a).tolist(), [
            '<1',
             1.0,
             1.0,
             1.0,
             3.4000000000000004,
             3.4000000000000004,
             '>1000',
             None,
             None
        ])


class TestUnique(unittest.TestCase):
    def setUp(self):
        self.encoder = lore.encoders.Unique('test')
        self.encoder.fit(pandas.DataFrame({'test': ['a', 'b', 'd', 'c', 'b']}))

    def test_cardinality(self):
        self.assertEqual(self.encoder.cardinality(), 7)

    def test_avoiding_0(self):
        data = pandas.DataFrame({'test': ['a', 'b', 'e', None]})
        transform = self.encoder.transform(data)
        self.assertFalse(0 in transform)

    def test_minimum_occurrences(self):
        self.encoder = lore.encoders.Unique('test', minimum_occurrences=2)
        self.encoder.fit(pandas.DataFrame({'test': ['a', 'b', 'd', 'c', 'b']}))
        self.assertEqual(self.encoder.cardinality(), 4)
        b = self.encoder.reverse_transform(
            self.encoder.transform(pandas.DataFrame({'test': ['a', 'notafitlabel']}))
        ).tolist()
        self.assertEqual(['LONG_TAIL', 'LONG_TAIL'], b)

    def test_reverse_transform(self):
        a = ['a', 'b', 'c']
        b = self.encoder.reverse_transform(
            self.encoder.transform(pandas.DataFrame({'test': a}))
        ).tolist()
        self.assertEqual(a, b)

    def test_handles_missing_labels(self):
        a = [None, float('nan')]
        b = self.encoder.reverse_transform(
            self.encoder.transform(pandas.DataFrame({'test': a}))
        ).tolist()
        self.assertEqual(['MISSING_VALUE', 'MISSING_VALUE'], b)
        self.assertEqual(self.encoder.cardinality(), 7)

    def test_handles_stratify(self):
        encoder = lore.encoders.Unique(
            'test', stratify='user_id', minimum_occurrences=2)
        encoder.fit(pandas.DataFrame({
            'test': ['a', 'b', 'd', 'c', 'b', 'a'],
            'user_id': [0, 1, 2, 3, 4, 0]
        }))
        self.assertEqual(encoder.cardinality(), 4)


class TestOneHot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.encoder = lore.encoders.OneHot('test')
        cls.input_df = pandas.DataFrame({'test': [i for s in [[j]*j for j in range(1, 4)] for i in s]})
        cls.test_df = pandas.DataFrame({'test': [i for s in [[j]*j for j in range(1, 5)] for i in s]})
        cls.encoder.fit(cls.input_df)

    def test_onehot(self):
        output_df = self.encoder.transform(self.test_df)
        result_matrix = numpy.zeros((10, 3))
        for j in range(result_matrix.shape[1]):
            for i in range(int(j*(j + 1)/2), int(j*(j+1)/2) + j + 1):
                result_matrix[i, j] = 1
        self.assertEqual((output_df.values == result_matrix).all(), True)

    def test_compessed_one_hot(self):
        self.encoder = lore.encoders.OneHot('test', compressed=True, minimum_occurrences=2)
        self.encoder.fit(self.input_df)
        output_df = self.encoder.transform(self.test_df)
        result_matrix = numpy.zeros((10, 2))
        for j in range(result_matrix.shape[1]):
            for i in range(int((j + 1)*(j + 2)/2), int((j + 1)*(j+2)/2) + j + 2):
                result_matrix[i, j] = 1
        self.assertEqual((output_df.values == result_matrix).all(), True)


class TestToken(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.encoder = lore.encoders.Token('test', sequence_length=3)
        cls.encoder.fit(pandas.DataFrame({'test': ['apple, !orange! carrot']}))

    def test_cardinality(self):
        self.assertEqual(self.encoder.cardinality(), 6)

    def test_reverse_transform(self):
        a = ['apple orange carrot', 'carrot orange apple']
        b = self.encoder.reverse_transform(
            self.encoder.transform(pandas.DataFrame({'test': a}))
        ).tolist()
        self.assertEqual(a, b)

    def test_unicode(self):
        a = ['漢 字']
        b = self.encoder.reverse_transform(
            self.encoder.transform(pandas.DataFrame({'test': a}))
        ).tolist()
        self.assertEqual(b, ['LONG_TAIL LONG_TAIL MISSING_VALUE'])
        
    def test_handles_missing_labels(self):
        a = ['thisisnotavalidwordintheembeddings', float('nan'), None]
        b = self.encoder.reverse_transform(
            self.encoder.transform(pandas.DataFrame({'test': a}))
        ).tolist()
        self.assertEqual(['LONG_TAIL MISSING_VALUE MISSING_VALUE',
                          'MISSING_VALUE MISSING_VALUE MISSING_VALUE',
                          'MISSING_VALUE MISSING_VALUE MISSING_VALUE'], b)
        self.assertEqual(self.encoder.cardinality(), 6)


class TestMiddleOut(unittest.TestCase):
    def test_simple(self):
        enc = lore.encoders.MiddleOut('test', depth=2)
        res = enc.transform(pandas.DataFrame({'test': numpy.arange(8) + 1}))
        self.assertEqual(res.tolist(), [0, 1, 2, 2, 2, 2, 3, 4])
        self.assertEqual(enc.cardinality(), 5)

    def test_even(self):
        enc = lore.encoders.MiddleOut('test', depth=5)
        res = enc.transform(pandas.DataFrame({'test': numpy.arange(4) + 1}))
        self.assertEqual(res.tolist(), [0, 1, 9, 10])
        self.assertEqual(enc.cardinality(), 11)

    def test_odd(self):
        enc = lore.encoders.MiddleOut('test', depth=5)
        res = enc.transform(pandas.DataFrame({'test': numpy.arange(5) + 1}))
        self.assertEqual(res.tolist(), [0, 1, 5, 9, 10])
        self.assertEqual(enc.cardinality(), 11)

class TestTwins(unittest.TestCase):
    def test_unique(self):
        encoder = lore.encoders.Unique('test', twin=True)
        df = pandas.DataFrame({'test': [1, 2, 3, 4, 5], 'test_twin': [4, 5, 6, 1, 2]})
        encoder.fit(df)
        res = encoder.transform(df)
        self.assertEqual(res.tolist(), [2, 3, 4, 5, 6, 5, 6, 7, 2, 3])
        self.assertEqual(encoder.cardinality(), 9)

    def test_token(self):
        encoder = lore.encoders.Token('product_name', sequence_length=3, twin=True)
        df = pandas.DataFrame({'product_name': ['organic orange juice', 'organic apple juice'], 'product_name_twin': ['healthy orange juice', 'naval orange juice']})
        encoder.fit(df)
        res = encoder.transform(df['product_name'])
        self.assertEqual(res.reshape(-1).tolist(), [7,6,4,7,2,4])
        self.assertEqual(encoder.cardinality(), 9)
        res = encoder.transform(df['product_name_twin'])
        self.assertEqual(res.reshape(-1).tolist(), [3, 6, 4, 5, 6, 4])
