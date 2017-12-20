import unittest

import lore.transformers
import numpy
import pandas
import datetime

class TestAreaCode(unittest.TestCase):
    def setUp(self):
        self.transformer = lore.transformers.AreaCode('phone')

    def test_phone_formats(self):
        values = pandas.DataFrame({
            'phone': [
                '12345678901',
                '+12345678901',
                '1(234)567-8901',
                '1 (234) 567-8901',
                '1.234.567.8901',
                '1-234-567-8901',
                '2345678901',
                '234.567.8901',
                '(234)5678901',
                '(234) 567-8901',
            ]
        })
        result = self.transformer.transform(values)
        self.assertEqual(result.tolist(), numpy.repeat('234', len(values)).tolist())

    def test_bad_data(self):
        values = pandas.DataFrame({
            'phone': [
                '1234567',
                '(123)4567',
                '',
                None,
                12345678901,
            ]
        })
        result = self.transformer.transform(values)
        self.assertEqual(result.tolist(), ['', '', '', None, ''])
    
        
class TestEmailDomain(unittest.TestCase):
    def setUp(self):
        self.transformer = lore.transformers.EmailDomain('email')

    def test_transform(self):
        values = pandas.DataFrame({
            'email': [
                'montana@instacart.com',
                'sue-bob+anne@instacart.com'
            ]
        })
        result = self.transformer.transform(values)
        self.assertEqual(result.tolist(), numpy.repeat('instacart.com', len(values)).tolist())


class TestNameFamilial(unittest.TestCase):
    def setUp(self):
        self.transformer = lore.transformers.NameFamilial('name')

    def test_transform(self):
        values = pandas.DataFrame({
            'name': [
                'mom',
                'Dad',
                'sue bob'
            ]
        })
        result = self.transformer.transform(values)
        self.assertEqual(result.tolist(), [True, True, False])


class TestDateTime(unittest.TestCase):
    def test_transform_day_of_week(self):
        transformer = lore.transformers.DateTime('test', 'dayofweek')
        data = pandas.DataFrame({'test': [datetime.datetime(2016, 12, 31), datetime.date(2017, 1, 1)]})
        transformed = transformer.transform(data)
        self.assertEqual(transformed.iloc[0] + 1, transformed.iloc[1])


class TestAge(unittest.TestCase):
    def test_transform_age(self):
        transformer = lore.transformers.Age('test', 'days')
        yesterday = datetime.datetime.now() - datetime.timedelta(days=1)

        data = pandas.DataFrame({'test': [datetime.datetime.now(), yesterday]})
        transformed = transformer.transform(data)
        self.assertEqual(transformed.astype(int).tolist(), [0, 1])
