import unittest
import datetime
import math

import numpy
import pandas

import lore.transformers


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
        transformer = lore.transformers.Age('test', unit='days')
        yesterday = datetime.datetime.now() - datetime.timedelta(days=1)

        data = pandas.DataFrame({'test': [datetime.datetime.now(), yesterday]})
        transformed = transformer.transform(data)
        self.assertEqual(transformed.astype(int).tolist(), [0, 1])


class TestNameAge(unittest.TestCase):
    def test_transform_name(self):
        transformer = lore.transformers.NameAge('test')
        
        data = pandas.DataFrame({'test': ['bob', 'Bob']})
        transformed = transformer.transform(data)
        self.assertTrue(transformed.iloc[0] > 0)
        self.assertEqual(transformed.iloc[0], transformed.iloc[1])


class TestNameSex(unittest.TestCase):
    def test_transform_name(self):
        transformer = lore.transformers.NameSex('test')
        
        data = pandas.DataFrame({'test': ['bob', 'Bob']})
        transformed = transformer.transform(data)
        self.assertTrue(transformed.iloc[0] > 0)
        self.assertEqual(transformed.iloc[0], transformed.iloc[1])


class TestNamePopulation(unittest.TestCase):
    def test_transform_name(self):
        transformer = lore.transformers.NamePopulation('test')
        
        data = pandas.DataFrame({'test': ['bob', 'Bob']})
        transformed = transformer.transform(data)
        self.assertTrue(transformed.iloc[0] > 0)
        self.assertEqual(transformed.iloc[0], transformed.iloc[1])


class TestStringLower(unittest.TestCase):
    def test_transform_name(self):
        transformer = lore.transformers.String('test', 'lower')
        
        data = pandas.DataFrame({'test': ['bob', 'Bob']})
        transformed = transformer.transform(data)
        self.assertEqual(transformed.iloc[0], 'bob')
        self.assertEqual(transformed.iloc[1], 'bob')


class TestGeoIP(unittest.TestCase):
    def test_transform_latitude(self):
        transformer = lore.transformers.GeoIP('test', 'latitude')
        
        data = pandas.DataFrame({'test': ['124.0.0.1', '124.0.0.2']})
        transformed = transformer.transform(data)
        self.assertEqual(transformed.iloc[0], 37.5112)
        self.assertEqual(transformed.iloc[1], 37.5112)

    def test_transform_longitude(self):
        transformer = lore.transformers.GeoIP('test', 'longitude')
    
        data = pandas.DataFrame({'test': ['124.0.0.1', '124.0.0.2']})
        transformed = transformer.transform(data)
        self.assertEqual(transformed.iloc[0], 126.97409999999999)
        self.assertEqual(transformed.iloc[1], 126.97409999999999)

    def test_transform_accuracy(self):
        transformer = lore.transformers.GeoIP('test', 'accuracy')
    
        data = pandas.DataFrame({'test': ['124.0.0.1', '124.0.0.2']})
        transformed = transformer.transform(data)
        self.assertEqual(transformed.iloc[0], 200)
        self.assertEqual(transformed.iloc[1], 200)

    def test_missing_ip(self):
        transformer = lore.transformers.GeoIP('test', 'accuracy')
        data = pandas.DataFrame({'test': ['127.0.0.2']})
        transformed = transformer.transform(data)
        self.assertTrue(math.isnan(transformed.iloc[0]))

    
class TestDistance(unittest.TestCase):
    def test_distance(self):
        data = pandas.DataFrame({
            'a_lat': [0., 52.2296756],
            'b_lat': [0., 52.406374],
            'a_lon': [0., 21.0122287],
            'b_lon': [0., 16.9251681]
        })
        
        transformer = lore.transformers.Distance(
            lat_a='a_lat',
            lat_b='b_lat',
            lon_a='a_lon',
            lon_b='b_lon',
        )
        
        transformed = transformer.transform(data)
        self.assertEqual(transformed.iloc[0], 0)
        self.assertEqual(transformed.iloc[1], 278.54558935106695)
    
    def test_ip(self):
        data = pandas.DataFrame({'a': ['124.0.0.1', '124.0.0.2'], 'b': ['124.0.0.1', '127.0.0.2']})
        
        transformer = lore.transformers.Distance(
            lat_a=lore.transformers.GeoIP('a', 'latitude'),
            lat_b=lore.transformers.GeoIP('b', 'latitude'),
            lon_a=lore.transformers.GeoIP('a', 'longitude'),
            lon_b=lore.transformers.GeoIP('b', 'longitude'),
        )
        
        transformed = transformer.transform(data)
        self.assertEqual(transformed.iloc[0], 0)
        self.assertTrue(math.isnan(transformed.iloc[1]))
