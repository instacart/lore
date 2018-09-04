from __future__ import unicode_literals, absolute_import
from abc import ABCMeta, abstractmethod

import csv
import datetime
import logging
import os
import re

import lore
from lore.util import timer
from lore.env import require

require(
    lore.dependencies.NUMPY +
    lore.dependencies.INFLECTION +
    lore.dependencies.PANDAS
)

from numpy import sin, cos, sqrt, arctan2, radians

import inflection
import numpy
import pandas


logger = logging.getLogger(__name__)


class Base(object):
    __metaclass__ = ABCMeta

    def __init__(self, column):
        self.column = column
        if isinstance(self.column, Base):
            self.name = self.column.name
        else:
            self.name = self.column
        self.name += '_' + inflection.underscore(self.__class__.__name__)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    @abstractmethod
    def transform(self, data):
        pass

    def series(self, data):
        if isinstance(self.column, Base):
            return self.column.transform(data)
        else:
            if isinstance(data, pandas.Series):
                return data
            else:
                return data[self.column]

    def other_series(self, data):
        if (not hasattr(self, 'other')) or self.other is None:
            return None

        if isinstance(self.other, Base):
            return self.other.transform(data)
        else:
            if isinstance(data, pandas.Series):
                return data
            else:
                return data[self.other]

    @property
    def source_column(self):
        column = self.column
        if isinstance(column, list):
            for i in range(len(column)):
                if isinstance(column[i], Base):
                    column[i] = column[i].source_column
            column = list(set(column))
        elif isinstance(column, Base):
            column = column.source_column
        return column


class IsNull(Base):
    def transform(self, data):
        with timer('transform %s' % self.name, logging.DEBUG):
            return self.series(data).isnull()


class Map(Base):
    def transform(self, data):
        with timer('transform %s' % self.name, logging.DEBUG):
            return self.series(data).map(self.__class__.MAP)


class DateTime(Base):
    """
    For available operators see:
    https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DatetimeIndex.html#pandas.DatetimeIndex
    """

    def __init__(self, column, operator):
        super(DateTime, self).__init__(column)
        self.operator = operator
        self.name += '_' + self.operator

    def transform(self, data):
        with timer('transform %s' % self.name, logging.DEBUG):
            return getattr(self.series(data).dt, self.operator)


class Age(Base):
    def __init__(self, column, reference=None, unit='seconds'):
        super(Age, self).__init__(column)
        self.unit = unit
        self.other = reference
        if isinstance(self.other, Base):
            self.name += '_' + self.other.name
        elif self.other is not None:
            self.name += '_' + self.other

    def transform(self, data):
        with timer('transform %s' % self.name, logging.DEBUG):
            series = self.series(data)
            other = self.other_series(data)
            if other is None:
                other = datetime.datetime.now()
            elif other.dtype != 'datetime64[ns]':
                logger.warning('%s is not a datetime. Converting to datetime64[ns]' % self.column)
                other = pandas.to_datetime(other).astype('datetime64[ns]')

            if series.dtype != 'datetime64[ns]':
                logger.warning('%s is not a datetime. Converting to datetime64[ns]' % self.column)
                series = pandas.to_datetime(series).astype('datetime64[ns]')

            age = (other - series)
            if self.unit in ['nanosecond', 'nanoseconds']:
                return age

            seconds = age.dt.total_seconds()
            if self.unit in ['second', 'seconds']:
                return seconds
            if self.unit in ['minute', 'minutes']:
                return seconds / 60
            if self.unit in ['hour', 'hours']:
                return seconds / 3600
            if self.unit in ['day', 'days']:
                return seconds / 86400
            if self.unit in ['week', 'weeks']:
                return seconds / 604800
            if self.unit in ['month', 'months']:
                return seconds / 2592000
            if self.unit in ['year', 'years']:
                return seconds / 31536000

            raise NameError('Unknown unit: %s' % self.unit)


class String(Base):
    def __init__(self, column, operator, *args, **kwargs):
        super(String, self).__init__(column)
        self.operator = operator
        self.args = args
        self.kwargs = kwargs
        self.name += '_' + self.operator

    def transform(self, data):
        with timer('transform %s' % self.name, logging.DEBUG):
            series = self.series(data).astype(object)
            return getattr(series.str, self.operator)(*self.args, **self.kwargs)


class Extract(String):
    def __init__(self, column, regex):
        super(Extract, self).__init__(column, 'extract', pat=regex, expand=False)


class Length(String):
    def __init__(self, column):
        super(Length, self).__init__(column, 'len')


class Log(Base):
    def transform(self, data):
        with timer('transform %s' % self.name, logging.DEBUG):
            return numpy.log(self.series(data))


class LogPlusOne(Base):
    def transform(self, data):
        with timer('transform %s' % self.name, logging.DEBUG):
            series = self.series(data)
            null = series.isnull()
            series = numpy.log1p(numpy.maximum(series.fillna(0), 0))
            series[null] = float('nan')
            return series


class AreaCode(Base):
    """Transforms various phone number formats into area codes (strings)

    e.g. '12345678901' => '234'
         '+1 (234) 567-8901' => '234'
         '1234567' => ''
         float.nan => None
    """

    COUNTRY_DIGITS = re.compile(r'^\+?1(\d{10})$', re.UNICODE)
    PUNCTUATED = re.compile(r'(?:1[.\-]?)?\s?\(?(\d{3})\)?\s?[.\-]?[\d]{3}[.\-]?[\d]{4}', re.UNICODE)

    def transform(self, data):
        with timer('transform %s' % self.name, logging.DEBUG):
            series = self.series(data).astype(object)
            countries = series.str.extract(AreaCode.COUNTRY_DIGITS, expand=False)
            countries = countries.str[0:3]
            punctuated = series.str.extract(AreaCode.PUNCTUATED, expand=False)
            areacodes = countries
            areacodes[areacodes.isnull()] = punctuated
            areacodes[areacodes.isnull()] = ''
            areacodes[series.isnull()] = None
            return areacodes


class EmailDomain(Base):
    """Transforms email addresses into their full domain name

    e.g. 'bob@bob.com' => 'bob.com'
    """
    NAIVE = re.compile(r'^[^@]+@(.+)$', re.UNICODE)

    def transform(self, data):
        with timer('transform %s' % self.name, logging.DEBUG):
            domains = self.series(data).str.extract(EmailDomain.NAIVE, expand=False)
            domains[domains.isnull()] = ''
            return domains


class NameAge(Map):
    MAP = {}

    with open(os.path.join(os.path.dirname(__file__), 'data', 'names.csv'), 'r') as file:
        reader = csv.reader(file)
        for line in reader:
            MAP[line[0]] = float(line[2])

    def transform(self, data):
        with timer('transform %s' % self.name, logging.DEBUG):
            return self.series(data).str.lower().map(self.__class__.MAP)


class NamePopulation(NameAge):
    MAP = {}

    with open(os.path.join(os.path.dirname(__file__), 'data', 'names.csv'), 'r') as file:
        reader = csv.reader(file)
        for line in reader:
            MAP[line[0]] = float(line[3])


class NameSex(NameAge):
    MAP = {}

    with open(os.path.join(os.path.dirname(__file__), 'data', 'names.csv'), 'r') as file:
        reader = csv.reader(file)
        for line in reader:
            MAP[line[0]] = float(line[1])


class NameFamilial(Base):
    NAIVE = re.compile(r'\b(mom|dad|mother|father|mama|papa|bro|brother|sis|sister)\b', re.UNICODE | re.IGNORECASE)

    def transform(self, data):
        with timer('transform %s' % self.name, logging.DEBUG):
            return ~self.series(data).str.extract(NameFamilial.NAIVE, expand=False).isnull()


class GeoIP(Base):
    reader = None

    def __init__(self, column, operator):
        import lore  # This is crazy, why is this statement necessary?
        require(lore.dependencies.GEOIP)
        import geoip2.database

        if GeoIP.reader is None:
            import lore.io
            import glob
            file = lore.io.download(
                'http://geolite.maxmind.com/download/geoip/database/GeoLite2-City.tar.gz',
                cache=True,
                extract=True
            )

            path = [file for file in glob.glob(file.split('.')[0] + '*') if os.path.isdir(file)][0]
            GeoIP.reader = geoip2.database.Reader(os.path.join(path, 'GeoLite2-City.mmdb'))

        super(GeoIP, self).__init__(column)
        self.operator = operator
        self.name += '_' + self.operator

    def transform(self, data):
        with timer('transform %s' % self.name, logging.DEBUG):
            if self.operator in {'lat', 'latitude'}:
                return self.series(data).apply(GeoIP.get_latitude)
            elif self.operator in {'lon', 'longitude'}:
                return self.series(data).apply(GeoIP.get_longitude)
            elif self.operator in {'acc', 'accuracy'}:
                return self.series(data).apply(GeoIP.get_accuracy)

            raise NameError('Unknown GeoIP operator [lat, lon, acc]: %s' % self.operator)

    @staticmethod
    def get_latitude(ip):
        import geoip2
        try:
            return GeoIP.reader._db_reader.get(ip)['location']['latitude']
        except (KeyError, TypeError, ValueError, geoip2.errors.AddressNotFoundError):
            return float('nan')

    @staticmethod
    def get_longitude(ip):
        import geoip2
        try:
            return GeoIP.reader._db_reader.get(ip)['location']['longitude']
        except (KeyError, TypeError, ValueError, geoip2.errors.AddressNotFoundError):
            return float('nan')

    @staticmethod
    def get_accuracy(ip):
        import geoip2
        try:
            return GeoIP.reader._db_reader.get(ip)['location']['accuracy_radius']
        except (KeyError, TypeError, ValueError, geoip2.errors.AddressNotFoundError):
            return float('nan')


class Distance(Base):
    def __init__(self, lat_a, lon_a, lat_b, lon_b, input='degrees'):
        self.column = [lat_a, lon_a, lat_b, lon_b]
        self.name = ('_').join([
            inflection.underscore(self.__class__.__name__),
            str(lat_a),
            str(lat_b),
            str(lon_a),
            str(lon_b)
        ])
        self.lat_a = lat_a
        self.lon_a = lon_a
        self.lat_b = lat_b
        self.lon_b = lon_b
        self.input = input

    def transform(self, data):
        with timer('transform %s' % self.name, logging.DEBUG):
            lat_a = self.radians(data, self.lat_a)
            lat_b = self.radians(data, self.lat_b)
            lon_a = self.radians(data, self.lon_a)
            lon_b = self.radians(data, self.lon_b)

            lon = lon_b - lon_a
            lat = lat_b - lat_a

            a = sin(lat / 2)**2 + cos(lat_a) * cos(lat_b) * sin(lon / 2)**2
            c = 2 * arctan2(sqrt(a), sqrt(1 - a))

            return c * 6373.0  # approximate radius of earth in km

    def radians(self, data, column):
        if isinstance(column, Base):
            series = column.transform(data)
        else:
            if isinstance(data, pandas.Series):
                series = data
            else:
                series = data[column]

        if self.input == 'degrees':
            null = series.isnull()
            series = radians(series.fillna(0))
            series[null] = float('nan')
            return series
        elif self.input == 'radians':
            return series

        raise NameError('Unknown Distance input [degrees, radians]: %s' % self.input)
