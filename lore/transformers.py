from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod

import csv
import os
import re

from past.builtins import basestring

import inflection
import numpy


class Base(object):
    __metaclass__ = ABCMeta
    
    def __init__(self, column):
        self.column = column
        self.name = self.column + '_' + inflection.underscore(self.__class__.__name__)
    
    @abstractmethod
    def transform(self, data):
        pass


class Map(Base):
    def transform(self, data):
        return data.loc[:, self.column].map(self.__class__.MAP)


class Log(Base):
    def transform(self, data):
        return numpy.log(data.loc[:, self.column])


class LogPlusOne(Base):
    def transform(self, data):
        return numpy.log1p(numpy.maximum(data.loc[:, self.column], 0))


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
        series = data[self.column].astype(object)
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
        domains = data[self.column].str.extract(EmailDomain.NAIVE, expand=False)
        domains[domains.isnull()] = ''
        return domains


class NameAge(Map):
    MAP = {}
    
    with open(os.path.join(os.path.dirname(__file__), 'data', 'names.csv'), 'r') as file:
        reader = csv.reader(file)
        for line in reader:
            MAP[line[0]] = float(line[2])


class NamePopulation(Map):
    MAP = {}
    
    with open(os.path.join(os.path.dirname(__file__), 'data', 'names.csv'), 'r') as file:
        reader = csv.reader(file)
        for line in reader:
            MAP[line[0]] = float(line[3])


class NameSex(Map):
    MAP = {}
    
    with open(os.path.join(os.path.dirname(__file__), 'data', 'names.csv'), 'r') as file:
        reader = csv.reader(file)
        for line in reader:
            MAP[line[0]] = float(line[1])


class NameFamilial(Base):
    NAIVE = re.compile(r'\b(mom|dad|mother|father|mama|papa|bro|brother|sis|sister)\b', re.UNICODE | re.IGNORECASE)
    
    def transform(self, data):
        return ~data[self.column].str.extract(NameFamilial.NAIVE, expand=False).isnull()
