# -*- coding: utf-8 -*-
import datetime
from abc import ABCMeta, abstractmethod

import lore
from lore.env import require

require(
    lore.dependencies.PANDAS +
    lore.dependencies.INFLECTION
)

import pandas
import inflection


class Base(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self._data = pandas.DataFrame()

    @abstractmethod
    def key(self):
        """
        :return: Composite or a single key for index
        """
        pass

    @abstractmethod
    def get_data(self):
        pass

    @abstractmethod
    def publish(self):
        """
        Publish the feature to store ( S3, Memcache, Redis, Cassandra etc)
        :return: None
        """
        pass

    @property
    def version(self, version=str(datetime.date.today())):
        """
        Feature version : Override this method if you want to manage versions yourself
        ex 'v1', 'v2'
        By default will you date as the version information
        :param version:
        :return:
        """
        return version

    def name(self):
        return inflection.underscore(self.__class__.__name)

    def values(self):
        return self._data.columns.values.tolist()

    def _features_as_kv(self):
        """
        Return features row as kv pairs so that they can be stored in memcache or redis and
        used at serving layer
        :return: a nested hash for each column
        """
        self._data = self.get_data()
        key_list = self.key()
        values_list = self.values()
        result = {}
        for column in values_list:
            key_prefix = self.cache_key_prefix() + "#" + column
            self._data['cache_key'] = self._data[key_list].apply(lambda xdf: key_prefix + "=" + '#'.join(xdf.astype(str).values), axis=1)
            result[column] = dict(zip(self._data.cache_key.values, self._data[column].values))
        return result

    def cache_key_prefix(self):
        return ('#').join(self.key())

    def generate_row_keys(self):
        """
        Method for generating key features at serving time or prediction time
        :param data: Pass in the data that is necessary for generating the keys
         Example :
            Feature : User warehouse searches and conversions
            Keys will be of the form 'user_id#warehouse_id#searches=23811676#3'
            Keys will be of the form 'user_id#warehouse_id#conversions=23811676#3'
            data Frame should have values for all the columns as feature_key in this case ['user_id','warehouse_id']
        :return:
        """
        keys = self.key
        columns = self.values
        if not self._data:
            self._data = self.get_data()

        for column in columns:
            key_prefix = self.cache_key_prefix() + "#" + column
            self._data['cache_key'] = self._data[keys].apply(lambda xdf: key_prefix + "=" + '#'.join(xdf.astype(str).values),
                                                 axis=1)
        return list(self._data['cache_key'].values)


    def __repr__(self):
        return (
            """
                Version      : {}
                Name         : {}
                Keys         : {}
                Values       : {}
                Rows         : {}
            """.format(self.version, self.name(), self.key(), self.values(), len(self._data))
            # """.format(self.version, self.name, self.key, self.values, len([1]))
        )

    def metadata(self):
        return {
            "version": self.version,
            "name": self.name(),
            "keys": self.key(),
            "values": self.values(),
            "num_rows": len(self._data)
        }


    def distribute(self, cache):
        """
        Sync features to a key value compliant cache. Should adhere to cache protocol
        :param cache:
        :return: None
        """
        data = self._features_as_kv()
        for key in data.keys():
            cache.batch_set(data[key])

