# -*- coding: utf-8 -*-
import datetime
from abc import ABCMeta, abstractmethod

import lore
from lore.env import require
from lore.util import convert_df_columns_to_json

require(
    lore.dependencies.PANDAS +
    lore.dependencies.INFLECTION
)

import pandas
import inflection


class BaseFeatureExporter(object):
    __metaclass__ = ABCMeta

    def __init__(self, collection_ts=datetime.datetime.now()):
        self.collection_ts = collection_ts

    @property
    def key(self):
        """
        :return: Composite or a single key for index
        """
        raise NotImplementedError

    @property
    def timestamp(self):
        return datetime.datetime.combine(self.collection_ts.date(),
                                         datetime.datetime.min.time())

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
    def _raw_data(self):
        return self.get_data()

    @property
    def version(self, version=str(datetime.date.today())):
        """
        Feature version : Override this method if you want to manage versions yourself
        ex 'v1', 'v2'
        By default will you date as the version information
        :param version:
        :return:
        """
        return 'v1'

    @property
    def name(self):
        return inflection.underscore(self._value)

    @property
    def _values(self):
        value_cols = set(self._raw_data.columns.values.tolist()) - set(self.key)
        if len(value_cols) > 1:
            raise ValueError('Only one feature column allowed')
        return list(value_cols)

    @property
    def _value(self):
        return self._values[0]

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

    @property
    def cache_key_prefix(self):
        return ('#').join(self.key)

    def _generate_row_keys(self, df):
        """
        Method to generate rows keys for storage in the DB
        :param df: DataFrame to generate rows keys forecast

        This method will use the key definition initially provided
        and convert those columns into a JSON column
        :return:
        """
        keys = self.key
        return convert_df_columns_to_json(df, keys)

    def _generate_row_keys_for_serving(self, df):
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
        key_prefix = self.cache_key_prefix
        cache_keys = df[keys].apply(lambda xdf: key_prefix + "=" + '#'.join(xdf.astype(str).values),
                                    axis=1)
        return list(cache_keys)

    def __repr__(self):
        return (
            """
                Version      : {}
                Name         : {}
                Keys         : {}
                Rows         : {}
            """.format(self.version, self.name, self.key, len(self._data))
        )

    def metadata(self):
        return {
            "version": self.version,
            "name": self.name,
            "keys": self.key,
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


class BaseFeatureImporter(object):
    def __init__(self, entity_name, feature_name, version, start_date, end_date):
        self.entity_name = entity_name
        self.feature_name = feature_name
        self.version = version
        self.start_date = start_date
        self.end_date = end_date

    @property
    def feature_data(self):
        raise NotImplementedError


