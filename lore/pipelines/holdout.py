from __future__ import absolute_import

from abc import ABCMeta, abstractmethod
from collections import OrderedDict, Iterable
import gc
import logging
import multiprocessing

import lore
from lore.env import require
from lore.util import timer, timed
from lore.pipelines import Observations

require(
    lore.dependencies.NUMPY +
    lore.dependencies.PANDAS +
    lore.dependencies.SKLEARN
)
import numpy
import pandas
from sklearn.model_selection import train_test_split


logger = logging.getLogger(__name__)


class Base(object):
    __metaclass__ = ABCMeta

    test_size = 0.1

    def __init__(self):
        self.name = self.__module__ + '.' + self.__class__.__name__
        self.stratify = None
        self.subsample = None
        self.split_seed = 1
        self.index = []
        self.multiprocessing = False
        self.workers = None
        self._data = None
        self._encoders = None
        self._training_data = None
        self._test_data = None
        self._validation_data = None
        self._output_encoder = None
        self._encoded_training_data = None
        self._encoded_validation_data = None
        self._encoded_test_data = None

    def __getstate__(self):
        state = dict(self.__dict__)
        # bloat can be restored via self.__init__() + self.build()
        for bloat in [
            '_data',
            '_training_data',
            '_test_data',
            '_validation_data',
            '_encoded_training_data',
            '_encoded_validation_data',
            '_encoded_test_data',
        ]:
            state[bloat] = None
        return state

    def __setstate__(self, dict):
        self.__dict__ = dict
        backward_compatible_defaults = {
            'index': [],
            'multiprocessing': False,
            'workers': None,
        }
        for key, default in backward_compatible_defaults.items():
            if key not in self.__dict__.keys():
                self.__dict__[key] = default

    @abstractmethod
    def get_data(self):
        pass

    @abstractmethod
    def get_encoders(self):
        pass

    @abstractmethod
    def get_output_encoder(self):
        pass

    @property
    def encoders(self):
        if self._encoders is None:
            with timer('fit encoders'):
                self._encoders = self.get_encoders()

                # Ensure we have an iterable for all single encoder cases
                if not isinstance(self._encoders, Iterable):
                    if len((self._encoders, )) == 1:
                        self._encoders = (self._encoders, )

                if self.multiprocessing:
                    pool = multiprocessing.Pool(self.workers)
                    results = []
                    for encoder in self._encoders:
                        results.append(pool.apply_async(self.fit, (encoder, self.training_data)))
                    self._encoders = [result.get() for result in results]

                else:
                    for encoder in self._encoders:
                        encoder.fit(self.training_data)

        return self._encoders

    @property
    def output_encoder(self):
        if self._output_encoder is None:
            with timer('fit output encoder'):
                self._output_encoder = self.get_output_encoder()
                self._output_encoder.fit(self.training_data)

        return self._output_encoder

    @property
    def training_data(self):
        if self._training_data is None:
            self._split_data()

        return self._training_data

    @property
    def validation_data(self):
        if self._validation_data is None:
            self._split_data()

        return self._validation_data

    @property
    def test_data(self):
        if self._test_data is None:
            self._split_data()

        return self._test_data

    @property
    def encoded_training_data(self):
        if not self._encoded_training_data:
            with timer('encode training data'):
                self._encoded_training_data = self.observations(self.training_data)

        return self._encoded_training_data

    @property
    def encoded_validation_data(self):
        if not self._encoded_validation_data:
            with timer('encode validation data'):
                self._encoded_validation_data = self.observations(self.validation_data)

        return self._encoded_validation_data

    @property
    def encoded_test_data(self):
        if not self._encoded_test_data:
            with timer('encode test data'):
                self._encoded_test_data = self.observations(self.test_data)

        return self._encoded_test_data

    def observations(self, data):
        return Observations(x=self.encode_x(data), y=self.encode_y(data))

    @timed(logging.INFO)
    def encode_x(self, data):
        """
        :param data: unencoded input dataframe
        :return: a dict with encoded values
        """
        encoded = OrderedDict()
        if self.multiprocessing:
            pool = multiprocessing.Pool(self.workers)
            results = []
            for encoder in self.encoders:
                results.append((encoder, pool.apply_async(self.transform, (encoder, data))))

            for encoder, result in results:
                self.merged_transformed(encoded, encoder, result.get())

        else:
            for encoder in self.encoders:
                self.merged_transformed(encoded, encoder, self.transform(encoder, data), append_twin=False)
                if encoder.twin:
                    self.merged_transformed(encoded, encoder, self.transform(encoder, data, append_twin = True), append_twin=True)

        for column in self.index:
            encoded[column] = self.read_column(data, column)

        # Using a DataFrame as a container temporarily requires double the memory,
        # as pandas copies all data on __init__. This is justified by having a
        # type supported by all dependent libraries (heterogeneous dict is not)
        dataframe = pandas.DataFrame(encoded)
        if self.index:
            dataframe.set_index(self.index)
        return dataframe

    def fit(self, encoder, data):
        encoder.fit(data)
        return encoder

    def transform(self, encoder, data, append_twin=False):
        if append_twin:
            return encoder.transform(self.read_column(data, encoder.twin_column))
        else:
            return encoder.transform(self.read_column(data, encoder.source_column))

    @staticmethod
    def merged_transformed(encoded, encoder, transformed, append_twin=False):
        if hasattr(encoder, 'sequence_length'):
            for i in range(encoder.sequence_length):
                if isinstance(transformed, pandas.DataFrame):
                    if append_twin:
                        encoded[encoder.sequence_name(i, suffix="_twin")] = transformed.iloc[:, i]
                    else:
                        encoded[encoder.sequence_name(i)] = transformed.iloc[:, i]
                else:
                    if append_twin:
                        encoded[encoder.sequence_name(i, suffix="_twin")] = transformed[:, i]
                    else:
                        encoded[encoder.sequence_name(i)] = transformed[:, i]

        else:
            if append_twin:
                encoded[encoder.twin_name] = transformed
            else:
                encoded[encoder.name] = transformed



    @timed(logging.INFO)
    def encode_y(self, data):
        if self.output_encoder.source_column in data.columns:
            return self.output_encoder.transform(self.read_column(data, self._output_encoder.source_column))
        else:
            return None

    @timed(logging.INFO)
    def decode(self, data):
        decoded = OrderedDict()
        for encoder in self.encoders:
            decoded[encoder.name.split('_', 1)[-1]] = encoder.reverse_transform(data[encoder.name])
        return pandas.DataFrame(decoded)

    def read_column(self, data, column):
        """
        Implemented so subclasses can overide handle different types of columnar data

        :param dataframe:
        :param column:
        :return:
        """
        return data[column]

    @timed(logging.INFO)
    def _split_data(self):
        if self._data:
            return

        numpy.random.seed(self.split_seed)
        logger.debug('random seed set to: %i' % self.split_seed)

        self._data = self.get_data()
        gc.collect()
        if self.subsample:

            if self.stratify:
                logger.debug('subsampling stratified by `%s`: %s' % (
                    self.stratify, self.subsample))
                ids = self._data[[self.stratify]].drop_duplicates()
                ids = ids.sample(self.subsample)
                self._data = pandas.merge(self._data, ids, on=self.stratify)
            else:
                logger.debug('subsampling rows: %s' % self.subsample)
                self._data = self._data.sample(self.subsample)
            gc.collect()

        if self.stratify:
            ids = self._data[self.stratify].drop_duplicates()

            train_ids, validate_ids = train_test_split(
                ids,
                test_size=self.test_size,
                random_state=self.split_seed
            )
            gc.collect()
            train_ids, test_ids = train_test_split(
                train_ids,
                test_size=self.test_size,
                random_state=self.split_seed
            )
            gc.collect()

            rows = self._data[self.stratify].values
            self._training_data = self._data.iloc[numpy.isin(rows, train_ids.values)]
            self._validation_data = self._data.iloc[numpy.isin(rows, validate_ids.values)]
            self._test_data = self._data.iloc[numpy.isin(rows, test_ids.values)]
        else:
            self._training_data, self._validation_data = train_test_split(
                self._data,
                test_size=self.test_size,
                random_state=self.split_seed
            )

            self._training_data, self._test_data = train_test_split(
                self._training_data,
                test_size=self.test_size,
                random_state=self.split_seed
            )
        gc.collect()

        self._data = None
        gc.collect()

        # It's import to reset these indexes after split so in case
        # these dataframes are copied, the missing split rows are
        # not re-materialized later full of nans.
        self._training_data.reset_index(drop=True, inplace=True)
        gc.collect()
        self._validation_data.reset_index(drop=True, inplace=True)
        gc.collect()
        self._test_data.reset_index(drop=True, inplace=True)
        gc.collect()

        logger.debug('training: %i | validation: %i | test: %i' % (
            len(self._training_data),
            len(self._validation_data),
            len(self._test_data)
        ))
