from abc import ABCMeta, abstractmethod
from collections import namedtuple
import inspect
import logging
import warnings

import numpy
import pandas
from lore.util import timer, timed
from sklearn.model_selection import train_test_split


logger = logging.getLogger(__name__)


Observations = namedtuple('Observations', 'x y')


class Holdout(object):
    __metaclass__ = ABCMeta
    
    test_size = 0.1

    def __init__(self):
        self.name = self.__module__ + '.' + self.__class__.__name__
        self.stratify = None
        self.subsample = None
        self.split_seed = 1
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
            '_encoded_validation_data'
            '_encoded_test_data',
        ]:
            state[bloat] = None
        return state

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
            with timer('fit encoders:'):
                self._encoders = self.get_encoders()
                for encoder in self._encoders:
                    encoder.fit(self.training_data)
            
        return self._encoders
    
    @property
    def output_encoder(self):
        if self._output_encoder is None:
            with timer('fit output encoder:'):
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
            with timer('encode training data:'):
                self._encoded_training_data = self.observations(self.training_data)
    
        return self._encoded_training_data

    @property
    def encoded_validation_data(self):
        if not self._encoded_validation_data:
            with timer('encode validation data:'):
                self._encoded_validation_data = self.observations(self.validation_data)
    
        return self._encoded_validation_data

    @property
    def encoded_test_data(self):
        if not self._encoded_test_data:
            with timer('encode test data:'):
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
        encoded = {}
        for encoder in self.encoders:
            transformed = encoder.transform(data)
            if hasattr(encoder, 'sequence_length'):
                for i in range(encoder.sequence_length):
                    encoded[encoder.sequence_name(i)] = transformed[:,i]
            else:
                encoded[encoder.name] = transformed
        
        # Using a DataFrame as a container temporairily requires double the memory,
        # as pandas copies all data on __init__. This is justified by having a
        # type supported by all dependent libraries (heterogeneous dict is not)
        return pandas.DataFrame(encoded)
    
    @timed(logging.INFO)
    def encode_y(self, data):
        return self.output_encoder.transform(data)

    @timed(logging.INFO)
    def decode(self, predictions):
        return {encoder.name: encoder.reverse_transform(predictions) for encoder in self.encoder}

    @timed(logging.INFO)
    def _split_data(self):
        if self._data:
            return

        numpy.random.seed(self.split_seed)
        logger.debug('random seed set to: %i' % self.split_seed)

        self._data = self.get_data()
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
    
        if self.stratify:
            ids = self._data[self.stratify].drop_duplicates()

            train_ids, validate_ids = train_test_split(
                ids,
                test_size=self.test_size,
                random_state=1
            )
            train_ids, test_ids = train_test_split(
                train_ids,
                test_size=self.test_size,
                random_state=1
            )
        
            rows = self._data[self.stratify].values
            self._training_data = self._data.iloc[numpy.in1d(rows, train_ids.values)]
            self._validation_data = self._data.iloc[numpy.in1d(rows, validate_ids.values)]
            self._test_data = self._data.iloc[numpy.in1d(rows, test_ids.values)]
        else:
            self._training_data, self._validation_data = train_test_split(
                self._data,
                test_size=self.test_size,
                random_state=1
            )
            self._training_data, self._test_data = train_test_split(
                self._training_data,
                test_size=self.test_size,
                random_state=1
            )
        # It's import to reset these indexes after split so in case
        # these dataframes are copied, the missing split rows are
        # not re-materialized later full of nans.
        self._training_data.reset_index(drop=True, inplace=True)
        self._validation_data.reset_index(drop=True, inplace=True)
        self._test_data.reset_index(drop=True, inplace=True)

        logger.debug('data: %i | training: %i | validation: %i | test: %i' % (
            len(self._data),
            len(self._training_data),
            len(self._validation_data),
            len(self._test_data)
        ))


class TimeSeries(Holdout):
    __metaclass__ = ABCMeta

    def __init__(self, test_size = 0.1, sort_by = None):
        super(TimeSeries, self).__init__()
        self.sort_by = sort_by
        self.test_size = test_size

    @timed(logging.INFO)
    def _split_data(self):
        if self._data:
            return

        logger.debug('No shuffle test train split')

        self._data = self.get_data()

        if self.sort_by:
            self._data = self._data.sort_values(by=self.sort_by, ascending=True)
        test_rows = int(len(self._data) * self.test_size)
        valid_rows = test_rows
        train_rows = int(len(self._data) - test_rows - valid_rows)
        self._training_data = self._data.iloc[:train_rows]
        self._validation_data = self._data[train_rows:train_rows+valid_rows]
        self._test_data = self._data.iloc[-test_rows:]


class TrainTestSplit(Holdout):
    def __init__(self, **kwargs):
        warnings.showwarning('TrainTestSplit has been renamed to Holdout. Please update your code.', DeprecationWarning,
                             __file__, inspect.currentframe().f_back.f_lineno)
        super(TrainTestSplit, self).__init__(**kwargs)


class SortedTrainTestSplit(TimeSeries):
    def __init__(self, **kwargs):
        warnings.showwarning('SortedTrainTestSplit has been renamed to TimeSeries. Please update your code.', DeprecationWarning,
                             __file__, inspect.currentframe().f_back.f_lineno)
        super(SortedTrainTestSplit, self).__init__(**kwargs)
