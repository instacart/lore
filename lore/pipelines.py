from abc import ABCMeta, abstractmethod
import codecs
from collections import namedtuple
import gc
import hashlib
import inspect
import logging
import os
import sqlite3
import types
import warnings

import numpy
import pandas
from sklearn.model_selection import train_test_split

from lore.util import timer, timed
import lore.stores


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
            gc.collect()

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
            self._training_data = self._data.iloc[numpy.isin(rows, train_ids.values)]
            self._validation_data = self._data.iloc[numpy.isin(rows, validate_ids.values)]
            self._test_data = self._data.iloc[numpy.isin(rows, test_ids.values)]
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


class LowMemory(Holdout):
    connection = sqlite3.connect(os.path.join(lore.env.data_dir, 'low_memory.sqlite'), isolation_level=None)
    connection.execute('PRAGMA synchronous = OFF')
    connection.execute('PRAGMA journal_mode = MEMORY')

    @timed(logging.INFO)
    def __init__(self):
        super(LowMemory, self).__init__()
        self._training_key = None
        self._validation_key = None
        self._test_key = None
        self._table = None
        self.chunksize = None
        self.loaded = False
        self._columns = None
        self._length = None
    
    @property
    def table(self):
        if self._table is None:
            self._table = 'pipeline_' + self.__class__.__name__.lower() + '_' + hashlib.sha1(str(self.__dict__)).hexdigest()
            
        return self._table
    
    @property
    def table_training(self):
        return self.table + '_training'
    
    @property
    def table_validation(self):
        return self.table + '_validation'
    
    @property
    def table_test(self):
        return self.table + '_test'

    @property
    def training_data(self):
        return self.read_table(self.table_training)
        
    @property
    def validation_data(self):
        return self.read_table(self.table_validation)

    @property
    def test_data(self):
        return self.read_table(self.table_test)
        
    def read_table(self, name):
        if not self.loaded:
            self._split_data()
    
        return pandas.read_sql("SELECT * FROM {name}".format(name=name), self.connection, chunksize=self.chunksize)

    @property
    def encoded_training_data(self):
        if not self._encoded_training_data:
            with timer('encode training data:'):
                self._encoded_training_data = self.observations(self.table_training)
    
        return self._encoded_training_data

    @property
    def encoded_validation_data(self):
        if not self._encoded_validation_data:
            with timer('encode validation data:'):
                self._encoded_validation_data = self.observations(self.table_validation)
    
        return self._encoded_validation_data

    @property
    def encoded_test_data(self):
        if not self._encoded_test_data:
            with timer('encode test data:'):
                self._encoded_test_data = self.observations(self.table_test)
    
        return self._encoded_test_data

    @property
    def encoders(self):
        if self._encoders is None:
            with timer('fit encoders:'):
                self._encoders = self.get_encoders()
                for encoder in self._encoders:
                    encoder.fit(self.read_column(self.table_training, encoder.source_column))
                    gc.collect()

        return self._encoders
    
    def read_column(self, table, column):
        return pandas.read_sql(
            'SELECT {column} FROM {table}'.format(
                column=self.quote(column),
                table=self.quote(table)
            ),
            self.connection
        )
    
    @property
    def output_encoder(self):
        if self._output_encoder is None:
            with timer('fit output encoder:'):
                self._output_encoder = self.get_output_encoder()
                self._output_encoder.fit(self.read_column(self.table_training, self._output_encoder.source_column))
        
        return self._output_encoder
    
    def observations(self, data):
        return Observations(x=self.encode_x(data), y=self.encode_y(data))
    
    @timed(logging.INFO)
    def encode_x(self, data):
        if isinstance(data, pandas.DataFrame):
            return super(LowMemory, self).encode_x(data)
        
        encoded = {}
        for encoder in self.encoders:
            transformed = encoder.transform(self.read_column(data, encoder.source_column))
            if hasattr(encoder, 'sequence_length'):
                for i in range(encoder.sequence_length):
                    encoded[encoder.sequence_name(i)] = transformed[:, i]
            else:
                encoded[encoder.name] = transformed
        
        # Using a DataFrame as a container temporairily requires double the memory,
        # as pandas copies all data on __init__. This is justified by having a
        # type supported by all dependent libraries (heterogeneous dict is not)
        return pandas.DataFrame(encoded)
    
    @timed(logging.INFO)
    def encode_y(self, data):
        if isinstance(data, pandas.DataFrame):
            return super(LowMemory, self).encode_x(data)
            
        return self.output_encoder.transform(self.read_column(data, self._output_encoder.source_column))
    
    @timed(logging.INFO)
    def decode(self, predictions):
        return {encoder.name: encoder.reverse_transform(predictions) for encoder in self.encoder}
    
    def _split_data(self):
        self.loaded = True
    
        if self.connection.execute(
            """
                SELECT name
                FROM sqlite_master
                WHERE type='table'
                  AND name={name}
            """.format(
                name=self.quote(self.table)
            )
        ).fetchone():
            return
    
        self._data = self.get_data()
    
        if not isinstance(self._data, types.GeneratorType):
            raise TypeError('LowMemory pipelines must be passed a generator rather than a dataframe. Did you forget to pass a `chunksize` to lore.io.Connection.dataframe()?')
    
        self.length = 0
        for i, dataframe in enumerate(self._data):
            logger.info('appending %i rows to sqlite' % len(dataframe))
            if i == 0:
                self._columns = dataframe.columns
                self.chunksize = len(dataframe)
        
            self.length += len(dataframe)
            dataframe.to_sql(self.name, self.connection, index=False, if_exists="append")
    
        if self.subsample:
            self.connection.executescript(
                """
                    BEGIN;
                    
                    CREATE TABLE {subsample} AS
                    SELECT * FROM {name}
                    ORDER BY random()
                    LIMIT {limit};
                    
                    DROP TABLE {name};
                    
                    ALTER TABLE {subsample} RENAME TO {name};
                    
                    COMMIT;
                """.format(
                    name=self.quote(self.name),
                    subsample=self.quote(self.name + '_subsample'),
                    limit=self.subsample
                )
            )
            self.length = self.subsample

        self._random_split(self.table_training, 0, 0.8, stratify=self.stratify)
        self._random_split(self.table_validation, 0.8, 0.9, stratify=self.stratify)
        self._random_split(self.table_test, 0.9, 1, stratify=self.stratify)
        
        logger.debug('data: %i | training: %i | validation: %i | test: %i' % (
            self.length,
            self.table_length(self.table_training),
            self.table_length(self.table_validation),
            self.table_length(self.table_test),
        ))

    def __getitem__(self, column):
        return self.connection.execute(
            'SELECT {column} FROM {table}'.format(
                column=self.quote(column),
                table=self.quote(self.name),
            )
        )

    def __len__(self):
        if self._length is None:
            self._length = self.table_length(self.name)
        return self._length

    def table_length(self, name):
        return self.connection.execute(
            'SELECT count(_ROWID_) FROM {table}'.format(
                table=self.quote(name)
            )
        ).fetchone()[0]
    
    @property
    def columns(self):
        if not self.loaded:
            self._split_data()
    
        if self._columns is None:
            self._columns = [
                row[1] for row in self.connection.execute(
                    'PRAGMA table_info({table});'.format(table=self.quote(self.name))
                )
            ]
    
        return self._columns

    def _random_split(self, name, start, stop, stratify=None):
        random = self.quote(name + '_random')
        name = self.quote(name)
        
        if stratify is not None:
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS {random} AS
                SELECT DISTINCT {stratify}
                FROM {source}
                ORDER BY random();
            """.format(
                random=random,
                source=self.quote(self.name),
                stratify=self.quote(stratify)
            ))
            random_column = self.quote(stratify)
    
        else:
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS {random} AS
                SELECT _rowid_ as other_rowid
                FROM {source}
                ORDER BY random();
            """.format(
                random=random,
                source=self.quote(self.name)
            ))
            stratify = '_rowid_'
            random_column = 'other_rowid'
    
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS {split} AS
            SELECT {table}.*
            FROM {table}
            JOIN {random}
              ON {random}.{random_column} = {table}.{stratify}
            WHERE {random}._rowid_ - 1 >= (select max(_rowid_) from {random}) * {start}
              AND {random}._rowid_ - 1 < (select max(_rowid_) from {random}) * {stop}
        """.format(
            split=name,
            table=self.quote(self.name),
            random=random,
            random_column=random_column,
            start=start,
            stop=stop,
            stratify=stratify
        ))

    def quote(self, identifier, errors="strict"):
        """
        https://stackoverflow.com/questions/6514274/how-do-you-escape-strings-for-sqlite-table-column-names-in-python

        :param identifier:
        :param errors:
        :return:
        """
        encodable = identifier.encode("utf-8", errors).decode("utf-8")
    
        nul_index = encodable.find("\x00")
    
        if nul_index >= 0:
            error = UnicodeEncodeError(
                "NUL-terminated utf-8",
                encodable,
                nul_index,
                nul_index + 1,
                "NUL not allowed"
            )
            error_handler = codecs.lookup_error(errors)
            replacement, _ = error_handler(error)
            encodable = encodable.replace("\x00", replacement)
    
        return "\"" + encodable.replace("\"", "\"\"") + "\""


class TrainTestSplit(Holdout):
    def __init__(self, **kwargs):
        frame, filename, line_number, function_name, lines, index = inspect.stack()[1]
        warnings.showwarning(
            'TrainTestSplit has been renamed to Holdout. Please update your code.',
            DeprecationWarning,
            filename,
            line_number
        )
        super(TrainTestSplit, self).__init__(**kwargs)


class SortedTrainTestSplit(TimeSeries):
    def __init__(self, **kwargs):
        frame, filename, line_number, function_name, lines, index = inspect.stack()[1]
        warnings.showwarning(
            'SortedTrainTestSplit has been renamed to TimeSeries. Please update your code.',
            DeprecationWarning,
            filename,
            line_number
        )
        super(SortedTrainTestSplit, self).__init__(**kwargs)
