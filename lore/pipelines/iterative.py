from __future__ import absolute_import

from abc import ABCMeta
import codecs
import hashlib
import io
import logging
import os
import pickle
import sqlite3
import types
import threading


import lore
from lore.env import require
from lore.util import timer, timed
from lore.pipelines import Observations

require(lore.dependencies.PANDAS)
import pandas



logger = logging.getLogger(__name__)


class Base(lore.pipelines.holdout.Base):
    __metaclass__ = ABCMeta

    @timed(logging.INFO)
    def __init__(self):
        super(Base, self).__init__()
        self._training_key = None
        self._validation_key = None
        self._test_key = None
        self._table = None
        self.chunksize = None
        self.loaded = False
        self._columns = None
        self._length = None
        self._datetime_columns = None
        self._connection = None
        self._connected_on_thread = None

    @property
    def connection(self):
        current_thread = threading.current_thread().ident
        if self._connected_on_thread and self._connected_on_thread != current_thread:
            logger.warning('Pipeline accessed via thread, reconnecting to sqlite (generators can not be shared)')
            self._connection = None

        if not self._connection:
            self._connection = sqlite3.connect(os.path.join(lore.env.DATA_DIR, 'low_memory.sqlite'),
                                               isolation_level=None)
            self._connection.execute('PRAGMA synchronous = OFF')
            self._connection.execute('PRAGMA journal_mode = MEMORY')
            self._connected_on_thread = current_thread

        return self._connection

    @property
    def table(self):
        if self._table is None:
            self._table = 'pipeline_' + self.__class__.__name__.lower() + '_' + hashlib.sha1(
                str(self.__dict__).encode('utf-8')).hexdigest()

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
        return self.generator(self.table_training)

    @property
    def validation_data(self):
        return self.generator(self.table_validation)

    @property
    def test_data(self):
        return self.generator(self.table_test)

    @property
    def datetime_columns(self):
        if not self.loaded:
            self._split_data()

        if self._datetime_columns is None:
            with open(self.metadata_path, 'rb') as f:
                self._datetime_columns = pickle.load(f)

        return self._datetime_columns

    @property
    def encoders(self):
        if self._encoders is None:
            with timer('fit encoders'):
                self._encoders = self.get_encoders()
                for encoder in self._encoders:
                    encoder.fit(self.read_column(self.table_training, encoder.source_column))

        return self._encoders

    @property
    def output_encoder(self):
        if self._output_encoder is None:
            with timer('fit output encoder'):
                self._output_encoder = self.get_output_encoder()
                self._output_encoder.fit(self.read_column(self.table_training, self._output_encoder.source_column))

        return self._output_encoder

    @property
    def encoded_training_data(self):
        return self.generator(self.table_training, encoded=True)

    @property
    def encoded_validation_data(self):
        return self.generator(self.table_validation, encoded=True)

    @property
    def encoded_test_data(self):
        return self.generator(self.table_test, encoded=True)

    def generator(self, table, orient='row', encoded=False, stratify=False, chunksize=None):
        if not self.loaded:
            self._split_data()

        if orient == 'column':
            if encoded:
                for encoder in self.encoders:
                    transformed = encoder.transform(self.read_column(table, encoder.source_column))
                    encoded = {}
                    if hasattr(encoder, 'sequence_length'):
                        for i in range(encoder.sequence_length):
                            encoded[encoder.sequence_name(i)] = transformed[:, i]
                    else:
                        encoded[encoder.name] = transformed
                    yield Observations(x=pandas.DataFrame(encoded), y=None)

            else:
                for column in self.columns:
                    yield self.read_column(table, column)

        elif orient == 'row':
            if stratify:
                if not self.stratify:
                    raise ValueError("Can't stratify a generator for a pipeline with no stratify")

                if chunksize is None:
                    chunksize = 1
                min, max = self.connection.execute(
                    """
                        SELECT min({stratify}), max({stratify})
                        FROM (
                            SELECT {stratify}
                            FROM {table}
                            ORDER BY {stratify} ASC
                            LIMIT :chunksize
                        )
                    """.format(
                        stratify=self.quote(self.stratify),
                        table=self.quote(table),
                    ),
                    {'chunksize': chunksize}
                ).fetchone()
                while min and max:
                    dataframe = pandas.read_sql(
                        """
                            SELECT * FROM {table} WHERE {stratify} BETWEEN :min AND :max
                        """.format(
                            stratify=self.quote(self.stratify),
                            table=self.quote(table),
                        ),
                        self.connection,
                        parse_dates=self.datetime_columns,
                        params={'min': min, 'max': max}
                    )
                    if encoded:
                        dataframe = Observations(x=self.encode_x(dataframe), y=self.encode_y(dataframe))
                    yield dataframe

                    min, max = self.connection.execute(
                        """
                            SELECT min({stratify}), max({stratify})
                            FROM (
                                SELECT {stratify}
                                FROM {table}
                                WHERE {stratify} > :max
                                ORDER BY {stratify} ASC
                                LIMIT :chunksize
                            )
                        """.format(
                            stratify=self.quote(self.stratify),
                            table=self.quote(table),
                        ),
                        {'max': max, 'chunksize': chunksize}
                    ).fetchone()

            else:
                if chunksize is None:
                    chunksize = self.chunksize
                for dataframe in pandas.read_sql(
                        "SELECT * FROM {name}".format(name=self.quote(table)),
                        self.connection,
                        chunksize=chunksize,
                        parse_dates=self.datetime_columns
                ):
                    if encoded:
                        dataframe = Observations(x=self.encode_x(dataframe), y=self.encode_y(dataframe))
                    yield dataframe
        else:
            raise ValueError('orient "%s" not in "[row, column]"' % orient)

    @timed(logging.INFO)
    def read_column(self, table, column):
        if isinstance(table, pandas.DataFrame):
            return super(Base, self).read_column(table, column)

        return pandas.read_sql(
            'SELECT {column} FROM {table}'.format(
                column=self.quote(column),
                table=self.quote(table)
            ),
            self.connection,
            parse_dates=set(self.datetime_columns).intersection([column])
        )

    @timed(logging.INFO)
    def decode(self, predictions):
        return {encoder.name: encoder.reverse_transform(predictions) for encoder in self.encoder}

    @property
    def metadata_path(self):
        return os.path.join(lore.env.DATA_DIR, self.table + '.pickle')

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
            raise TypeError(
                'Iterative pipelines must be passed a generator rather than a dataframe. Did you forget to pass a `chunksize` to lore.io.Connection.dataframe()?')

        self.length = 0

        # This loop is unfortunately CPU limited rather than IO bound, based on dataframe creation
        # TODO parrallelize with multiple processes?
        for i, dataframe in enumerate(self._data):
            if i == 0:
                self._columns = dataframe.columns
                self.chunksize = len(dataframe)
                buffer = io.StringIO()
                dataframe.info(buf=buffer, memory_usage='deep')
                logger.info(buffer.getvalue())
                logger.info(dataframe.head())

                self._datetime_columns = [col for col, type in dataframe.dtypes.iteritems() if type == 'datetime64[ns]']
                with open(self.metadata_path, 'wb') as f:
                    pickle.dump(self.datetime_columns, f, pickle.HIGHEST_PROTOCOL)

            logger.info('appending %i rows to sqlite' % len(dataframe))
            self.length += len(dataframe)
            dataframe.to_sql(self.table, self.connection, index=False, if_exists="append")

        if self.subsample:
            with timer('subsample'):
                self.connection.executescript(
                    """
                        BEGIN;

                        CREATE TABLE {subsample} AS
                        SELECT * FROM {table}
                        ORDER BY random()
                        LIMIT {limit};

                        DROP TABLE {table};

                        ALTER TABLE {subsample} RENAME TO {table};

                        COMMIT;
                    """.format(
                        table=self.quote(self.table),
                        subsample=self.quote(self.table + '_subsample'),
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

    def __len__(self):
        if self._length is None:
            self._length = self.table_length(self.table)
        return self._length

    @timed(logging.INFO)
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
                    'PRAGMA table_info({table});'.format(table=self.quote(self.table))
                ).fetchall()
            ]

        return self._columns

    @timed(logging.INFO)
    def _random_split(self, name, start, stop, stratify=None):
        random = self.quote(name + '_random')

        if stratify is not None:
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS {random} AS
                SELECT DISTINCT {stratify}
                FROM {source}
                ORDER BY random();
            """.format(
                random=random,
                source=self.quote(self.table),
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
                source=self.quote(self.table)
            ))
            stratify = '_rowid_'
            random_column = 'other_rowid'

        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS {split} AS
            SELECT {source}.*
            FROM {source}
            JOIN {random}
              ON {random}.{random_column} = {source}.{stratify}
            WHERE {random}._rowid_ - 1 >= (SELECT max(_rowid_) FROM {random}) * :start
              AND {random}._rowid_ - 1 < (SELECT max(_rowid_) FROM {random}) * :stop
        """.format(
            split=self.quote(name),
            source=self.quote(self.table),
            random=random,
            random_column=random_column,
            stratify=stratify
        ), {
            'start': start,
            'stop': stop
        })

        if self.stratify:
            self.connection.execute("""
                CREATE INDEX IF NOT EXISTS {name} ON {table} ({column})
            """.format(
                name=self.quote('index_' + name + '_stratify'),
                table=self.quote(name),
                column=self.quote(self.stratify),
            ))

    def quote(self, identifier, errors="strict"):
        """
        https://stackoverflow.com/questions/6514274/how-do-you-escape-strings-for-sqlite-table-column-names-in-python

        :param identifier:
        :param errors:
        :return:
        """
        if identifier is None:
            return None

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
