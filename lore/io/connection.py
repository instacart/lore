import hashlib
import inspect
import logging
import os
import re
import sys
import tempfile
import csv
import gzip
from io import StringIO

import pandas
import sqlalchemy

import lore
from lore.util import timer
from lore.stores import query_cached


logger = logging.getLogger(__name__)


class Connection(object):
    UNLOAD_PREFIX = os.path.join(lore.env.name, 'unloads')
    IAM_ROLE = os.environ.get('IAM_ROLE', None)
    
    def __init__(self, url, **kwargs):
        for int_value in ['pool_size', 'pool_recycle', 'max_overflow']:
            if int_value in kwargs:
                kwargs[int_value] = int(kwargs[int_value])
        if 'poolclass' in kwargs:
            kwargs['poolclass'] = getattr(sqlalchemy.pool, kwargs['poolclass'])
        self._engine = sqlalchemy.create_engine(url, **kwargs)
        self._connection = None
        self._transactions = []
    
    def __enter__(self):
        if self._connection is None:
            self._connection = self._engine.connect()
        self._transactions.append(self._connection.begin())
        return self
    
    def __exit__(self, type, value, traceback):
        transaction = self._transactions.pop()
        if type is None:
            transaction.commit()
        else:
            transaction.rollback()

    @staticmethod
    def path(filename, extension='.sql'):
        return os.path.join(
            lore.env.root, lore.env.project, 'extracts',
            filename + extension)

    def execute(self, sql=None, filename=None, **kwargs):
        self.__execute(self.__prepare(sql, filename), kwargs)

    def insert(self, table, dataframe, batch_size=None):
        if batch_size is None:
            batch_size = len(dataframe)

        if self._connection is None:
            self._connection = self._engine.connect()

        with timer('INSERT ' + table):
            offset = 0
            while offset < len(dataframe):
                dataframe[offset:(offset + batch_size)].to_sql(
                    table,
                    self._connection,
                    if_exists='append',
                    index=False
                )
                offset += batch_size

    def replace(self, table, dataframe, batch_size=None):
        with timer('REPLACE ' + table):
            with self as transaction:
                transaction.execute("TRUNCATE " + table)
                self.insert(table, dataframe, batch_size)
                transaction.execute("ANALYZE " + table)

    def select(self, sql=None, filename=None, **kwargs):
        cache = kwargs.pop('cache', False)
        sql = self.__prepare(sql, filename)
        return self._select(sql, kwargs, cache=cache)

    @query_cached
    def _select(self, sql, bindings):
        return self.__execute(sql, bindings)

    def unload(self, sql=None, filename=None, **kwargs):
        cache = kwargs.pop('cache', False)
        sql = self.__prepare(sql, filename)
        return self._unload(sql, kwargs, cache=cache)
    
    @query_cached
    def _unload(self, sql, bindings):
        key = hashlib.sha1(str(sql).encode('utf-8')).hexdigest()

        match = re.match(r'.*?select\s(.*)from.*', sql, flags=re.IGNORECASE | re.UNICODE | re.DOTALL)
        if match:
            columns = []
            nested = 0
            potential = match[1].split(',')
            for column in potential:
                nested += column.count('(')
                nested -= column.count(')')
                if nested == 0:
                    columns.append(column.split()[-1].split('.')[-1].strip())
                elif column == potential[-1]:
                    column = re.split('from', column, flags=re.IGNORECASE)[0].strip()
                    columns.append(column.split()[-1].split('.')[-1].strip())
        else:
            columns = []
        logger.warning("Redshift unload requires poorly parsing column names from sql, found: {}".format(columns))

        sql = "UNLOAD ('" + sql.replace('\\', '\\\\').replace("'", "\\'") + "') "
        sql += "TO 's3://" + os.path.join(
            lore.io.bucket.name,
            self.UNLOAD_PREFIX,
            key,
            ''
        ) + "' "
        if Connection.IAM_ROLE:
            sql += "IAM_ROLE '" + Connection.IAM_ROLE + "' "
        sql += "DELIMITER '|' ADDQUOTES GZIP ALLOWOVERWRITE"
        if re.match(r'(.*?)(limit\s+\d+)(.*)', sql, re.IGNORECASE | re.UNICODE | re.DOTALL):
            logger.warning('LIMIT clause is not supported by unload, returning full set.')
            sql = re.sub(r'(.*?)(limit\s+\d+)(.*)', r'\1\3', sql, flags=re.IGNORECASE | re.UNICODE | re.DOTALL)
        self.__execute(sql, bindings)
        return key, columns

    @query_cached
    def load(self, key, columns):
        result = [columns]
        with timer('load:'):
            for entry in lore.io.bucket.objects.filter(
                Prefix=os.path.join(self.UNLOAD_PREFIX, key)
            ):
                temp = tempfile.NamedTemporaryFile()
                lore.io.bucket.download_file(entry.key, temp.name)
                with gzip.open(temp.name, 'rt') as gz:
                    result += list(csv.reader(gz, delimiter='|', quotechar='"'))
        
            return result
    
    @query_cached
    def load_dataframe(self, key, columns):
        with timer('load_dataframe:'):
            frames = []
            for entry in lore.io.bucket.objects.filter(
                Prefix=os.path.join(self.UNLOAD_PREFIX, key)
            ):
                temp = tempfile.NamedTemporaryFile()
                lore.io.bucket.download_file(entry.key, temp.name)
                dataframe = pandas.read_csv(
                    temp.name,
                    delimiter='|',
                    quotechar='"',
                    compression='gzip',
                    error_bad_lines=False
                )
                dataframe.columns = columns
                frames.append(dataframe)

            result = pandas.concat(frames)
            result.columns = columns
            buffer = StringIO()
            result.info(buf=buffer, memory_usage='deep')
            logger.info(buffer.getvalue())
            logger.info(result.head())
            return result
        
    def dataframe(self, sql=None, filename=None, **kwargs):
        cache = kwargs.pop('cache', False)
        sql = self.__prepare(sql, filename)
        dataframe = self._dataframe(sql, kwargs, cache=cache)
        buffer = StringIO()
        dataframe.info(buf=buffer, memory_usage='deep')
        logger.info(buffer.getvalue())
        logger.info(dataframe.head())
        return dataframe
        
    @query_cached
    def _dataframe(self, sql, bindings):
        sql = self.__caller_annotation(stack_depth=2) + sql
        logger.debug(str(bindings) + '\n' + sql)
        with timer("dataframe:"):
            if self._connection is None:
                self._connection = self._engine.connect()
            dataframe = pandas.read_sql(sql=sql, con=self._connection, params=bindings)
            return dataframe

    def __prepare(self, sql, filename):
        if sql is None and filename is not None:
            filename = Connection.path(filename, '.sql')
            logger.debug("READ SQL FILE: " + filename)
            with open(filename) as file:
                sql = file.read()
        # support mustache style bindings
        sql = re.sub(r'\{(\w+?)\}', r'%(\1)s', sql)
        return sql

    def __execute(self, sql, bindings):
        sql = self.__caller_annotation() + sql
        logger.debug(str(bindings) + '\n' + sql)
        with timer("sql:"):
            if self._connection is None:
                self._connection = self._engine.connect()
            self._connection.execute(sql, bindings)

    def __caller_annotation(self, stack_depth=3):
        caller = inspect.stack()[stack_depth]
        if sys.version_info.major == 3:
            caller = (caller.function, caller.filename, caller.lineno)
        return "/* %s | %s:%d in %s */\n" % (lore.env.project, caller[1], caller[2], caller[0])
