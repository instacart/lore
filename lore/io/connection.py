import csv
import gc
import gzip
import hashlib
import inspect
import io
import logging
import math
import os
import re
import sys
import tempfile
import threading
import uuid

from datetime import datetime

import lore
from lore.env import require
from lore.util import timer, scrub_url
from lore.stores import query_cached

require(
    lore.dependencies.PANDAS +
    lore.dependencies.SQL +
    lore.dependencies.JINJA
)

import pandas

import sqlalchemy
from sqlalchemy import event
from sqlalchemy.schema import DropTable
from sqlalchemy.ext.compiler import compiles

import jinja2
jinja2_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(
        os.path.join(lore.env.ROOT, lore.env.APP, 'extracts')
    ),
    trim_blocks=True,
    lstrip_blocks=True
)

try:
    from psycopg2 import OperationalError as Psycopg2OperationalError
except lore.env.ModuleNotFoundError:
    class Psycopg2OperationalError(lore.env.StandardError):
        pass

try:
    from snowflake.connector.errors import ProgrammingError as SnowflakeProgrammingError
except lore.env.ModuleNotFoundError:
    class SnowflakeProgrammingError(lore.env.StandardError):
        pass

logger = logging.getLogger(__name__)

if sqlalchemy:
    @compiles(DropTable, 'postgresql')
    def _compile_drop_table(element, compiler, **kwargs):
        return compiler.visit_drop_table(element) + ' CASCADE'


    _after_replace_callbacks = []
    def after_replace(func):
        global _after_replace_callbacks
        _after_replace_callbacks.append(func)

    sqlalchemy_logger = logging.getLogger('sqlalchemy.engine')
    log_levels = {
        lore.env.DEVELOPMENT: logging.INFO,
        lore.env.TEST: logging.INFO,
    }
    sqlalchemy_logger.setLevel(log_levels.get(lore.env.NAME, logging.WARN))


class Connection(object):
    UNLOAD_PREFIX = os.path.join(lore.env.NAME, 'unloads')
    IAM_ROLE = os.environ.get('IAM_ROLE', None)

    def __init__(self, url, name='connection', watermark=True, **kwargs):
        if not sqlalchemy:
            raise lore.env.ModuleNotFoundError('No module named sqlalchemy. Please add it to requirements.txt.')

        parsed = lore.env.parse_url(url)
        self.adapter = parsed.scheme

        if self.adapter == 'postgres':
            require(lore.dependencies.POSTGRES)
        if self.adapter == 'snowflake':
            require(lore.dependencies.SNOWFLAKE)
            if 'numpy' not in parsed.query:
                logger.error('You should add `?numpy=True` query param to your snowflake connection url to ensure proper compatibility')

        for int_value in ['pool_size', 'pool_recycle', 'max_overflow']:
            if int_value in kwargs:
                kwargs[int_value] = int(kwargs[int_value])
        if 'poolclass' in kwargs:
            kwargs['poolclass'] = getattr(sqlalchemy.pool, kwargs['poolclass'])
        if '__name__' in kwargs:
            del kwargs['__name__']
        if 'echo' not in kwargs:
            kwargs['echo'] = False
        logger.info("Creating engine: %s %s" % (scrub_url(url), kwargs))
        self._engine = sqlalchemy.create_engine(url, **kwargs).execution_options(autocommit=True)
        self._metadata = None
        self.name = name
        self.url = url
        self._transactions = []
        self.__thread_local = threading.local()

        @event.listens_for(self._engine, "before_cursor_execute", retval=True)
        def comment_sql_calls(conn, cursor, statement, parameters, context, executemany):
            conn.info.setdefault('query_start_time', []).append(datetime.now())
            if watermark:
                stack = inspect.stack()[1:-1]
                if sys.version_info.major == 3:
                    stack = [(x.filename, x.lineno, x.function) for x in stack]
                else:
                    stack = [(x[1], x[2], x[3]) for x in stack]

                paths = [x[0] for x in stack]
                origin = next((x for x in paths if x.startswith(lore.env.ROOT)), None)
                if origin is None:
                    origin = next((x for x in paths if 'sqlalchemy' not in x), None)
                if origin is None:
                    origin = paths[0]
                caller = next(x for x in stack if x[0] == origin)

                statement = "/* %s | %s:%d in %s */\n" % (lore.env.APP, caller[0], caller[1], caller[2]) + statement
            return statement, parameters

        @event.listens_for(self._engine, "after_cursor_execute")
        def time_sql_calls(conn, cursor, statement, parameters, context, executemany):
            total = datetime.now() - conn.info['query_start_time'].pop(-1)
            logger.info("SQL: %s" % total)

        @event.listens_for(self._engine, "connect")
        def receive_connect(dbapi_connection, connection_record):
            if hasattr(dbapi_connection, 'get_dsn_parameters'):
                logger.info("connect: %s" % dbapi_connection.get_dsn_parameters())

    def __enter__(self):
        self._transactions.append(self._connection.begin())
        return self

    def __exit__(self, type, value, traceback):
        if self._transactions:
            transaction = self._transactions.pop()
            if type is None:
                transaction.commit()
            else:
                transaction.rollback()
        else:
            logger.warning("Closed connection aborted transaction")

    @property
    def _connection(self):
        if not hasattr(self.__thread_local, 'connection') or self.__thread_local.connection is None:
            self.__thread_local.connection = self._engine.connect()
        return self.__thread_local.connection

    @staticmethod
    def path(extract, extension='.sql'):
        return os.path.join(lore.env.ROOT, lore.env.APP, 'extracts', extract + extension)

    def execute(self, sql=None, extract=None, filename=None, **kwargs):
        self.__execute(self.__prepare(sql=sql, extract=extract, filename=filename, **kwargs), kwargs)

    def insert(self, table, dataframe, batch_size=10 ** 5):
        for i in range(int(math.ceil(float(len(dataframe)) / batch_size))):
            batch = dataframe.iloc[i * batch_size:(i + 1) * batch_size]
            if self._engine.dialect.name in ['postgresql', 'redshift']:
                # postgres can bulk load from a buffer of csv
                if sys.version_info[0] == 2:
                    rows = io.BytesIO()
                else:
                    rows = io.StringIO()
                batch.to_csv(rows, index=False, header=False, sep='|', na_rep='\\N', quoting=csv.QUOTE_NONE)
                rows.seek(0)
                self._connection.connection.cursor().copy_from(rows, table, null='\\N', sep='|', columns=dataframe.columns)
                self._connection.connection.commit()
                del rows

            elif self._engine.dialect.name == 'snowflake':
                # snowflake requires s3 upload to bulk insert
                try:
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.csv.gz')
                    tmp.close()
                    batch.to_csv(tmp.name, index=False, header=False, sep='|', na_rep='\\N', quoting=csv.QUOTE_NONE, compression='gzip')
                    suffix = str(uuid.uuid4())
                    stage_name = 'staged_' + suffix
                    self.execute('REMOVE @~/%(stage_name)s' % {'stage_name': stage_name})
                    self._connection.connection.cursor().execute('PUT file://%(path)s @~/%(stage_name)s;' % {'path': tmp.name, 'stage_name': stage_name})
                    self._connection.connection.cursor().execute(
                        'COPY INTO %(table)s '
                        'FROM @~/%(stage_name)s '
                        'FILE_FORMAT = (TYPE = CSV FIELD_DELIMITER = \'|\' SKIP_HEADER = 0 COMPRESSION = GZIP) '
                        'PURGE = TRUE' % {
                            'table': table,
                            'stage_name': stage_name
                        }
                    )
                    self.execute('REMOVE @~/%(stage_name)s' % {'stage_name': stage_name})
                finally:
                    os.remove(tmp.name)

            else:
                batch.to_sql(
                    table,
                    self._connection,
                    if_exists='append',
                    index=False,
                    chunksize=batch_size
                )

    def close(self):
        for transaction in reversed(self._transactions):
            logger.warning("Closing connection with active transactions causes rollback")
            transaction.rollback()
        self._transactions = []
        self.__thread_local = threading.local()
        self._engine.dispose()

    def replace(self, table, dataframe, batch_size=10 ** 5):
        import migrate.changeset
        global _after_replace_callbacks

        with timer('REPLACE ' + table):
            suffix = datetime.now().strftime('_%Y%m%d%H%M%S').encode('utf-8')
            self.metadata
            temp = 'tmp_'.encode('utf-8')
            source = sqlalchemy.Table(table, self.metadata, autoload=True, autoload_with=self._engine)
            destination_name = 'tmp_' + hashlib.sha256(temp + table.encode('utf-8') + suffix).hexdigest()[0:56]
            destination = sqlalchemy.Table(destination_name, self.metadata, autoload=False)
            for column in source.columns:
                destination.append_column(column.copy())
            destination.create()

            original_names = {}
            for index in source.indexes:
                # make sure the name is < 63 chars with the suffix
                name = hashlib.sha256(temp + index.name.encode('utf-8') + suffix).hexdigest()[0:60]
                original_names[name] = index.name
                columns = []
                for column in index.columns:
                    columns.append(next(x for x in destination.columns if x.name == column.name))
                new = sqlalchemy.Index(name, *columns)
                new.unique = index.unique
                new.table = destination
                new.create(bind=self._connection)
            self.insert(destination.name, dataframe, batch_size=batch_size)
            self.execute(sql="BEGIN; SET LOCAL statement_timeout = '1min'; ANALYZE %s; COMMIT;" % self.quote_identifier(table))

            with self as transaction:
                backup = sqlalchemy.Table(table + '_b', self.metadata)
                backup.drop(bind=self._connection, checkfirst=True)
                source.rename(name=source.name + '_b', connection=self._connection)
                destination.rename(name=table, connection=self._connection)
                for index in source.indexes:
                    index.rename(index.name[0:-2] + '_b', connection=self._connection)
                for index in destination.indexes:
                    index.rename(original_names[index.name], connection=self._connection)

        for func in _after_replace_callbacks:
            func(destination, source)

    @property
    def metadata(self):
        if not self._metadata:
            self._metadata = sqlalchemy.MetaData(bind=self._engine)

        return self._metadata

    def select(self, sql=None, extract=None, filename=None, **kwargs):
        cache = kwargs.pop('cache', False)
        return self._select(self.__prepare(sql=sql, extract=extract, filename=filename, **kwargs), kwargs, cache=cache)

    @query_cached
    def _select(self, sql, bindings):
        return self.__execute(sql, bindings).fetchall()

    def unload(self, sql=None, extract=None, filename=None, **kwargs):
        cache = kwargs.pop('cache', False)
        sql = self.__prepare(sql=sql, extract=extract, filename=filename)
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
        with timer('load'):
            for entry in lore.io.bucket.objects.filter(
                Prefix=os.path.join(self.UNLOAD_PREFIX, key)
            ):
                temp = tempfile.NamedTemporaryFile()
                lore.io.bucket.download_file(entry.key, temp.name)
                with gzip.open(temp.name, 'rt') as gz:
                    result += list(csv.reader(gz, delimiter='|', quotechar='"'))

            return result

    @query_cached
    def load_dataframe(self, key, columns, log_verbose=False,):
        with timer('load_dataframe'):
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
            if log_verbose is True:
                buffer = io.StringIO()
                result.info(buf=buffer, memory_usage='deep')
                logger.info(buffer.getvalue())
                logger.info(result.head())
            return result

    def dataframe(self, sql=None, extract=None, filename=None, log_verbose=False, **kwargs):
        cache = kwargs.pop('cache', False)
        chunksize = kwargs.pop('chunksize', None)
        if chunksize and cache:
            raise ValueError('Chunking is incompatible with caching. Choose to pass either "cache" or "chunksize".')
        sql = self.__prepare(sql=sql, extract=extract, filename=filename, **kwargs)
        dataframe = self._dataframe(sql, kwargs, cache=cache, chunksize=chunksize)
        if log_verbose is True and chunksize is None:
            buffer = io.StringIO()
            dataframe.info(buf=buffer, memory_usage='deep')
            logger.info(buffer.getvalue())
            logger.info(dataframe.head())
        return dataframe

    @query_cached
    def _dataframe(self, sql, bindings, chunksize=None):
        with timer("dataframe:"):
            try:
                return pandas.read_sql_query(sql=sql, con=self._connection, params=bindings, chunksize=chunksize)
            except (sqlalchemy.exc.DBAPIError, Psycopg2OperationalError, SnowflakeProgrammingError) as e:
                if not self._transactions and (isinstance(e, Psycopg2OperationalError) or e.connection_invalidated):
                    logger.warning('Reconnect and retry due to invalid connection')
                    self.close()
                    return pandas.read_sql_query(sql=sql, con=self._connection, params=bindings, chunksize=chunksize)
                elif not self._transactions and (isinstance(e, SnowflakeProgrammingError) or e.connection_invalidated):
                    if hasattr(e, 'msg') and e.msg and "authenticate" in e.msg.lower():
                        logger.warning('Reconnect and retry due to unauthenticated connection')
                        self.close()
                        return pandas.read_sql_query(sql=sql, con=self._connection, params=bindings, chunksize=chunksize)
                    else:
                        logger.warning('Failed to execute db query with error type {}. Reason : {}'.format(type(e).__name__, e.msg))
                        raise
                else:
                    raise

    def temp_table(self, tablename, sql=None, extract=None, filename=None, drop=True, **kwargs):
        tablename = self.quote_identifier(tablename)
        with timer("temptable:"):
            if drop:
                self.execute(sql='DROP TABLE IF EXISTS ' + tablename)
            self.__execute('CREATE TEMPORARY TABLE ' + tablename + ' AS ' + self.__prepare(sql=sql, extract=extract, filename=filename, **kwargs), kwargs)

    def quote_identifier(self, identifier):
        return self._engine.dialect.identifier_preparer.quote(identifier)

    def __prepare(self, sql=None, extract=None, filename=None, **kwargs):
        if extract is None:
            extract = filename
        if sql is None and extract is not None:
            sql_filename = Connection.path(extract, '.sql')
            template_filename = Connection.path(extract, '.sql.j2')
            if os.path.exists(sql_filename):
                logger.debug('READ SQL FILE: ' + sql_filename)
                if os.path.exists(template_filename):
                    logger.warning('ignoring template with the same base file name')
                with open(sql_filename) as file:
                    sql = file.read()
            elif os.path.exists(template_filename):
                logger.debug('READ SQL TEMPLATE: ' + template_filename)
                sql = jinja2_env.get_template(extract + '.sql.j2').render(**kwargs)
            else:
                raise IOError('There is no template or sql file for %s' % extract)

        # support mustache style bindings
        sql = re.sub(r'\{(\w+?)\}', r'%(\1)s', sql)

        return sql

    def __execute(self, sql, bindings):
        try:
            return self._connection.execute(sql, bindings)
        except (sqlalchemy.exc.DBAPIError, Psycopg2OperationalError, SnowflakeProgrammingError) as e:
            if not self._transactions and (isinstance(e, Psycopg2OperationalError) or e.connection_invalidated):
                logger.warning('Reconnect and retry due to invalid connection')
                self.close()
                return self._connection.execute(sql, bindings)
            elif not self._transactions and (isinstance(e, SnowflakeProgrammingError) or e.connection_invalidated):
                if hasattr(e, 'msg') and e.msg and "authenticate" in e.msg.lower():
                    logger.warning('Reconnect and retry due to unauthenticated connection')
                    self.close()
                    return self._connection.execute(sql, bindings)
                else:
                    logger.warning('Failed to execute db query with error type {}. Reason : {}'.format(type(e).__name__, e.msg))
                    raise
            else:
                raise
