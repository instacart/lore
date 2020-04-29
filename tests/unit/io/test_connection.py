# coding=utf-8
from __future__ import unicode_literals

import unittest
from threading import Thread
import datetime
import time
import math

import lore.io.connection
import sqlalchemy
from sqlalchemy import event
from sqlalchemy.engine import Engine
import pandas
import psycopg2
import lore


calls = 0
@event.listens_for(Engine, "after_cursor_execute")
def count_sql_calls(conn, cursor, statement, parameters, context, executemany):
    global calls
    calls += 1


class TestConnection(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.table = sqlalchemy.Table(
            'tests_users', lore.io.main.metadata,
            sqlalchemy.Column('id', sqlalchemy.Integer, primary_key=True),
            sqlalchemy.Column('first_name', sqlalchemy.String(50), nullable=False),
            sqlalchemy.Column('last_name', sqlalchemy.String(50), nullable=False),
            sqlalchemy.Column('nullable', sqlalchemy.Float(), nullable=True),
            sqlalchemy.Column('zero', sqlalchemy.Integer(), nullable=False),
            sqlalchemy.Column('pi', sqlalchemy.Float(), nullable=False),
            sqlalchemy.Index('index_tests_users_first_name_last_name', 'first_name', 'last_name', unique=True),
            sqlalchemy.Index('long_name_long_name_long_name_long_name_long_name_long_name_63_', 'first_name', unique=True),
        )
        lore.io.main.metadata.create_all()

    def setUp(self):
        self.table.drop()
        self.dataframe = pandas.DataFrame({
            'id': range(1001),
            'first_name': [str(i) for i in range(1001)],
            'last_name': [str(i) for i in range(1001)],
            'nullable': [i if i % 2 == 0 else None for i in range(1001)],
            'zero': [0] * 1001,
            'pi': [math.pi] * 1001,
        })

        lore.io.main.metadata.create_all()

    def test_connection(self):
        self.assertTrue(hasattr(lore.io, 'main'))

    def test_replace(self):
        lore.io.main.replace(self.table.name, self.dataframe)
        selected = lore.io.main.dataframe(sql='select * from tests_users')
        self.assertEqual(self.dataframe['first_name'].tolist(), selected['first_name'].tolist())
        self.assertEqual(self.dataframe['last_name'].tolist(), selected['last_name'].tolist())

    def test_insert_bulk(self):
        global calls
        calls = 0
        lore.io.main.insert(self.table.name, self.dataframe)
        self.assertEqual(calls, 0)

    def test_insert_batches(self):
        global calls
        calls = 0
        lore.io.main.insert(self.table.name, self.dataframe, batch_size=100)
        self.assertEqual(calls, 0)

    def test_multiple_statements(self):
        users = lore.io.main.select(sql='''
            insert into tests_users(first_name, last_name, zero, pi)
            values ('1', '2', 0, 3.14);
            insert into tests_users(first_name, last_name, zero, pi)
            values ('3', '4', 0, 3.14);
            insert into tests_users(first_name, last_name, zero, pi)
            values ('4', '5', 0, 3.14);
            select * from tests_users;
        ''')
        self.assertEqual(len(users), 3)

    def test_persistent_temp_tables(self):
        lore.io.main.execute(sql='create temporary table tests_persistence(id integer not null primary key)')
        lore.io.main.execute(sql='insert into tests_persistence values (1), (2), (3)')
        temps = lore.io.main.select(sql='select count(*) from tests_persistence')
        self.assertEqual(temps[0][0], 3)

    def test_connection_temp_table_isolation(self):
        lore.io.main.execute(sql='create temporary table tests_temp(id integer not null primary key)')
        lore.io.main.execute(sql='insert into tests_temp values (1), (2), (3)')
        lore.io.main_two.execute(sql='create temporary table tests_temp(id integer not null primary key)')
        lore.io.main_two.execute(sql='insert into tests_temp values (1), (2), (3)')

        temps = lore.io.main.select(sql='select count(*) from tests_temp')
        temps_two = lore.io.main_two.select(sql='select count(*) from tests_temp')
        self.assertEqual(temps[0][0], 3)
        self.assertEqual(temps_two[0][0], 3)

    def test_connection_autocommit_isolation(self):
        lore.io.main.execute(sql='drop table if exists tests_autocommit')
        lore.io.main.execute(sql='create table tests_autocommit(id integer not null primary key)')
        lore.io.main.execute(sql='insert into tests_autocommit values (1), (2), (3)')
        temps_two = lore.io.main_two.select(sql='select count(*) from tests_autocommit')
        self.assertEqual(temps_two[0][0], 3)

    def test_transaction_rollback(self):
        lore.io.main.execute(sql='drop table if exists tests_autocommit')
        lore.io.main.execute(sql='create table tests_autocommit(id integer not null primary key)')

        lore.io.main.execute(sql='insert into tests_autocommit values (0)')
        with self.assertRaises(sqlalchemy.exc.IntegrityError):
            with lore.io.main as transaction:
                transaction.execute(sql='insert into tests_autocommit values (1), (2), (3)')
                transaction.execute(sql='insert into tests_autocommit values (1), (2), (3)')

        inserted = lore.io.main_two.select(sql='select count(*) from tests_autocommit')[0][0]

        self.assertEqual(inserted, 1)

    def test_out_of_transaction_does_not_block_concurrent_writes(self):
        lore.io.main.execute(sql='drop table if exists tests_autocommit')
        lore.io.main.execute(sql='create table tests_autocommit(id integer not null primary key)')

        priors = []
        posts = []
        thrown = []

        def insert(delay=0):
            try:
                priors.append(lore.io.main.select(sql='select count(*) from tests_autocommit')[0][0])
                lore.io.main.execute(sql='insert into tests_autocommit values (1), (2), (3)')
                posts.append(lore.io.main.select(sql='select count(*) from tests_autocommit')[0][0])
                time.sleep(delay)
            except psycopg2.IntegrityError as ex:
                thrown.append(True)

        slow = Thread(target=insert, args=(1,))
        fast = Thread(target=insert, args=(0,))

        slow.start()
        time.sleep(0.5)
        fast.start()

        fast.join()
        fast_done = datetime.datetime.now()
        slow.join()
        slow_done = datetime.datetime.now()

        self.assertEqual(priors, [0, 3])
        self.assertEqual(posts, [3])
        self.assertEqual(thrown, [True])
        self.assertAlmostEqual((slow_done - fast_done).total_seconds(), 0.5, 1)

    def test_close(self):
        lore.io.main.execute(sql='create temporary table tests_close(id integer not null primary key)')
        lore.io.main.execute(sql='insert into tests_close values (1), (2), (3)')
        lore.io.main.close()
        reopened = lore.io.main.select(sql='select 1')
        self.assertEquals(reopened, [(1,)])
        with self.assertRaises(psycopg2.ProgrammingError):
            lore.io.main.select(sql='select count(*) from tests_close')

    def test_reconnect_and_retry(self):
        original_execute = lore.io.main._connection_execute

        def raise_dbapi_error_on_first_call(sql, bindings):
            lore.io.main._connection_execute = original_execute
            e = lore.io.connection.Psycopg2OperationalError('server closed the connection unexpectedly. This probably means the server terminated abnormally before or while processing the request.')
            raise sqlalchemy.exc.DBAPIError('select 1', [], e, True)

        exceptions = lore.env.STDOUT_EXCEPTIONS
        lore.env.STDOUT_EXCEPTIONS = False
        connection = lore.io.main._connection
        lore.io.main._connection_execute = raise_dbapi_error_on_first_call

        result = lore.io.main.select(sql='select 1')
        lore.env.STDOUT_EXCEPTIONS = exceptions

        self.assertNotEquals(connection, lore.io.main._connection)
        self.assertEquals(result[0][0], 1)

    def test_tuple_interpolation(self):
        lore.io.main.execute(sql='create temporary table tests_interpolation(id integer not null primary key)')
        lore.io.main.execute(sql='insert into tests_interpolation values (1), (2), (3)')
        temps = lore.io.main.select(sql='select * from tests_interpolation where id in {ids}', ids=(1, 2, 3))
        self.assertEqual(len(temps), 3)

    def test_reconnect_and_retry_on_expired_connection(self):
        original_execute = lore.io.main._connection_execute

        def raise_snowflake_programming_error_on_first_call(sql, bindings):
            lore.io.main._connection_execute = original_execute
            e = lore.io.connection.SnowflakeProgrammingError('Authentication token has expired.  The user must authenticate again')
            raise sqlalchemy.exc.DBAPIError('select 1', [], e, True)

        exceptions = lore.env.STDOUT_EXCEPTIONS
        lore.env.STDOUT_EXCEPTIONS = False
        connection = lore.io.main._connection
        lore.io.main._connection_execute = raise_snowflake_programming_error_on_first_call

        result = lore.io.main.select(sql='select 1')
        lore.env.STDOUT_EXCEPTIONS = exceptions

        self.assertNotEquals(connection, lore.io.main._connection)
        self.assertEquals(result[0][0], 1)

