# coding=utf-8
from __future__ import unicode_literals

import unittest
from threading import Thread
import datetime
import time

import lore.io.connection
import sqlalchemy
from sqlalchemy import event
from sqlalchemy.engine import Engine
import pandas

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
            sqlalchemy.Column('first_name', sqlalchemy.String(50)),
            sqlalchemy.Column('last_name', sqlalchemy.String(50)),
            sqlalchemy.Column('nullable', sqlalchemy.Float()),
            sqlalchemy.Index('index_tests_users_first_name_last_name', 'first_name', 'last_name', unique=True),
            sqlalchemy.Index('long_name_long_name_long_name_long_name_long_name_long_name_63_', 'first_name', unique=True),
        )
        lore.io.main.metadata.create_all()

    def setUp(self):
        self.table.drop()
        self.dataframe = pandas.DataFrame({
            'id': range(1000),
            'first_name': [str(i) for i in range(1000)],
            'last_name': [str(i) for i in range(1000)],
            'nullable': [i if i % 2 == 0 else None for i in range(1000)]
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
            insert into tests_users(first_name, last_name)
            values ('1', '2');
            insert into tests_users(first_name, last_name)
            values ('3', '4');
            insert into tests_users(first_name, last_name)
            values ('4', '5');
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
    
        def insert(connection, delay=0):
            try:
                priors.append(connection.select(sql='select count(*) from tests_autocommit')[0][0])
                connection.execute(sql='insert into tests_autocommit values (1), (2), (3)')
                posts.append(connection.select(sql='select count(*) from tests_autocommit')[0][0])
                time.sleep(delay)
            except sqlalchemy.exc.IntegrityError as ex:
                thrown.append(True)
    
        slow = Thread(target=insert, args=(lore.io.main, 1))
        fast = Thread(target=insert, args=(lore.io.main_two, 0))

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
