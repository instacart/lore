# coding=utf-8
from __future__ import unicode_literals

import unittest
from datetime import datetime

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
            sqlalchemy.Index('index_tests_users_first_name_last_name', 'first_name', 'last_name', unique=True)
        )
        lore.io.main.metadata.create_all()

    def setUp(self):
        self.table.drop()
        self.dataframe = pandas.DataFrame({
            'id': range(1000),
            'first_name': [str(i) for i in range(1000)],
            'last_name': [str(i) for i in range(1000)],
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
        self.assertEqual(calls, 2)

    def test_insert_batches(self):
        global calls
        calls = 0
        lore.io.main.insert(self.table.name, self.dataframe, batch_size=100)
        self.assertEqual(calls, 11)

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
        lore.io.main.execute(sql='''
            create temporary table tests_temp(id integer not null primary key);
        ''')
        lore.io.main.execute(sql='''
            insert into tests_temp values (1), (2), (3)
        ''')
        temps = lore.io.main.select(sql='''
            select count(*) from tests_temp
        ''')
        self.assertEqual(temps[0][0], 3)
