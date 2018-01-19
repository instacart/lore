import datetime

import pandas
import sqlalchemy

from lore.encoders import Unique, Pass, Token, Boolean, Enum
from lore.transformers import DateTime
import lore.io
from lore.pipelines import Holdout, LowMemory, TimeSeries


class Xor(Holdout):
    def get_data(self):
        return pandas.DataFrame({
            'a': [0, 1, 0, 1] * 1000,
            'b': [0, 0, 1, 1] * 1000,
            'words': ['is false', 'is true', 'is not false', 'is not true' ] * 1000,
            'xor': [0, 1, 1, 0] * 1000
        })
    
    def get_encoders(self):
        return (
            Unique('a'),
            Unique('b'),
            Token('words')
        )
    
    def get_output_encoder(self):
        return Pass('xor')


class MockData(TimeSeries):
    def get_data(self):
        return pandas.DataFrame({
            'a': [1,2,3,4,5,6,7,8,9,10],
            'b': [21,22,23,24,25,26,27,28,29,30],
            'target': [1,0,1,0,1,0,1,0,1,0]
        })

    def get_encoders(self):
        return (
            Unique('a'),
            Unique('b'),
        )

    def get_output_encoder(self):
        return Pass('target')


class Users(LowMemory):
    dataframe = pandas.DataFrame({
        'id': range(1000),
        'first_name': [str(i) for i in range(1000)],
        'last_name': [str(i % 100) for i in range(1000)],
        'subscriber': [i % 2 == 0 for i in range(1000)],
        'signup_at': [datetime.datetime.now()] * 1000
    })
    sqlalchemy_table = sqlalchemy.Table(
        'tests_low_memory_users', lore.io.main.metadata,
        sqlalchemy.Column('id', sqlalchemy.Integer, primary_key=True),
        sqlalchemy.Column('first_name', sqlalchemy.String(50)),
        sqlalchemy.Column('last_name', sqlalchemy.String(50)),
        sqlalchemy.Column('subscriber', sqlalchemy.Boolean()),
        sqlalchemy.Column('signup_at', sqlalchemy.DateTime()),
    )
    sqlalchemy_table.drop(checkfirst=True)
    lore.io.main.metadata.create_all()
    lore.io.main.insert('tests_low_memory_users', dataframe)

    def _split_data(self):
        self.connection.execute('drop table if exists {name};'.format(name=self.table))
        super(Users, self)._split_data()
        
    def get_data(self):
        return lore.io.main.dataframe(sql='select * from tests_low_memory_users', chunksize=2)
    
    def get_encoders(self):
        return (
            Unique('id'),
            Unique('first_name'),
            Unique('last_name'),
            Boolean('subscriber'),
            Enum(DateTime('signup_at', 'dayofweek')),
        )
    
    def get_output_encoder(self):
        return Pass('subscriber')
