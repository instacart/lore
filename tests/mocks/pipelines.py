import datetime

import pandas
import sqlalchemy

from lore.encoders import Unique, Pass, Token, Boolean, Enum, Continuous, OneHot, NestedUnique, NestedNorm
from lore.transformers import DateTime
import lore.io
import lore.pipelines.holdout
import lore.pipelines.iterative
import lore.pipelines.time_series


class Xor(lore.pipelines.holdout.Base):
    def get_data(self):
        return pandas.DataFrame({
            'a': [0, 1, 0, 1] * 1000,
            'b': [0, 0, 1, 1] * 1000,
            'words': ['is false', 'is true', 'is not false', 'is not true'] * 1000,
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


class XorSingle(Xor):
    def __init__(
        self,
        type
    ):
        super(XorSingle, self).__init__()
        self.type = type

    def get_encoders(self):
        if self.type == 'tuple':
            return (
                Unique('a'),
            )
        elif self.type == 'len1':
            return (
                Unique('a')
            )
        elif self.type == 'single':
            return Unique('a')


class XorMulti(Xor):
    def get_output_encoder(self):
        return OneHot('xor')


class MockData(lore.pipelines.time_series.Base):
    def get_data(self):
        return pandas.DataFrame({
            'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'b': [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
            'target': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        })

    def get_encoders(self):
        return (
            Unique('a'),
            Unique('b'),
        )

    def get_output_encoder(self):
        return Pass('target')


class TwinData(lore.pipelines.time_series.Base):
    def get_data(self):
        return pandas.DataFrame({
            'a': [100, 200, 300],
            'a_twin': [300, 500, 100],
            'b': [500, 100, 700],
            'b_twin': [100, 400, 500],
            'c': ["orange", "orange juice", "organic orange juice"],
            'c_twin': ["navel orange", "orange juice", "organic orange juice"],
            'user_id': [1,2,3],
            'price': [1.99, 2.99, 3.99],
            'target': [1, 0, 1]
        })

    def get_encoders(self):
        return (
            Unique('a', twin=True),
            Unique('b', twin=True),
            Token('c', twin=True, sequence_length=3),
            Unique('user_id'),
            Pass('price')
        )

    def get_output_encoder(self):
        return Pass('target')


class TwinDataWithVaryingEmbedScale(lore.pipelines.time_series.Base):
    def get_data(self):
        return pandas.DataFrame({
            'a': [100, 200, 300],
            'a_twin': [300, 500, 100],
            'b': [500, 100, 700],
            'b_twin': [100, 400, 500],
            'c': ["orange", "orange juice", "organic orange juice"],
            'c_twin': ["navel orange", "orange juice", "organic orange juice"],
            'user_id': [1,2,3],
            'price': [1.99, 2.99, 3.99],
            'target': [1, 0, 1]
        })

    def get_encoders(self):
        return (
            Unique('a', embed_scale=3, twin=True),
            Unique('b', embed_scale=4, twin=True),
            Token('c', embed_scale=5, twin=True, sequence_length=3),
            Unique('user_id'),
            Pass('price')
        )

    def get_output_encoder(self):
        return Pass('target')

class Users(lore.pipelines.iterative.Base):
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


class MockNestedData(lore.pipelines.time_series.Base):
    def get_data(self):
        return pandas.DataFrame({
            'a': [['a', 'b'], ['a', 'b', 'c'], ['c', 'd'], ['a', 'e'], None],
            'b': [[0, 1, 2], None, [2, 3, 4, 5], [1], [-1, 10]],
            'target': [1, 0, 1, 0, 1]
        })

    def get_encoders(self):
        return (
            NestedUnique('a'),
            NestedNorm('b'),
        )

    def get_output_encoder(self):
        return Pass('target')


class OneHotPipeline(lore.pipelines.holdout.Base):
    def get_data(self):
        return pandas.DataFrame({
            'a': [1, 1, 2, 3] * 1000,
            'b': [0, 0, 1, 1] * 1000,
            'words': ['is false', 'is true', 'is not false', 'is not true'] * 1000,
            'xor': [0, 1, 1, 0] * 1000
        })

    def get_encoders(self):
        return (
            OneHot('a', minimum_occurrences=1001, compressed=True),
            OneHot('b'),
            Token('words')
        )

    def get_output_encoder(self):
        return Pass('xor')


class NaivePipeline(lore.pipelines.holdout.Base):
    def get_data(self):
        return pandas.DataFrame({
            'x': [1, 2, 3]*1000,
            'y': [0, 1, 1]*1000
        })

    def get_encoders(self):
        return(Pass('x'))

    def get_output_encoder(self):
        return Pass('y')
