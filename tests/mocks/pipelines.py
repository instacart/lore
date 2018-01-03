import pandas
from lore.encoders import Unique, Pass, Token

from lore.pipelines import Holdout, TimeSeries


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

