import pandas
from lore.encoders import Unique, Pass

from lore.pipelines import TrainTestSplit, SortedTrainTestSplit


class Xor(TrainTestSplit):
    def get_data(self):
        return pandas.DataFrame({
            'a': [0, 1, 0, 1] * 1000,
            'b': [0, 0, 1, 1] * 1000,
            'xor': [0, 1, 1, 0] * 1000
        })
    
    def get_encoders(self):
        return (
            Unique('a'),
            Unique('b'),
        )
    
    def get_output_encoder(self):
        return Pass('xor')


class MockData(SortedTrainTestSplit):
    def get_data(self):
        return pandas.DataFrame({
            'a': [1,2,3,4,5,6,7,8,9,10],
            'b': [21,22,23,24,25,26,27,28,29,30]
        })

