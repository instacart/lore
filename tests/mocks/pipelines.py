import pandas
from lore.encoders import Unique, Pass

from lore.pipelines import TrainTestSplit


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
