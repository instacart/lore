from __future__ import absolute_import
import lore
from lore.env import require

require(
    lore.dependencies.NUMPY +
    lore.dependencies.PANDAS +
    lore.dependencies.PYSPARK
)
from lore.encoders import Uniform, Unique, Token, Enum, Quantile, Norm
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, StandardScaler, QuantileDiscretizer, MinMaxScaler
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

class Pipelines(object):
    def __init__(self, encoders):
        self._encoders = encoders
        self._pipeline = None

    def fit(self, data):
        indexers = []
        for encoder in self._encoders:                        
            input_name = encoder.column            
            encoded_name = "{}_enc".format(encoder.column)

            if isinstance(encoder, Unique):                
                indexer = StringIndexer(inputCol=input_name, outputCol=encoded_name)
                indexers.append(indexer)
            elif isinstance(encoder, Uniform):
                vectorized = "{}_vectorized".format(input_name)
                assembler = VectorAssembler(inputCols=[input_name],outputCol=vectorized)
                indexers.append(assembler)
                indexer = MinMaxScaler(inputCol=vectorized, outputCol=encoded_name)
                indexers.append(indexer)
            elif isinstance(encoder, Norm):
                vectorized = "{}_vectorized".format(input_name)
                assembler = VectorAssembler(inputCols=[input_name],outputCol=vectorized)
                indexers.append(assembler)
                indexer = StandardScaler(withStd=True, withMean=True, inputCol=input_name, outputCol=encoded_name)
                indexers.append(indexer)
            elif isinstance(encoder, Quantile):
                indexer = QuantileDiscretizer(numBuckets=encoder.quantiles, inputCol=encoder.column, outputCol=encoded_name)
                indexers.append(indexer)
        pipeline = Pipeline(stages=indexers)
        self._pipeline = pipeline.fit(data)
        return self._pipeline

    def transform(self, data):        
        return self._pipeline.transform(data)