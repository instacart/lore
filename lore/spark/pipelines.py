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
from pyspark.ml.feature import StringIndexer, StandardScaler, QuantileDiscretizer, MinMaxScaler, Tokenizer, CountVectorizer
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

class Pipelines(object):
    def __init__(self, encoders):
        self._encoders = encoders
        self._pipeline = None
        self._str2idx = {}
        self._indexer_map = {}

    def fit(self, data):
        indexers = []
        for encoder in self._encoders:                        
            input_name = encoder.column            
            encoded_name = "{}_enc".format(encoder.column)

            # Normalizer, Standard Scaler, MinMaxScaler act on vector and not scalars
            if isinstance(encoder, (Uniform, Norm)):
                vectorized = "{}_vectorized".format(input_name)
                assembler = VectorAssembler(inputCols=[input_name],outputCol=vectorized)
                indexers.append(assembler)

            ## For Token type create map
            ### HACK!!!!!!! Ideally we should write a transformer/estimator for doing this. Still trying to figure
            ### how to write it in pyspark

            print("XXX Encoder : {}, {}".format(encoder, isinstance(encoder, Token)))
            if isinstance(encoder, Token):
                tokenizer = Tokenizer(inputCol=input_name, outputCol="{}_tokens".format(input_name))                
                indexers.append(tokenizer)
            
            if isinstance(encoder, Token):
                indexer = CountVectorizer(inputCol="{}_tokens".format(input_name), outputCol="ignored".format(input_name), minDF=1.0)                
                self._indexer_map[str(indexer)] = encoder.column
            elif isinstance(encoder, Unique):                
                indexer = StringIndexer(inputCol=input_name, outputCol=encoded_name)                
            elif isinstance(encoder, Uniform):                                
                indexer = MinMaxScaler(inputCol=vectorized, outputCol=encoded_name)
            elif isinstance(encoder, Norm):
                indexer = StandardScaler(withStd=True, withMean=True, inputCol=vectorized, outputCol=encoded_name)                
            elif isinstance(encoder, Quantile):
                indexer = QuantileDiscretizer(numBuckets=encoder.quantiles, inputCol=encoder.column, outputCol=encoded_name)            
            indexers.append(indexer)

        pipeline = Pipeline(stages=indexers)
        self._pipeline = pipeline.fit(data)
        for indexer in self._pipeline.stages:
            if str(indexer) in self._indexer_map:
                self._str2idx[self._indexer_map[str(indexer)]] = dict([(x, i) for i,x in enumerate(indexer.vocabulary)])
        return self._pipeline

    def transform(self, data): 

        ## For Token type create map
            ### HACK!!!!!!! Ideally we should write a transformer/estimator for doing this. Still trying to figure
            ### how to write it in pyspark
       
        def translate(mapping):
            def translate_(col, pad = 10):
                tmp = [mapping.get(x, -1) for x in col]
                return  [-1]* (pad - len(tmp)) + tmp
            return udf(translate_, StringType())

        result =  self._pipeline.transform(data)
        for indexer in self._pipeline.stages:
            idx_key = str(indexer)
            if idx_key in self._indexer_map:
                column = self._indexer_map[idx_key]
                input_name = "{}_tokens".format(column)
                result = result.withColumn("{}_enc".format(column), translate(self._str2idx[column])(input_name))
        return result