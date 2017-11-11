from __future__ import absolute_import
import atexit
import logging
import sys

import pandas
import keras
import keras.backend
from keras.callbacks import EarlyStopping, TensorBoard, TerminateOnNaN
from keras.layers import Input, Embedding, Dense, Reshape, Concatenate, Dropout, LSTM
from keras.optimizers import Adam
from sklearn.base import BaseEstimator
import tensorflow
from tensorflow.python.client.timeline import Timeline

import lore.io
from lore.callbacks import ReloadBest
from lore.encoders import Continuous
from lore.pipelines import Observations
from lore.util import timed

logger = logging.getLogger(__name__)


def cleanup_tensorflow():
    # prevents random gc exception at exit
    keras.backend.clear_session()
atexit.register(cleanup_tensorflow)


class Keras(BaseEstimator):
    def __init__(
            self,
            model=None,
            embed_size=10,
            hidden_width=1024,
            hidden_layers=4,
            layer_shrink=0.5,
            dropout=0,
            batch_size=32,
            learning_rate=0.001,
            decay=0.,
            optimizer=None,
            hidden_activation='relu',
            hidden_activity_regularizer=None,
            hidden_bias_regularizer=None,
            hidden_kernel_regularizer=None,
            monitor='val_acc',
            loss='categorical_crossentropy'
    ):
        super(Keras, self).__init__()
        self.embed_size = embed_size
        self.hidden_width = hidden_width
        self.hidden_layers = hidden_layers
        self.layer_shrink = layer_shrink
        self.dropout = dropout
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.decay = decay
        self.optimizer = optimizer
        self.hidden_activation = hidden_activation
        self.hidden_activity_regularizer = hidden_activity_regularizer
        self.hidden_bias_regularizer = hidden_bias_regularizer
        self.hidden_kernel_regularizer = hidden_kernel_regularizer
        self.monitor = monitor
        self.loss = loss
        self.keras = None
        self.history = None
        self.session = None
        self.model = model
    
    def __getstate__(self):
        state = super(Keras, self).__getstate__()
        # bloat can be restored via self.__init__() + self.build()
        for bloat in [
            'keras',
            'optimizer',
            'session',
            'model'
        ]:
            state[bloat] = None
        return state
    
    @property
    def description(self):
        return '\n'.join([
            '\n  %s' % self.__module__ + '.' + self.__class__.__name__,
            '==========================================================',
            '| embed | hidden | layer | layer  |         | model      |',
            '| size  | layers | width | shrink | dropout | parameters |',
            '----------------------------------------------------------',
            '| %5i | %6i | %5i | %6.4f | %7.5f | %10i |' % (
                self.embed_size,
                self.hidden_layers,
                self.hidden_width,
                self.layer_shrink,
                self.dropout,
                self.keras.count_params()
            ),
            '=========================================================='
        ])
    
    def callbacks(self):
        return []
    
    @timed(logging.INFO)
    def build(self, log_device_placement=False):
        keras.backend.clear_session()
        self.session = tensorflow.Session(config=tensorflow.ConfigProto(log_device_placement=log_device_placement))
        keras.backend.set_session(self.session)
        with self.session.as_default():
            inputs = self.build_inputs()
            embedding_layer = self.build_embedding_layer(inputs)
            hidden_layers = self.build_hidden_layers(embedding_layer)
            output = self.build_output_layer(hidden_layers)
            
            self.keras = keras.models.Model(inputs=list(inputs.values()), outputs=output)
            self.optimizer = Adam(lr=self.learning_rate, decay=self.decay)
            self.keras._make_predict_function()
        logger.info('\n\n' + self.description + '\n\n')
    
    @timed(logging.INFO)
    def build_inputs(self):
        inputs = {}
        for encoder in self.model.pipeline.encoders:
            if hasattr(encoder, 'sequence_length'):
                for i in range(encoder.sequence_length):
                    name = encoder.name + '_' + str(i)
                    inputs[name] = Input(shape=(1,), name=name)
            else:
                inputs[encoder.name] = Input(shape=(1,), name=encoder.name)
        return inputs
    
    @timed(logging.INFO)
    def build_embedding_layer(self, inputs):
        embeddings = {}
        reshape = Reshape(target_shape=(self.embed_size,))
        for encoder in self.model.pipeline.encoders:
            if isinstance(encoder, Continuous):
                embedding = Dense(self.embed_size, activation='relu', name='embed_' + encoder.name)
                embeddings[encoder.name] = embedding(inputs[encoder.name])
            elif hasattr(encoder, 'sequence_length'):
                for i in range(encoder.sequence_length):
                    name = encoder.name + '_' + str(i)
                    embedding = Embedding(encoder.cardinality(), self.embed_size, name='embed_' + name)
                    embeddings[name] = reshape(embedding(inputs[name]))
            else:
                logger.debug("%s: %s %s" % (encoder.name, encoder.dtype, encoder.cardinality()))
                embedding = Embedding(encoder.cardinality(), self.embed_size, name='embed_' + encoder.name)
                embeddings[encoder.name] = reshape(embedding(inputs[encoder.name]))
        
        return Concatenate()(list(embeddings.values()))
    
    @timed(logging.INFO)
    def build_hidden_layers(self, input_layer):
        hidden_layers = input_layer

        hidden_width = self.hidden_width
        for i in range(self.hidden_layers):
            hidden_layers = Dense(int(hidden_width),
                                  activation=self.hidden_activation,
                                  activity_regularizer=self.hidden_activity_regularizer,
                                  kernel_regularizer=self.hidden_kernel_regularizer,
                                  bias_regularizer=self.hidden_bias_regularizer,
                                  name='hidden_%i' % i)(hidden_layers)
            if self.dropout > 0:
                hidden_layers = Dropout(self.dropout)(hidden_layers)
            if self.layer_shrink < 1:
                hidden_width *= self.layer_shrink
            else:
                hidden_width -= self.layer_shrink
            hidden_width = max(1, hidden_width)
            
        return hidden_layers
    
    @timed(logging.INFO)
    def build_output_layer(self, hidden_layers):
        return Dense(1, activation='sigmoid')(hidden_layers)
    
    @timed(logging.INFO)
    def fit(self, x, y, epochs=100, patience=0, verbose=None, min_delta=0, tensorboard=False, timeline=True):

        if isinstance(x, pandas.DataFrame):
            x = x.to_dict(orient='series')
        
        if isinstance(self.model.pipeline.encoded_validation_data.x, pandas.DataFrame):
            validation_data = Observations(
                x=self.model.pipeline.encoded_validation_data.x.to_dict(orient='series'),
                y=self.model.pipeline.encoded_validation_data.y
            )
        else:
            validation_data = self.model.pipeline.encoded_validation_data
            
        if not self.keras or not self.optimizer:
            self.build()
            
        with self.session.as_default():
            if timeline:
                run_metadata = tensorflow.RunMetadata()
            else:
                run_metadata = None
            self.keras.compile(
                loss=self.loss,
                optimizer=self.optimizer,
                options=tensorflow.RunOptions(trace_level=tensorflow.RunOptions.FULL_TRACE),
                run_metadata=run_metadata
            )
        if verbose is None:
            verbose = 1 if lore.env.name == lore.env.DEVELOPMENT else 0
        
        logger.info(
            '\n'.join([
                '\n\n\n  Fitting',
                '==============================',
                '| batch | learning |         |',
                '| size  | rate     |   decay |',
                '------------------------------',
                '| %5i | %8.6f | %7.5f |' % (
                    self.batch_size,
                    self.learning_rate,
                    self.decay,
                ),
                '==============================\n\n'
            ])
        )
        
        reload_best = ReloadBest(
            filepath=self.model.serializer.checkpoint_path,
            monitor=self.monitor,
            mode='auto',
        )
        
        callbacks = self.callbacks()
        callbacks += [
            reload_best,
            TerminateOnNaN(),
            EarlyStopping(
                monitor=self.monitor,
                min_delta=min_delta,
                patience=patience,
                verbose=verbose,
                mode='auto',
            ),
        ]
        if tensorboard:
            callbacks += [TensorBoard(
                log_dir=self.model.serializer.tensorboard_path,
                histogram_freq=1,
                batch_size=self.batch_size,
                write_graph=True,
                write_grads=True,
                write_images=True,
                embeddings_freq=1,
                embeddings_metadata=None
            )]
        
        with self.session.as_default():
            self.history = self.keras.fit(
                x=x,
                y=y,
                validation_data=validation_data,
                batch_size=self.batch_size,
                epochs=epochs,
                verbose=verbose,
                callbacks=callbacks
            ).history

        if timeline:
            with open(self.model.serializer.timeline_path, 'w') as f:
                f.write(Timeline(step_stats=run_metadata.step_stats).generate_chrome_trace_format())

        return {
            'epochs': len(self.history),
            'train': reload_best.train_loss,
            'validate': reload_best.validate_loss,
            'timeline': timeline,
        }

    @timed(logging.INFO)
    def predict(self, dataframe):
        if isinstance(dataframe, pandas.DataFrame):
            dataframe = dataframe.to_dict(orient='series')
        with self.session.as_default():
            return self.keras.predict(dataframe, batch_size=self.batch_size)
    
    @timed(logging.INFO)
    def score(self, x, y):
        if isinstance(x, pandas.DataFrame):
            x = x.to_dict(orient='series')
        with self.session.as_default():
            return 1 / self.keras.evaluate(x, y, batch_size=self.batch_size)


class LSTMEmbeddings(Keras):
    @timed(logging.INFO)
    def build_embedding_layer(self, inputs):
        embeddings = {}
        reshape = Reshape(target_shape=(self.embed_size,))
        for encoder in self.model.pipeline.encoders:
            if isinstance(encoder, Continuous):
                embedding = Dense(self.embed_size, activation='relu', name='embed_' + encoder.name)
                embeddings[encoder.name] = embedding(inputs[encoder.name])
            elif hasattr(encoder, 'sequence_length'):
                adapter = Embedding(encoder.cardinality(), self.embed_size, name='embed_' + encoder.name)
                embedding = LSTM(self.embed_size, name=encoder.name)
                tokens = []
                for i in range(encoder.sequence_length):
                    name = encoder.name + '_' + str(i)
                    tokens.append(adapter(inputs[name]))
                embeddings[encoder.name] = embedding(
                    Reshape((encoder.sequence_length, self.embed_size))(Concatenate()(tokens))
                )
            else:
                logger.debug("%s: %s %s" % (encoder.name, encoder.dtype, encoder.cardinality()))
                embedding = Embedding(encoder.cardinality(), self.embed_size, name='embed_' + encoder.name)
                embeddings[encoder.name] = reshape(embedding(inputs[encoder.name]))
        
        return Concatenate()(list(embeddings.values()))
