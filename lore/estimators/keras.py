from __future__ import absolute_import
import atexit
import logging

import pandas
import keras
import keras.backend
from keras.callbacks import EarlyStopping, TensorBoard, TerminateOnNaN
from keras.layers import Input, Embedding, Dense, Reshape, Concatenate, Dropout, SimpleRNN, Flatten, LSTM, GRU
from keras.optimizers import Adam
from sklearn.base import BaseEstimator
import tensorflow
from tensorflow.python.client.timeline import Timeline

import lore.io
from lore.callbacks import ReloadBest
from lore.encoders import Continuous
from lore.pipelines import Observations
from lore.util import timed

from tensorflow.python.client import device_lib
available_gpus = len([x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU'])

if available_gpus:
    from keras.layers import CuDNNLSTM, CuDNNGRU

logger = logging.getLogger(__name__)


# prevents random gc exception at exit
atexit.register(keras.backend.clear_session)


class Keras(BaseEstimator):
    def __init__(
            self,
            model=None,
            embed_size=10,
            sequence_embedding='flatten',
            sequence_embed_size=10,
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
            loss='categorical_crossentropy',
            towers=1,
            cudnn=True,
            multi_gpu_model=True,
    ):
        super(Keras, self).__init__()
        self.towers = towers
        self.embed_size = embed_size
        self.sequence_embed_size = 10
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
        self.sequence_embedding = sequence_embedding
        self.sequence_embed_size = sequence_embed_size
        self.cudnn = cudnn
        self.multi_gpu_model = multi_gpu_model
    
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

    def __setstate__(self, dict):
        self.__dict__ = dict
        backward_compatible_defaults = {
            'towers': 1,
            'cudnn': False,
            'multi_gpu_model': None,
        }
        for key, default in backward_compatible_defaults.items():
            if key not in self.__dict__.keys():
                self.__dict__[key] = default

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
        self.session = tensorflow.Session(
            config=tensorflow.ConfigProto(
                allow_soft_placement=available_gpus > 0,
                log_device_placement=log_device_placement
            )
        )
        keras.backend.set_session(self.session)
        with self.session.as_default():
            inputs = self.build_inputs()
            outputs = []
            for i in range(self.towers):
                if available_gpus > 0 and not self.multi_gpu_model:
                    chip = 'gpu'
                    node = i % available_gpus
                else:
                    chip = 'cpu'
                    node = 0
                with tensorflow.device('/%s:%i' % (chip, node)):
                    embedding_layer = self.build_embedding_layer(inputs, i)
                    hidden_layers = self.build_hidden_layers(embedding_layer, i)
                    outputs.append(self.build_output_layer(hidden_layers, i))
        
            self.keras = keras.models.Model(inputs=list(inputs.values()), outputs=outputs)
            if self.multi_gpu_model and available_gpus > 0:
                self.keras = keras.utils.multi_gpu_model(self.keras, gpus=available_gpus)
            self.optimizer = Adam(lr=self.learning_rate, decay=self.decay)
            self.keras._make_predict_function()
        logger.info('\n\n' + self.description + '\n\n')

    @timed(logging.INFO)
    def build_inputs(self):
        inputs = {}
        for encoder in self.model.pipeline.encoders:
            if hasattr(encoder, 'sequence_length'):
                for i in range(encoder.sequence_length):
                    inputs[encoder.sequence_name(i)] = Input(shape=(1,), name=encoder.sequence_name(i))
            else:
                inputs[encoder.name] = Input(shape=(1,), name=encoder.name)
        return inputs

    @timed(logging.INFO)
    def build_embedding_layer(self, inputs, tower):
        embeddings = {}
        for encoder in self.model.pipeline.encoders:
            embed_name = str(tower) + '_embed_' + encoder.name
            embed_size = encoder.embed_scale * self.embed_size

            if isinstance(encoder, Continuous):
                embedding = Dense(embed_size, activation='relu', name=embed_name)
            else:
                embedding = Embedding(encoder.cardinality(), embed_size, name=embed_name)

            if hasattr(encoder, 'sequence_length'):
                embeddings[embed_name] = self.build_sequence_embedding(encoder, embedding, inputs, embed_name)
            else:
                embeddings[embed_name] = Reshape(target_shape=(embed_size,))(embedding(inputs[encoder.name]))

        return Concatenate()(list(embeddings.values()))
    
    def build_sequence_embedding(self, encoder, embedding, inputs, embed_name, suffix=''):
        embed_size = encoder.embed_scale * self.embed_size
    
        sequence = []
        for i in range(encoder.sequence_length):
            sequence.append(embedding(inputs[encoder.sequence_name(i, suffix)]))
        embed_sequence = Concatenate(name=embed_name + '_sequence' + suffix)(sequence)
    
        if self.sequence_embedding == 'flatten':
            embedding = Flatten(name=embed_name + '_flatten' + suffix)(embed_sequence)
        else:
            sequence_embed_size = encoder.embed_scale * self.sequence_embed_size
            shaped_sequence = Reshape(target_shape=(encoder.sequence_length, embed_size))(embed_sequence)
            if self.sequence_embedding == 'lstm':
                lstm = LSTM
                if self.cudnn:
                    if available_gpus > 0:
                        lstm = CuDNNLSTM
                    else:
                        raise ValueError('Your estimator self.cuddn is True, but there are no GPUs available to tensorflow')
                embedding = lstm(sequence_embed_size, name=embed_name + '_lstm' + suffix)(shaped_sequence)
            elif self.sequence_embedding == 'gru':
                gru = GRU
                if self.cudnn:
                    if available_gpus > 0:
                        gru = CuDNNGRU
                    else:
                        raise ValueError('Your estimator self.cuddn is True, but there are no GPUs available to tensorflow')
                embedding = gru(sequence_embed_size, name=embed_name + '_gru' + suffix)(shaped_sequence)
            elif self.sequence_embedding == 'simple_rnn':
                embedding = SimpleRNN(sequence_embed_size, name=embed_name + '_rnn' + suffix)(shaped_sequence)
            else:
                raise ValueError("Unknown sequence_embedding type: %s" % self.sequence_embedding)
        return embedding

    @timed(logging.INFO)
    def build_hidden_layers(self, input_layer, tower):
        hidden_layers = input_layer

        hidden_width = self.hidden_width
        for i in range(self.hidden_layers):
            hidden_layers = Dense(int(hidden_width),
                                  activation=self.hidden_activation,
                                  activity_regularizer=self.hidden_activity_regularizer,
                                  kernel_regularizer=self.hidden_kernel_regularizer,
                                  bias_regularizer=self.hidden_bias_regularizer,
                                  name='%i_hidden_%i' % (tower, i))(hidden_layers)
            if self.dropout > 0:
                hidden_layers = Dropout(self.dropout)(hidden_layers)
            if self.layer_shrink < 1:
                hidden_width *= self.layer_shrink
            else:
                hidden_width -= self.layer_shrink
            hidden_width = max(1, hidden_width)
            
        return hidden_layers
    
    @timed(logging.INFO)
    def build_output_layer(self, hidden_layers, tower):
        return Dense(1, activation='sigmoid', name='%i_output' % tower)(hidden_layers)
    
    @timed(logging.INFO)
    def fit(self, x, y, validation_data=None, epochs=100, patience=0, verbose=None, min_delta=0, tensorboard=False, timeline=False, **keras_kwargs):

        if validation_data is None:
            validation_data = self.model.pipeline.encoded_validation_data

        if isinstance(x, pandas.DataFrame):
            x = x.to_dict(orient='series')
        
        if isinstance(validation_data.x, pandas.DataFrame):
            validation_data = Observations(
                x=validation_data.x.to_dict(orient='series'),
                y=validation_data.y
            )
            
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
            filepath=self.model.checkpoint_path(),
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
                y=[y] * self.towers,
                validation_data=Observations(x=validation_data.x, y=[validation_data.y] * self.towers),
                batch_size=self.batch_size,
                epochs=epochs,
                verbose=verbose,
                callbacks=callbacks,
                **keras_kwargs
            ).history

        if timeline:
            with open(self.model.timeline_path(), 'w') as f:
                f.write(Timeline(step_stats=run_metadata.step_stats).generate_chrome_trace_format())

        return {
            'epochs': len(self.history),
            'train': reload_best.train_loss,
            'validate': reload_best.validate_loss,
            'timeline': timeline,
        }

    @timed(logging.DEBUG)
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
