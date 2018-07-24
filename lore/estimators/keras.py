from __future__ import absolute_import
import atexit
import inspect
import logging
import warnings

import lore.io
from lore.callbacks import ReloadBest
from lore.encoders import Continuous, Pass
from lore.pipelines import Observations
from lore.util import timed, before_after_callbacks
from lore.env import require

require(
    lore.dependencies.KERAS +
    lore.dependencies.NUMPY +
    lore.dependencies.PANDAS +
    lore.dependencies.SKLEARN
)

import keras
import keras.backend
from keras.callbacks import EarlyStopping, TensorBoard, TerminateOnNaN
from keras.layers import Input, Embedding, Dense, Reshape, Concatenate, Dropout, SimpleRNN, Flatten, LSTM, GRU, BatchNormalization
from keras.optimizers import Adam
import numpy
import pandas
from sklearn.base import BaseEstimator
import tensorflow
from tensorflow.python.client.timeline import Timeline
from tensorflow.python.client import device_lib


available_gpus = len([x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU'])

if available_gpus:
    from keras.layers import CuDNNLSTM, CuDNNGRU

logger = logging.getLogger(__name__)


# prevents random gc exception at exit
atexit.register(keras.backend.clear_session)


class Base(BaseEstimator):
    def __init__(
        self,
        model=None,
        embed_size=10,
        sequence_embedding='flatten',
        sequence_embed_size=None,
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
        output_activation=None,
        monitor='val_loss',
        loss=None,
        towers=1,
        cudnn=False,
        multi_gpu_model=True,
        short_names=False,
        batch_norm=False,
    ):
        super(Base, self).__init__()
        if output_activation == 'sigmoid' and loss in ['mse', 'mae', 'mean_squared_error', 'mean_absolute_error']:
            logger.warning("Passing output_activation='sigmoid' restricts predictions between 0 and 1. If you have a binary classification problem, you should consider passing loss='binary_crossentropy'. Otherwise you should consider setting output_activation='linear'")

        self.towers = towers
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
        self.output_activation = output_activation
        self.monitor = monitor
        self.loss = loss
        self.keras = None
        self.history = None
        self.session = None
        self.model = model
        self.sequence_embedding = sequence_embedding
        self.sequence_embed_size = sequence_embed_size or embed_size
        self.cudnn = cudnn
        self.multi_gpu_model = multi_gpu_model
        self.short_names = short_names
        self.batch_norm = batch_norm

    def __getstate__(self):
        state = super(Base, self).__getstate__()
        # bloat can be restored via self.__init__() + self.build()
        for bloat in [
            'keras',
            'optimizer',
            'session',
            'model',
        ]:
            state[bloat] = None
        return state

    def __setstate__(self, dict):
        self.__dict__ = dict
        backward_compatible_defaults = {
            'towers': 1,
            'cudnn': False,
            'multi_gpu_model': None,
            'output_activation': 'sigmoid',
            'short_names': False,
            'batch_norm': False,
        }
        for key, default in backward_compatible_defaults.items():
            if key not in self.__dict__.keys():
                self.__dict__[key] = default

    @property
    def description(self):
        return '\n'.join([
            '\n  %s' % self.__module__ + '.' + self.__class__.__name__,
            '===================================================================',
            '| towers | embed | hidden | layer | layer  |         | model      |',
            '|        | size  | layers | width | shrink | dropout | parameters |',
            '-------------------------------------------------------------------',
            '| %6i | %5i | %6i | %5i | %6.4f | %7.5f | %10i |' % (
                self.towers,
                self.embed_size,
                self.hidden_layers,
                self.hidden_width,
                self.layer_shrink,
                self.dropout,
                self.keras.count_params()
            ),
            '==================================================================='
        ])

    def callbacks(self):
        return []

    @before_after_callbacks
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
                    if encoder.twin:
                        inputs[encoder.sequence_name(i, suffix='_twin')] = Input(shape=(1,), name=encoder.sequence_name(i, suffix='_twin'))
            else:
                inputs[encoder.name] = Input(shape=(1,), name=encoder.name)
                if encoder.twin:
                    inputs[encoder.twin_name] = Input(shape=(1,), name=encoder.twin_name)
        return inputs

    @timed(logging.INFO)
    def build_embedding_layer(self, inputs, tower):
        embeddings = {}
        for i, encoder in enumerate(self.model.pipeline.encoders):
            if self.short_names:
                number = i * self.towers + tower
                suffix = 't'
                embed_name = 'e%x' % number
                embed_name_twin = embed_name + suffix
                reshape_name = 'r%x' % number
                reshape_name_twin = reshape_name + suffix
                concatenate_name = 'c%x' % number
            else:
                suffix = '_twin'
                embed_name = str(tower) + '_embed_' + encoder.name
                embed_name_twin = str(tower) + '_embed_' + encoder.name + suffix
                reshape_name = None
                reshape_name_twin = None
                concatenate_name = None

            embed_size = encoder.embed_scale * self.embed_size
            if isinstance(encoder, Pass):
                embeddings[embed_name] = inputs[encoder.name]
            else:
                if isinstance(encoder, Continuous):
                    embedding = Dense(embed_size, activation='relu', name=embed_name)
                else:
                    embedding = Embedding(encoder.cardinality(), embed_size, name=embed_name)

                if hasattr(encoder, 'sequence_length'):
                    embeddings[embed_name], layer = self.build_sequence_embedding(encoder, embedding, inputs, embed_name)
                    if encoder.twin:
                        embeddings[embed_name_twin], _ = self.build_sequence_embedding(encoder, embedding, inputs, embed_name, suffix=suffix, layer=layer)
                else:
                    embeddings[embed_name] = Reshape(target_shape=(embed_size,), name=reshape_name)(embedding(inputs[encoder.name]))
                    if encoder.twin:
                        embeddings[embed_name_twin] = Reshape(target_shape=(embed_size,), name=reshape_name_twin)(embedding(inputs[encoder.twin_name]))

        return Concatenate(name=concatenate_name)(list(embeddings.values()))

    def build_sequence_embedding(self, encoder, embedding, inputs, embed_name, suffix='', layer=None):
        sequence_embed_size = encoder.embed_scale * self.sequence_embed_size
        sequence = []
        for i in range(encoder.sequence_length):
            sequence.append(embedding(inputs[encoder.sequence_name(i, suffix)]))

        if self.short_names:
            embed_sequence_name = embed_name + 's' + suffix
            embed_rnn_name = embed_name + 'r' + suffix
        else:
            embed_sequence_name = embed_name + '_sequence' + suffix
            embed_rnn_name = embed_name + '_' + self.sequence_embedding + suffix

        embed_sequence = Concatenate(name=embed_sequence_name)(sequence)

        if self.sequence_embedding == 'flatten' and not layer:
            layer = Flatten
        elif self.sequence_embedding in ['lstm', 'gru', 'simple_rnn'] and not layer:
            if self.sequence_embedding == 'lstm':
                layer = LSTM
                if self.cudnn:
                    if available_gpus > 0:
                        layer = CuDNNLSTM
                    else:
                        raise ValueError('Your estimator self.cuddn is True, but there are no GPUs available to tensorflow')
            elif self.sequence_embedding == 'gru':
                layer = GRU
                if self.cudnn:
                    if available_gpus > 0:
                        layer = CuDNNGRU
                    else:
                        raise ValueError('Your estimator self.cuddn is True, but there are no GPUs available to tensorflow')
            elif self.sequence_embedding == 'simple_rnn':
                layer = SimpleRNN
            else:
                raise ValueError("Unknown sequence_embedding type: %s" % self.sequence_embedding)

        if self.sequence_embedding in ['lstm', 'gru', 'simple_rnn']:
            embedding = layer(sequence_embed_size, name=embed_rnn_name)(embed_sequence)
        else:
            embedding = layer(name=embed_rnn_name)(embed_sequence)
        return embedding, layer

    @timed(logging.INFO)
    def build_hidden_layers(self, input_layer, tower):
        hidden_layers = input_layer

        hidden_width = self.hidden_width
        for i in range(self.hidden_layers):
            number = (i * self.towers + tower)
            if self.short_names:
                name = 'h%x' % number
            else:
                name = '%i_hidden_%i' % (tower, i)

            hidden_layers = Dense(int(hidden_width),
                                  activation=self.hidden_activation,
                                  activity_regularizer=self.hidden_activity_regularizer,
                                  kernel_regularizer=self.hidden_kernel_regularizer,
                                  bias_regularizer=self.hidden_bias_regularizer,
                                  name=name)(hidden_layers)
            if self.dropout > 0:
                if self.short_names:
                    name = 'd%x' % number
                else:
                    name = '%i_dropout_%i' % (tower, i)
                hidden_layers = Dropout(self.dropout, name=name)(hidden_layers)

            if self.batch_norm:
                if self.short_names:
                    name = 'b%x' % number
                else:
                    name = '%i_batchnorm_%i' % (tower, i)
                hidden_layers = BatchNormalization(name=name)(hidden_layers)

            if self.layer_shrink is None or self.layer_shrink == 0:
                pass
            elif self.layer_shrink < 1:
                hidden_width *= self.layer_shrink
            else:
                hidden_width -= self.layer_shrink

            hidden_width = max(1, hidden_width)

        return hidden_layers

    @timed(logging.INFO)
    def build_output_layer(self, hidden_layers, tower):
        if self.short_names:
            name = 'o%x' % tower
        else:
            name = '%i_output' % tower

        return Dense(1, activation=self.output_activation, name=name)(hidden_layers)

    @before_after_callbacks
    @timed(logging.INFO)
    def fit(self, x, y, validation_x=None, validation_y=None, epochs=100, patience=0, verbose=None, min_delta=0, tensorboard=False, timeline=False, **keras_kwargs):

        if isinstance(x, pandas.DataFrame):
            x = x.to_dict(orient='series')

        if isinstance(validation_x, pandas.DataFrame):
            validation_x = validation_x.to_dict(orient='series')

        if not self.keras or not self.optimizer:
            self.build()

        with self.session.as_default():
            if timeline:
                run_metadata = tensorflow.RunMetadata()
                options = tensorflow.RunOptions(trace_level=tensorflow.RunOptions.FULL_TRACE)
            else:
                run_metadata = None
                options = None
            self.keras.compile(
                loss=self.loss,
                optimizer=self.optimizer,
                options=options,
                run_metadata=run_metadata
            )
        if verbose is None:
            verbose = 1 if lore.env.NAME == lore.env.DEVELOPMENT else 0

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
                log_dir=self.model.tensorboard_path(),
                histogram_freq=1,
                batch_size=self.batch_size,
                write_graph=True,
                write_grads=True,
                write_images=True,
                embeddings_freq=1,
                embeddings_metadata=None
            )]

        try:
            with self.session.as_default():
                self.history = self.keras_fit(
                    x=x,
                    y=[y] * self.towers,
                    validation_data=Observations(x=validation_x, y=[validation_y] * self.towers),
                    batch_size=self.batch_size,
                    epochs=epochs,
                    verbose=verbose,
                    callbacks=callbacks,
                    **keras_kwargs
                ).history
        except KeyboardInterrupt:
            logger.warning('Caught SIGINT. Training aborted, and its history lost.')
            return {'loss': []}

        if timeline:
            with open(self.model.timeline_path(), 'w') as f:
                f.write(Timeline(step_stats=run_metadata.step_stats).generate_chrome_trace_format())

        return {
            'epochs': len(self.history['loss']),
            'train': reload_best.train_loss,
            'validate': reload_best.validate_loss,
        }

    def keras_fit(self, **kwargs):
        return self.keras.fit(**kwargs)

    @before_after_callbacks
    @timed(logging.DEBUG)
    def predict(self, dataframe):
        if isinstance(dataframe, pandas.DataFrame):
            dataframe = dataframe.to_dict(orient='series')

        with self.session.as_default():
            result = self.keras.predict(dataframe, batch_size=self.batch_size)

        if self.towers > 1:
            result = numpy.mean(result, axis=0).squeeze()

        return result

    @before_after_callbacks
    @timed(logging.INFO)
    def evaluate(self, x, y):
        if isinstance(x, pandas.DataFrame):
            x = x.to_dict(orient='series')

        if self.towers > 1:
            y = [y] * self.towers

        result = self.keras_evaluate(x, y)

        if self.towers > 1:
            result = numpy.mean(result, axis=0) / self.towers

        return result.squeeze()

    def keras_evaluate(self, x, y):
        with self.session.as_default():
            return self.keras.evaluate(x, y, batch_size=self.batch_size, verbose=0)

    @before_after_callbacks
    def score(self, x, y):
        return 1 / self.evaluate(x, y)


class Regression(Base):
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
            output_activation='linear',
            monitor='val_loss',
            loss='mean_squared_error',
            towers=1,
            cudnn=False,
            multi_gpu_model=True,
            short_names=False,
            batch_norm=False,
    ):
        kwargs = locals()
        kwargs.pop('self')
        kwargs.pop('__class__', None)
        super(Regression, self).__init__(**kwargs)


class BinaryClassifier(Base):
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
            output_activation='sigmoid',
            monitor='val_loss',
            loss='binary_crossentropy',
            towers=1,
            cudnn=False,
            multi_gpu_model=True,
            short_names=False,
            batch_norm=False,
    ):
        kwargs = locals()
        kwargs.pop('self')
        kwargs.pop('__class__', None)
        super(BinaryClassifier, self).__init__(**kwargs)


class MultiClassifier(Base):
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
            output_activation='softmax',
            monitor='val_loss',
            loss='categorical_crossentropy',
            towers=1,
            cudnn=False,
            multi_gpu_model=True,
            short_names=False,
            batch_norm=False,
    ):
        kwargs = locals()
        kwargs.pop('self')
        kwargs.pop('__class__', None)
        super(MultiClassifier, self).__init__(**kwargs)

    @timed(logging.INFO)
    def build_output_layer(self, hidden_layers, tower):
        return Dense(
            self.model.pipeline.output_encoder.cardinality(),
            activation=self.output_activation,
            name='%i_output' % tower
        )(hidden_layers)

    @timed(logging.DEBUG)
    def predict(self, dataframe):
        result = super(MultiClassifier, self).predict(dataframe)
        # map softmax to argmax
        return self.softmax
        if isinstance(dataframe, pandas.DataFrame):
            dataframe = dataframe.to_dict(orient='series')
        with self.session.as_default():
            return self.keras.predict(dataframe, batch_size=self.batch_size)
