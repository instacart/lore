import os
from os.path import join, dirname
import logging
import botocore

import lore
from lore.util import timer
from lore.env import require

require(lore.dependencies.H5PY)
import h5py


try:
    FileExistsError
except NameError:
    FileExistsError = OSError


logger = logging.getLogger(__name__)


class Base(lore.models.base.Base):
    def __init__(self, pipeline, estimator):
        super(Base, self).__init__(pipeline, estimator)

    def weights_path(self):
        return join(self.fitting_path(), 'weights.h5')

    def checkpoint_path(self):
        return join(self.fitting_path(), 'checkpoints/{epoch}.h5')

    def tensorboard_path(self):
        return join(self.fitting_path(), 'tensorboard')

    def timeline_path(self):
        return join(self.fitting_path(), 'timeline.json')

    def remote_weights_path(self):
        if self.fitting:
            return join(self.remote_path(), str(self.fitting), 'weights.h5')
        else:
            return join(self.remote_path(), 'weights.h5')

    @property
    def fitting(self):
        return self._fitting

    @fitting.setter
    def fitting(self, value):
        self._fitting = value
        if self._fitting is not None:

            if not os.path.exists(dirname(self.checkpoint_path())):
                try:
                    os.makedirs(dirname(self.checkpoint_path()))
                except FileExistsError as ex:
                    pass  # race to create

            if not os.path.exists(dirname(self.tensorboard_path())):
                try:
                    os.makedirs(dirname(self.tensorboard_path()))
                except FileExistsError as ex:
                    pass  # race to create

    def save(self, stats=None):
        super(Base, self).save(stats)

        with timer('save weights'):
            # Only save weights, because saving named layers that have shared
            # weights causes an error on reload
            self.estimator.keras.save_weights(self.weights_path())

        # Patch for keras 2 models saved with optimizer weights:
        # https://github.com/gagnonlg/explore-ml/commit/c05b01076c7eb99dae6a480a05ac14765ef08e4b
        with h5py.File(self.weights_path(), 'a') as f:
            if 'optimizer_weights' in f.keys():
                del f['optimizer_weights']

    @classmethod
    def load(cls, fitting=None):
        model = super(Base, cls).load(fitting)

        if hasattr(model, 'estimator'):
            # HACK to set estimator model, and model serializer
            model.estimator = model.estimator

            # Rely on build + load_weights rather than loading the named layers
            # w/ Keras for efficiency (and also because it causes a
            # deserialization issue) as of Keras 2.0.4:
            # https://github.com/fchollet/keras/issues/5442
            model.estimator.build()

            try:
                with timer('load weights %i' % model.fitting):
                    model.estimator.keras.load_weights(model.weights_path())
            except ValueError as ex:
                if model.estimator.multi_gpu_model and not lore.estimators.keras.available_gpus:
                    error = "Trying to load a multi_gpu_model when no GPUs are present is not supported"
                    logger.exception(error)
                    raise NotImplementedError(error)
                else:
                    raise

        else:
            model.build()
            with timer('load weights'):
                model.keras.load_weights(model.weights_path())

        return model

    def upload(self):
        super(Base, self).upload()
        lore.io.upload(self.weights_path(), self.remote_weights_path())

    @classmethod
    def download(cls, fitting=0):
        model = cls(None, None)
        model.fitting = fitting
        try:
            lore.io.download(model.remote_weights_path(), model.weights_path())
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                model.fitting = None
                lore.io.download(model.remote_weights_path(), model.weights_path())
        return super(Base, cls).download(fitting)
