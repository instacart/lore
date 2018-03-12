import pickle
import os
from os.path import join
import h5py
import json
import re
import warnings
import inspect

import lore
from lore import io
from lore.util import timer


class Base(object):
    def __init__(self, klass=None, model=None):
        frame, filename, line_number, function_name, lines, index = inspect.stack()[1]
        warnings.showwarning('Serializers are now directly implemented in models. Please use the model class directly instead"', DeprecationWarning,
                             filename, line_number)
        self.local_path = None
        self.remote_path = None
        self.fitting_path = None
        self.model_path = None
        self.remote_model_path = None
        self.model = model
        self._fitting = None

        if model is not None:
            self.local_path = join(lore.env.models_dir, model.__module__, model.__class__.__name__)
            self.remote_path = join(model.__module__, model.__class__.__name__)
        elif klass is not None:
            self.local_path = join(lore.env.models_dir, klass.__module__, klass.__name__)
            self.remote_path = join(klass.__module__, klass.__name__)
        else:
            raise ValueError('You must pass name or model')
        self.fitting = self.last_fitting()

    def last_fitting(self):
        if not os.path.exists(self.local_path):
            return 1
            
        fittings = [int(d) for d in os.listdir(self.local_path) if re.match(r'^\d+$', d)]
        if not fittings:
            return 1
        
        return sorted(fittings)[-1]

    @property
    def fitting(self):
        return self._fitting
    
    @fitting.setter
    def fitting(self, value):
        self._fitting = value
        self.fitting_path = join(self.local_path, str(self._fitting))
        model_file = 'model.pickle'
        self.model_path = join(self.fitting_path, model_file)
        self.remote_model_path = join(self.remote_path, model_file)
        if not os.path.exists(self.fitting_path):
            try:
                os.makedirs(self.fitting_path)
            except os.FileExistsError as ex:
                pass  # race to create

    def save(self, stats=None):
        with timer('pickle model'):
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)

        with open(join(self.fitting_path, 'params.json'), 'w') as f:
            if hasattr(self.model.estimator, 'params'):
                params = self.model.estimator.params
            else:
                params = {}
                for key, value in self.model.estimator.__getstate__().items():
                    if not key.startswith('_'):
                        params[key] = value.__repr__()
            json.dump(params, f, indent=2, sort_keys=True)

        if stats:
            with open(join(self.fitting_path, 'stats.json'), 'w') as f:
                json.dump(stats, f, indent=2, sort_keys=True)
        
    def load(self, fitting=None):
        if fitting:
            self.fitting = fitting
        
        with timer('unpickle model'):
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)

    def upload(self):
        self.save()
        io.upload(self.model_path, self.remote_model_path)

    def download(self):
        self.fitting = 0
        io.download(self.remote_model_path, self.model_path)
        return self.load()


class Keras(Base):
    def __init__(self, klass=None, model=None):
        self.weights_path = None
        self.remote_weights_path = None
        self.checkpoint_path = None
        self.tensorboard_path = None
        self.timeline_path = None
        super(Keras, self).__init__(klass=klass, model=model)

    @Base.fitting.setter
    def fitting(self, value):
        Base.fitting.fset(self, value)  # this is a "pythonic" super() call :(
        weights_file = 'weights.h5'
        self.weights_path = join(self.fitting_path, weights_file)
        self.remote_weights_path = join(self.remote_path, weights_file)
        self.checkpoint_path = join(self.fitting_path, 'checkpoints/{epoch}.h5')
        if not os.path.exists(os.path.dirname(self.checkpoint_path)):
            try:
                os.makedirs(os.path.dirname(self.checkpoint_path))
            except os.FileExistsError as ex:
                pass  # race to create
        self.tensorboard_path = join(self.fitting_path, 'tensorboard')
        if not os.path.exists(os.path.dirname(self.tensorboard_path)):
            try:
                os.makedirs(os.path.dirname(self.tensorboard_path))
            except os.FileExistsError as ex:
                pass  # race to create
        self.timeline_path = join(self.fitting_path, 'timeline.json')
        if not os.path.exists(os.path.dirname(self.timeline_path)):
            try:
                os.makedirs(os.path.dirname(self.timeline_path))
            except os.FileExistsError as ex:
                pass  # race to create

    def save(self, stats=None):
        super(Keras, self).save(stats)
        
        with timer('save weights'):
            # Only save weights, because saving named layers that have shared
            # weights causes an error on reload
            self.model.estimator.keras.save_weights(self.weights_path)

        # Patch for keras 2 models saved with optimizer weights:
        # https://github.com/gagnonlg/explore-ml/commit/c05b01076c7eb99dae6a480a05ac14765ef08e4b
        with h5py.File(self.weights_path, 'a') as f:
            if 'optimizer_weights' in f.keys():
                del f['optimizer_weights']
        
    def load(self, fitting=None):
        super(Keras, self).load(fitting)

        if hasattr(self.model, 'estimator'):
            # HACK to set estimator model, and model serializer
            self.model.estimator = self.model.estimator
    
            # Rely on build + load_weights rather than loading the named layers
            # w/ Keras for efficiency (and also because it causes a
            # deserialization issue) as of Keras 2.0.4:
            # https://github.com/fchollet/keras/issues/5442
            self.model.estimator.build()
            
            with timer('load weights'):
                self.model.estimator.keras.load_weights(self.weights_path)
        else:
            self.model.build()
            with timer('load weights'):
                self.model.keras.load_weights(self.weights_path)

        return self.model

    def upload(self):
        super(Keras, self).upload()
        io.upload(self.weights_path, self.remote_weights_path)

    def download(self):
        self.fitting = 0
        io.download(self.remote_weights_path, self.weights_path)
        return super(Keras, self).download()
