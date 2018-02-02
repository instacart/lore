import os
from os.path import join, dirname
import h5py

import lore
from lore.util import timer


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
        return join(self.remote_path(), 'weights.h5')

    @property
    def fitting(self):
        return self._fitting
    
    @fitting.setter
    def fitting(self, value):
        self._fitting = value
        if self._fitting is not None:
            if not os.path.exists(dirname(self.checkpoint_path())):
                os.makedirs(dirname(self.checkpoint_path()))
                
            if not os.path.exists(dirname(self.tensorboard_path())):
                os.makedirs(dirname(self.tensorboard_path()))
    
    def save(self, stats=None):
        super(Base, self).save(stats)
        
        with timer('save weights:'):
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
            
            with timer('load weights %i:' % model.fitting):
                model.estimator.keras.load_weights(model.weights_path())
        else:
            model.build()
            with timer('load weights:'):
                model.keras.load_weights(model.weights_path())
        
        return model
    
    def upload(self):
        super(Base, self).upload()
        lore.io.upload(self.weights_path(), self.remote_weights_path())
    
    @classmethod
    def download(cls, fitting=0):
        model = cls(None, None)
        model.fitting = fitting
        lore.io.download(model.weights_path(), model.remote_weights_path())
        return super(Base, cls).download(fitting)
