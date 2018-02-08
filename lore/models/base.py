from __future__ import absolute_import

import inspect
import json
import logging
import os.path
from os.path import join
import pickle
import re

from tabulate import tabulate
import lore.estimators
import lore.serializers
from lore.util import timer

from sklearn.model_selection import RandomizedSearchCV

logger = logging.getLogger(__name__)


class Base(object):
    def __init__(self, pipeline=None, estimator=None):
        self.name = self.__module__ + '.' + self.__class__.__name__
        self._estimator = None
        self.estimator = estimator
        self.fitting = None
        self.pipeline = pipeline
    
    def __getstate__(self):
        return dict(self.__dict__)
    
    @property
    def estimator(self):
        return self._estimator
    
    @estimator.setter
    def estimator(self, value):
        self._estimator = value
        
        # Keras models require access to the pipeline during build,
        # and the serializer during fit for extended functionality
        if hasattr(self._estimator, 'model'):
            self._estimator.model = self

    def fit(self, **estimator_kwargs):
        self.fitting = self.__class__.last_fitting() + 1

        self.stats = self.estimator.fit(
            x=self.pipeline.encoded_training_data.x,
            y=self.pipeline.encoded_training_data.y,
            **estimator_kwargs
        )
        self.save(stats=self.stats)
        logger.info(
            '\n\n' + tabulate([self.stats.keys(), self.stats.values()], tablefmt="grid", headers='firstrow') + '\n\n')
    
    def predict(self, dataframe):
        return self.estimator.predict(self.pipeline.encode_x(dataframe))
    
    def hyper_parameter_search(
            self,
            param_distributions,
            n_iter=10,
            scoring=None,
            fit_params={},
            n_jobs=1,
            iid=True,
            refit=True,
            cv=None,
            verbose=0,
            pre_dispatch='2*njobs',
            random_state=None,
            error_score='raise',
            return_train_score=True
    ):
        """Random search hyper params
        """
        if scoring is None:
            scoring = None
        result = RandomizedSearchCV(
            self.estimator,
            param_distributions,
            n_iter=n_iter,
            scoring=scoring,
            n_jobs=n_jobs,
            iid=iid,
            refit=refit,
            cv=cv,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            random_state=random_state,
            error_score=error_score,
            return_train_score=return_train_score
        ).fit(
            self.pipeline.encoded_training_data.x,
            y=self.pipeline.encoded_training_data.y,
            **fit_params
        )
        self.estimator = result.best_estimator_
        
        return result
    
    @classmethod
    def local_path(cls):
        return join(lore.env.models_dir, cls.remote_path())
    
    @classmethod
    def remote_path(cls):
        return join(cls.__module__, cls.__name__)
    
    @classmethod
    def last_fitting(cls):
        if not os.path.exists(cls.local_path()):
            return 0
        
        fittings = [int(d) for d in os.listdir(cls.local_path()) if re.match(r'^\d+$', d)]
        if not fittings:
            return 0
        
        return sorted(fittings)[-1]
    
    def fitting_path(self):
        if self.fitting is None:
            self.fitting = self.last_fitting()
        
        return join(self.local_path(), str(self.fitting))
    
    def model_path(self):
        return join(self.fitting_path(), 'model.pickle')
    
    def remote_model_path(self):
        return join(self.remote_path(), 'model.pickle')
    
    def save(self, stats=None):
        if self.fitting is None:
            raise ValueError("This model has not been fit yet. There is no point in saving.")

        if not os.path.exists(self.fitting_path()):
            os.makedirs(self.fitting_path())

        with timer('pickle model:'):
            with open(self.model_path(), 'wb') as f:
                pickle.dump(self, f)
        
        with open(join(self.fitting_path(), 'params.json'), 'w') as f:
            params = {}
            for child in [self.estimator, self.pipeline]:
                param = child.__module__ + '.' + child.__class__.__name__
                params[param] = {}
                for key, value in child.__getstate__().items():
                    if not key.startswith('_'):
                        params[param][key] = value.__repr__()
            json.dump(params, f, indent=2, sort_keys=True)
        
        if stats:
            with open(join(self.fitting_path(), 'stats.json'), 'w') as f:
                json.dump(stats, f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, fitting=None):
        model = cls()
        if fitting is None:
            model.fitting = model.last_fitting()
        else:
            model.fitting = int(fitting)
        
        with timer('unpickle model:'):
            with open(model.model_path(), 'rb') as f:
                loaded = pickle.load(f)
                loaded.fitting = model.fitting
                return loaded
    
    def upload(self):
        self.fitting = 0
        self.save()
        lore.io.upload(self.model_path(), self.remote_model_path())
     
    @classmethod
    def download(cls, fitting=0):
        model = cls(None, None)
        model.fitting = int(fitting)
        lore.io.download(model.model_path(), model.remote_model_path())
        return cls.load(fitting)
