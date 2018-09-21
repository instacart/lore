from __future__ import absolute_import

import json
import logging
import os.path
from os.path import join
import pickle
import re
import inspect
import warnings
import botocore

import lore.ansi
import lore.estimators
from lore.env import require
from lore.util import timer, timed, before_after_callbacks

require(
    lore.dependencies.TABULATE +
    lore.dependencies.SKLEARN +
    lore.dependencies.SHAP
)
import shap
from tabulate import tabulate
from sklearn.model_selection import RandomizedSearchCV


logger = logging.getLogger(__name__)

try:
    FileExistsError
except NameError:
    FileExistsError = OSError


class Base(object):
    def __init__(self, pipeline=None, estimator=None):
        self.name = self.__module__ + '.' + self.__class__.__name__
        self._estimator = None
        self.estimator = estimator
        self.fitting = None
        self.pipeline = pipeline
        self._shap_explainer = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state['_shap_explainer'] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        backward_compatible_defaults = {
            '_shap_explainer': None,
        }
        for key, default in backward_compatible_defaults.items():
            if key not in self.__dict__.keys():
                self.__dict__[key] = default

    @property
    def estimator(self):
        return self._estimator

    @estimator.setter
    def estimator(self, value):
        self._estimator = value

        # Keras models require access to the pipeline during build,
        # and fit for extended functionality
        if hasattr(self._estimator, 'model'):
            self._estimator.model = self

    @before_after_callbacks
    @timed(logging.INFO)
    def fit(self, test=True, score=True, **estimator_kwargs):
        self.fitting = self.__class__.last_fitting() + 1

        self.stats = self.estimator.fit(
            x=self.pipeline.encoded_training_data.x,
            y=self.pipeline.encoded_training_data.y,
            validation_x=self.pipeline.encoded_validation_data.x,
            validation_y=self.pipeline.encoded_validation_data.y,
            **estimator_kwargs
        )

        if test:
            self.stats['test'] = self.evaluate(self.pipeline.test_data)

        if score:
            self.stats['score'] = self.score(self.pipeline.test_data)

        self.save(stats=self.stats)
        logger.info(
            '\n\n' + tabulate([self.stats.keys(), self.stats.values()], tablefmt="grid", headers='firstrow') + '\n\n')

    @before_after_callbacks
    @timed(logging.INFO)
    def predict(self, dataframe):
        predictions = self.estimator.predict(self.pipeline.encode_x(dataframe))
        return self.pipeline.output_encoder.reverse_transform(predictions)

    @before_after_callbacks
    @timed(logging.INFO)
    def evaluate(self, dataframe):
        return self.estimator.evaluate(
            self.pipeline.encode_x(dataframe),
            self.pipeline.encode_y(dataframe)
        )

    @before_after_callbacks
    @timed(logging.INFO)
    def score(self, dataframe):
        return self.estimator.score(
            self.pipeline.encode_x(dataframe),
            self.pipeline.encode_y(dataframe)
        )

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
            validation_x=self.pipeline.encoded_validation_data.x,
            validation_y=self.pipeline.encoded_validation_data.y,
            **fit_params
        )
        self.estimator = result.best_estimator_

        return result

    @classmethod
    def local_path(cls):
        return join(lore.env.MODELS_DIR, cls.remote_path())

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
        if self.fitting:
            return join(self.remote_path(), str(self.fitting), 'model.pickle')
        else:
            return join(self.remote_path(), 'model.pickle')

    def save(self, stats=None):
        if self.fitting is None:
            raise ValueError("This model has not been fit yet. There is no point in saving.")

        if not os.path.exists(self.fitting_path()):
            try:
                os.makedirs(self.fitting_path())
            except FileExistsError as ex:
                pass  # race to create

        with timer('pickle model'):
            with open(self.model_path(), 'wb') as f:
                pickle.dump(self, f)

        with open(join(self.fitting_path(), 'params.json'), 'w') as f:
            params = {}
            for child in [self.estimator, self.pipeline]:
                param = child.__module__ + '.' + child.__class__.__name__
                params[param] = {}
                if hasattr(child, '__getstate__'):
                    state = child.__getstate__()
                else:
                    state = child.__dict__
                for key, value in state.items():
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

        with timer('unpickle model'):
            with open(model.model_path(), 'rb') as f:
                loaded = pickle.load(f)
                loaded.fitting = model.fitting
                return loaded

    def upload(self):
        self.save()
        lore.io.upload(self.model_path(), self.remote_model_path())

    @classmethod
    def download(cls, fitting=None):
        frame, filename, line_number, function_name, lines, index = inspect.stack()[1]
        warnings.showwarning('Please start using explicit fitting number when downloading the model ex "Keras.download(10)". Default Keras.download() will be deprecated in 0.7.0',
                             DeprecationWarning,
                             filename, line_number)
        model = cls(None, None)
        if not fitting:
            fitting = model.last_fitting()
        model.fitting = int(fitting)
        try:
            lore.io.download(model.remote_model_path(), model.model_path(), cache=True)
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                model.fitting = None
                lore.io.download(model.remote_model_path(), model.model_path(), cache=True)
        return cls.load(fitting)

    def shap_values(self, i, nsamples=1000):
        instance = self.pipeline.encoded_test_data.x.iloc[i, :]
        display = self.pipeline.decode(instance.to_frame().transpose()).iloc[0, :]
        return self.shap_explainer.shap_values(instance, nsamples=nsamples), display

    def shap_force_plot(self, i, nsamples=1000):
        return shap.force_plot(*self.shap_values(i, nsamples))

    @property
    def shap_explainer(self):
        if self._shap_explainer is None:
            with timer('fitting shap'):
                shap_data = self.pipeline.encoded_training_data.x.sample(100)

                def f(X):
                    return self.estimator.predict([X[:, i] for i in range(X.shape[1])]).flatten()

                self._shap_explainer = shap.KernelExplainer(f, shap_data)

        return self._shap_explainer
