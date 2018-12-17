from __future__ import absolute_import

import datetime
import json
import logging
import os.path
from os.path import join
import pickle
import inspect
import warnings
import botocore

import lore.ansi
import lore.estimators
import lore.metadata
from lore.env import require
from lore.util import timer, timed, before_after_callbacks, convert_df_columns_to_json

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
        self.pipeline = pipeline
        self._shap_explainer = None
        self.fitting = None

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

    def __repr__(self):
        properties = ['%s=%s' % (key, value) for key, value in self.__dict__.items() if key[0] != '_']
        return '<%s(%s)>' % (self.__class__.__name__, ', '.join(properties))

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
    def fit(self, test=True, score=True, custom_data=None, save=True, **estimator_kwargs):
        self.fitting = lore.metadata.Fitting.create(
            model=self.name,
            custom_data=custom_data,
            snapshot=lore.metadata.Snapshot(pipeline=self.pipeline.name,
                                            head=str(self.pipeline.training_data.head(2)),
                                            tail=str(self.pipeline.training_data.tail(2))
                                            )
        )

        self.stats = self.estimator.fit(
            x=self.pipeline.encoded_training_data.x,
            y=self.pipeline.encoded_training_data.y,
            validation_x=self.pipeline.encoded_validation_data.x,
            validation_y=self.pipeline.encoded_validation_data.y,
            **estimator_kwargs
        )

        self.estimator_kwargs = estimator_kwargs
        if test:
            self.stats['test'] = self.evaluate(self.pipeline.test_data)

        if score:
            self.stats['score'] = self.score(self.pipeline.test_data)

        self.complete_fitting()

        if save:
            self.save()

        logger.info(
            '\n\n' + tabulate([self.stats.keys(), self.stats.values()], tablefmt="grid", headers='firstrow') + '\n\n')

    def create_predictions_for_logging(self, dataframe, predictions, key_cols, custom_data=None):
        require(lore.dependencies.PANDAS)
        import pandas

        keys = convert_df_columns_to_json(dataframe, key_cols)
        features = convert_df_columns_to_json(dataframe, dataframe.columns)
        df = pandas.DataFrame({'key': keys,
                               'features': features})
        df['value'] = predictions
        df['custom_data'] = custom_data
        df['created_at'] = datetime.datetime.utcnow()
        df['fitting_id'] = self.fitting.id
        return df

    def log_predictions(self, dataframe, predictions, key_cols, custom_data=None):
        import lore.io
        predictions = self.create_predictions_for_logging(dataframe, predictions, key_cols, custom_data)
        lore.io.metadata.insert("predictions", predictions)

    @before_after_callbacks
    @timed(logging.INFO)
    def predict(self, dataframe, log_predictions=False, key_cols=None, custom_data=None):
        if log_predictions is True and key_cols is None:
            raise ValueError("Key columns cannot be null when logging predictions")
        predictions = self.estimator.predict(self.pipeline.encode_x(dataframe))
        if log_predictions:
            self.log_predictions(dataframe, predictions, key_cols, custom_data)
        return self.pipeline.output_encoder.reverse_transform(predictions)

    @before_after_callbacks
    @timed(logging.INFO)
    def predict_proba(self, dataframe):
        try:
            probs = self.estimator.predict_proba(self.pipeline.encode_x(dataframe))
            return probs
        except AttributeError:
            raise AttributeError('Estimator does not define predict_proba')

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
            return_train_score=True,
            test=True,
            score=True,
            save=True,
            custom_data=None
    ):
        """Random search hyper params
        """
        params = locals()
        params.pop('self')

        self.fitting = lore.metadata.Fitting.create(
            model=self.name,
            custom_data=custom_data,
            snapshot=lore.metadata.Snapshot(pipeline=self.pipeline.name,
                                            head=str(self.pipeline.training_data.head(2)),
                                            tail=str(self.pipeline.training_data.tail(2))
                                            )
        )

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
        self.stats = {}
        self.estimator_kwargs = self.estimator.__getstate__()

        if test:
            self.stats['test'] = self.evaluate(self.pipeline.test_data)

        if score:
            self.stats['score'] = self.score(self.pipeline.test_data)

        self.complete_fitting()

        if save:
            self.save()
        return result

    def complete_fitting(self):
        self.fitting.completed_at = datetime.datetime.now()
        self.fitting.args = self.estimator_kwargs
        self.fitting.stats = self.stats
        self.fitting.iterations = self.stats.get('epochs', None)
        self.fitting.train = self.stats.get('train', None)
        self.fitting.validate = self.stats.get('validate', None)
        self.fitting.test = self.stats.get('test', None)
        self.fitting.score = self.stats.get('score', None)

        self.fitting.save()

        logger.info(
            '\n\n' + tabulate([self.stats.keys(), self.stats.values()], tablefmt="grid", headers='firstrow') + '\n\n')


    @classmethod
    def local_path(cls):
        return join(lore.env.MODELS_DIR, cls.remote_path())

    @classmethod
    def remote_path(cls):
        return join(cls.__module__, cls.__name__)

    @classmethod
    def last_fitting(cls):
        return lore.metadata.Fitting.last(model=cls.__module__ + '.' + cls.__name__)

    def fitting_path(self):
        return join(self.local_path(), str(self.fitting.id))

    def model_path(self):
        return ''.join([self.fitting_path(), '/model.pickle'])

    def remote_model_path(self):
        return join(self.remote_path(), str(self.fitting.id), 'model.pickle')

    def save(self, upload=False):
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

        if self.stats:
            with open(join(self.fitting_path(), 'stats.json'), 'w') as f:
                json.dump(self.stats, f, indent=2, sort_keys=True)

        if upload:
            url = self.upload()
            self.fitting.url = url
            self.fitting.uploaded_at = datetime.datetime.utcnow()
            self.fitting.save()

    @classmethod
    def load(cls, fitting_id=None):
        model = cls()
        if fitting_id is None:
            model.fitting = model.last_fitting()
        else:
            model.fitting = lore.metadata.Fitting.get(fitting_id)

        if model.fitting is None:
            logger.warning(
                "Attempting to download a model from outside of the metadata store is deprecated and will be removed in 0.8.0")
            model.fitting = lore.metadata.Fitting(id=fitting_id or 0)

        with timer('unpickle model'):
            with open(model.model_path(), 'rb') as f:
                loaded = pickle.load(f)
                loaded.fitting = model.fitting
                return loaded

    def upload(self):
        lore.io.upload(self.model_path(), self.remote_model_path())
        return self.remote_model_path()

    @classmethod
    def download(cls, fitting_id=None):
        model = cls()
        if fitting_id is None:
            model.fitting = model.last_fitting()
        else:
            model.fitting = lore.metadata.Fitting.get(fitting_id)

        if model.fitting is None:
            logger.warning("Attempting to download a model from outside of the metadata store is deprecated and will be removed in 0.8.0")
            model.fitting = lore.metadata.Fitting(id=0)

        try:
            lore.io.download(model.remote_model_path(), model.model_path(), cache=True)
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                model.fitting.id = None
                logger.warning("Attempting to download a model without a fitting id is deprecated and will be removed in 0.8.0")
                lore.io.download(model.remote_model_path(), model.model_path(), cache=True)
        return cls.load(model.fitting.id)

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
