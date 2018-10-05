from __future__ import absolute_import

import datetime
import string
import json
import logging
import os.path
from os.path import join
import pickle
import random
import inspect
import warnings
import botocore

import lore.ansi
import lore.estimators
import lore.metadata
from lore.env import require
from lore.util import timer, timed, before_after_callbacks, \
    convert_df_columns_to_json, sql_alchemy_object_as_dict, \
    memoized_property
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy import func

require(
    lore.dependencies.TABULATE +
    lore.dependencies.SKLEARN +
    lore.dependencies.SHAP
)
import shap
from tabulate import tabulate
from sklearn.model_selection import RandomizedSearchCV

Session = scoped_session(sessionmaker(bind=lore.io.metadata._engine))


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
        self.metadata = None
        self.fit_complete = False

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

        self.fit_complete = True

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
        df['fitting_name'] = self.fitting_name
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
        self.stats = {}
        self.estimator_kwargs = None
        self.fit_complete = True
        return result

    @classmethod
    def local_path(cls):
        return join(lore.env.MODELS_DIR, cls.remote_path())

    @classmethod
    def remote_path(cls):
        return join(cls.__module__, cls.__name__)

    @classmethod
    def last_fitting(cls):
        session = Session()
        fitting_name = (session.query(func.max(lore.metadata.Fitting.fitting_name))
                        .filter_by(model='lore_test.models.Boost')
                        .scalar())
        session.close()
        return fitting_name

    @memoized_property
    def fitting_name(self):
        try:
            return self._fitting_name
        except AttributeError:
            current_time = datetime.datetime.utcnow().strftime("%Y%m%d%H%m")
            model_suffix = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))
            fitting_name = current_time + '_' + model_suffix
            return fitting_name

    @fitting_name.setter
    def fitting_name(self, name=None):
        self._fitting_name = name

    @memoized_property
    def fitting_path(self):
        return join(self.local_path(), str(self.fitting_name))

    @memoized_property
    def model_path(self):
        return ''.join([self.fitting_path, '/model.pickle'])

    @memoized_property
    def remote_model_path(self):
        return join(self.remote_path(), self.fitting_name, 'model.pickle')

    def save(self, custom_data=None, upload=False):
        if self.fit_complete is False:
            raise ValueError("This model has not been fit yet. There is no point in saving.")

        if self.metadata:
            raise ValueError("This model has already been saved")

        commit = lore.metadata.Commit.from_git()
        commit.get_or_create(sha=commit.sha)
        fitting = lore.metadata.Fitting.create(
            model='.'.join([self.__class__.__module__, self.__class__.__name__]),
            commit=None,
            name=self.fitting_name,
            custom_data=custom_data,
            snapshot=lore.metadata.Snapshot(pipeline='.'.join([self.pipeline.__class__.__module__,
                                                               self.pipeline.__class__.__name__]),
                                            commit=None,
                                            head=str(self.pipeline.training_data.head(2)),
                                            tail=str(self.pipeline.training_data.tail(2))
                                            )
        )

        try:
            fitting.iterations = self.stats['epochs']
        except KeyError:
            fitting.iterations = None
        fitting.completed_at = datetime.datetime.now()
        fitting.args = self.estimator_kwargs
        fitting.stats = self.stats
        try:
            fitting.train = self.stats['train']
            fitting.validate = self.stats['validate']
            fitting.test = self.stats['test']
            fitting.score = self.stats['score']
        except KeyError:
            fitting.test = None
            fitting.score = None

        self.metadata = sql_alchemy_object_as_dict(fitting)

        if not os.path.exists(self.fitting_path):
            try:
                os.makedirs(self.fitting_path)
            except FileExistsError as ex:
                pass  # race to create

        with timer('pickle model'):
            with open(self.model_path, 'wb') as f:
                pickle.dump(self, f)

        with open(join(self.fitting_path, 'params.json'), 'w') as f:
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
            with open(join(self.fitting_path, 'stats.json'), 'w') as f:
                json.dump(self.stats, f, indent=2, sort_keys=True)

        if upload:
            url = self.upload()
            fitting.url = url
            fitting.uploaded_at = datetime.datetime.utcnow()
        fitting.save()

    @classmethod
    def load(cls, fitting_name=None):
        model = cls()
        if fitting_name is None:
            fitting_name = model.last_fitting()
        model.fitting_name = fitting_name

        with timer('unpickle model'):
            with open(model.model_path, 'rb') as f:
                loaded = pickle.load(f)
                return loaded

    def upload(self):
        if self.metadata is None:
            raise ValueError("Please save model first, before uplaoding")
        lore.io.upload(self.model_path, self.remote_model_path)
        return self.remote_model_path

    @classmethod
    def download(cls, fitting_name=None):
        frame, filename, line_number, function_name, lines, index = inspect.stack()[1]
        warnings.showwarning('Please start using explicit fitting name when downloading the model ex "Keras.download(10)". Default Keras.download() will be deprecated in 0.7.0',
                             DeprecationWarning,
                             filename, line_number)
        model = cls()
        if fitting_name is None:
            fitting_name = model.last_fitting()
            # If still none, then either no model was fit or user is trying to download a lore model pre-0.7
            if fitting_name is None:
                raise ValueError('No fittings found for this model. If you are looking for fittings created with a ' +
                                 'prior version of lore, please explicitly specify the fitting number')
        model.fitting_name = fitting_name
        try:
            lore.io.download(model.remote_model_path, model.model_path, cache=True)
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                lore.io.download(model.remote_model_path, model.model_path, cache=True)
        return cls.load(fitting_name)

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
