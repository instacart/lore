from __future__ import absolute_import
import inspect
import logging
import warnings

from sklearn.base import BaseEstimator
import xgboost
from lore.util import timed


class Base(object):
    def __init__(self, **xgboost_train_params):
        super(Base, self).__init__()
        self.bst = None
        self.params = xgboost_train_params

    @timed(logging.INFO)
    def fit(self, x, y, **xgboost_kwargs):
        self.bst = xgboost.train(
            self.params,
            xgboost.DMatrix(x, label=y),
            **xgboost_kwargs
        )

        # TODO interesting XGBoost fitting stats
        return {}

    @timed(logging.INFO)
    def predict(self, dataframe):
        return self.bst.predict(xgboost.DMatrix(dataframe))

    @timed(logging.INFO)
    def evaluate(self, x, y):
        return self.bst.eval(xgboost.DMatrix(x, label=y))

    @timed(logging.INFO)
    def score(self, x, y):
        return 0
        # return 1 / self.evaluate(x, y)


class XGBoost(BaseEstimator):
    def __init__(self, **kwargs):
        frame, filename, line_number, function_name, lines, index = inspect.stack()[1]
        warnings.showwarning('Please import XGBoost with "from lore.estimators.xgboost import Base"',
                             DeprecationWarning,
                             filename, line_number)
        super(XGBoost, self).__init__(**kwargs)


class Regression(xgboost.XGBRegressor):
    def __init__(
        self,
        max_depth=3,
        learning_rate=0.1,
        n_estimators=100,
        silent=True,
        objective='reg:linear',
        # booster='gbtree',
        # n_jobs=1,
        nthread=8,
        gamma=0,
        min_child_weight=1,
        max_delta_step=0,
        subsample=1,
        colsample_bytree=1,
        colsample_bylevel=1,
        reg_alpha=0,
        reg_lambda=1,
        scale_pos_weight=1,
        base_score=0.5,
        # random_state=0,
        seed=0,
        missing=None,
        **kwargs
    ):
        kwargs = locals()
        kwargs.pop('self')
        kwargs.pop('__class__', None)
        kwargs = dict(kwargs, **(kwargs.pop('kwargs', {})))
        super(Regression, self).__init__(**kwargs)

    @timed(logging.INFO)
    def fit(self, x, y, **xgboost_kwargs):
        super(Regression, self).fit(x, y, **xgboost_kwargs)
        # super
        # self.bst = xgboost.train(
        #     self.params,
        #     xgboost.DMatrix(x, label=y),
        #     **xgboost_kwargs
        # )

        # TODO interesting XGBoost fitting stats
        return {}

    @timed(logging.INFO)
    def predict(self, dataframe):
        return super(Regression, self).predict(dataframe)

    @timed(logging.INFO)
    def evaluate(self, x, y):
        return 0
        # return super(BinaryClassifier, self).eval(xgboost.DMatrix(x, label=y))

    @timed(logging.INFO)
    def score(self, x, y):
        return 0
        # return 1 / self.evaluate(x, y)


class BinaryClassifier(xgboost.XGBClassifier):
    def __init__(
        self,
        max_depth=3,
        learning_rate=0.1,
        n_estimators=100,
        silent=True,
        objective='binary:logistic',
        # booster='gbtree',
        # n_jobs=1,
        nthread=4,
        gamma=0,
        min_child_weight=1,
        max_delta_step=0,
        subsample=1,
        colsample_bytree=1,
        colsample_bylevel=1,
        reg_alpha=0,
        reg_lambda=1,
        scale_pos_weight=1,
        base_score=0.5,
        # random_state=0,
        seed=0,
        missing=None,
        **kwargs
    ):
        kwargs = locals()
        kwargs.pop('self')
        kwargs.pop('__class__', None)
        kwargs = dict(kwargs, **(kwargs.pop('kwargs', {})))
        super(BinaryClassifier, self).__init__(**kwargs)

    @timed(logging.INFO)
    def fit(self, x, y, **xgboost_kwargs):
        super(BinaryClassifier, self).fit(x, y, **xgboost_kwargs)
        # super
        # self.bst = xgboost.train(
        #     self.params,
        #     xgboost.DMatrix(x, label=y),
        #     **xgboost_kwargs
        # )

        # TODO interesting XGBoost fitting stats
        return {}

    @timed(logging.INFO)
    def predict(self, dataframe):
        return super(BinaryClassifier, self).predict(dataframe)

    @timed(logging.INFO)
    def evaluate(self, x, y):
        return 0
        # return super(BinaryClassifier, self).eval(xgboost.DMatrix(x, label=y))

    @timed(logging.INFO)
    def score(self, x, y):
        return 0
        # return 1 / self.evaluate(x, y)


MutliClassifier = BinaryClassifier
