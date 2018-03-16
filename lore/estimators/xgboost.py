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
        return 1 / self.evaluate(x, y)


class XGBoost(BaseEstimator):
    def __init__(self, **kwargs):
        frame, filename, line_number, function_name, lines, index = inspect.stack()[1]
        warnings.showwarning('Please import XGBoost with "from lore.estimators.xgboost import Base"',
                             DeprecationWarning,
                             filename, line_number)
        super(XGBoost, self).__init__(**kwargs)


class Regression(Base, xgboost.XGBRegressor):
    def __init__(self, **kwargs):
        super(Regression, self).__init__(kwargs)


class BinaryClassifier(Base, xgboost.XGBClassifier):
    def __init__(self, **kwargs):
        super(BinaryClassifier, self).__init__(kwargs)


MutliClassifier = BinaryClassifier
