from __future__ import absolute_import
import inspect
import logging
import warnings

from sklearn.base import BaseEstimator
import xgboost
from lore.util import timed


class Base(BaseEstimator):
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
        # TODO
        return 0

    @timed(logging.INFO)
    def score(self, x, y):
        # TODO
        return 0


class XGBoost(Base):
    def __init__(self, **kwargs):
        frame, filename, line_number, function_name, lines, index = inspect.stack()[1]
        warnings.showwarning('Please import XGBoost with "from lore.estimators.xgboost import Base"',
                             DeprecationWarning,
                             filename, line_number)
        super(XGBoost, self).__init__(**kwargs)


class Regression(Base):
    pass


class BinaryClassifier(Base):
    pass


class MutliClassifier(Base):
    pass
