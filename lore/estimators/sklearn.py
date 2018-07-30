from __future__ import absolute_import
import inspect
import logging
import warnings

import lore
from lore.env import require
from lore.util import timed

require(lore.dependencies.SKLEARN)

from sklearn.base import BaseEstimator


class Base(BaseEstimator):
    def __init__(self, estimator):
        super(Base, self).__init__()
        self.sklearn = estimator

    @timed(logging.INFO)
    def fit(self, x, y, validation_x=None, validation_y=None, **sklearn_kwargs):
        self.sklearn.fit(x, y=y, **sklearn_kwargs)

        # TODO interesting SKLearn fitting stats
        return {}

    @timed(logging.INFO)
    def predict(self, dataframe):
        return self.sklearn.predict(dataframe)

    @timed(logging.INFO)
    def evaluate(self, x, y):
        # TODO
        return 0

    @timed(logging.INFO)
    def score(self, x, y):
        # TODO
        return 0


class SKLearn(Base):
    def __init__(self, estimator):
        frame, filename, line_number, function_name, lines, index = inspect.stack()[1]
        warnings.showwarning('Please import SKLearn with "from lore.estimators.sklearn import Base"',
                             DeprecationWarning,
                             filename, line_number)
        super(SKLearn, self).__init__(estimator)


class Regression(Base):
    pass


class BinaryClassifier(Base):
    pass


class MutliClassifier(Base):
    pass
