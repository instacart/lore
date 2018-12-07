# -*- coding: utf-8 -*-
"""
scikit-learn Estimator
****************
This estimator allows you to use any scikit-learn estimator of your choice.
Note that the underlying estimator can always be accessed as ``Base(estimator).sklearn``
"""
from __future__ import absolute_import
import inspect
import logging
import warnings

import lore
import lore.estimators
from lore.env import require
from lore.util import timed, before_after_callbacks

require(lore.dependencies.SKLEARN)


class Base(lore.estimators.Base):
    def __init__(self, estimator):
        super(Base, self).__init__()
        self.sklearn = estimator

    @before_after_callbacks
    @timed(logging.INFO)
    def fit(self, x, y, validation_x=None, validation_y=None, **sklearn_kwargs):
        self.sklearn.fit(x, y=y, **sklearn_kwargs)

        # TODO interesting SKLearn fitting stats
        return {}

    @before_after_callbacks
    @timed(logging.INFO)
    def predict(self, dataframe):
        return self.sklearn.predict(dataframe)

    @before_after_callbacks
    @timed(logging.INFO)
    def evaluate(self, x, y):
        # TODO
        return 0

    @before_after_callbacks
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
    @before_after_callbacks
    @timed(logging.INFO)
    def predict_proba(self, dataframe):
        """Predict probabilities using the model
        :param dataframe: Dataframe against which to make predictions
        """
        return self.sklearn.predict_proba(dataframe)


class MutliClassifier(Base):
    pass
