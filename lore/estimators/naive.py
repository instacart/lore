from __future__ import absolute_import
import inspect
import logging
import lore
from lore.env import require
import lore.estimators
from lore.util import timed, before_after_callbacks

require(lore.dependencies.NUMPY)
import numpy


class Base(lore.estimators.Base):
    def __init__(self):
        super(Base, self).__init__()

    @before_after_callbacks
    @timed(logging.INFO)
    def fit(self, x, y, **kwargs):
        self.mean = numpy.mean(y)
        return {}

    @before_after_callbacks
    @timed(logging.INFO)
    def predict(self, dataframe):
        pass

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


class Naive(Base):
    def __init__(self):
        frame, filename, line_number, function_name, lines, index = inspect.stack()[1]
        super(Naive, self).__init__()


class Regression(Base):
    @before_after_callbacks
    @timed(logging.INFO)
    def predict(self, dataframe):
        return numpy.ones(dataframe.shape[0])*self.mean


class BinaryClassifier(Base):
    @before_after_callbacks
    @timed(logging.INFO)
    def predict(self, dataframe):
        if self.mean > 0.5:
            return numpy.ones(dataframe.shape[0])
        else:
            return numpy.zeros(dataframe.shape[0])

    @before_after_callbacks
    @timed(logging.INFO)
    def predict_proba(self, dataframe):
        return numpy.ones(dataframe.shape[0])*self.mean
