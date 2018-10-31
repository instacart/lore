# -*- coding: utf-8 -*-
"""
Naive Estimator
****************
A naive estimator is a useful baseline against which to benchmark more complex models.
A naive estimator will return the mean of the outcome for regression models and
the plurality class for classification models. Note that currently only binary classification
is implemented. For binary classifiers, the majority class will be returned
"""
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
    """Base class for the Naive estimator. Implements functionality common to all Naive models"""
    def __init__(self):
        super(Base, self).__init__()

    @before_after_callbacks
    @timed(logging.INFO)
    def fit(self, x, y, **kwargs):
        """
        Fit a naive model
        :param x: Predictors to use for fitting the data (this will not be used in naive models)
        :param y: Outcome
        """
        self.mean = numpy.mean(y)
        return {}

    @before_after_callbacks
    @timed(logging.INFO)
    def predict(self, dataframe):
        """
        .. _naive_base_predict
        Predict using the model
        :param dataframe: Dataframe against which to make predictions
        """
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
        """See :ref:`Base Estimator for Naive _naive_base_predict`"""
        return numpy.full(dataframe.shape[0], self.mean)


class BinaryClassifier(Base):
    @before_after_callbacks
    @timed(logging.INFO)
    def predict(self, dataframe):
        """See :ref:`Base Estimator for Naive _naive_base_predict`"""
        if self.mean > 0.5:
            return numpy.ones(dataframe.shape[0])
        else:
            return numpy.zeros(dataframe.shape[0])

    @before_after_callbacks
    @timed(logging.INFO)
    def predict_proba(self, dataframe):
        """Predict probabilities using the model
        :param dataframe: Dataframe against which to make predictions
        """
        ret = numpy.ones((dataframe.shape[0], 2))
        ret[:, 0] = (1 - self.mean)
        ret[:, 1] = self.mean
        return ret
