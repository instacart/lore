"""
lore.estimators
"""
from __future__ import absolute_import

from abc import ABCMeta, abstractmethod
import logging
from sklearn.base import BaseEstimator

from lore.util import timed, before_after_callbacks


class Base(BaseEstimator):
    __metaclass__ = ABCMeta

    @before_after_callbacks
    @timed(logging.INFO)
    @abstractmethod
    def fit(self):
        pass

    @before_after_callbacks
    @timed(logging.INFO)
    @abstractmethod
    def predict(self):
        pass

    @before_after_callbacks
    @timed(logging.INFO)
    @abstractmethod
    def evaluate(self):
        pass

    @before_after_callbacks
    @timed(logging.INFO)
    @abstractmethod
    def score(self):
        pass
