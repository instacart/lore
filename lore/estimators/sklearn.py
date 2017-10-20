from __future__ import absolute_import
import logging
from sklearn.base import BaseEstimator
from lore.util import timed


class SKLearn(BaseEstimator):
    def __init__(self, estimator):
        super(SKLearn, self).__init__()
        self.sklearn = estimator

    @timed(logging.INFO)
    def fit(self, x, y,  **sklearn_kwargs):
        self.sklearn.fit(x, y=y, **sklearn_kwargs)

        # TODO interesting SKLearn fitting stats
        return {}

    @timed(logging.INFO)
    def predict(self, dataframe):
        return self.sklearn.predict(dataframe)
