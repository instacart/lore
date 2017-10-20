from __future__ import absolute_import
import logging

from sklearn.base import BaseEstimator
import xgboost
from lore.util import timed


class XGBoost(BaseEstimator):
    def __init__(self, **xgboost_train_params):
        super(XGBoost, self).__init__()
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