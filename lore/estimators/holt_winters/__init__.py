import logging

from lore.env import require
from lore.util import timed
import lore.estimators.holt_winters.holtwinters

require(lore.dependencies.SKLEARN)

from sklearn.base import BaseEstimator


class HoltWinters(BaseEstimator):

  def __init__(self, **kwargs):
    super(HoltWinters, self).__init__()
    self.periodicity = kwargs.get('periodicity')
    self.forecasts = kwargs.get('days_to_forecast')
    self.kwargs = kwargs
    self.params = None

  @timed(logging.INFO)
  def fit(self, x, y=None):
    results = holtwinters.additive(x, self.periodicity, self.forecasts,
      alpha=self.kwargs.get('alpha'),
      beta=self.kwargs.get('beta'),
      gamma=self.kwargs.get('gamma'))
    self.params = {'alpha': results[1], 'beta': results[2], 'gamma': results[3]}
    self.rmse = results[4]
    return {'alpha': results[1], 'beta': results[2], 'gamma': results[3], 'RMSE': self.rmse}

  @timed(logging.INFO)
  def predict(self, X):
    return holtwinters.additive(X, self.periodicity, self.forecasts,
      **self.params)[0]
