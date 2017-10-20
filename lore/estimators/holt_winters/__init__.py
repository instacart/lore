import logging

from sklearn.base import BaseEstimator
import holtwinters
from lore.util import timed


class HoltWinters(BaseEstimator):

  def __init__(self, **kwargs):
    super(HoltWinters, self).__init__()
    self.periodicity = kwargs.get('periodicity')
    self.forecasts = kwargs.get('days_to_forecast')
    self.kwargs = kwargs

  @timed(logging.INFO)
  def predict(self, X):
    return holtwinters.additive(X, self.periodicity, self.forecasts, 
      alpha=self.kwargs.get('alpha'),
      beta=self.kwargs.get('beta'),
      gamma=self.kwargs.get('gamma'))
    