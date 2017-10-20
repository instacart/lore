from lore.estimators import keras, xgboost, sklearn
import warnings, inspect

class Keras(keras.Keras):

  def __init__(self, **kwargs):
    warnings.showwarning('Please import Keras with "from lore.estimators.keras import Keras"', DeprecationWarning, __file__, inspect.currentframe().f_back.f_lineno)
    super(Keras, self).__init__(**kwargs)

class XGBoost(xgboost.XGBoost):

  def __init__(self, **kwargs):
    warnings.showwarning('Please import XGBoost with "from lore.estimators.xgboost import XGBoost"', DeprecationWarning, __file__, inspect.currentframe().f_back.f_lineno)
    super(XGBoost, self).__init__(**kwargs)

class SKLearn(sklearn.SKLearn):

  def __init__(self, estimator):
    warnings.showwarning('Please import SKLearn with "from lore.estimators.sklearn import SKLearn"', DeprecationWarning, __file__, inspect.currentframe().f_back.f_lineno)
    super(SKLearn, self).__init__(estimator)

