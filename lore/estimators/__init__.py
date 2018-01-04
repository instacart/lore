from lore.estimators import keras, xgboost, sklearn
import warnings, inspect

class Keras(keras.Keras):

    def __init__(self, **kwargs):
        frame, filename, line_number, function_name, lines, index = inspect.stack()[1]
        warnings.showwarning('Please import Keras with "from lore.estimators.keras import Keras"', DeprecationWarning,
                             filename, line_number)
        super(Keras, self).__init__(**kwargs)

class XGBoost(xgboost.XGBoost):

    def __init__(self, **kwargs):
        frame, filename, line_number, function_name, lines, index = inspect.stack()[1]
        warnings.showwarning('Please import XGBoost with "from lore.estimators.xgboost import XGBoost"', DeprecationWarning,
                             filename, line_number)
        super(XGBoost, self).__init__(**kwargs)
 
class SKLearn(sklearn.SKLearn):

    def __init__(self, estimator):
        frame, filename, line_number, function_name, lines, index = inspect.stack()[1]
        warnings.showwarning('Please import SKLearn with "from lore.estimators.sklearn import SKLearn"', DeprecationWarning,
                             filename, line_number)
        super(SKLearn, self).__init__(estimator)

