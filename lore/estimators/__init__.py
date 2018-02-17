"""
lore.estimators
"""
from __future__ import absolute_import
import sys
import inspect
import warnings


if not (sys.version_info.major == 3 and sys.version_info.minor >= 6):
    ModuleNotFoundError = ImportError

try:
    import keras as installed
    from lore.estimators.keras import Base

    class Keras(Base):
    
        def __init__(self, **kwargs):
            frame, filename, line_number, function_name, lines, index = inspect.stack()[1]
            warnings.showwarning('Please import Keras with "from lore.estimators.keras import Base"', DeprecationWarning,
                                 filename, line_number)
            super(Keras, self).__init__(**kwargs)

except ModuleNotFoundError as e:
    pass


try:
    import xgboost as installed
    from lore.estimators.xgboost import Base


    class XGBoost(Base):
    
        def __init__(self, **kwargs):
            frame, filename, line_number, function_name, lines, index = inspect.stack()[1]
            warnings.showwarning('Please import XGBoost with "from lore.estimators.xgboost import Base"', DeprecationWarning,
                                 filename, line_number)
            super(XGBoost, self).__init__(**kwargs)

except ModuleNotFoundError as e:
    pass


try:
    import sklearn as installed
    from lore.estimators.sklearn import Base

    class SKLearn(Base):
    
        def __init__(self, estimator):
            frame, filename, line_number, function_name, lines, index = inspect.stack()[1]
            warnings.showwarning('Please import SKLearn with "from lore.estimators.sklearn import Base"',
                                 DeprecationWarning,
                                 filename, line_number)
            super(SKLearn, self).__init__(estimator)

except ModuleNotFoundError as e:
    pass
