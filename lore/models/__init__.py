import inspect
import warnings
from lore.models import base


class Base(base.Base):

    def __init__(self, *args, **kwargs):
        frame, filename, line_number, function_name, lines, index = inspect.stack()[1]
        warnings.showwarning('lore.models.Base does not fully support lore.estimators.keras.Base. You should inherit from a specific Base from available modules: lore.models.[keras|sklearn|xgboost].Base', DeprecationWarning,
                             filename, line_number)
        super(Base, self).__init__(*args, **kwargs)
