from __future__ import absolute_import

from abc import ABCMeta
from collections import namedtuple
import inspect
import warnings

Observations = namedtuple('Observations', 'x y')

from lore.pipelines.holdout import Base


class Holdout(Base):
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        frame, filename, line_number, function_name, lines, index = inspect.stack()[1]
        warnings.showwarning(
            'Holdout has moved to lore.pipelines.holdout.Base. Please update your code. This class will be removed in version 0.5.1',
            DeprecationWarning,
            filename,
            line_number
        )
        super(Holdout, self).__init__(**kwargs)


class TimeSeries(Base):
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        frame, filename, line_number, function_name, lines, index = inspect.stack()[1]
        warnings.showwarning(
            'TimeSeries has moved to lore.pipelines.time_series.Base. Please update your code. This class will be removed in version 0.5.1',
            DeprecationWarning,
            filename,
            line_number
        )
        super(TimeSeries, self).__init__(**kwargs)


class TrainTestSplit(Holdout):
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        frame, filename, line_number, function_name, lines, index = inspect.stack()[1]
        warnings.showwarning(
            'TrainTestSplit has moved to lore.pipelines.holdout.Base. Please update your code. This class will be removed in version 0.5.1',
            DeprecationWarning,
            filename,
            line_number
        )
        super(TrainTestSplit, self).__init__(**kwargs)


class SortedTrainTestSplit(TimeSeries):
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        frame, filename, line_number, function_name, lines, index = inspect.stack()[1]
        warnings.showwarning(
            'SortedTrainTestSplit has moved to lore.pipelines.time_series.Base. Please update your code. This class will be removed in version 0.5.1',
            DeprecationWarning,
            filename,
            line_number
        )
        super(SortedTrainTestSplit, self).__init__(**kwargs)
