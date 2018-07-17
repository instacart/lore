from __future__ import absolute_import

from abc import ABCMeta
import logging

import lore
from lore.util import timed
import lore.pipelines.holdout

logger = logging.getLogger(__name__)


class Base(lore.pipelines.holdout.Base):
    __metaclass__ = ABCMeta

    def __init__(self, test_size=0.1, sort_by=None):
        super(Base, self).__init__()
        self.sort_by = sort_by
        self.test_size = test_size

    @timed(logging.INFO)
    def _split_data(self):
        if self._data:
            return

        logger.debug('No shuffle test train split')

        self._data = self.get_data()

        if self.sort_by:
            self._data = self._data.sort_values(by=self.sort_by, ascending=True)
        test_rows = int(len(self._data) * self.test_size)
        valid_rows = test_rows
        train_rows = int(len(self._data) - test_rows - valid_rows)
        self._training_data = self._data.iloc[:train_rows]
        self._validation_data = self._data.iloc[train_rows:train_rows+valid_rows]
        self._test_data = self._data.iloc[-test_rows:]
