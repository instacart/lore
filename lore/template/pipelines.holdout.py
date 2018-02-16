import lore.pipelines
from lore.util import timed

import logging

logger = logging.getLogger(__name__)


class Holdout(lore.pipelines.holdout.Base):
    def __init__(self):
        super(Holdout, self).__init__()
    
    @timed(logging.INFO)
    def get_data(self):
        return  # TODO
    
    @timed(logging.INFO)
    def get_encoders(self):
        return  # TODO

    @timed(logging.INFO)
    def get_output_encoder(self):
        return  # TODO

