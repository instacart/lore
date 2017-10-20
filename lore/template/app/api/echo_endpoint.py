import logging

from hub import endpoint


logger = logging.getLogger(__name__)


@endpoint
class EchoEndpoint(object):
    def perform(self, payload):
        logger.info(payload["msg"])
        return payload["msg"]
