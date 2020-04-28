import random
import logging
import re

from lore.io.connection import Connection
from lore.util import scrub_url

logger = logging.getLogger(__name__)


class MultiConnectionProxy(object):

    SQL_RUNNING_METHODS = ['dataframe', 'unload', 'select', 'execute', 'temp_table']

    def __init__(self, urls, name='connection', watermark=True, **kwargs):
        sticky = kwargs.pop('sticky_connection', None)
        sticky = False if sticky is None else (sticky.lower() == 'true')

        self._urls = urls
        self._sticky = sticky
        self._connections = []
        self._active_connection = None

        self.parse_connections(name, watermark, **kwargs)

    def parse_connections(self, name, watermark, **kwargs):
        kwargs.pop('url', None)
        for url in re.split(r'\s+', self._urls):
            c = Connection(url, name=name, watermark=watermark, **kwargs)
            self._connections.append(c)
        self.shuffle_connections()

    def shuffle_connections(self):
        if len(self._connections) == 0:
            return
        if len(self._connections) == 1:
            self._active_connection = self._connections[0]
        else:
            filtered = list(filter(lambda x: x is not self._active_connection, self._connections))
            self._active_connection = filtered[0] if len(filtered) == 1 else random.choice(filtered)
        self.log_connection()

    def log_connection(self):
        logger.debug("using database connection {}".format(scrub_url(self._active_connection.url)))

    # proxying - forward getattr to self._active_connection if not defined in MultiConnectionProxy

    def __getattr__(self, attr):
        if not self._sticky and attr in self.SQL_RUNNING_METHODS:
            self.shuffle_connections()
        return getattr(self._active_connection, attr)
