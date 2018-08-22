# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
import json
import os
import tempfile

import lore
from lore.features.base import Base
from lore.io import upload


class S3(Base):
    __metaclass__ = ABCMeta

    @abstractmethod
    def serialization(self):
        pass

    def publish(self, compression='gzip'):
        temp_file, temp_path = tempfile.mkstemp(dir=lore.env.DATA_DIR)
        data = self.get_data()

        if self.serialization() == 'csv':
            data.to_csv(temp_path, index=False, compression=compression)
        elif self.serialization() == 'pickle':
            data.to_pickle(temp_path, compression=compression)
        else:
            raise "Invalid serialization"
        upload(temp_path, self.data_path())

        with open(temp_path, 'w') as f:
            f.write(json.dumps(self.metadata()))
        upload(temp_path, self.metadata_path())
        os.close(temp_file)
        os.remove(temp_path)

    def data_path(self):
        return "{}/{}/data.{}".format(self.version, self.name(), self.serialization())

    def metadata_path(self):
        return "{}/{}/metadata.json".format(self.version, self.name())
