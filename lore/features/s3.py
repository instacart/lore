# -*- coding: utf-8 -*-

from lore.features.base import Base
from abc import ABCMeta, abstractmethod
from lore.io import upload
import json
import tempfile

class S3(Base):
    __metaclass__ = ABCMeta

    def canonical_store(self):
        return 's3'

    def publish(self):
        temp_file, temp_path = tempfile.mkstemp()
        data = self.get_data()

        if self.serialization() == 'csv':
            data.to_csv(temp_path, index=False)
        else:
            data.to_pickle(temp_path)
        upload(temp_path, self.data_path())

        with open(temp_path, 'w') as f:
            f.write(json.dumps(self.json_meta_data()))
        upload(temp_path, self.metadata_path())

    @abstractmethod
    def data_path(self):
        pass

    @abstractmethod
    def metadata_path(self):
        pass

    @abstractmethod
    def serialization(self):
        pass
