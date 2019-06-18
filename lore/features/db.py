# -*- coding: utf-8 -*-

from abc import ABCMeta
from datetime import datetime
import lore.io
from lore.features.base import BaseFeatureExporter
import lore.metadata


class DBFeatureExporter(BaseFeatureExporter):
    __metaclass__ = ABCMeta

    @property
    def entity_name(self):
        raise NotImplementedError

    def publish(self):
        df = self._data
        df.drop(self._values, inplace=True, axis=1)
        df.drop(self.key, inplace=True, axis=1)
        feature_metadata = lore.metadata.FeatureMetaData.create(created_at=datetime.utcnow(),
                                                                entity_name=self.entity_name,
                                                                feature_name=self.name,
                                                                version=self.version,
                                                                s3_url=None)
        df['feature_metadata_id'] = feature_metadata.id
        lore.io.metadata.insert('features', df)
