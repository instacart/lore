# -*- coding: utf-8 -*-

from abc import ABCMeta
from datetime import datetime
import lore.io
from lore.features.base import Base
from lore.util import convert_df_columns_to_json
import lore.metadata


class DB(Base):
    __metaclass__ = ABCMeta
    MANDATORY_COLUMNS = ['entity_id']

    @property
    def entity_name(self):
        raise NotImplementedError

    @property
    def feature_validity(self):
        return None

    def publish(self):
        self._raw_data = self.get_data()
        df = self._raw_data.copy()
        self._data = df
        df['key'] = self._generate_row_keys(df)
        df['created_at'] = datetime.utcnow()
        if self.feature_validity is not None:
            df['starts_on'], df['ends_before'] = self.feature_validity
        df['entity_name'] = self.entity_name
        df['version'] = self.version
        df['feature_name'] = self.name
        df['feature_data'] = convert_df_columns_to_json(df, self._values)
        df.drop(self._values, inplace=True, axis=1)
        df.drop(self.key, inplace=True, axis=1)
        feature_metadata = lore.metadata.FeatureMetaData.create(created_at=datetime.utcnow(),
                                                                s3_url=None)
        df['feature_metadata_id'] = feature_metadata.id
        lore.io.metadata.insert('features', df)
