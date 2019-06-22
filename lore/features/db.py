# -*- coding: utf-8 -*-

from abc import ABCMeta
from datetime import datetime
import lore.io
from lore.features.base import BaseFeatureExporter, BaseFeatureImporter
import lore.metadata
from lore.metadata import FeatureMetaData
from sqlalchemy.orm import sessionmaker, scoped_session
import pandas
from lore.util import convert_df_columns_to_json
import json

engine = lore.io.metadata._engine
Session = scoped_session(sessionmaker(bind=engine))


class DBFeatureExporter(BaseFeatureExporter):
    __metaclass__ = ABCMeta

    @property
    def entity_name(self):
        raise NotImplementedError

    @property
    def _data(self):
        df = self._raw_data
        df['key'] = self._generate_row_keys(df)
        df['created_at'] = datetime.utcnow()
        df['feature_data'] = df[self._values]
        df.drop(self._values, inplace=True, axis=1)
        df.drop(self.key, inplace=True, axis=1)
        return df

    @property
    def dtypes(self):
        df = self._raw_data
        dtypes = (df[self.key + self._values]
                  .dtypes
                  .to_frame('dtype'))
        dtypes = dtypes['dtype'].astype(str).to_dict()
        return dtypes

    def publish(self):
        df = self._data
        feature_metadata = lore.metadata.FeatureMetaData.create(created_at=datetime.utcnow(),
                                                                entity_name=self.entity_name,
                                                                feature_name=self.name,
                                                                version=self.version,
                                                                snapshot_at=self.timestamp,
                                                                feature_dtypes=self.dtypes,
                                                                s3_url=None)
        df['feature_metadata_id'] = feature_metadata.id
        lore.io.metadata.insert('features', df)


class DBFeatureImporter(BaseFeatureImporter):
    @property
    def feature_data(self):
        session = Session()
        metadata = (session.query(FeatureMetaData)
                    .filter_by(entity_name=self.entity_name,
                               feature_name=self.feature_name,
                               version=self.version,
                               s3_url=None)
                    .filter(FeatureMetaData.snapshot_at.between(self.start_date, self.end_date)))
        if metadata.count() == 0:
            return pandas.DataFrame()

        metadata_ids = [str(m.id) for m in metadata]
        feature_name = metadata[0].feature_name
        dtypes = metadata[0].feature_dtypes
        sql = """SELECT key, feature_data FROM features where feature_metadata_id in ({feature_metadata_ids})"""
        sql = sql.format(feature_metadata_ids=','.join(metadata_ids))
        df = lore.io.metadata.dataframe(sql, feature_metadata_ids=metadata_ids)
        if lore.io.metadata.adapter == 'sqlite':
            key_df = pandas.io.json.json_normalize(df.key.apply(json.loads))
        else:
            key_df = pandas.io.json.json_normalize(df.key)
        df.drop('key', axis=1, inplace=True)
        df = pandas.concat([df, key_df], axis=1)
        df = df.rename({'feature_data': feature_name}, axis=1)
        df = df.astype(dtypes)
        return df
