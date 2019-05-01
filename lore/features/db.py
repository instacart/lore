# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
import json
import os
import tempfile
import lore
import lore.io
from lore.features.base import Base
from lore.io import upload
from lore.util import add_random_delay, get_random_string

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DB(Base):
    __metaclass__ = ABCMeta

    @abstractmethod
    def table_name(self):
        pass

    @abstractmethod
    def db_conn(self):
        pass

    def publish(self, schema, upsert=True):
        data = self.get_data()

    def _stage_features_in_temp_table(schema, features_df):
        staging_table_name = "{}_{}".format(self.table_name(), )
        qualified_staging_table_name = schema + "." + staging_table_name
        try:
            create_table_query = """
                CREATE TABLE %s AS
                SELECT * FROM %s WHERE 0=1
            """ % (qualified_staging_table_name, table_name)
            logger.info("Create new staging table query : {}".format(create_table_query))

            self.db_conn().execute(create_table_query)
            self.db_conn().insert(qualified_staging_table_name, features_df)
        except Exception as e:
            raise(e)

        add_random_delay(max_delay_value=300)

        return qualified_staging_table_name


    @timed(logging.INFO)
    def _upsert_in_features_table(features_temp_table):
        did_upsert_succeed = True

        updated_features_query = """
            UPDATE %s AS main_table
                SET features = CAST(temp.features AS jsonb), updated_at = now()
            FROM %s AS temp
            WHERE 1=1
                AND main_table.engine = temp.engine
                AND main_table.feature_type = temp.feature_type
                AND main_table.feature_key = temp.feature_key
        """ % (self.table_name(), features_temp_table)
        logger.info("Update query for replacement features : {}".format(updated_features_query))
        try:
            self.db_conn().execute(sql=updated_features_query)
        except Exception as e:
            did_upsert_succeed = False
            logger.error("UPDATE failed. Reason : {}".format(e))

        add_random_delay(max_delay_value=300)

        insert_features_query = """
            INSERT INTO %s(engine, feature_type, feature_key, features, created_at, updated_at)
            SELECT DISTINCT
                temp.engine,
                temp.feature_type,
                temp.feature_key,
                CAST(temp.features AS jsonb) AS features,
                now() AS created_at,
                now() AS updated_at
            FROM %s temp
            LEFT OUTER JOIN %s rmf ON (
                temp.engine = rmf.engine AND
                temp.feature_type = rmf.feature_type AND
                temp.feature_key = rmf.feature_key
            )
            WHERE rmf.feature_key IS NULL
        """ % (self.table_name(), features_temp_table, self.table_name())
        logger.info("Insert query for generating features : {}".format(insert_features_query))

        try:
            self.db_conn().execute(sql=insert_replacement_features_query)
        except Exception as e:
            did_upsert_succeed = False
            logger.error("INSERT failed. Reason : {}".format(e))

        try:
            logger.info("Dropping feature staging table : {}".format(features_temp_table))
            self.db_conn().execute("DROP TABLE " + features_temp_table)
        except Exception as e:
            logger.error("Failed to drop table {}. Reason : {}".format(features_temp_table, e))

        return did_upsert_succeed