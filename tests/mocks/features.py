import logging
from lore.features.base import Base
from lore.features.s3 import S3
import pandas as pd
import inflection
logger = logging.getLogger(__name__)

class UserWarehouseSearchesFeature(S3):
    def __init__(self):
        Base.__init__(self)
        S3.__init__(self)

    def name(self):
        return inflection.underscore(UserWarehouseSearchesFeature.__name__)

    def key(self):
        return ['user_id', 'warehouse_id']

    def data_path(self):
        return "{}/{}/data.{}".format(self.version, self.name(), self.serialization())

    def metadata_path(self):
        return "{}/{}/metadata.json".format(self.version, self.name())

    def serialization(self):
        return 'csv'

    def get_data(self):
        return pd.DataFrame({'user_id': [1, 1, 2], 'warehouse_id': [1, 2, 1], 'searches': [10, 20, 30], 'conversions': [1, 2, 3]})

        # write entry in ddatabase w/ metadata
        # lore.io.s3[mymetadatakey] = self (upload, pickle/json/csv)



### Ideas

# feature = UserWarehouseSearchesFeature()
# print(feature.get_data())
# print(feature)
# feature.publish()
# print(feature.features_as_kv())
# feature.distribute(Redis(lore.io.redis_conn))
# building.publish()
#
# downloaded = UserWarehouseSearchesFeature(version=1)
# downloaded._data # filled from s3 previous publish
#
# downloaded.distribute(lore.io.redis)
#
# building.metadata # dataframe
# building.metadata.to_sql('features', lore.io.customers)
#
# lore.io.customers.insert('features', building.metadata)
# lore.io.customers.replace('features', building.metadata)
