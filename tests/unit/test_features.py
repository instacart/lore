import unittest
from tests.mocks.features import UserWarehouseSearchesFeature
from moto import mock_s3
import boto3

class TestFeatures(unittest.TestCase):

    @mock_s3
    def test_s3_features(self):
        conn = boto3.resource('s3')
        # We need to create the bucket since this is all in Moto's 'virtual' AWS account
        conn.create_bucket(Bucket='lore-test')

        user_warehouse_feature = UserWarehouseSearchesFeature()
        print(user_warehouse_feature)
        user_warehouse_feature.publish()
        print(user_warehouse_feature.features_as_kv())
        self.assertTrue(user_warehouse_feature)