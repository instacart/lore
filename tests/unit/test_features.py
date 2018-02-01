import unittest
import tempfile

from moto import mock_s3
import boto3
import pandas

from lore.io import download

from tests.mocks.features import UserWarehouseSearchesFeature


class TestFeatures(unittest.TestCase):

    @mock_s3
    def test_s3_features(self):
        s3 = boto3.resource('s3')
        # We need to create the bucket since this is all in Moto's 'virtual' AWS account
        s3.create_bucket(Bucket='lore-test')

        user_warehouse_feature = UserWarehouseSearchesFeature()
        user_warehouse_feature.publish()

        temp_file, temp_path = tempfile.mkstemp()
        download(temp_path, user_warehouse_feature.data_path(), cache=False)

        fetched_data = pandas.read_csv(temp_path)
        self.assertTrue(len(user_warehouse_feature.get_data()) == 3)
        self.assertTrue(user_warehouse_feature.get_data().equals(fetched_data))