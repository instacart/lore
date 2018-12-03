import unittest
import tests.mocks.models
import lore.metadata
import datetime


def truncate_table(table_name):
    if lore.io.metadata.adapter == 'postgres':
        lore.io.metadata.execute('TRUNCATE {} RESTART IDENTITY CASCADE'.format(table_name))
    else:
        lore.io.metadata.execute('DELETE FROM {}'.format(table_name))


def truncate_metadata_tables():
    truncate_table('fittings')
    truncate_table('commits')
    truncate_table('predictions')
    truncate_table('snapshots')


def setUpModule():
    truncate_metadata_tables()


class TestCrud(unittest.TestCase):
    def test_lifecycle(self):
        commit = lore.metadata.Commit.create(sha='abc')
        self.assertEqual(commit.__class__, lore.metadata.Commit)
        self.assertIsNotNone(commit.sha)

        commit.created_at = datetime.datetime.now()
        commit.save()

        all = lore.metadata.Commit.all()
        self.assertEqual(len(all), 1)

        first = lore.metadata.Commit.first()
        self.assertEqual(first.sha, commit.sha)

        last = lore.metadata.Commit.last()
        self.assertEqual(first.sha, last.sha)

        commit.delete()

        first = lore.metadata.Commit.first()
        self.assertIsNone(first)


class TestFitting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = tests.mocks.models.XGBoostRegressionWithPredictionLogging()
        cls.df = cls.model.pipeline.training_data

    def test_model_fit(self):
        self.model.fit()
        fitting = lore.metadata.Fitting.last()
        self.assertEqual(fitting.id, self.model.fitting.id)
        self.assertEqual(fitting.model, self.model.name)
        self.assertEqual(fitting.id, self.model.last_fitting().id)


class TestPredictionLogging(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = tests.mocks.models.XGBoostRegressionWithPredictionLogging()
        cls.df = cls.model.pipeline.training_data

    def test_prediction_logging(self):
        self.model.fit()
        self.model.predict(self.df, log_predictions=True, key_cols=['a', 'b'])
        prediction_metadata = lore.metadata.Prediction.first(fitting_id=self.model.fitting.id)
        self.assertIsNotNone(prediction_metadata)
