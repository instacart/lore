import unittest
import tests.mocks.models
import lore.metadata
from sqlalchemy.orm import sessionmaker, scoped_session

Session = scoped_session(sessionmaker(bind=lore.io.metadata._engine))
adapter = lore.io.metadata.adapter


def truncate_table(table_name):
    if adapter == 'postgres':
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


def tearDownModule():
    truncate_metadata_tables()


class FitLogging(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = tests.mocks.models.XGBoostRegressionWithPredictionLogging()
        cls.df = cls.model.pipeline.training_data

    def test_basic_logging_(self):
        self.model.fit()
        self.model.save()
        session = Session()
        model_metadata = session.query(lore.metadata.Fitting).filter_by(name=self.model.fitting_name).first()
        session.close()
        self.assertIsNotNone(model_metadata)


class PredictionLogging(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = tests.mocks.models.XGBoostRegressionWithPredictionLogging()
        cls.df = cls.model.pipeline.training_data

    def test_prediction_logging_(self):
        self.model.fit()
        self.model.save()
        self.model.predict(self.df, log_predictions=True, key_cols=['a', 'b'])
        session = Session()
        prediction_metadata = (session.query(lore.metadata.Prediction)
                               .filter_by(fitting_name=self.model.fitting_name).first())
        session.close()
        self.assertIsNotNone(prediction_metadata)
