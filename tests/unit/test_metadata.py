import unittest
import tests.mocks.models
import lore.metadata
from sqlalchemy.orm import sessionmaker, scoped_session

Session = scoped_session(sessionmaker(bind=lore.io.metadata._engine))


def truncate_table(table_name):
    lore.io.metadata.execute('TRUNCATE {} RESTART IDENTITY CASCADE'.format(table_name))


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
        session.commit()
        session.close()
        self.assertIsNotNone(model_metadata)

    def test_fit_number_from_db(self):
        pass
