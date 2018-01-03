import unittest

import tests.mocks.models
import scipy.stats


class TestKeras(unittest.TestCase):
    def test_lifecycle(self):
        model = tests.mocks.models.Keras()
        model.fit(epochs=1)
        model.save()

        loaded = tests.mocks.models.Keras.load()
        self.assertEqual(loaded.fitting, model.fitting)

    def test_hyper_param_search(self):
        model = tests.mocks.models.Keras()
        result = model.hyper_parameter_search(
            {'embed_size': scipy.stats.randint(low=1, high=10)},
            n_iter=2,
            fit_params={'epochs': 2}
        )
        self.assertEqual(model.estimator, result.best_estimator_)

    def test_lstm_embeddings(self):
        model = tests.mocks.models.Keras()
        model.estimator.sequence_embedding = 'lstm'
        model.fit(epochs=1)
        assert True

    def test_gru_embeddings(self):
        model = tests.mocks.models.Keras()
        model.estimator.sequence_embedding = 'gru'
        model.fit(epochs=1)
        assert True

    def test_rnn_embeddings(self):
        model = tests.mocks.models.Keras()
        model.estimator.sequence_embedding = 'simple_rnn'
        model.fit(epochs=1)
        assert True

    def test_flat_embeddings(self):
        model = tests.mocks.models.Keras()
        model.estimator.sequence_embedding = 'flatten'
        model.fit(epochs=1)
        assert True

    def test_towers(self):
        model = tests.mocks.models.Keras()
        model.estimator.towers = 2
        model.fit(epochs=1)
        assert True


class TestXGBoost(unittest.TestCase):
    def test_lifecycle(self):
        model = tests.mocks.models.XGBoost()
        model.fit()
        model.save()

        loaded = tests.mocks.models.XGBoost.load()
        self.assertEqual(loaded.fitting, model.fitting)


class TestSKLearn(unittest.TestCase):
    def test_lifecycle(self):
        model = tests.mocks.models.SVM()
        model.fit()
        model.save()
    
        loaded = tests.mocks.models.SVM.load()
        self.assertEqual(loaded.fitting, model.fitting)
