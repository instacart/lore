import unittest

import tests.mocks.models
import scipy.stats


class TestModels(unittest.TestCase):
    def test_keras(self):
        model = tests.mocks.models.Keras()
        model.fit(epochs=1)
        assert True

    def test_xgboost(self):
        model = tests.mocks.models.XGBoost()
        model.fit()
        assert True

    def test_svm(self):
        model = tests.mocks.models.SVM()
        model.fit()
        assert True

    def test_hyper_param_search(self):
        model = tests.mocks.models.Keras()
        result = model.hyper_parameter_search(
            {'embed_size': scipy.stats.randint(low=1, high=10)},
            n_iter=2,
            fit_params={'epochs': 2}
        )
        self.assertEqual(model.estimator, result.best_estimator_)
