import unittest
import tests.mocks.models_other

import numpy

class TestXGBoostRegression(unittest.TestCase):
    def test_lifecycle(self):
        model = tests.mocks.models_other.XGBoostRegression()
        model.fit()
        model.save()

        loaded = tests.mocks.models_other.XGBoostRegression.load()
        self.assertEqual(loaded.fitting, model.fitting)


class TestXGBoostBinaryClassifier(unittest.TestCase):
    def test_lifecycle(self):
        model = tests.mocks.models_other.XGBoostBinaryClassifier()
        model.fit()
        model.save()

        loaded = tests.mocks.models_other.XGBoostBinaryClassifier.load()
        self.assertEqual(loaded.fitting, model.fitting)

    def test_probs(self):
        model = tests.mocks.models_other.XGBoostBinaryClassifier()
        model.fit()
        model.predict_proba(model.pipeline.test_data)
        assert True


class TestSKLearn(unittest.TestCase):
    def test_lifecycle(self):
        model = tests.mocks.models_other.SVM()
        model.fit()
        model.save()

        loaded = tests.mocks.models_other.SVM.load()
        self.assertEqual(loaded.fitting, model.fitting)

    def test_before_after_hooks(self):
        model = tests.mocks.models_other.SVM()
        model.fit(test=True, score=True)
        model.predict(model.pipeline.test_data)
        self.assertTrue(model.called_before_fit)
        self.assertTrue(model.called_after_fit)
        self.assertTrue(model.called_before_predict)
        self.assertTrue(model.called_after_predict)
        self.assertTrue(model.called_before_evaluate)
        self.assertTrue(model.called_after_evaluate)
        self.assertTrue(model.called_before_score)
        self.assertTrue(model.called_after_score)


class TestOneHotBinaryClassifier(unittest.TestCase):
    def test_lifecycle(self):
        model = tests.mocks.models_other.OneHotBinaryClassifier()
        model.fit()
        model.save()

        loaded = tests.mocks.models_other.OneHotBinaryClassifier.load()
        self.assertEqual(loaded.fitting, model.fitting)

    def test_before_after_hooks(self):
        model = tests.mocks.models_other.OneHotBinaryClassifier()
        model.fit(test=True, score=True)
        model.predict(model.pipeline.test_data)

        self.assertTrue(model.called_before_fit)
        self.assertTrue(model.called_after_fit)
        self.assertTrue(model.called_before_predict)
        self.assertTrue(model.called_after_predict)
        self.assertTrue(model.called_before_evaluate)
        self.assertTrue(model.called_after_evaluate)
        self.assertTrue(model.called_before_score)
        self.assertTrue(model.called_after_score)


class TestNaiveBinaryClassifier(unittest.TestCase):
    def test_lifecycle(self):
        model = tests.mocks.models_other.NaiveBinaryClassifier()
        model.fit()
        model.save()

        loaded = tests.mocks.models_other.NaiveBinaryClassifier.load()
        self.assertEqual(loaded.fitting, model.fitting)

    def test_before_after_hooks(self):
        model = tests.mocks.models_other.NaiveBinaryClassifier()
        model.fit(test=True, score=True)
        model.predict(model.pipeline.test_data)

        self.assertTrue(model.called_before_fit)
        self.assertTrue(model.called_after_fit)
        self.assertTrue(model.called_before_predict)
        self.assertTrue(model.called_after_predict)
        self.assertTrue(model.called_before_evaluate)
        self.assertTrue(model.called_after_evaluate)
        self.assertTrue(model.called_before_score)
        self.assertTrue(model.called_after_score)

    def test_preds(self):
        model = tests.mocks.models_other.NaiveBinaryClassifier()
        model.fit(test=True, score=True)
        preds = model.predict(model.pipeline.test_data)
        self.assertTrue((preds == 1).all())

    def test_probs(self):
        model = tests.mocks.models_other.NaiveBinaryClassifier()
        model.fit(test=True, score=True)
        probs = model.predict_proba(model.pipeline.test_data)[:, 1]
        self.assertTrue((numpy.abs(probs - 0.667) < 0.001).all())
