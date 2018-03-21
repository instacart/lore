import unittest

import lore.serializers

import lore.env

import lore
import lore.models

import tests.mocks.models
import numpy.testing


class TestKeras(unittest.TestCase):
    def test_save(self):
        m1 = tests.mocks.models.Keras()
        m1.fit(epochs=1)
        s1 = lore.serializers.Keras(model=m1)
        s1.save()
        
        s2 = lore.serializers.Keras(klass=tests.mocks.models.Keras)
        m2 = s2.load()

        # m3 will have uninitialized, isolated weights
        m3 = tests.mocks.models.Keras()
        m3.estimator.build()
        
        with m1.estimator.session.as_default():
            weights1 = m1.estimator.keras.get_layer('0_hidden_0').get_weights()[0][0]
        with m2.estimator.session.as_default():
            weights2 = m2.estimator.keras.get_layer('0_hidden_0').get_weights()[0][0]
        with m3.estimator.session.as_default():
            weights3 = m3.estimator.keras.get_layer('0_hidden_0').get_weights()[0][0]

        self.assertTrue(numpy.all([numpy.all(x == y) for x, y in zip(weights1, weights2)]))
        self.assertFalse(numpy.all([numpy.all(x == y) for x, y in zip(weights1, weights3)]))


class TestXGBoost(unittest.TestCase):
    def test_save(self):
        m1 = tests.mocks.models.XGBoostBinaryClassifier()
        m1.fit()
        s1 = lore.serializers.Base(model=m1)
        s1.save()
        
        s2 = lore.serializers.Base(klass=tests.mocks.models.XGBoostBinaryClassifier)
        m2 = s2.load()
        self.assertTrue(True)


class TestSKLearn(unittest.TestCase):
    def test_save(self):
        m1 = tests.mocks.models.SVM()
        m1.fit()
        s1 = lore.serializers.Base(model=m1)
        s1.save()
    
        s2 = lore.serializers.Base(klass=tests.mocks.models.SVM)
        m2 = s2.load()
        self.assertTrue(True)
