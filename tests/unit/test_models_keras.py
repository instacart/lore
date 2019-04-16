import unittest

import tests.mocks.models_keras
import scipy.stats
import numpy


class TestKeras(unittest.TestCase):
    def test_lifecycle(self):
        model = tests.mocks.models_keras.Keras()
        model.fit(epochs=1)
        model.save()

        loaded = tests.mocks.models_keras.Keras.load()
        self.assertEqual(loaded.fitting, model.fitting)

    def test_before_after_hooks(self):
        model = tests.mocks.models_keras.SaimeseTwinsClassifier()
        model.fit(epochs=1, test=True, score=True)
        model.predict(model.pipeline.test_data)
        self.assertTrue(model.called_before_fit)
        self.assertTrue(model.called_after_fit)
        self.assertTrue(model.called_before_predict)
        self.assertTrue(model.called_after_predict)
        self.assertTrue(model.called_before_evaluate)
        self.assertTrue(model.called_after_evaluate)
        self.assertTrue(model.called_before_score)
        self.assertTrue(model.called_after_score)

    def test_hyper_param_search(self):
        model = tests.mocks.models_keras.Keras()
        result = model.hyper_parameter_search(
            {'embed_size': scipy.stats.randint(low=1, high=10)},
            n_iter=2,
            fit_params={'epochs': 2}
        )
        self.assertEqual(model.estimator, result.best_estimator_)

    def test_lstm_embeddings(self):
        model = tests.mocks.models_keras.Keras()
        model.estimator.sequence_embedding = 'lstm'
        model.fit(epochs=1)
        assert True

    def test_gru_embeddings(self):
        model = tests.mocks.models_keras.Keras()
        model.estimator.sequence_embedding = 'gru'
        model.fit(epochs=1)
        assert True

    def test_rnn_embeddings(self):
        model = tests.mocks.models_keras.Keras()
        model.estimator.sequence_embedding = 'simple_rnn'
        model.fit(epochs=1)
        assert True

    def test_flat_embeddings(self):
        model = tests.mocks.models_keras.Keras()
        model.estimator.sequence_embedding = 'flatten'
        model.fit(epochs=1)
        assert True

    def test_towers(self):
        model = tests.mocks.models_keras.Keras()
        model.estimator.towers = 2
        model.fit(epochs=1)
        assert True

    def test_short_names(self):
        model = tests.mocks.models_keras.Keras()
        model.estimator.short_names = True
        model.estimator.build()
        assert True

    def test_batch_norm(self):
        model = tests.mocks.models_keras.Keras()
        model.estimator.batch_norm = True
        model.estimator.build()
        assert True

    def test_kernel_initializer(self):
        model = tests.mocks.models_keras.Keras()
        model.estimator.kernel_initializer = 'he_uniform'
        model.estimator.build()
        assert True


class TestKerasSingle(unittest.TestCase):
    def test_single_encoder_a(self):
        model = tests.mocks.models_keras.KerasSingle(type='tuple')
        model.estimator.build()

    def test_single_encoder_b(self):
        model = tests.mocks.models_keras.KerasSingle(type='len1')
        model.estimator.build()

    def test_single_encoder_c(self):
        model = tests.mocks.models_keras.KerasSingle(type='single')
        model.estimator.build()


class TestNestedKeras(unittest.TestCase):
    def test_lifecycle(self):
        model = tests.mocks.models_keras.NestedKeras()
        model.fit(epochs=1)
        model.save()

        loaded = tests.mocks.models_keras.NestedKeras.load()
        self.assertEqual(loaded.fitting, model.fitting)


class TestKerasMulti(unittest.TestCase):
    def test_multi(self):
        model = tests.mocks.models_keras.KerasMulti()
        model.fit(epochs=1)
        assert True


class TestSiameseArchitectureBinaryClassifier(unittest.TestCase):

    def test_siamese_architecture_twin_sequence_pair_shapes(self):
        model = tests.mocks.models_keras.SaimeseTwinsClassifier()
        model.fit()
        model.save()

        keras_model = model.estimator.keras
        twin_layers = [l.name for l in keras_model.layers if "twin" in l.name]

        for twin_layer_name in twin_layers:
            original_layer_name = twin_layer_name.replace("_twin", "")
            siamese_original_layer = keras_model.get_layer(original_layer_name)
            siamese_twin_layer = keras_model.get_layer(twin_layer_name)
            self.assertEqual(siamese_twin_layer.input_shape, siamese_original_layer.input_shape)
            self.assertEqual(siamese_twin_layer.output_shape, siamese_original_layer.output_shape)


class TestBinaryClassifier(unittest.TestCase):
    def test_lifecycle(self):
        model = tests.mocks.models_keras.BinaryClassifier()
        model.estimator.sequence_embedding = 'lstm'
        model.fit()
        model.save()
        # loaded = tests.mocks.models_other.BinaryClassifier.load()
        # self.assertEqual(loaded.fitting, model.fitting)

    def test_rnn_embeddings(self):
        model = tests.mocks.models_keras.BinaryClassifier()
        model.estimator.sequence_embedding = 'simple_rnn'
        model.fit(epochs=1)
        assert True

    def test_flatten_embeddings(self):
        model = tests.mocks.models_keras.BinaryClassifier()
        model.estimator.sequence_embedding = 'flatten'
        model.fit(epochs=1)
        assert True
