from sklearn import svm

import lore.models
import lore.models.keras
import lore.models.sklearn
import lore.models.xgboost
import lore.estimators.keras
import lore.estimators.sklearn
import lore.estimators.xgboost
import tests.mocks.pipelines


class Keras(lore.models.keras.Base):
    def __init__(
        self,
        embed_size=10
    ):
        super(Keras, self).__init__(
            tests.mocks.pipelines.Xor(),
            lore.estimators.keras.Base(
                batch_size=1024,
                embed_size=embed_size,
                hidden_layers=1,
                hidden_width=100,
                loss='binary_crossentropy',
                monitor='val_loss',
                cudnn=False
            )
        )


class KerasSingle(lore.models.keras.Base):
    def __init__(
        self,
        type
    ):
        super(KerasSingle, self).__init__(
            tests.mocks.pipelines.XorSingle(type=type),
            lore.estimators.keras.Base(loss='binary_crossentropy')
        )


class NestedKeras(lore.models.keras.Base):
    def __init__(
        self,
        embed_size=10
    ):
        super(NestedKeras, self).__init__(
            tests.mocks.pipelines.MockNestedData(),
            lore.estimators.keras.Base(
                batch_size=1024,
                embed_size=embed_size,
                hidden_layers=1,
                hidden_width=100,
                loss='binary_crossentropy',
                monitor='loss',
                cudnn=False
            )
        )


class KerasMulti(lore.models.keras.Base):
    def __init__(self):
        super(KerasMulti, self).__init__(
            tests.mocks.pipelines.XorMulti(),
            lore.estimators.keras.MultiClassifier(
                batch_size=1024,
                embed_size=10,
                hidden_layers=1,
                hidden_width=100
            )
        )


class XGBoostBinaryClassifier(lore.models.xgboost.Base):
    def __init__(self):
        super(XGBoostBinaryClassifier, self).__init__(
            tests.mocks.pipelines.Xor(),
            lore.estimators.xgboost.BinaryClassifier()
        )


class XGBoostRegression(lore.models.xgboost.Base):
    def __init__(self):
        super(XGBoostRegression, self).__init__(
            tests.mocks.pipelines.Xor(),
            lore.estimators.xgboost.Regression()
        )


class XGBoostRegressionWithPredictionLogging(lore.models.xgboost.Base):
    def __init__(self):
        super(XGBoostRegressionWithPredictionLogging, self).__init__(
            tests.mocks.pipelines.Xor(),
            lore.estimators.xgboost.Regression()
        )


class SVM(lore.models.sklearn.Base):
    def __init__(self):
        super(SVM, self).__init__(
            tests.mocks.pipelines.Xor(),
            lore.estimators.sklearn.Base(
                svm.SVC()
            )
        )

    def before_fit(self, *args, **kwargs):
        self.called_before_fit = True

    def after_fit(self, *args, **kwargs):
        self.called_after_fit = True

    def before_predict(self, *args, **kwargs):
        self.called_before_predict = True

    def after_predict(self, *args, **kwargs):
        self.called_after_predict = True

    def before_evaluate(self, *args, **kwargs):
        self.called_before_evaluate = True

    def after_evaluate(self, *args, **kwargs):
        self.called_after_evaluate = True

    def before_score(self, *args, **kwargs):
        self.called_before_score = True

    def after_score(self, *args, **kwargs):
        self.called_after_score = True


class BinaryClassifier(lore.models.keras.Base):
    def __init__(
        self,
        embed_size=10
    ):
        super(BinaryClassifier, self).__init__(
        tests.mocks.pipelines.TwinData(test_size=0.5),
        lore.estimators.keras.BinaryClassifier(
            batch_size=1024,
            embed_size=embed_size,
            hidden_layers=1,
            hidden_width=100,
            cudnn=False
        )
    )


class SaimeseTwinsClassifier(lore.models.keras.Base):
    def __init__(
        self,
        embed_size=10,
        sequence_embed_size=50,
    ):
        super(SaimeseTwinsClassifier, self).__init__(
            tests.mocks.pipelines.TwinDataWithVaryingEmbedScale(test_size=0.5),
            lore.estimators.keras.BinaryClassifier(
                batch_size=1024,
                embed_size=embed_size,
                sequence_embed_size=sequence_embed_size,
                sequence_embedding='lstm',
                hidden_layers=1,
                hidden_width=100,
                cudnn=False
            )
        )

    def before_fit(self, *args, **kwargs):
        self.called_before_fit = True

    def after_fit(self, *args, **kwargs):
        self.called_after_fit = True

    def before_predict(self, *args, **kwargs):
        self.called_before_predict = True

    def after_predict(self, *args, **kwargs):
        self.called_after_predict = True

    def before_evaluate(self, *args, **kwargs):
        self.called_before_evaluate = True

    def after_evaluate(self, *args, **kwargs):
        self.called_after_evaluate = True

    def before_score(self, *args, **kwargs):
        self.called_before_score = True

    def after_score(self, *args, **kwargs):
        self.called_after_score = True


class OneHotBinaryClassifier(lore.models.xgboost.Base):
    def __init__(self):
        super(OneHotBinaryClassifier, self).__init__(
            tests.mocks.pipelines.OneHotPipeline(),
            lore.estimators.xgboost.BinaryClassifier())

    def before_fit(self, *args, **kwargs):
        self.called_before_fit = True

    def after_fit(self, *args, **kwargs):
        self.called_after_fit = True

    def before_predict(self, *args, **kwargs):
        self.called_before_predict = True

    def after_predict(self, *args, **kwargs):
        self.called_after_predict = True

    def before_evaluate(self, *args, **kwargs):
        self.called_before_evaluate = True

    def after_evaluate(self, *args, **kwargs):
        self.called_after_evaluate = True

    def before_score(self, *args, **kwargs):
        self.called_before_score = True

    def after_score(self, *args, **kwargs):
        self.called_after_score = True
