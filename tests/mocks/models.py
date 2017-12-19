from sklearn import svm

import lore.models
import lore.models.keras
import tests.mocks.pipelines


class Keras(lore.models.keras.Keras):
    def __init__(
        self,
        embed_size=10
    ):
        super(Keras, self).__init__(
            tests.mocks.pipelines.Xor(),
            lore.estimators.keras.Keras(
                batch_size=1024,
                embed_size=embed_size,
                hidden_layers=1,
                hidden_width=100,
                loss='binary_crossentropy',
                monitor='val_loss'
            )
        )


class XGBoost(lore.models.Base):
    def __init__(self):
        super(XGBoost, self).__init__(
            tests.mocks.pipelines.Xor(),
            lore.estimators.xgboost.XGBoost(
                silent=1,
                objective='binary:logistic'
            )
        )


class SVM(lore.models.Base):
    def __init__(self):
        super(SVM, self).__init__(
            tests.mocks.pipelines.Xor(),
            lore.estimators.sklearn.SKLearn(
                svm.SVC()
            )
        )
