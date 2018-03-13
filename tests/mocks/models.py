from sklearn import svm

import lore.models
import lore.models.keras
import lore.models.sklearn
import lore.models.xgboost
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


class XGBoost(lore.models.xgboost.Base):
    def __init__(self):
        super(XGBoost, self).__init__(
            tests.mocks.pipelines.Xor(),
            lore.estimators.xgboost.Base(
                silent=1,
                objective='binary:logistic'
            )
        )


class SVM(lore.models.sklearn.Base):
    def __init__(self):
        super(SVM, self).__init__(
            tests.mocks.pipelines.Xor(),
            lore.estimators.sklearn.Base(
                svm.SVC()
            )
        )


class BinaryClassifier(lore.models.keras.Base):
    def __init__(
        self,
        embed_size=10
    ):
        super(BinaryClassifier, self).__init__(
        tests.mocks.pipelines.TwinData(),
        lore.estimators.keras.BinaryClassifier(
            batch_size=1024,
            embed_size=embed_size,
            hidden_layers=1,
            hidden_width=100,
            cudnn=False
        )
    )
