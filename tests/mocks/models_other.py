from sklearn import svm
import tests

import lore.models.sklearn
import lore.models.xgboost
import lore.models.naive
import lore.estimators.sklearn
import lore.estimators.xgboost
import lore.estimators.naive


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


class NaiveBinaryClassifier(lore.models.naive.Base):
    def __init__(self):
        super(NaiveBinaryClassifier, self).__init__(
            tests.mocks.pipelines.NaivePipeline(),
            lore.estimators.naive.BinaryClassifier())

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
