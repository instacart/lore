import logging

from tabulate import tabulate
import lore.estimators
import lore.serializers

from sklearn.model_selection import RandomizedSearchCV

logger = logging.getLogger(__name__)


class Base(object):
    def __init__(self, pipeline, estimator):
        self.name = self.__module__ + '.' + self.__class__.__name__
        self._estimator = None
        self.serializer = None
        self.pipeline = pipeline
        self.estimator = estimator
    
    def __getstate__(self):
        return dict(self.__dict__)
    
    @property
    def estimator(self):
        return self._estimator
    
    @estimator.setter
    def estimator(self, value):
        self._estimator = value
        # TODO Should we unify serializers to avoid this check?
        if isinstance(self._estimator, lore.estimators.keras.Keras):
            self.serializer = lore.serializers.Keras(model=self)
        else:
            self.serializer = lore.serializers.Base(model=self)
        
        # Keras models require access to the pipeline during build,
        # and the serializer during fit for extended functionality
        if hasattr(self._estimator, 'model'):
            self._estimator.model = self
            
    def fit(self, **estimator_kwargs):
        self.serializer.fitting += 1
        
        self.stats = self.estimator.fit(
            x=self.pipeline.encoded_training_data.x,
            y=self.pipeline.encoded_training_data.y,
            **estimator_kwargs
        )
        self.serializer.save(stats=self.stats)
        logger.info('\n\n' + tabulate([self.stats.keys(), self.stats.values()], tablefmt="grid", headers='firstrow') + '\n\n')

    def predict(self, dataframe):
        return self.estimator.predict(self.pipeline.encode_x(dataframe))
        
    def hyper_parameter_search(
            self,
            param_distributions,
            n_iter=10,
            scoring=None,
            fit_params={},
            n_jobs=1,
            iid=True,
            refit=True,
            cv=None,
            verbose=0,
            pre_dispatch='2*njobs',
            random_state=None,
            error_score='raise',
            return_train_score=True
    ):
        """Random search hyper params
        """
        if scoring is None:
            scoring = None
        result = RandomizedSearchCV(
            self.estimator,
            param_distributions,
            n_iter=n_iter,
            scoring=scoring,
            n_jobs=n_jobs,
            iid=iid,
            refit=refit,
            cv=cv,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            random_state=random_state,
            error_score=error_score,
            return_train_score=return_train_score
        ).fit(
            self.pipeline.encoded_training_data.x,
            y=self.pipeline.encoded_training_data.y,
            **fit_params
        )
        self.estimator = result.best_estimator_
        
        return result
