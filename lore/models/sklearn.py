from sklearn.model_selection import RandomizedSearchCV

import lore.models.base


class Base(lore.models.base.Base):
    
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
        self.estimator.sklearn = result.best_estimator_
        return result

