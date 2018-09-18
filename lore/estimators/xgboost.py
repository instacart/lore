from __future__ import absolute_import
import inspect
import logging
import warnings
import threading

import lore.env
import lore.estimators
from lore.util import timed, before_after_callbacks

lore.env.require(
    lore.dependencies.XGBOOST +
    lore.dependencies.SKLEARN
)

import xgboost


logger = logging.getLogger(__name__)


class Base(object):
    def __init__(self, **xgboost_params):
        self.eval_metric = xgboost_params.pop('eval_metric', None)
        self.xgboost_lock = threading.RLock()
        self.missing = None
        super(Base, self).__init__(**xgboost_params)

    def __getstate__(self):
        state = super(Base, self).__getstate__()
        state['xgboost_lock'] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.xgboost_lock = threading.RLock()

        backward_compatible_defaults = {
            'n_jobs': state.pop('nthread', -1),
            'random_state': state.pop('seed', 0)
        }
        for key, default in backward_compatible_defaults.items():
            if key not in self.__dict__.keys():
                self.__dict__[key] = default

    @before_after_callbacks
    @timed(logging.INFO)
    def fit(self, x, y, validation_x=None, validation_y=None, patience=0, verbose=None, **xgboost_kwargs):
        eval_set = [(x, y)]
        if validation_x is not None and validation_y is not None:
            eval_set += [(validation_x, validation_y)]
        if verbose is None:
            verbose = True if lore.env.NAME == lore.env.DEVELOPMENT else False

        try:
            super(Base, self).fit(
                X=x,
                y=y,
                eval_set=eval_set,
                eval_metric=self.eval_metric,
                verbose=verbose,
                early_stopping_rounds=patience,
                **xgboost_kwargs
            )
        except KeyboardInterrupt:
            logger.warning('Caught SIGINT. Training aborted.')

        evals = super(Base, self).evals_result()
        results = {
            'train': evals['validation_0'][self.eval_metric][self.best_iteration],
            'best_iteration': self.best_iteration
        }
        if validation_x is not None:
            results['validate'] = evals['validation_1'][self.eval_metric][self.best_iteration]
        return results

    @before_after_callbacks
    @timed(logging.INFO)
    def predict(self, dataframe, ntree_limit=None):
        if ntree_limit is None:
            ntree_limit = self.best_ntree_limit or 0
        with self.xgboost_lock:
            return super(Base, self).predict(dataframe, ntree_limit=ntree_limit)

    @before_after_callbacks
    @timed(logging.INFO)
    def evaluate(self, x, y):
        with self.xgboost_lock:
            return float(self.get_booster().eval(xgboost.DMatrix(x, label=y)).split(':')[-1])

    @before_after_callbacks
    @timed(logging.INFO)
    def score(self, x, y):
        return 1 / self.evaluate(x, y)


class XGBoost(lore.estimators.Base):
    def __init__(self, **kwargs):
        frame, filename, line_number, function_name, lines, index = inspect.stack()[1]
        warnings.showwarning('Please import XGBoost with "from lore.estimators.xgboost import Base"',
                             DeprecationWarning,
                             filename, line_number)
        super(XGBoost, self).__init__(**kwargs)


class Regression(Base, xgboost.XGBRegressor):
    def __init__(
        self,
        max_depth=3,
        learning_rate=0.1,
        n_estimators=100,
        silent=True,
        objective='reg:linear',
        booster='gbtree',
        n_jobs=-1,
        gamma=0,
        min_child_weight=1,
        max_delta_step=0,
        subsample=1,
        colsample_bytree=1,
        colsample_bylevel=1,
        reg_alpha=0,
        reg_lambda=1,
        scale_pos_weight=1,
        base_score=0.5,
        random_state=0,
        missing=None,
        eval_metric='rmse',
        **kwargs
    ):
        kwargs = locals()
        kwargs.pop('self')
        kwargs.pop('__class__', None)
        kwargs = dict(kwargs, **(kwargs.pop('kwargs', {})))
        if 'random_state' not in kwargs and 'seed' in kwargs:
            kwargs['random_state'] = kwargs.pop('seed')
        if 'n_jobs' not in kwargs and 'nthread' in kwargs:
            kwargs['n_jobs'] = kwargs.pop('nthread')
        super(Regression, self).__init__(**kwargs)


class BinaryClassifier(Base, xgboost.XGBClassifier):
    def __init__(
        self,
        max_depth=3,
        learning_rate=0.1,
        n_estimators=100,
        silent=True,
        objective='binary:logistic',
        booster='gbtree',
        n_jobs=-1,
        gamma=0,
        min_child_weight=1,
        max_delta_step=0,
        subsample=1,
        colsample_bytree=1,
        colsample_bylevel=1,
        reg_alpha=0,
        reg_lambda=1,
        scale_pos_weight=1,
        base_score=0.5,
        random_state=0,
        missing=None,
        eval_metric='logloss',
        **kwargs
    ):
        kwargs = locals()
        kwargs.pop('self')
        kwargs.pop('__class__', None)
        kwargs = dict(kwargs, **(kwargs.pop('kwargs', {})))
        if 'random_state' not in kwargs and 'seed' in kwargs:
            kwargs['random_state'] = kwargs.pop('seed')
        if 'n_jobs' not in kwargs and 'nthread' in kwargs:
            kwargs['n_jobs'] = kwargs.pop('nthread')
        super(BinaryClassifier, self).__init__(**kwargs)


MutliClassifier = BinaryClassifier
