'''
Author: George Zhao
Date: 2022-02-24 22:25:02
LastEditors: George Zhao
LastEditTime: 2022-04-12 20:31:47
Description: 
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''
import sys
sys.path.append('src')
import functools

from . import common
import libpyga
import utils

import lightgbm as lgb
import numpy as np

from sklearn.semi_supervised import LabelPropagation
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from skopt import BayesSearchCV


class lgbm_optim(common.Model_Selection):
    def __init__(
        self,
        lgb_param_o,
        cv,
        default_param={'verbose': -1},
    ) -> None:
        super().__init__(param_o=lgb_param_o, cv=cv, default_param=default_param)
        self.model_0 = None
        # ? Optimited Para.
        self.para = self.default_param

        self.lgb_param_o = lgb_param_o

        self.gs = None
        pass

    def best_fit(self, X, y, step_param, n_jobs=1, verbose=-1):
        super().best_fit(X=X, y=y, verbose=verbose, n_jobs=n_jobs)
        self.para = self.default_param
        self.find_parm(X, y, verbose=verbose,
                       n_jobs=n_jobs, step_param=step_param)
        self.model_0 = lgb.LGBMClassifier(
            **self.para
        ).fit(X, y)
        return self

    def fit(self, X, y):
        super().fit(X=X, y=y)
        self.model_0 = lgb.LGBMClassifier(
            **self.para
        ).fit(X, y)
        return self

    def predict_proba(self, X):
        super().predict_proba(X=X)
        return self.model_0.predict_proba(X)[:, 1]

    def find_parm(self, X, y, verbose, n_jobs, step_param):
        super().find_parm(X=X, y=y, n_jobs=n_jobs, verbose=verbose)
        self.gs = libpyga.libpyga.bastion3_ga_cv(
            estimator=functools.partial(
                lgb.LGBMClassifier,
                **self.default_param
            ),
            param_o=self.lgb_param_o,
            n_jobs=n_jobs,
            verbose_bar=1,
            fit_parament={'verbose': False},
            cv=self.cv
        ).fit(X, y, step_param=step_param)
        self.para.update(self.gs.best_param_)
        return self


class skl_grid_optim(common.Model_Selection):
    def __init__(self, param_o, cv, default_param, model_constructor) -> None:
        super().__init__(param_o, cv, default_param)
        self.model_0 = None
        self.para = self.default_param
        self.model_constructor = model_constructor
        pass

    def best_fit(self, X, y, verbose, n_jobs):
        super().best_fit(X, y, verbose=verbose, n_jobs=n_jobs)
        self.para = self.default_param
        self.find_parm(X, y, verbose=verbose, n_jobs=n_jobs)
        self.fit(X, y)
        return self

    def fit(self, X, y):
        super().fit(X, y)
        self.model_0 = self.model_constructor(
            **self.para
        ).fit(X, y)
        return self

    def predict_proba(self, X):
        super().predict_proba(X)
        return self.model_0.predict_proba(X)[:, 1]

    def find_parm(self, X, y, n_jobs, verbose):
        super().find_parm(X, y, n_jobs, verbose)
        gs = GridSearchCV(
            estimator=functools.partial(
                self.model_constructor,
                **self.default_param
            )(),
            param_grid=self.param_o,
            cv=self.cv,
            scoring='roc_auc',
            n_jobs=n_jobs,
            verbose=verbose
        ).fit(X, y)
        self.para.update(gs.best_params_)
        return self


class mlp_optim(skl_grid_optim):
    def __init__(self, param_o, cv, default_param={
        'verbose': False,
        'early_stopping': True,
        'max_iter': 10000
    }) -> None:
        super().__init__(param_o=param_o, cv=cv,
                         default_param=default_param, model_constructor=MLPClassifier)
        pass


class xgb_optim(skl_grid_optim):
    def __init__(self, param_o, cv, default_param={
        'verbosity': 0,
        'n_jobs': 1
    }) -> None:
        super().__init__(param_o=param_o, cv=cv,
                         default_param=default_param, model_constructor=XGBClassifier)
        pass


class rf_optim(skl_grid_optim):
    def __init__(self, param_o, cv, default_param={
        'verbose': 0
    }) -> None:
        super().__init__(param_o=param_o, cv=cv,
                         default_param=default_param, model_constructor=RandomForestClassifier)
        pass


class lr_optim(skl_grid_optim):
    def __init__(self, param_o, cv, default_param={
        'verbose': 0,
        'max_iter': 1000
    }) -> None:
        super().__init__(param_o=param_o, cv=cv,
                         default_param=default_param, model_constructor=LogisticRegression)
        pass


class gbc_optim(skl_grid_optim):
    def __init__(self, param_o, cv, default_param={
        'verbose': 0
    }) -> None:
        super().__init__(param_o=param_o, cv=cv,
                         default_param=default_param, model_constructor=GradientBoostingClassifier)
        pass


class etc_optim(skl_grid_optim):
    def __init__(self, param_o, cv, default_param={
        'verbose': 0
    }) -> None:
        super().__init__(param_o=param_o, cv=cv,
                         default_param=default_param, model_constructor=ExtraTreesClassifier)
        pass


class svm_optim(skl_grid_optim):
    def __init__(self, param_o, cv, default_param={
        'verbose': False,
        'probability': True
    }) -> None:
        super().__init__(param_o=param_o, cv=cv,
                         default_param=default_param, model_constructor=SVC)
        pass


class knn_optim(skl_grid_optim):
    def __init__(self, param_o, cv, default_param={}) -> None:
        super().__init__(param_o=param_o, cv=cv,
                         default_param=default_param, model_constructor=KNeighborsClassifier)
        pass


class nb_optim(skl_grid_optim):
    def __init__(self, ) -> None:
        super().__init__(param_o=None, cv=None,
                         default_param={}, model_constructor=GaussianNB)
        pass

    def best_fit(self, X, y, verbose=None, n_jobs=None):
        return self.fit(X, y)


class lpa_model(skl_grid_optim):
    def __init__(self, param_o, cv, default_param={'n_jobs': -1}) -> None:
        super().__init__(param_o=param_o, cv=cv, default_param=default_param,
                         model_constructor=LabelPropagation)
        self.X = None
        self.Y = None
        pass

    def fit(self, X, y):
        self.X = X
        self.y = y
        return self

    def predict_proba(self, X):
        traning_f, traning_l = self.make_training_data(X)
        self.model_0 = LabelPropagation(
            **self.para).fit(traning_f, traning_l)
        return super().predict_proba(X)  # predict_proba( on model_0

    def make_training_data(self, pred_X):
        if self.X is None or self.y is None:
            raise RuntimeError(f'Fit model First.')
        return utils.ds_preprocess.make_merge(
            t_p_f=self.X,
            t_p_l=self.y,
            t_n_f=pred_X,
            t_n_l=np.ones(shape=pred_X.shape[0]) * (-1)
        )
