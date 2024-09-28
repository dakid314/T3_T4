'''
Author: George Zhao
Date: 2022-03-13 13:17:12
LastEditors: George Zhao
LastEditTime: 2022-06-26 16:48:24
Description: 
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''
import os
import json
import pickle
import functools

from . import model_optimite
from . import common
import utils
import libpybiofeature
from libpybiofeature.pssmcode import get_all_task_feature

import numpy as np

from sklearn.model_selection import ShuffleSplit
from sklearn.naive_bayes import GaussianNB

n_jobs = 16
allow_load_from_disk = False
submodel_choise = None
submodel_desc = None
only_tt_model = False

svm_param_o = [{
    "kernel": ["linear", ],
    "C": [10**i for i in range(-6, 6 + 1)],
}, {
    "kernel": ["poly", "rbf", "sigmoid"],
    "gamma": [10**i for i in range(-6, 6 + 1)],
    "C": [10**i for i in range(-6, 6 + 1)],
}]

etc_param_o = {
    "n_estimators": [2 ** i for i in range(4, 8)],
    "criterion": ["gini", "entropy"]
}

rf_param_o = {
    "n_estimators": [2 ** i for i in range(4, 8)],
    "criterion": ["gini", "entropy"]
}

gbc_param_o = [{
    "loss": ['exponential', 'deviance'],
    'learning_rate': [10**i for i in range(-5, 5)],
    "n_estimators": [2 ** i for i in range(4, 8)],
    "criterion": ['friedman_mse', 'mse', 'squared_error'],
}, ]

xgb_param_o = [{
    "eta": [10 ** i for i in range(-4, 1, 1)],
    "n_estimators": [2 ** i for i in range(4, 13, 2)],
    "reg_lambda": [i * 0.3 for i in range(0, 11, 1)],
    "booster": ['gblinear', ],
}, {
    "eta": [10 ** i for i in range(-4, 1, 1)],
    "n_estimators": [2 ** i for i in range(4, 7, 1)],
    "gamma": [10**i for i in range(-5, 4, 1)],
    "max_depth": [i for i in range(2, 7, 1)],
    "booster": ['gbtree', ]
}, {
    "eta": [10 ** i for i in range(-4, 1, 1)],
    "n_estimators": [2 ** i for i in range(4, 7, 1)],
    "gamma": [10**i for i in range(-5, 4, 1)],
    "max_depth": [i for i in range(2, 7, 1)],
    "booster": ['dart', ]
}]

lr_param_o = [{
    "penalty": ['none'],
    "solver": ['newton-cg', 'lbfgs', 'sag', 'saga']
}, {
    "penalty": ['elasticnet', ],
    'l1_ratio': [0.1 * i for i in range(0, 11, 1)],
    "C": [10**i for i in range(-6, 6)],
    "solver": ['saga']
}, ]

knn_optim_o = {
    'n_neighbors': [i for i in range(1, 6)]
}


class PredT4SEStacker_Model(common.Model_Final):
    def __init__(self, cv, desc):
        super().__init__(cv, desc=desc)
        self.model_svm = None
        self.model_nb = None
        self.model_knn = None
        self.model_lr = None
        self.model_rf = None
        self.model_etc = None
        self.model_gbm = None
        self.model_xgb = None
        self.model_meta = None
        pass

    def tranmodel(self, f, l):
        super().tranmodel(f, l)
        self._tranmodel_base_step1(f, l)
        self._tranmodel_base_step2(
            self._predict_step1(f), l
        )
        return self

    def predict(self, f):
        super().predict(f)
        if submodel_choise is not None:
            if isinstance(submodel_choise, int) == False or submodel_choise < 0 or submodel_choise >= 8:
                raise ValueError(f"Wrong submodel_choise: {submodel_choise}")
            return self._predict_step1(f)[:, submodel_choise]

        return self._predict_step2(
            self._predict_step1(f)
        )

    def _tranmodel_base_step1(
        self,
        f,
        l,
    ):
        self.model_svm = model_optimite.svm_optim(
            param_o=svm_param_o,
            cv=self.cv
        ).best_fit(f, l, verbose=-1, n_jobs=n_jobs)

        self.model_nb = GaussianNB().fit(
            f, l
        )

        self.model_knn = model_optimite.knn_optim(
            param_o=knn_optim_o,
            cv=self.cv
        ).best_fit(f, l, verbose=-1, n_jobs=n_jobs)

        self.model_lr = model_optimite.lr_optim(
            param_o=lr_param_o,
            cv=self.cv
        ).best_fit(f, l, verbose=-1, n_jobs=n_jobs)

        self.model_rf = model_optimite.rf_optim(
            param_o=rf_param_o,
            cv=self.cv
        ).best_fit(f, l, verbose=-1, n_jobs=n_jobs)

        self.model_etc = model_optimite.etc_optim(
            param_o=etc_param_o,
            cv=self.cv
        ).best_fit(f, l, verbose=-1, n_jobs=n_jobs)

        self.model_gbm = model_optimite.gbc_optim(
            param_o=gbc_param_o,
            cv=self.cv
        ).best_fit(f, l, verbose=-1, n_jobs=n_jobs)

        self.model_xgb = model_optimite.xgb_optim(
            param_o=xgb_param_o,
            cv=self.cv
        ).best_fit(f, l, verbose=-1, n_jobs=n_jobs)

        return self

    def _tranmodel_base_step2(self, f, l,):
        self.model_meta = model_optimite.lr_optim(
            param_o=lr_param_o,
            cv=self.cv
        ).best_fit(f, l, verbose=-1, n_jobs=n_jobs)

        return self

    def _predict_step1(self, f,):
        return np.stack([
            self.model_svm.predict_proba(f),
            self.model_nb.predict_proba(f)[:, 1],
            self.model_knn.predict_proba(f),
            self.model_lr.predict_proba(f),
            self.model_rf.predict_proba(f),
            self.model_etc.predict_proba(f),
            self.model_gbm.predict_proba(f),
            self.model_xgb.predict_proba(f),
        ]).T

    def _predict_step2(
            self,
            f,):
        return self.model_meta.predict_proba(f)


def Five_Cross_Get_model(
    model_construct_funtion,
    t_5C: list,
    v_f,
    v_l,
    path_to_model_pickle: str,
    path_to_model_result: str,
    size_of_data: str,
):
    model_set = list()
    model_result_set = list()
    loaded_from_disk = False
    if allow_load_from_disk == True:
        if os.path.exists(path_to_model_pickle) == True:
            with open(path_to_model_pickle, 'br') as f:
                model_set = pickle.load(f)
            loaded_from_disk = True
    submodel_desc_ = submodel_desc
    if submodel_choise is not None:
        if submodel_desc_ is None:
            submodel_desc_ = submodel_choise
        path_tmp = os.path.split(path_to_model_result)
        path_to_model_result = os.path.join(
            path_tmp[0], submodel_desc_ + '_' + path_tmp[1])
    for i in range(len(t_5C)):
        train_fl, test_fl = t_5C[i]

        model: PredT4SEStacker_Model = None
        if loaded_from_disk == True:
            model = model_set[i]
        else:
            model: PredT4SEStacker_Model = model_construct_funtion()
            model.tranmodel(
                train_fl[0], train_fl[1]
            )
            model_set.append(model)

        model_result_set.append({
            "training": {
                "origin": {
                    f'pred': list(model.predict(
                        train_fl[0]
                    )),
                    f'label': list(train_fl[1])},
                "evaluation": {
                }, "option": {
                }
            },
            "testing": {
                "origin": {
                    f'pred': list(model.predict(
                        test_fl[0]
                    )),
                    f'label': list(test_fl[1])},
                "evaluation": {
                }, "option": {
                }
            },
            "validated": {
                "origin": {
                    f'pred': list(model.predict(
                        v_f
                    )),
                    f'label': list(v_l)},
                "evaluation": {
                }, "option": {
                }
            },
            "detail": {
                "model": 'PredT4SEStacker',
                'desc': model.desc,
                'iteration': i,
                "size_of_data": size_of_data,
            }
        })
        if submodel_choise is not None:
            model_result_set[-1]['detail']['model'] += f'_{submodel_desc_}'

    model_result_set = utils.ds_preprocess.Five_Cross_Evaluation(
        model_result_set,
        pro_cutoff=0.5,
        mode='loo' if size_of_data == 'small' else None
    )

    if os.path.exists(os.path.split(path_to_model_pickle)[0]) == False:
        os.makedirs(os.path.split(path_to_model_pickle)[0])
    if os.path.exists(os.path.split(path_to_model_result)[0]) == False:
        os.makedirs(os.path.split(path_to_model_result)[0])

    if loaded_from_disk == False:
        with open(path_to_model_pickle, 'bw+') as f:
            pickle.dump(model_set, f)

    with open(path_to_model_result, 'w+', encoding='UTF-8') as f:
        json.dump(model_result_set, f, cls=utils.ds_preprocess.MyEncoder)

    return model_set, model_result_set


def research(path_dict: dict):
    seq_id_dict = None
    with open(path_dict['seq_id'], 'r', encoding='UTF-8') as f:
        seq_id_dict = json.load(f)

    possum_index_dict = None
    with open(path_dict['possum']['index'], 'r', encoding='UTF-8') as f:
        possum_index_dict = json.load(f)

    possum_t_p_f, possum_t_n_f, possum_v_p_f, possum_v_n_f = get_all_task_feature(
        possum_index_dict=possum_index_dict,
        path_to_json_seq_id=path_dict['seq_id'],
        feature_name_list=['pssm_composition', ],
        path_to_fasta_pattern=path_dict['possum']['fasta_pattern'],
        path_to_with_pattern=path_dict['possum']['pssm_fdb_pattern']
    )

    t_p_l = utils.ds_preprocess.make_binary_label(
        size=possum_t_p_f.shape[0], label=True)

    t_n_l = utils.ds_preprocess.make_binary_label(
        size=possum_t_n_f.shape[0], label=False)

    v_p_l = utils.ds_preprocess.make_binary_label(
        size=possum_v_p_f.shape[0], label=True)

    v_n_l = utils.ds_preprocess.make_binary_label(
        size=possum_v_n_f.shape[0], label=False)

    t_5C = utils.ds_preprocess.get_5C_data(
        shape_to_chiose=possum_t_p_f.shape[0],
        t_p_f=possum_t_p_f,
        t_p_l=t_p_l,
        t_n_f=possum_t_n_f,
        t_n_l=t_n_l,
        shufflesplit={
            'shufflesplit_index_file': path_dict['shufflesplit_index_file']
        } if path_dict['model']['size'] == 'small' else None
    )

    t_f, t_l = utils.ds_preprocess.make_merge(
        t_p_f=possum_t_p_f,
        t_p_l=t_p_l,
        t_n_f=possum_t_n_f,
        t_n_l=t_n_l
    )

    v_f, v_l = utils.ds_preprocess.make_merge(
        t_p_f=possum_v_p_f,
        t_p_l=v_p_l,
        t_n_f=possum_v_n_f,
        t_n_l=v_n_l
    )

    model_cv = 10
    if path_dict['model']['size'] == 'small':
        model_cv = ShuffleSplit(n_splits=10, test_size=0.5, random_state=0)

    if only_tt_model == False:
        Five_Cross_Get_model(
            model_construct_funtion=functools.partial(
                PredT4SEStacker_Model, desc=path_dict['model']['cv']['desc'], cv=model_cv),
            t_5C=t_5C,
            v_f=v_f,
            v_l=v_l,
            path_to_model_pickle=path_dict['model']['cv']['model_pickle'],
            path_to_model_result=path_dict['model']['cv']['model_result'],
            size_of_data=path_dict['model']['size']
        )
    Five_Cross_Get_model(
        model_construct_funtion=functools.partial(
            PredT4SEStacker_Model, desc=path_dict['model']['tt']['desc'], cv=model_cv),
        t_5C=(((t_f, t_l),
               (v_f, v_l)),),
        v_f=v_f,
        v_l=v_l,
        path_to_model_pickle=path_dict['model']['tt']['model_pickle'],
        path_to_model_result=path_dict['model']['tt']['model_result'],
        size_of_data=None
    )
