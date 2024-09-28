'''
Author: George Zhao
Date: 2022-03-12 15:02:23
LastEditors: George Zhao
LastEditTime: 2022-06-26 21:38:06
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

import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.naive_bayes import GaussianNB

from skopt.space import Real, Categorical, Integer

n_jobs = 30

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

rf_param_o = {
    "n_estimators": [2 ** i for i in range(4, 15)],
}

mlp_param_o = {
    "solver": ['adam', 'sgd', 'lbfgs'],
    "activation": ['relu', 'tanh', 'logistic', 'identity'],
    'hidden_layer_sizes': [(100,), (50,), (25,), (10,), ],
    "learning_rate": ['adaptive', 'invscaling', 'constant']
}

knn_optim_o = {
    'n_neighbors': [i for i in range(1, 6)]
}

rf_optim: model_optimite.rf_optim = functools.partial(
    model_optimite.rf_optim,
    default_param={
        'verbose': 0,
        'criterion': 'gini'
    }
)

import pandas as pd


def load_from_bert_dir(
    path: str,
    id_list: list,
    len_: int = 100,
):
    result = pickle.load(open(path, "rb"))['value'][:, :len_, :]

    return pd.DataFrame(np.reshape(result, (result.shape[0], -1,)), index=id_list)


class PyPredT6_Model(common.Model_Final):
    def __init__(self, cv, desc):
        super().__init__(cv, desc=desc)
        self.model_svm = None
        self.model_nb = None
        self.model_mlp = None
        self.model_knn = None
        self.model_rf = None
        pass

    def tranmodel(self, f, l):
        super().tranmodel(f, l)

        self.model_mlp = model_optimite.mlp_optim(
            param_o=mlp_param_o,
            cv=self.cv,
        ).best_fit(
            f, l, verbose=-1, n_jobs=n_jobs
        )

        self.model_svm = model_optimite.svm_optim(
            param_o=svm_param_o,
            cv=self.cv,
        ).best_fit(
            f, l, verbose=-1, n_jobs=n_jobs
        )

        self.model_nb = GaussianNB().fit(
            f, l
        )

        self.model_knn = model_optimite.knn_optim(
            param_o=knn_optim_o,
            cv=self.cv
        ).best_fit(
            f, l, verbose=-1, n_jobs=n_jobs
        )

        self.model_rf = rf_optim(
            param_o=rf_param_o,
            cv=self.cv,
        ).best_fit(
            f, l, verbose=-1, n_jobs=n_jobs
        )

        return self

    def predict(self, f):
        super().predict(f)
        result = np.stack([
            self.model_svm.predict_proba(f),
            self.model_nb.predict_proba(f)[:, 1],
            self.model_knn.predict_proba(f),
            self.model_rf.predict_proba(f),
            self.model_mlp.predict_proba(f),
        ]).T
        if submodel_choise is None:
            result = (result >= 0.5)
            return result.sum(axis=1) / 5
        if isinstance(submodel_choise, int) == False or submodel_choise < 0 or submodel_choise >= 5:
            raise ValueError(f"Wrong submodel_choise: {submodel_choise}")
        return result[:, submodel_choise]


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

        model: PyPredT6_Model = None
        if loaded_from_disk == True:
            model = model_set[i]
        else:
            model: PyPredT6_Model = model_construct_funtion()
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
                "model": 'PyPredT6',
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

    aac_t_p_f = libpybiofeature.featurebuilder.build_acc_feature(
        path_to_fasta=path_dict['fasta']['t']['p'],
        seq_id_list=seq_id_dict['t']['p'],
        desc='t_p'
    )

    aac_t_n_f = libpybiofeature.featurebuilder.build_acc_feature(
        path_to_fasta=path_dict['fasta']['t']['n'],
        seq_id_list=seq_id_dict['t']['n'],
        desc='t_n'
    )

    aac_v_p_f = libpybiofeature.featurebuilder.build_acc_feature(
        path_to_fasta=path_dict['fasta']['v']['p'],
        seq_id_list=seq_id_dict['v']['p'],
        desc='v_p'
    )

    aac_v_n_f = libpybiofeature.featurebuilder.build_acc_feature(
        path_to_fasta=path_dict['fasta']['v']['n'],
        seq_id_list=seq_id_dict['v']['n'],
        desc='v_n'
    )

    dac_t_p_f = libpybiofeature.featurebuilder.build_dac_feature(
        path_to_fasta=path_dict['fasta']['t']['p'],
        seq_id_list=seq_id_dict['t']['p'],
        desc='t_p'
    )

    dac_t_n_f = libpybiofeature.featurebuilder.build_dac_feature(
        path_to_fasta=path_dict['fasta']['t']['n'],
        seq_id_list=seq_id_dict['t']['n'],
        desc='t_n'
    )

    dac_v_p_f = libpybiofeature.featurebuilder.build_dac_feature(
        path_to_fasta=path_dict['fasta']['v']['p'],
        seq_id_list=seq_id_dict['v']['p'],
        desc='v_p'
    )

    dac_v_n_f = libpybiofeature.featurebuilder.build_dac_feature(
        path_to_fasta=path_dict['fasta']['v']['n'],
        seq_id_list=seq_id_dict['v']['n'],
        desc='v_n'
    )

    etpp_t_p_f = libpybiofeature.featurebuilder.build_etpp_feature(
        path_to_fasta=path_dict['fasta']['t']['p'],
        seq_id_list=seq_id_dict['t']['p'],
        desc='t_p'
    )

    etpp_t_n_f = libpybiofeature.featurebuilder.build_etpp_feature(
        path_to_fasta=path_dict['fasta']['t']['n'],
        seq_id_list=seq_id_dict['t']['n'],
        desc='t_n'
    )

    etpp_v_p_f = libpybiofeature.featurebuilder.build_etpp_feature(
        path_to_fasta=path_dict['fasta']['v']['p'],
        seq_id_list=seq_id_dict['v']['p'],
        desc='v_p'
    )

    etpp_v_n_f = libpybiofeature.featurebuilder.build_etpp_feature(
        path_to_fasta=path_dict['fasta']['v']['n'],
        seq_id_list=seq_id_dict['v']['n'],
        desc='v_n'
    )

    cj_t_p_f = libpybiofeature.featurebuilder.build_conjoint_td_feature(
        path_to_fasta=path_dict['fasta']['t']['p'],
        seq_id_list=seq_id_dict['t']['p'],
        desc='t_p'
    )

    cj_t_n_f = libpybiofeature.featurebuilder.build_conjoint_td_feature(
        path_to_fasta=path_dict['fasta']['t']['n'],
        seq_id_list=seq_id_dict['t']['n'],
        desc='t_n'
    )

    cj_v_p_f = libpybiofeature.featurebuilder.build_conjoint_td_feature(
        path_to_fasta=path_dict['fasta']['v']['p'],
        seq_id_list=seq_id_dict['v']['p'],
        desc='v_p'
    )

    cj_v_n_f = libpybiofeature.featurebuilder.build_conjoint_td_feature(
        path_to_fasta=path_dict['fasta']['v']['n'],
        seq_id_list=seq_id_dict['v']['n'],
        desc='v_n'
    )

    aacpro_t_p_f = load_from_bert_dir(
        path=path_dict['sa'].format(db_type="t_p"),
        id_list=seq_id_dict['t']['p']
    )
    aacpro_t_n_f = load_from_bert_dir(
        path=path_dict['sa'].format(db_type="t_n"),
        id_list=seq_id_dict['t']['n']
    )
    aacpro_v_p_f = load_from_bert_dir(
        path=path_dict['sa'].format(db_type="v_p"),
        id_list=seq_id_dict['v']['p']
    )
    aacpro_v_n_f = load_from_bert_dir(
        path=path_dict['sa'].format(db_type="v_n"),
        id_list=seq_id_dict['v']['n']
    )

    ssapro_t_p_f = load_from_bert_dir(
        path=path_dict['ss'].format(db_type="t_p"),
        id_list=seq_id_dict['t']['p']
    )
    ssapro_t_n_f = load_from_bert_dir(
        path=path_dict['ss'].format(db_type="t_n"),
        id_list=seq_id_dict['t']['n']
    )
    ssapro_v_p_f = load_from_bert_dir(
        path=path_dict['ss'].format(db_type="v_p"),
        id_list=seq_id_dict['v']['p']
    )
    ssapro_v_n_f = load_from_bert_dir(
        path=path_dict['ss'].format(db_type="v_n"),
        id_list=seq_id_dict['v']['n']
    )

    t_p_f = utils.ds_preprocess.merge_pd_list([
        aac_t_p_f, dac_t_p_f, etpp_t_p_f, cj_t_p_f, aacpro_t_p_f, ssapro_t_p_f
    ])
    t_n_f = utils.ds_preprocess.merge_pd_list([
        aac_t_n_f, dac_t_n_f, etpp_t_n_f, cj_t_n_f, aacpro_t_n_f, ssapro_t_n_f
    ])
    v_p_f = utils.ds_preprocess.merge_pd_list([
        aac_v_p_f, dac_v_p_f, etpp_v_p_f, cj_v_p_f, aacpro_v_p_f, ssapro_v_p_f
    ])
    v_n_f = utils.ds_preprocess.merge_pd_list([
        aac_v_n_f, dac_v_n_f, etpp_v_n_f, cj_v_n_f, aacpro_v_n_f, ssapro_v_n_f
    ])

    t_p_l = utils.ds_preprocess.make_binary_label(
        size=t_p_f.shape[0], label=True)
    t_n_l = utils.ds_preprocess.make_binary_label(
        size=t_n_f.shape[0], label=False)

    v_p_l = utils.ds_preprocess.make_binary_label(
        size=v_p_f.shape[0], label=True)
    v_n_l = utils.ds_preprocess.make_binary_label(
        size=v_n_f.shape[0], label=False)

    t_5C = utils.ds_preprocess.get_5C_data(
        shape_to_chiose=t_p_f.shape[0],
        t_p_f=t_p_f,
        t_p_l=t_p_l,
        t_n_f=t_n_f,
        t_n_l=t_n_l,
        shufflesplit={
            'shufflesplit_index_file': path_dict['shufflesplit_index_file']
        } if path_dict['model']['size'] == 'small' else None
    )

    t_f, t_l = utils.ds_preprocess.make_merge(
        t_p_f=t_p_f,
        t_p_l=t_p_l,
        t_n_f=t_n_f,
        t_n_l=t_n_l,
    )

    v_f, v_l = utils.ds_preprocess.make_merge(
        t_p_f=v_p_f,
        t_p_l=v_p_l,
        t_n_f=v_n_f,
        t_n_l=v_n_l,
    )

    model_cv = 10
    if path_dict['model']['size'] == 'small':
        model_cv = ShuffleSplit(n_splits=10, test_size=0.5, random_state=0)

    if only_tt_model == False:
        Five_Cross_Get_model(
            model_construct_funtion=functools.partial(
                PyPredT6_Model, desc=path_dict['model']['cv']['desc'], cv=model_cv),
            t_5C=t_5C,
            v_f=v_f,
            v_l=v_l,
            path_to_model_pickle=path_dict['model']['cv']['model_pickle'],
            path_to_model_result=path_dict['model']['cv']['model_result'],
            size_of_data=path_dict['model']['size']
        )
    Five_Cross_Get_model(
        model_construct_funtion=functools.partial(
            PyPredT6_Model, desc=path_dict['model']['tt']['desc'], cv=model_cv),
        t_5C=(((t_f, t_l),
               (v_f, v_l)),),
        v_f=v_f,
        v_l=v_l,
        path_to_model_pickle=path_dict['model']['tt']['model_pickle'],
        path_to_model_result=path_dict['model']['tt']['model_result'],
        size_of_data=None
    )
