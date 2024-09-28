'''
Author: George Zhao
Date: 2022-03-24 16:45:47
LastEditors: George Zhao
LastEditTime: 2022-06-23 21:24:04
Description: 
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''
import os
import sys
import pickle
import json
import math
import functools

import utils
import libpybiofeature

from . import model_optimite
from . import common

from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

n_jobs = 8

rf_param_o = {
    "n_estimators": [500 * i for i in range(1, 6, 2)],
    "criterion": ["gini", "entropy"],
    "max_features": [i for i in range(1, 41, 5)]
}

import numpy as np
import pandas as pd


def onehot_code_from_logist(
    dim: int,
    arr: np.ndarray
):
    return np.eye(dim)[np.argmax(arr, axis=-1)]


def load_from_bert_dir(
    ss: str,
    sa: str,
    id_list: list,
    len_: int = 100,
):
    result = np.concatenate(
        [
            onehot_code_from_logist(3, pickle.load(open(ss, "rb"))['value']),
            onehot_code_from_logist(2, pickle.load(open(sa, "rb"))['value']),
        ],
        axis=-1
    )[:, :len_, :]

    return pd.DataFrame(np.reshape(result, (result.shape[0], -1,)), index=id_list)


class T3SPs_Model(common.Model_Final):
    def __init__(self, cv, desc):
        super().__init__(cv, desc=desc)
        self.model = None
        self.scaler = None

    def tranmodel(self, f, l):
        super().tranmodel(f, l)
        self.scaler = MinMaxScaler()
        self.scaler.fit(f)

        self.model = model_optimite.rf_optim(
            param_o=rf_param_o,
            cv=self.cv
        ).best_fit(X=self.scaler.transform(f), y=l, verbose=-1, n_jobs=n_jobs)
        return self

    def predict(self, f):
        super().predict(f)
        return self.model.predict_proba(self.scaler.transform(f))


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
    for i in range(len(t_5C)):
        train_fl, test_fl = t_5C[i]

        model: T3SPs_Model = model_construct_funtion()
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
                "model": 'T3SPs',
                'desc': model.desc,
                'iteration': i,
                "size_of_data": size_of_data,
            }
        })

    model_result_set = utils.ds_preprocess.Five_Cross_Evaluation(
        model_result_set,
        pro_cutoff=0.5,
        mode='loo' if size_of_data == 'small' else None
    )

    if os.path.exists(os.path.split(path_to_model_pickle)[0]) == False:
        os.makedirs(os.path.split(path_to_model_pickle)[0])
    if os.path.exists(os.path.split(path_to_model_result)[0]) == False:
        os.makedirs(os.path.split(path_to_model_result)[0])

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

    s_t_p = load_from_bert_dir(
        ss=path_dict['ss'].format(db_type="t_p"),
        sa=path_dict['sa'].format(db_type="t_p"),
        id_list=seq_id_dict['t']['p']
    )
    s_t_n = load_from_bert_dir(
        ss=path_dict['ss'].format(db_type="t_n"),
        sa=path_dict['sa'].format(db_type="t_n"),
        id_list=seq_id_dict['t']['n']
    )
    s_v_p = load_from_bert_dir(
        ss=path_dict['ss'].format(db_type="v_p"),
        sa=path_dict['sa'].format(db_type="v_p"),
        id_list=seq_id_dict['v']['p']
    )
    s_v_n = load_from_bert_dir(
        ss=path_dict['ss'].format(db_type="v_n"),
        sa=path_dict['sa'].format(db_type="v_n"),
        id_list=seq_id_dict['v']['n']
    )

    expasy_t_p_f = libpybiofeature.libdataloader.expasy.get_expasy_t3sps(
        path_to_json=path_dict['expasy']['t']['p'],
        seq_id_list=seq_id_dict['t']['p']
    )
    expasy_t_n_f = libpybiofeature.libdataloader.expasy.get_expasy_t3sps(
        path_to_json=path_dict['expasy']['t']['n'],
        seq_id_list=seq_id_dict['t']['n']
    )
    expasy_v_p_f = libpybiofeature.libdataloader.expasy.get_expasy_t3sps(
        path_to_json=path_dict['expasy']['v']['p'],
        seq_id_list=seq_id_dict['v']['p']
    )
    expasy_v_n_f = libpybiofeature.libdataloader.expasy.get_expasy_t3sps(
        path_to_json=path_dict['expasy']['v']['n'],
        seq_id_list=seq_id_dict['v']['n']
    )

    t_p_f = utils.ds_preprocess.merge_pd_list([
        aac_t_p_f, s_t_p, expasy_t_p_f
    ])
    t_n_f = utils.ds_preprocess.merge_pd_list([
        aac_t_n_f, s_t_n, expasy_t_n_f
    ])
    v_p_f = utils.ds_preprocess.merge_pd_list([
        aac_v_p_f, s_v_p, expasy_v_p_f
    ])
    v_n_f = utils.ds_preprocess.merge_pd_list([
        aac_v_n_f, s_v_n, expasy_v_n_f
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

    os.makedirs(
        os.path.dirname(path_dict['model']['tt']['model_pickle']), exist_ok=True
    )

    Five_Cross_Get_model(
        model_construct_funtion=functools.partial(
            T3SPs_Model, desc=path_dict['model']['cv']['desc'], cv=model_cv),
        t_5C=t_5C,
        v_f=v_f,
        v_l=v_l,
        path_to_model_pickle=path_dict['model']['cv']['model_pickle'],
        path_to_model_result=path_dict['model']['cv']['model_result'],
        size_of_data=path_dict['model']['size']
    )
    Five_Cross_Get_model(
        model_construct_funtion=functools.partial(
            T3SPs_Model, desc=path_dict['model']['tt']['desc'], cv=model_cv),
        t_5C=(((t_f, t_l),
               (v_f, v_l)),),
        v_f=v_f,
        v_l=v_l,
        path_to_model_pickle=path_dict['model']['tt']['model_pickle'],
        path_to_model_result=path_dict['model']['tt']['model_result'],
        size_of_data=None
    )
