'''
Author: George Zhao
Date: 2022-03-18 17:43:52
LastEditors: George Zhao
LastEditTime: 2022-06-24 10:22:49
Description: 
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''
import os
import json
import pickle
import math
import functools

from . import model_optimite
from . import common
import utils
import libpybiofeature
from libpybiofeature.pssmcode import get_all_task_feature

import numpy as np

from sklearn.model_selection import ShuffleSplit

n_jobs = 16

svm_param_o = {
    "gamma": [10**i for i in range(-6, 6 + 1)],
    "C": [10**i for i in range(-6, 6 + 1)],
}

svm_optim: model_optimite.svm_optim = functools.partial(
    model_optimite.svm_optim,
    default_param={
        'verbose': False,
        'probability': True,
        'kernel': 'rbf',
    }
)


class T4EffPred_Model(common.Model_Final):
    def __init__(self, cv, desc):
        super().__init__(cv, desc=desc)
        self.model = None

    def tranmodel(self, f, l):
        super().tranmodel(f, l)
        self.model = svm_optim(
            param_o=svm_param_o,
            cv=self.cv
        ).best_fit(X=f, y=l, verbose=-1, n_jobs=n_jobs)
        return self

    def predict(self, f):
        super().predict(f)
        return self.model.predict_proba(f)


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

        model: T4EffPred_Model = model_construct_funtion()
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
                "model": 'T4EffPred',
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

    possum_index_dict = None
    with open(path_dict['possum']['index'], 'r', encoding='UTF-8') as f:
        possum_index_dict = json.load(f)

    possum_t_p_f, possum_t_n_f, possum_v_p_f, possum_v_n_f = libpybiofeature.pssmcode.get_all_task_feature(
        possum_index_dict=possum_index_dict,
        path_to_json_seq_id=path_dict['seq_id'],
        feature_name_list=['pssm_ac', 'pssm_composition'],
        path_to_fasta_pattern=path_dict['possum']['fasta_pattern'],
        path_to_with_pattern=path_dict['possum']['pssm_fdb_pattern']
    )

    # %%
    t_p_f = utils.ds_preprocess.merge_pd_list([
        aac_t_p_f, dac_t_p_f, possum_t_p_f
    ])
    t_n_f = utils.ds_preprocess.merge_pd_list([
        aac_t_n_f, dac_t_n_f, possum_t_n_f
    ])
    v_p_f = utils.ds_preprocess.merge_pd_list([
        aac_v_p_f, dac_v_p_f, possum_v_p_f
    ])
    v_n_f = utils.ds_preprocess.merge_pd_list([
        aac_v_n_f, dac_v_n_f, possum_v_n_f
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

    Five_Cross_Get_model(
        model_construct_funtion=functools.partial(
            T4EffPred_Model, desc=path_dict['model']['cv']['desc'], cv=model_cv),
        t_5C=t_5C,
        v_f=v_f,
        v_l=v_l,
        path_to_model_pickle=path_dict['model']['cv']['model_pickle'],
        path_to_model_result=path_dict['model']['cv']['model_result'],
        size_of_data=path_dict['model']['size']
    )
    Five_Cross_Get_model(
        model_construct_funtion=functools.partial(
            T4EffPred_Model, desc=path_dict['model']['tt']['desc'], cv=model_cv),
        t_5C=(((t_f, t_l),
               (v_f, v_l)),),
        v_f=v_f,
        v_l=v_l,
        path_to_model_pickle=path_dict['model']['tt']['model_pickle'],
        path_to_model_result=path_dict['model']['tt']['model_result'],
        size_of_data=None
    )
