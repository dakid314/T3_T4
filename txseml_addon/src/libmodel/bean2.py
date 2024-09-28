'''
Author: George Zhao
Date: 2022-03-07 17:36:37
LastEditors: George Zhao
LastEditTime: 2022-06-24 10:23:49
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

from sklearn.model_selection import ShuffleSplit

svm_param_o = [{
    "kernel": ["linear", ],
    "C": [10**i for i in range(-6, 6 + 1)],
}, {
    "kernel": ["poly", "rbf", "sigmoid"],
    "gamma": [10**i for i in range(-6, 6 + 1)],
    "C": [10**i for i in range(-6, 6 + 1)],
}]

n_jobs = 16


class BEAN2_Model(common.Model_Final):
    def __init__(self, cv, desc):
        super().__init__(cv, desc=desc)
        self.model = None

    def tranmodel(self, f, l):
        super().tranmodel(f, l)
        self.model = model_optimite.svm_optim(
            param_o=svm_param_o,
            cv=self.cv
        ).best_fit(X=f, y=l, verbose=-1, n_jobs=n_jobs)
        return self

    def predict(self, f):
        super().predict(f)
        return self.model.predict_proba(f)


def Five_Cross_Get_model(
    model_construct_funtion,
    Five_Cross_set: list,
    t_model_v_f,
    t_model_v_l,
    path_to_model_pickle: str,
    path_to_model_result: str,
        size_of_data: str,):
    model_set = list()
    model_result_set = list()
    for i in range(len(Five_Cross_set)):
        train, test = Five_Cross_set[i]

        validated_f = t_model_v_f

        model: BEAN2_Model = model_construct_funtion()
        model.tranmodel(train[0], train[1])
        model_set.append(model)
        model_result_set.append({
            "training": {
                "origin": {
                    f'pred': list(model.predict(train[0])),
                    f'label': list(train[1])},
                "evaluation": {
                }, "option": {
                }
            },
            "testing": {
                "origin": {
                    f'pred': list(model.predict(test[0])),
                    f'label': list(test[1])},
                "evaluation": {
                }, "option": {
                }
            },
            "validated": {
                "origin": {
                    f'pred': list(model.predict(validated_f)),
                    f'label': list(t_model_v_l)},
                "evaluation": {
                }, "option": {
                }
            },
            "detail": {
                "model": 'BEAN2',
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
    id_dict = None
    with open(path_dict['possum']['index'], 'r', encoding='UTF-8') as f:
        id_dict = json.load(f)

    vector_t_p_f = libpybiofeature.HH_CKSAAP.build_HH_CKSAAP_feature(
        path_dict['fasta']['t']['p'],
        id_dict['data']['t_p'],
        path_dict['possum']['pssm_db_pattern'],
        seq_id_list=seq_id_dict['t']['p'],
        desc='t_p',
    )
    vector_t_n_f = libpybiofeature.HH_CKSAAP.build_HH_CKSAAP_feature(
        path_dict['fasta']['t']['n'],
        id_dict['data']['t_n'],
        path_dict['possum']['pssm_db_pattern'],
        seq_id_list=seq_id_dict['t']['n'],
        desc='t_n',
    )

    vector_v_p_f = libpybiofeature.HH_CKSAAP.build_HH_CKSAAP_feature(
        path_dict['fasta']['v']['p'],
        id_dict['data']['v_p'],
        path_dict['possum']['pssm_db_pattern'],
        seq_id_list=seq_id_dict['v']['p'],
        desc='v_p',
    )
    vector_v_n_f = libpybiofeature.HH_CKSAAP.build_HH_CKSAAP_feature(
        path_dict['fasta']['v']['n'],
        id_dict['data']['v_n'],
        path_dict['possum']['pssm_db_pattern'],
        seq_id_list=seq_id_dict['v']['n'],
        desc='v_n',
    )

    # %%
    vector_t_p_l = utils.ds_preprocess.make_binary_label(
        size=vector_t_p_f.shape[0],
        label=True
    )

    vector_t_n_l = utils.ds_preprocess.make_binary_label(
        size=vector_t_n_f.shape[0],
        label=False
    )

    vector_v_p_l = utils.ds_preprocess.make_binary_label(
        size=vector_v_p_f.shape[0],
        label=True
    )

    vector_v_n_l = utils.ds_preprocess.make_binary_label(
        size=vector_v_n_f.shape[0],
        label=False
    )

    # %%
    t_5C = utils.ds_preprocess.get_5C_data(
        shape_to_chiose=vector_t_p_f.shape[0],
        t_p_f=vector_t_p_f,
        t_p_l=vector_t_p_l,
        t_n_f=vector_t_n_f,
        t_n_l=vector_t_n_l,
        shufflesplit={
            'shufflesplit_index_file': path_dict['shufflesplit_index_file']
        } if path_dict['model']['size'] == 'small' else None
    )

    t_model_t_f, t_model_t_l = utils.ds_preprocess.make_merge(
        t_p_f=vector_t_p_f,
        t_p_l=vector_t_p_l,
        t_n_f=vector_t_n_f,
        t_n_l=vector_t_n_l
    )

    t_model_v_f, t_model_v_l = utils.ds_preprocess.make_merge(
        t_p_f=vector_v_p_f,
        t_p_l=vector_v_p_l,
        t_n_f=vector_v_n_f,
        t_n_l=vector_v_n_l
    )

    model_cv = 10
    if path_dict['model']['size'] == 'small':
        model_cv = ShuffleSplit(n_splits=10, test_size=0.5, random_state=0)

    Five_Cross_Get_model(
        model_construct_funtion=functools.partial(
            BEAN2_Model, desc=path_dict['model']['cv']['desc'], cv=model_cv),
        Five_Cross_set=t_5C,
        t_model_v_f=t_model_v_f,
        t_model_v_l=t_model_v_l,
        path_to_model_pickle=path_dict['model']['cv']['model_pickle'],
        path_to_model_result=path_dict['model']['cv']['model_result'],
        size_of_data=path_dict['model']['size']
    )
    Five_Cross_Get_model(
        model_construct_funtion=functools.partial(
            BEAN2_Model, desc=path_dict['model']['tt']['desc'], cv=model_cv),
        Five_Cross_set=(((t_model_t_f, t_model_t_l),
                         (t_model_v_f, t_model_v_l)),),
        t_model_v_f=t_model_v_f,
        t_model_v_l=t_model_v_l,
        path_to_model_pickle=path_dict['model']['tt']['model_pickle'],
        path_to_model_result=path_dict['model']['tt']['model_result'],
        size_of_data=None
    )
