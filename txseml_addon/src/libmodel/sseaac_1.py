'''
Author: George Zhao
Date: 2022-03-11 18:55:17
LastEditors: George Zhao
LastEditTime: 2022-08-06 15:33:27
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
import numpy as np
import pandas as pd

svm_param_o = {
    "gamma": [10**i for i in range(-6, 6 + 1)],
    "C": [10**i for i in range(-6, 6 + 1)],
}

n_jobs = 30

svm_optim: model_optimite.svm_optim = functools.partial(
    model_optimite.svm_optim,
    default_param={
        'verbose': False,
        'probability': True,
        'kernel': 'rbf',
    }
)


def ac_code_from_logist(
    dim: int,
    arr: np.ndarray
):
    return np.array([
        "A", "B", "C"
    ][:dim])[np.argmax(arr, axis=-1)]


def load_from_bert_dir(
    path: str,
    id_list: list,
    dim: int,
    len_: int = 100,
):
    result = ac_code_from_logist(dim, pickle.load(
        open(path, "rb"))['value'])[:, :len_]

    result = [
        libpybiofeature.AC.AAC(
            seq_aa="".join(item),
            aaorder=["A", "B", "C"][:dim]
        )
        for item in result
    ]

    return pd.DataFrame(result, index=id_list)


class SSEAAC_Model(common.Model_Final):
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

        model: SSEAAC_Model = model_construct_funtion()
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
                "model": 'SSEAAC',
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

    aacpro_t_p_f = load_from_bert_dir(
        path=path_dict['sa'].format(db_type="t_p"),
        id_list=seq_id_dict['t']['p'],
        dim=2
    )
    aacpro_t_n_f = load_from_bert_dir(
        path=path_dict['sa'].format(db_type="t_n"),
        id_list=seq_id_dict['t']['n'],
        dim=2
    )
    aacpro_v_p_f = load_from_bert_dir(
        path=path_dict['sa'].format(db_type="v_p"),
        id_list=seq_id_dict['v']['p'],
        dim=2
    )
    aacpro_v_n_f = load_from_bert_dir(
        path=path_dict['sa'].format(db_type="v_n"),
        id_list=seq_id_dict['v']['n'],
        dim=2
    )
    psipred_t_p_f = load_from_bert_dir(
        path=path_dict['ss'].format(db_type="t_p"),
        id_list=seq_id_dict['t']['p'],
        dim=3
    )
    psipred_t_n_f = load_from_bert_dir(
        path=path_dict['ss'].format(db_type="t_n"),
        id_list=seq_id_dict['t']['n'],
        dim=3
    )
    psipred_v_p_f = load_from_bert_dir(
        path=path_dict['ss'].format(db_type="v_p"),
        id_list=seq_id_dict['v']['p'],
        dim=3
    )
    psipred_v_n_f = load_from_bert_dir(
        path=path_dict['ss'].format(db_type="v_n"),
        id_list=seq_id_dict['v']['n'],
        dim=3
    )

    t_p_f = utils.ds_preprocess.merge_pd_list([
        psipred_t_p_f, aacpro_t_p_f
    ])
    t_n_f = utils.ds_preprocess.merge_pd_list([
        psipred_t_n_f, aacpro_t_n_f
    ])
    v_p_f = utils.ds_preprocess.merge_pd_list([
        psipred_v_p_f, aacpro_v_p_f
    ])
    v_n_f = utils.ds_preprocess.merge_pd_list([
        psipred_v_n_f, aacpro_v_n_f
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
            SSEAAC_Model, desc=path_dict['model']['cv']['desc'], cv=model_cv),
        t_5C=t_5C,
        v_f=v_f,
        v_l=v_l,
        path_to_model_pickle=path_dict['model']['cv']['model_pickle'],
        path_to_model_result=path_dict['model']['cv']['model_result'],
        size_of_data=path_dict['model']['size']
    )
    Five_Cross_Get_model(
        model_construct_funtion=functools.partial(
            SSEAAC_Model, desc=path_dict['model']['tt']['desc'], cv=model_cv),
        t_5C=(((t_f, t_l),
               (v_f, v_l)),),
        v_f=v_f,
        v_l=v_l,
        path_to_model_pickle=path_dict['model']['tt']['model_pickle'],
        path_to_model_result=path_dict['model']['tt']['model_result'],
        size_of_data=None
    )
