'''
Author: George Zhao
Date: 2022-06-22 14:48:41
LastEditors: George Zhao
LastEditTime: 2022-06-24 10:01:59
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

from Bio import SeqIO
import pandas as pd
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


class BPBaac_Model(common.Model_Final):
    def __init__(self, cv, desc,
                 cter: bool,
                 terlength: int = 100,
                 padding_ac='A',
                 ):
        super().__init__(cv, desc=desc,)
        self.model = None
        self.profile = None
        self.cter = cter
        self.terlength = terlength
        self.padding_ac = padding_ac
        pass

    def tranmodel(self, f, l):
        super().tranmodel(f, l)
        f = f[0].values.tolist()
        self.profile = {
            "p": libpybiofeature.BPBaac_psp.mat_constructor(
                fasta_db=[f[i] for i in range(len(f)) if l[i] == 1],
                cter=self.cter,
                terlength=self.terlength,
                padding_ac=self.padding_ac
            ),
            "n": libpybiofeature.BPBaac_psp.mat_constructor(
                fasta_db=[f[i] for i in range(len(f)) if l[i] == 0],
                cter=self.cter,
                terlength=self.terlength,
                padding_ac=self.padding_ac
            ),
        }

        self.model = model_optimite.svm_optim(
            param_o=svm_param_o,
            cv=self.cv
        ).best_fit(X=[libpybiofeature.BPBaac_psp.mat_mapper(
            seq=str(seq.seq),
            pmat=self.profile['p'],
            nmat=self.profile['n'],
            cter=self.cter,
            terlength=self.terlength,
            padding_ac=self.padding_ac
        )for seq in f], y=l, verbose=-1, n_jobs=n_jobs)
        return self

    def predict(self, f):
        super().predict(f)
        f = f[0].values.tolist()
        return self.model.predict_proba(
            [libpybiofeature.BPBaac_psp.mat_mapper(
                seq=str(seq.seq),
                pmat=self.profile['p'],
                nmat=self.profile['n'],
                cter=self.cter,
                terlength=self.terlength,
                padding_ac=self.padding_ac
            ) for seq in f]
        )


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

        model: BPBaac_Model = model_construct_funtion()
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
                "model": 'BPBaac',
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


from libpybiofeature.libdataloader.fasta_seq_loader import prepare_data


def research(path_dict: dict):
    seq_id_dict = None
    with open(path_dict['seq_id'], 'r', encoding='UTF-8') as f:
        seq_id_dict = json.load(f)

    t_p_f = prepare_data(
        path_to_fasta=path_dict['fasta']['t']['p'],
        seq_id_list=seq_id_dict['t']['p'],
    )

    t_n_f = prepare_data(
        path_to_fasta=path_dict['fasta']['t']['n'],
        seq_id_list=seq_id_dict['t']['n'],
    )

    v_p_f = prepare_data(
        path_to_fasta=path_dict['fasta']['v']['p'],
        seq_id_list=seq_id_dict['v']['p'],
    )

    v_n_f = prepare_data(
        path_to_fasta=path_dict['fasta']['v']['n'],
        seq_id_list=seq_id_dict['v']['n'],
    )

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
            BPBaac_Model, desc=path_dict['model']['cv']['desc'], cv=model_cv, cter=path_dict['model']['cter']),
        Five_Cross_set=t_5C,
        t_model_v_f=v_f,
        t_model_v_l=v_l,
        path_to_model_pickle=path_dict['model']['cv']['model_pickle'],
        path_to_model_result=path_dict['model']['cv']['model_result'],
        size_of_data=path_dict['model']['size']
    )
    Five_Cross_Get_model(
        model_construct_funtion=functools.partial(
            BPBaac_Model, desc=path_dict['model']['tt']['desc'], cv=model_cv, cter=path_dict['model']['cter']),
        Five_Cross_set=(((t_f, t_l),
                         (v_f, v_l)),),
        t_model_v_f=v_f,
        t_model_v_l=v_l,
        path_to_model_pickle=path_dict['model']['tt']['model_pickle'],
        path_to_model_result=path_dict['model']['tt']['model_result'],
        size_of_data=None
    )
