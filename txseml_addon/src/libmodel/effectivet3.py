'''
Author: George Zhao
Date: 2022-03-05 15:53:27
LastEditors: George Zhao
LastEditTime: 2022-06-23 21:55:02
Description: 
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''
import os
import sys
import pickle
import json
import functools
sys.path.append('src')

import utils
import libpybiofeature

from libmodel import common

from sklearn.naive_bayes import GaussianNB


class EffectiveT3_Model(common.Model_Final):
    def __init__(self, cv, desc):
        super().__init__(cv, desc=desc)
        self.model = None

    def tranmodel(self, f, l):
        super().tranmodel(f, l)
        self.model = GaussianNB()
        self.model.fit(X=f, y=l)
        return self

    def predict(self, f):
        super().predict(f)
        return self.model.predict_proba(f)[:, 1]


def Five_Cross_Get_model(
    model_construct_funtion,
    Five_Cross_set: list,
    v_f,
    v_l,
    path_to_model_pickle: str,
    path_to_model_result: str,
    size_of_data: str,
):
    model_set = list()
    model_result_set = list()
    for i in range(len(Five_Cross_set)):

        train, test = Five_Cross_set[i]
        validated_f = v_f

        model: EffectiveT3_Model = model_construct_funtion()
        model.tranmodel(f=train[0], l=train[1])

        model_set.append(model)

        model_result_set.append(
            {
                "training": {
                    "origin": {
                        f'pred': list(model.predict(train[0])),
                        f'label': list(train[1])},
                    "evaluation": {
                    },
                    "option": {
                    }
                },
                "testing": {
                    "origin": {
                        f'pred': list(model.predict(test[0])),
                        f'label': list(test[1])},
                    "evaluation": {
                    },
                    "option": {
                    }
                },
                "validated": {
                    "origin": {
                        f'pred': list(model.predict(validated_f)),
                        f'label': list(v_l)},
                    "evaluation": {
                    },
                    "option": {
                    }
                },
                "detail": {
                    "model": 'EffectiveT3',
                    'desc': model.desc,
                    'iteration': i,
                    "size_of_data": size_of_data,
                }
            }
        )

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

    t_p_f = libpybiofeature.featurebuilder.build_PPT_feature(
        path_to_fasta=path_dict['ppt']['t']['p'],
        seq_id_list=seq_id_dict['t']['p'],
        desc='t_p',
        cter=path_dict['ppt']['cter']
    )

    t_n_f = libpybiofeature.featurebuilder.build_PPT_feature(
        path_to_fasta=path_dict['ppt']['t']['n'],
        seq_id_list=seq_id_dict['t']['n'],
        desc='t_n',
        cter=path_dict['ppt']['cter']
    )

    v_p_f = libpybiofeature.featurebuilder.build_PPT_feature(
        path_to_fasta=path_dict['ppt']['v']['p'],
        seq_id_list=seq_id_dict['v']['p'],
        desc='v_p',
        cter=path_dict['ppt']['cter']
    )

    v_n_f = libpybiofeature.featurebuilder.build_PPT_feature(
        path_to_fasta=path_dict['ppt']['v']['n'],
        seq_id_list=seq_id_dict['v']['n'],
        desc='v_n',
        cter=path_dict['ppt']['cter']
    )

    t_p_l = utils.ds_preprocess.make_binary_label(
        size=t_p_f.shape[0], label=True)
    t_n_l = utils.ds_preprocess.make_binary_label(
        size=t_n_f.shape[0], label=False)

    v_p_l = utils.ds_preprocess.make_binary_label(
        size=v_p_f.shape[0], label=True)
    v_n_l = utils.ds_preprocess.make_binary_label(
        size=v_n_f.shape[0], label=False)

    t_f, t_l = utils.ds_preprocess.make_merge(t_p_f, t_p_l, t_n_f, t_n_l)
    v_f, v_l = utils.ds_preprocess.make_merge(v_p_f, v_p_l, v_n_f, v_n_l)

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

    Five_Cross_Get_model(
        model_construct_funtion=functools.partial(
            EffectiveT3_Model, desc=path_dict['model']['cv']['desc'], cv=None),
        Five_Cross_set=t_5C,
        v_f=v_f,
        v_l=v_l,
        path_to_model_pickle=path_dict['model']['cv']['model_pickle'],
        path_to_model_result=path_dict['model']['cv']['model_result'],
        size_of_data=path_dict['model']['size']
    )

    Five_Cross_Get_model(
        model_construct_funtion=functools.partial(
            EffectiveT3_Model, desc=path_dict['model']['tt']['desc'], cv=None),
        Five_Cross_set=(([t_f, t_l], [v_f, v_l]),),
        v_f=v_f,
        v_l=v_l,
        path_to_model_pickle=path_dict['model']['tt']['model_pickle'],
        path_to_model_result=path_dict['model']['tt']['model_result'],
        size_of_data=None
    )
