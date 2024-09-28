'''
Author: George Zhao
Date: 2022-03-05 18:20:46
LastEditors: George Zhao
LastEditTime: 2022-06-23 23:06:45
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

import numpy as np
import pandas as pd


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
    config = model_construct_funtion()
    config['path_to_tmp'] = os.path.join(
        config['path_to_tmp'], *[config['desc'], ]
    )
    if os.path.exists(config['path_to_tmp']) == False:
        os.makedirs(config['path_to_tmp'])
    for i in range(len(Five_Cross_set)):
        # train: Train set((seq,pssm), label)
        train, test = Five_Cross_set[i]

        # train[0]: (seq,pssm)
        model_config = {
            'path_to_model_dir': config['model_out_dir'].format(
                desc=f'_{config["desc"]}_{i+1}'),
            'label_tag': config['label_tag'],
        }
        # if os.path.exists(model_config['path_to_model_dir']) == False:
        #     os.makedirs(model_config['path_to_model_dir'])
        utils.fastprofkernel_opt.go_opt(
            p_seq_list=[d_s[0] for d_s, label in zip(
                train[0].values, train[1]) if label == 1],
            n_seq_list=[d_s[0] for d_s, label in zip(
                train[0].values, train[1]) if label == 0],
            p_seq_pssm_content=[d_s[1] for d_s, label in zip(
                train[0].values, train[1]) if label == 1],
            n_seq_pssm_content=[d_s[1] for d_s, label in zip(
                train[0].values, train[1]) if label == 0],
            path_to_exebin=config['path_to_exebin'],
            path_to_model_dir=model_config['path_to_model_dir'],
            path_to_tmp=config['path_to_tmp'],
            label_tag=model_config['label_tag'],
            verbose=False
        )

        model_set.append(model_config)
        model_result_set.append({
            "training": {
                "origin": {
                    f'pred': utils.fastprofkernel_opt.go_pred(
                        seq_list=train[0].values[:, 0],
                        seq_pssm_content=train[0].values[:, 1],
                        path_to_exebin=config['path_to_exebin'],
                        path_to_model_dir=model_config['path_to_model_dir'],
                        path_to_tmp=config['path_to_tmp'],
                        label_tag=model_config['label_tag'],
                        path_to_out=None,
                        verbose=False
                    ).loc[:, 'Score'].values,
                    f'label': train[1]},
                "evaluation": {
                }, "option": {
                }
            },
            "testing": {
                "origin": {
                    f'pred': utils.fastprofkernel_opt.go_pred(
                        seq_list=test[0][:, 0] if isinstance(
                            test[0], np.ndarray) == True else test[0].values[:, 0],
                        seq_pssm_content=test[0][:, 1] if isinstance(
                            test[0], np.ndarray) == True else test[0].values[:, 1],
                        path_to_exebin=config['path_to_exebin'],
                        path_to_model_dir=model_config['path_to_model_dir'],
                        path_to_tmp=config['path_to_tmp'],
                        label_tag=model_config['label_tag'],
                        path_to_out=None,
                        verbose=False
                    ).loc[:, 'Score'].values,
                    f'label': test[1]},
                "evaluation": {
                }, "option": {
                }
            },
            "validated": {
                "origin": {
                    f'pred': utils.fastprofkernel_opt.go_pred(
                        seq_list=v_f.values[:, 0],
                        seq_pssm_content=v_f.values[:, 1],
                        path_to_exebin=config['path_to_exebin'],
                        path_to_model_dir=model_config['path_to_model_dir'],
                        path_to_tmp=config['path_to_tmp'],
                        label_tag=model_config['label_tag'],
                        path_to_out=None,
                        verbose=False
                    ).loc[:, 'Score'].values,
                    f'label': v_l},
                "evaluation": {
                }, "option": {
                }
            },
            "detail": {
                "model": 'pEffect',
                'desc': config["desc"],
                'iteration': i,
                "model_config": model_config,
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

    t_p_f = pd.DataFrame(utils.fastprofkernel_opt.prepare_data(
        path_to_fasta=path_dict['fasta']['t']['p'],
        path_to_profile_dir=path_dict['pssm']['t']['p'],
        tag_of_profile='t_p',
        seq_id_list=seq_id_dict['t']['p']
    ))

    t_n_f = pd.DataFrame(utils.fastprofkernel_opt.prepare_data(
        path_to_fasta=path_dict['fasta']['t']['n'],
        path_to_profile_dir=path_dict['pssm']['t']['n'],
        tag_of_profile='t_n',
        seq_id_list=seq_id_dict['t']['n']
    ))

    v_p_f = pd.DataFrame(utils.fastprofkernel_opt.prepare_data(
        path_to_fasta=path_dict['fasta']['v']['p'],
        path_to_profile_dir=path_dict['pssm']['v']['p'],
        tag_of_profile='v_p',
        seq_id_list=seq_id_dict['v']['p']
    ))

    v_n_f = pd.DataFrame(utils.fastprofkernel_opt.prepare_data(
        path_to_fasta=path_dict['fasta']['v']['n'],
        path_to_profile_dir=path_dict['pssm']['v']['n'],
        tag_of_profile='v_n',
        seq_id_list=seq_id_dict['v']['n']
    ))

    t_p_l = utils.ds_preprocess.make_binary_label(
        size=t_p_f.shape[0],
        label=True
    )

    t_n_l = utils.ds_preprocess.make_binary_label(
        size=t_n_f.shape[0],
        label=False
    )

    v_p_l = utils.ds_preprocess.make_binary_label(
        size=v_p_f.shape[0],
        label=True
    )

    v_n_l = utils.ds_preprocess.make_binary_label(
        size=v_n_f.shape[0],
        label=False
    )

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
        t_n_l=t_n_l
    )
    v_f, v_l = utils.ds_preprocess.make_merge(
        t_p_f=v_p_f,
        t_p_l=v_p_l,
        t_n_f=v_n_f,
        t_n_l=v_n_l
    )

    def config_func(desc, cv=None):
        return {
            'model_out_dir': path_dict['model']['model_out_dir_pattern'],
            'path_to_exebin': 'profkernel-workflow',
            'desc': desc,
            'path_to_tmp': path_dict['model']['tmp_dir'],
            'label_tag': 'SEs'
        }

    Five_Cross_Get_model(
        model_construct_funtion=functools.partial(
            config_func, desc=path_dict['model']['cv']['desc'], cv=None),
        Five_Cross_set=t_5C,
        v_f=v_f,
        v_l=v_l,
        path_to_model_pickle=path_dict['model']['cv']['model_pickle'],
        path_to_model_result=path_dict['model']['cv']['model_result'],
        size_of_data=path_dict['model']['size']
    )

    Five_Cross_Get_model(
        model_construct_funtion=functools.partial(
            config_func, desc=path_dict['model']['tt']['desc'], cv=None),
        Five_Cross_set=(([t_f, t_l], [v_f, v_l]),),
        v_f=v_f,
        v_l=v_l,
        path_to_model_pickle=path_dict['model']['tt']['model_pickle'],
        path_to_model_result=path_dict['model']['tt']['model_result'],
        size_of_data=None
    )
