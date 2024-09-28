'''
Author: George Zhao
Date: 2022-03-06 15:21:13
LastEditors: George Zhao
LastEditTime: 2022-03-16 17:50:01
Description: 
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''
import itertools
import json

import numpy as np
import pandas as pd


def build_form_of_data_set_align(
    path_to_align_result: str,
    looking_key: str,
    looking_key2: str,
    path_to_json_seq_id: str
):
    align_result_json = None
    with open(path_to_align_result, 'r', encoding='utf-8') as f:
        align_result_json = json.load(f)

    seq_id_dict = None
    with open(path_to_json_seq_id, 'r', encoding='UTF-8') as f:
        seq_id_dict = json.load(f)

    data_35 = [
        item['Feature']['BLOSUM35']
        for item in align_result_json['data']
    ]
    data_40 = [
        item['Feature']['BLOSUM40']
        for item in align_result_json['data']
    ]
    data_45 = [
        item['Feature']['BLOSUM45']
        for item in align_result_json['data']
    ]

    data_index = [
        item['id']
        for item in align_result_json['data']
    ]

    a35_f = pd.DataFrame(data_35)
    a35_f.index = data_index
    a35_f = a35_f.loc[seq_id_dict[looking_key][looking_key2], :]

    a40_f = pd.DataFrame(data_40)
    a40_f.index = data_index
    a40_f = a40_f.loc[seq_id_dict[looking_key][looking_key2], :]

    a45_f = pd.DataFrame(data_45)
    a45_f.index = data_index
    a45_f = a45_f.loc[seq_id_dict[looking_key][looking_key2], :]

    return a35_f, a40_f, a45_f


def build_form_of_data_set_iDNA(
    path_to_iDNA: str,
    looking_key: str,
    path_to_json_seq_id: str
):
    iDNA_json = None
    with open(path_to_iDNA, 'r', encoding='utf-8') as f:
        iDNA_json = json.load(f)

    seq_id_dict = None
    with open(path_to_json_seq_id, 'r', encoding='UTF-8') as f:
        seq_id_dict = json.load(f)

    data = [
        [
            [
                item['Feature']['iDNA_Prot_dis']['data'],
            ]
            for item in iDNA_json['data'][p_n]
        ]
        for p_n in iDNA_json['data'].keys()
    ]

    data_index = [
        [

            item['id']
            for item in iDNA_json['data'][p_n]
        ]
        for p_n in iDNA_json['data'].keys()
    ]

    t_p_f = pd.DataFrame(data[0])
    t_p_f.columns = ['iDNA_Prot_dis']
    t_p_f.index = data_index[0]
    t_p_f = t_p_f.loc[seq_id_dict[looking_key]['p'], :]
    t_p_l = np.ones(shape=t_p_f.shape[0])

    t_n_f = pd.DataFrame(data[1])
    t_n_f.columns = ['iDNA_Prot_dis']
    t_n_f.index = data_index[1]
    t_n_f = t_n_f.loc[seq_id_dict[looking_key]['n'], :]
    t_n_l = np.zeros(shape=t_n_f.shape[0])

    return t_p_f, t_p_l, t_n_f, t_n_l


def build_form_of_data_set_Topm(
    path_to_Topm: str,
    looking_key: str,
    path_to_json_seq_id: str
):
    Topm_json = None
    with open(path_to_Topm, 'r', encoding='utf-8') as f:
        Topm_json = json.load(f)

    seq_id_dict = None
    with open(path_to_json_seq_id, 'r', encoding='UTF-8') as f:
        seq_id_dict = json.load(f)

    data = [
        [

            item['Feature']['Top_n_gram']['data']
            for item in Topm_json['data'][p_n]
        ]
        for p_n in Topm_json['data'].keys()
    ]

    data_index = [
        [

            item['id']
            for item in Topm_json['data'][p_n]
        ]
        for p_n in Topm_json['data'].keys()
    ]

    t_p_f = pd.DataFrame(data[0])
    t_p_f.index = data_index[0]
    t_p_f = t_p_f.loc[seq_id_dict[looking_key]['p'], :]
    t_p_l = np.ones(shape=t_p_f.shape[0])

    t_n_f = pd.DataFrame(data[1])
    t_n_f.index = data_index[1]
    t_n_f = t_n_f.loc[seq_id_dict[looking_key]['n'], :]
    t_n_l = np.zeros(shape=t_n_f.shape[0])

    return t_p_f, t_p_l, t_n_f, t_n_l


def build_form_of_data_set_PSE(
    path_to_PSE: str,
    looking_key: str,
    path_to_json_seq_id: str
):
    Topm_json = None
    with open(path_to_PSE, 'r', encoding='utf-8') as f:
        Topm_json = json.load(f)

    seq_id_dict = None
    with open(path_to_json_seq_id, 'r', encoding='UTF-8') as f:
        seq_id_dict = json.load(f)

    data = [
        [

            item['Feature']['PC_PSEACC']['data']
            for item in Topm_json['data'][p_n]
        ]
        for p_n in Topm_json['data'].keys()
    ]

    data_index = [
        [

            item['id']
            for item in Topm_json['data'][p_n]
        ]
        for p_n in Topm_json['data'].keys()
    ]

    t_p_f = pd.DataFrame(data[0])
    t_p_f.index = data_index[0]
    t_p_f = t_p_f.loc[seq_id_dict[looking_key]['p'], :]
    t_p_l = np.ones(shape=t_p_f.shape[0])

    t_n_f = pd.DataFrame(data[1])
    t_n_f.index = data_index[1]
    t_n_f = t_n_f.loc[seq_id_dict[looking_key]['n'], :]
    t_n_l = np.zeros(shape=t_n_f.shape[0])

    return t_p_f, t_p_l, t_n_f, t_n_l


def build_form_of_data_set_SPSE(
    path_to_PSE: str,
    looking_key: str,
    path_to_json_seq_id: str
):
    Topm_json = None
    with open(path_to_PSE, 'r', encoding='utf-8') as f:
        Topm_json = json.load(f)

    seq_id_dict = None
    with open(path_to_json_seq_id, 'r', encoding='UTF-8') as f:
        seq_id_dict = json.load(f)

    data = [
        [

            item['Feature']['SC_PSEACC']['data']
            for item in Topm_json['data'][p_n]
        ]
        for p_n in Topm_json['data'].keys()
    ]

    data_index = [
        [

            item['id']
            for item in Topm_json['data'][p_n]
        ]
        for p_n in Topm_json['data'].keys()
    ]

    t_p_f = pd.DataFrame(data[0])
    t_p_f.index = data_index[0]
    t_p_f = t_p_f.loc[seq_id_dict[looking_key]['p'], :]
    t_p_l = np.ones(shape=t_p_f.shape[0])

    t_n_f = pd.DataFrame(data[1])
    t_n_f.index = data_index[1]
    t_n_f = t_n_f.loc[seq_id_dict[looking_key]['n'], :]
    t_n_l = np.zeros(shape=t_n_f.shape[0])

    return t_p_f, t_p_l, t_n_f, t_n_l
