'''
Author: George Zhao
Date: 2021-10-09 19:10:15
LastEditors: George Zhao
LastEditTime: 2021-10-09 20:15:43
Description: 
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''

import json
import itertools
import os

import pandas as pd
import numpy as np

ss_dict = {
    "C": [1, 0, 0],
    "E": [0, 1, 0],
    "H": [0, 0, 1],
}


def get_cchmc_data(
    path_to_json: str,
    seq_id_list: list,
    desc: str,
    num_of_each_part: int = 25,
    length: int = 100,
    cter: bool = False,
):
    reversed_func = iter
    if cter == True:
        reversed_func = reversed
    json_DB = None
    with open(path_to_json, 'r', encoding='UTF-8') as f:
        json_DB = json.load(f)

    tag_name = f'{desc}_{num_of_each_part}'
    k_list = [k for k in json_DB['data'].keys() if k.find(tag_name) == 0]
    cchmc_data_list = list(itertools.chain(
        *[json_DB['data'][k] for k in k_list]))

    index_list = [
        item['id']
        for item in cchmc_data_list
    ]

    # RSA
    rsa_list = [
        [
            float(int(digit) / 10)
            for digit in reversed_func(item['Feature']['cchmc']['seaSeq'])
        ]
        for item in cchmc_data_list
    ]
    df_rsa = pd.DataFrame(rsa_list, index=index_list)
    df_rsa_f = df_rsa.loc[seq_id_list, :]

    # SS
    ss_list = [
        list(itertools.chain(*[
            ss_dict[character]
            for character in reversed_func(item['Feature']['cchmc']['ssSeq'])
        ]))
        for item in cchmc_data_list
    ]
    df_ss = pd.DataFrame(ss_list, index=index_list)
    df_ss_f = df_ss.loc[seq_id_list, :]

    df_rsa_f = df_rsa_f.iloc[:, :length * 1]
    df_ss_f = df_ss_f.iloc[:, :length * 3]
    df_rsa_f = df_rsa_f.fillna(0.0)
    df_ss_f = df_ss_f.fillna(0.0)
    return df_rsa_f, df_ss_f
