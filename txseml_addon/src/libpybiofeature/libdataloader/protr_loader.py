'''
Author: George Zhao
Date: 2022-02-24 21:04:26
LastEditors: George Zhao
LastEditTime: 2022-02-24 21:05:08
Description: 
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''
import json

import pandas as pd


def build_form_of_data_set_protr(
    path_to_p_protr: str,
    path_to_n_protr: str,
    path_to_seq_id: str,
    looking_key: str,
    path_to_json_seq_id: str
):
    seq_id_dict = None
    with open(path_to_json_seq_id, 'r', encoding='UTF-8') as f:
        seq_id_dict = json.load(f)

    id_list = None
    with open(path_to_seq_id, 'r', encoding='utf-8') as f:
        id_list = json.load(f)

    p_f = pd.read_csv(path_to_p_protr, index_col=0)
    p_f = p_f.loc[id_list[looking_key]['p'], :]

    p_f.loc[seq_id_dict[looking_key]['p'], :]

    n_f = pd.read_csv(path_to_n_protr, index_col=0)
    n_f = n_f.loc[id_list[looking_key]['n'], :]

    n_f.loc[seq_id_dict[looking_key]['n'], :]

    return p_f, n_f
