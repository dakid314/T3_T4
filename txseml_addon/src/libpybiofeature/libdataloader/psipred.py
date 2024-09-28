'''
Author: George Zhao
Date: 2021-08-21 17:16:49
LastEditors: George Zhao
LastEditTime: 2021-08-21 18:07:54
Description: 
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''
# %%
import json
import itertools
from Bio.SeqIO import index

import numpy as np
import pandas as pd
import tqdm


def get_psipred_datapair(
        path_to_json_list: list,
        seq_id_list: list = None,
        length: int = 100):
    psipred_data_list = []
    for path_to_json in path_to_json_list:
        with open(path_to_json, 'r', encoding='UTF-8') as f:
            d = json.load(f)
            psipred_data_list.append(d['data'])

    psipred_data_list = list(itertools.chain(*psipred_data_list))

    index_list = [
        item['id']
        for item in psipred_data_list
    ]

    data_list = [
        np.array(item['Feature']['psipred'])[
            :, 2:].astype(np.float_).reshape(-1)
        for item in psipred_data_list
    ]
    df_f = pd.DataFrame(data_list, index=index_list)
    if seq_id_list is not None:
        df_f = df_f.loc[seq_id_list, :]

    if df_f.shape[1] >= length * 3:
        df_f = df_f.iloc[:, 0:length * 3]
    else:
        df_f = pd.concat(
            [df_f, pd.DataFrame(np.zeros((df_f.shape[0], length * 3 - df_f.shape[1])), index=df_f.index)], axis=1)
    df_f = df_f.fillna(0)
    return df_f


# n = get_psipred_datapair(
#     path_to_json_list=[
#         "psipred_t_p.json",
#         "psipred_t_p_new.json"
#     ],
#     seq_id_list=None,
#     length=10000
# )
# %%
