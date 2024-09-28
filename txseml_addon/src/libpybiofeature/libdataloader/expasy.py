'''
Author: George Zhao
Date: 2021-08-21 15:31:51
LastEditors: George Zhao
LastEditTime: 2022-03-24 16:38:12
Description: 
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''
import os
import json
import tqdm

import pandas as pd

t3sps_col = ['pI', 'Nnc', 'Npc', 'Ins', 'Ali', 'Hydro']


def get_expasy_t3sps(
        path_to_json: list,
        seq_id_list: list = None):

    db = None
    with open(path_to_json, 'r', encoding='UTF-8') as f:
        db = json.load(f)

    data_ = [
        [
            element['Feature'][k]
            for k in t3sps_col
        ]
        for element in tqdm.tqdm(
            db['data'],
            desc=f"{os.path.split(path_to_json)[1]}_{db['param']['lenght']}_{db['param']['cter']}"
        )
    ]

    index_list = [
        element['id']
        for element in db['data']
    ]

    df_expasy_f = pd.DataFrame(data_, index=index_list, columns=t3sps_col)

    if seq_id_list is not None:
        df_expasy_f = df_expasy_f.loc[seq_id_list, :]

    # df_expasy_f = df_expasy_f.astype('float')
    return df_expasy_f
