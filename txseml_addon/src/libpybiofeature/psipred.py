'''
Author: George Zhao
Date: 2021-08-14 23:10:04
LastEditors: George Zhao
LastEditTime: 2022-08-06 15:33:03
Description: 
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''
import json
import itertools
import os

from . import AC

import pandas as pd
import numpy as np


def code_f(seq_: list, cter: bool):
    if cter == True:
        seq_ = list(reversed(seq_))
    return itertools.chain(*[
        aa
        for aa in seq_
    ])


def get_data(
        path_to_json: str,
        path_to_datadir: str,
        desc: str,
        seqid_list: list,
        length: int = 250,
        num_of_each_part: int = 10,
):
    length = length * 3
    uuid_dict = None
    with open(path_to_json, 'r', encoding='UTF-8') as f:
        uuid_dict = json.load(f)
    tag_name = f'{desc}_{num_of_each_part}'
    k_list = [k for k in uuid_dict['data'].keys() if k.find(tag_name) == 0]

    uuid_list = list(itertools.chain(
        *[uuid_dict['data'][k] for k in k_list]))

    aaorder = ['C', 'H', 'E']
    seq_index = [u['submission_name'] for u in uuid_list]
    codeaac = [
        AC.AAC(
            seq_aa=pd.read_csv(
                os.path.join(path_to_datadir, *
                             [tag_name, f'{uuid["UUID"]}.ss2']),
                sep=r'\s+',
                comment='#',
                header=None,
                index_col=0
            ).loc[:, 2].values.tolist(),
            aaorder=aaorder
        )
        for uuid in uuid_list
    ]

    df = pd.DataFrame(codeaac, index=seq_index, columns=aaorder)

    if seqid_list is None:
        pass
    else:
        df = df.loc[seqid_list, :]

    return df

# df = get_data(
#     '../../../out/T4SEs/data/Bastion4/DISOPRED3/datarecord.json.t4',
#     '../../../out/T4SEs/data/Bastion4/DISOPRED3/',
#     't_4_t_p',
#     ['YP_094338.1', ],
#     250
# )
