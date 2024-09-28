'''
Author: George Zhao
Date: 2021-08-21 16:03:22
LastEditors: George Zhao
LastEditTime: 2022-08-06 16:12:28
Description: 
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''
# %%
import json
import itertools
import functools

from .. import AC

import pandas as pd
import numpy as np

# %%


def code_f(seq_: list, coded_: bool, coded_dict: dict, cter: bool = False):
    seq_ = seq_ if cter == False else list(reversed(seq_))
    if coded_ == False:
        return seq_
    return itertools.chain(*[
        coded_dict[aa]
        for aa in seq_
    ])


def decode_context(content: str, coded_: bool, coded_dict: dict, cter: bool = False):
    return code_f(
        list(
            content.splitlines()[2]
        ),
        coded_=coded_,
        coded_dict=coded_dict,
        cter=cter
    )


def get_data(
        path_to_json_db: str,
        tag_name: str,
        seqid_list: list,
        coded_dict: dict,
        length: int = 250,
        dig_code: bool = False,
        desc: str = '',
        cter: bool = False
):
    if dig_code == True:
        length = length * len(coded_dict[''])

    t_db = None
    with open(path_to_json_db, 'r', encoding='UTF-8') as f:
        t_db = json.load(f)
    k_list = [k for k in t_db['data'].keys() if k.find(tag_name) == 0]
    db_choise_by_k = list(itertools.chain(
        *[t_db['data'][k] for k in k_list]))

    df = pd.DataFrame([
        decode_context(
            item['data'], coded_=dig_code, coded_dict=coded_dict, cter=cter
        )
        for item in db_choise_by_k
    ], index=[u['id'] for u in db_choise_by_k])

    if length <= df.shape[1]:
        df = df.iloc[:, 0:length]
    else:
        df = pd.concat(
            [df, pd.DataFrame(np.zeros((df.shape[0], length - df.shape[1])), index=df.index)], axis=1)

    if seqid_list is None:
        pass
    else:
        df = df.loc[seqid_list, :]

    if dig_code == True:
        df.fillna(0, inplace=True)
    else:
        df.fillna('', inplace=True)

    return df


def make_header(l: list):
    if '' in l:
        l.pop(l.index(''))

    return [
        l,
        AC._get_dac_order(aaorder=''.join(l)),
        AC._get_tac_order(aaorder=''.join(l))
    ]


def get_muti_stats(
    path_to_json_db: str,
    tag_name: str,
    seqid_list: list,
    coded_dict: dict,
    head_dict: dict,
    length: int = 250,
    desc: str = '',
    cter: bool = False
):
    o_df = get_data(
        path_to_json_db=path_to_json_db,
        tag_name=tag_name,
        seqid_list=seqid_list,
        coded_dict=coded_dict,
        length=length,
        dig_code=False,
        desc=desc,
        cter=cter
    )
    o_data = [''.join(r) for r in o_df.values]

    return pd.concat([pd.DataFrame([AC.AAC(seq_aa=seq, aaorder=head_dict[0])
                                    for seq in o_data], index=o_df.index, columns=head_dict[0]),
                      pd.DataFrame([AC.DAC(seq_aa=seq, dacorder=head_dict[1])
                                    for seq in o_data], index=o_df.index, columns=head_dict[1]),
                      pd.DataFrame([AC.TAC(seq_aa=seq, tacorder=head_dict[2])
                                    for seq in o_data], index=o_df.index, columns=head_dict[2])], axis=1)


def get_single_stats(
    path_to_json_db: str,
    tag_name: str,
    seqid_list: list,
    coded_dict: dict,
    head_dict: dict,
    length: int = 250,
    desc: str = '',
    cter: bool = False
):
    o_df = get_data(
        path_to_json_db=path_to_json_db,
        tag_name=tag_name,
        seqid_list=seqid_list,
        coded_dict=coded_dict,
        length=length,
        dig_code=False,
        desc=desc,
        cter=cter
    )
    o_data = [''.join(r) for r in o_df.values]

    return pd.DataFrame([AC.AAC(seq_aa=seq, aaorder=head_dict[0])
                         for seq in o_data], index=o_df.index, columns=head_dict[0])


class accpro:
    coded_dict = {
        'b': [1, 0],
        'e': [0, 1],
        '': [0, 0],
    }

    muti_header = make_header(list(coded_dict.keys()))

    get_func = functools.partial(
        get_data,
        coded_dict=coded_dict
    )

    get_single_func = functools.partial(
        get_single_stats,
        coded_dict=coded_dict,
        head_dict=muti_header
    )


class sspro:
    coded_dict = {
        'C': [1, 0, 0],
        'H': [0, 1, 0],
        'E': [0, 0, 1],
        '': [0, 0, 0],
    }

    muti_header = make_header(list(coded_dict.keys()))

    get_func = functools.partial(
        get_data,
        coded_dict=coded_dict
    )

    get_muti_func = functools.partial(
        get_muti_stats,
        coded_dict=coded_dict,
        head_dict=muti_header
    )


# df = sspro.get_func(
#     '../../../out/T3SEs/data/SSE_AAC/pyscratch_t3.json', 't3_t_n_1_acc', None, head_dict=accpro.muti_header
# )

# %%
