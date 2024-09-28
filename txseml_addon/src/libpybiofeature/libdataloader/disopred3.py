'''
Author: George Zhao
Date: 2021-08-14 23:10:04
LastEditors: George Zhao
LastEditTime: 2022-03-19 22:59:29
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

# pbdat
coded_dict = {
    '^': [0, 0, 1],
    '.': [0, 1, 0],
    '-': [1, 0, 0],
    '': [0, 0, 0],
}

# comb
coded2_dict = {
    '': [0, 0],
    '*': [1, 0],
    '.': [0, 1],
}


def code_f(seq_: list, coded_: bool, cter: bool):
    if cter == True:
        seq_ = list(reversed(seq_))
    if coded_ == False:
        return seq_
    return itertools.chain(*[
        coded_dict[aa]
        for aa in seq_
    ])


def get_pbdat_data(
        path_to_json: str,
        path_to_datadir: str,
        desc: str,
        seqid_list: list,
        length: int = 250,
        num_of_each_part: int = 10,
        dig_code: bool = False,
        cter: bool = False
):
    if dig_code == True:
        length = length * 3
    uuid_dict = None
    with open(path_to_json, 'r', encoding='UTF-8') as f:
        uuid_dict = json.load(f)
    tag_name = f'{desc}_{num_of_each_part}'
    k_list = [k for k in uuid_dict['data'].keys() if k.find(tag_name) == 0]

    uuid_list = list(itertools.chain(
        *[uuid_dict['data'][k] for k in k_list]))

    df = pd.DataFrame([
        code_f(pd.read_csv(os.path.join(path_to_datadir, *[tag_name, f'{uuid["UUID"]}.pbdat']),
                           sep=r'\s+', comment='#', header=None, index_col=0).loc[:, 2].values.tolist(), coded_=dig_code, cter=cter)
        for uuid in uuid_list
    ], index=[u['submission_name'] for u in uuid_list])

    if length <= df.shape[1]:
        df = df.iloc[:, 0:length]
    else:
        df = pd.concat(
            [df, pd.DataFrame(np.zeros((df.shape[0], length - df.shape[1])))], axis=1)

    if seqid_list is None:
        pass
    else:
        df = df.loc[seqid_list, :]

    if dig_code == True:
        df.fillna(0, inplace=True)
    else:
        df.fillna('', inplace=True)

    return df


def code2_f(seq_: list, coded_: bool, cter: bool):
    if cter == True:
        seq_ = list(reversed(seq_))
    if coded_ == False:
        return seq_
    return itertools.chain(*[
        coded2_dict[aa]
        for aa in seq_
    ])


def get_comb_data(
        path_to_json: str,
        path_to_datadir: str,
        desc: str,
        seqid_list: list,
        length: int = 250,
        num_of_each_part: int = 10,
        dig_code: bool = False,
        cter: bool = False
):
    if dig_code == True:
        length = length * 2
    uuid_dict = None
    with open(path_to_json, 'r', encoding='UTF-8') as f:
        uuid_dict = json.load(f)
    tag_name = f'{desc}_{num_of_each_part}'
    k_list = [k for k in uuid_dict['data'].keys() if k.find(tag_name) == 0]

    uuid_list = list(itertools.chain(
        *[uuid_dict['data'][k] for k in k_list]))

    df = pd.DataFrame([
        code2_f(pd.read_csv(os.path.join(path_to_datadir, *[tag_name, f'{uuid["UUID"]}.comb']),
                            sep=r'\s+', comment='#', header=None, index_col=0).loc[:, 2].values.tolist(), coded_=dig_code, cter=cter)
        for uuid in uuid_list
    ], index=[u['submission_name'] for u in uuid_list])

    if length <= df.shape[1]:
        df = df.iloc[:, 0:length]
    else:
        df = pd.concat(
            [df, pd.DataFrame(np.zeros((df.shape[0], length - df.shape[1])))], axis=1)

    if seqid_list is None:
        pass
    else:
        df = df.loc[seqid_list, :]

    if dig_code == True:
        df.fillna(0, inplace=True)
    else:
        df.fillna('', inplace=True)

    return df


def get_data(
        path_to_json: str,
        path_to_datadir: str,
        desc: str,
        seqid_list: list,
        length: int = 250,
        num_of_each_part: int = 10,
        dig_code: bool = False,
        cter: bool = False
):
    return pd.concat(
        [
            get_comb_data(
                path_to_json=path_to_json,
                path_to_datadir=path_to_datadir,
                desc=desc,
                seqid_list=seqid_list,
                length=length,
                num_of_each_part=num_of_each_part,
                dig_code=dig_code,
                cter=cter,
            ),
            get_pbdat_data(
                path_to_json=path_to_json,
                path_to_datadir=path_to_datadir,
                desc=desc,
                seqid_list=seqid_list,
                length=length,
                num_of_each_part=num_of_each_part,
                dig_code=dig_code,
                cter=cter,
            ),
        ], axis=1)
