'''
Author: George Zhao
Date: 2022-03-15 16:41:18
LastEditors: George Zhao
LastEditTime: 2022-03-19 18:40:06
Description: 
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''
import os
import sys
sys.path.append('..')
sys.path.append('../..')
import itertools
from collections.abc import Iterable
# %%
import utils
from . import libdataloader

import numpy as np
import pandas as pd
from Bio import SeqIO, Seq
import tqdm
# %%


def get_PSSM_SMTH(
    pssm_dict: dict,
    length: int,
    cter: bool,
    w,
):
    pssm_df: np.ndarray = pssm_dict['form_1'][1:]

    if cter == True:
        if pssm_df.shape[0] < length:
            pssm_df = np.concatenate(
                [
                    np.zeros(
                        shape=((length - pssm_df.shape[0]), pssm_df.shape[1])
                    ),
                    pssm_df,
                ]
            )
        pass
    else:
        if pssm_df.shape[0] < length:
            pssm_df = np.concatenate(
                [
                    pssm_df,
                    np.zeros(
                        shape=((length - pssm_df.shape[0]), pssm_df.shape[1])
                    ),
                ]
            )
        pass

    result = []
    pssm_df = pssm_dict['form_1'][:length]

    wlist = w
    if isinstance(w, int):
        wlist = [w, ]

    for w_ith in wlist:
        if w_ith < 1:
            raise RuntimeError("get_PSSM_SMTH: w < 1")
        for row_index in range(0, length - w_ith + 1):
            vector_: np.ndarray = pssm_df[row_index]
            for window_index in range(1, w_ith):
                vector_ = vector_ + pssm_df[row_index + window_index]
            result.append(vector_.tolist())

    return list(itertools.chain(*result))


def build_PSSM_SMTH_feature(
        path_to_fasta: str,
        order_list: list,
        path_with_pattern: str,
        seq_id_list: list,
        length: int,
        cter: bool,
        w: int,
        desc: str = 'undefine',
):
    pssm_file_content_list = libdataloader.pssm_tools.get_pssm_in_order(
        order_list,
        path_with_pattern
    )
    len_of_fasta = len(list(SeqIO.parse(
        path_to_fasta,
        'fasta'
    )))
    feature_json = [
        {
            'id': seq.id,
            'seq': len(seq.seq),
            'Feature': get_PSSM_SMTH(
                pssm_dict=libdataloader.pssm_tools.get_pssm_from_file(
                    pssm_content
                ),
                length=length,
                cter=cter,
                w=w,
            )
        } for seq, pssm_content in tqdm.tqdm(zip(
            SeqIO.parse(
                path_to_fasta,
                'fasta'
            ),
            pssm_file_content_list
        ), total=len_of_fasta,
            desc=f'{desc}_PSSM_SMTH')
    ]
    return pd.DataFrame(
        [
            item['Feature']
            for item in feature_json
        ],
        index=[
            item['id']
            for item in feature_json
        ],
    ).loc[seq_id_list, :]
