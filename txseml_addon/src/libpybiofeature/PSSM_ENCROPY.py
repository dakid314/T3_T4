'''
Author: George Zhao
Date: 2021-10-04 22:45:49
LastEditors: George Zhao
LastEditTime: 2022-03-28 14:59:38
Description: 
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''
import os
import sys
sys.path.append('../..')

import numpy as np
import pandas as pd
from Bio import SeqIO
from scipy.stats import entropy
import tqdm
from . import libdataloader


def get_seq_position_score_entropy(form_1: np.array):
    return (list(map(
        lambda x: entropy(np.unique(x, return_counts=True)[1]),
        form_1
    )))


def get_2_dim_feature(entropy_list: list):
    return {
        'average_bit': np.average(entropy_list),
        'Length': len(entropy_list)
    }


def build_feature_from_file(
        path_to_fasta: str,
        order_list: list,
        path_with_pattern: str,
        seq_id_list: str,
        desc: str = 'unDefine'):

    pssm_file_content_list = libdataloader.pssm_tools.get_pssm_in_order(
        order_list,
        path_with_pattern
    )
    len_fasta = len(list(SeqIO.parse(
        path_to_fasta,
        'fasta'
    )))
    data_dict = [
        {
            'id': seq.id,
            'seq': len(seq.seq),
            'Feature': get_2_dim_feature(
                get_seq_position_score_entropy(
                    libdataloader.pssm_tools.get_pssm_from_file(
                        pssm_content
                    )['form_1']
                )
            )
        } for seq, pssm_content in tqdm.tqdm(zip(
            SeqIO.parse(
                path_to_fasta,
                'fasta'
            ),
            pssm_file_content_list
        ), total=len_fasta, desc=desc)
    ]
    df_f = pd.DataFrame([
        item['Feature']['average_bit']
        for item in data_dict
    ],
        index=[
        item['id']
        for item in data_dict
    ],
        columns=['average_bit', ]
    )
    df_f = df_f.loc[seq_id_list, :]
    return df_f
