'''
Author: George Zhao
Date: 2021-08-09 21:38:12
LastEditors: George Zhao
LastEditTime: 2022-05-23 20:57:47
Description: 
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''
import json
import os

from Bio import SeqIO
from tqdm.std import tqdm
from .libdataloader import pssm_tools
import utils

import tqdm
import numpy as np
import pandas as pd


def get_all_task_pssmcode(
    possum_index_dict: dict,
    seq_id_dict: str,
    path_to_fasta_with_pattern: str,
    path_to_with_pattern: str,
    length: int,
    desc: str = 'unDefine',
    cter: bool = False
):
    result = list()
    for taskname in ['t_p', 't_n', 'v_p', 'v_n']:
        index_ = [seq.id for seq in SeqIO.parse(
            path_to_fasta_with_pattern.format(taskname=taskname), 'fasta')]
        data_array_list = list()
        tqdmer = tqdm.tqdm(desc=f'{desc}_pssmcode_{taskname}',
                           total=len(index_))
        for pssmfilecontent in pssm_tools.get_pssm_in_order(
            order_list=possum_index_dict['data'][taskname],
            path_with_pattern=path_to_with_pattern
        ):
            tqdmer.update()
            data_array = pssm_tools.get_pssm_from_file(
                content=pssmfilecontent)['form_1']
            if cter != False:
                data_array = data_array[::-1, :]
            if data_array.shape[0] >= length:
                data_array = data_array[0:length, :]
            else:
                data_array = np.concatenate(
                    [data_array, np.zeros(shape=(length - data_array.shape[0], 20))])
            data_array = np.reshape(data_array, (-1,))
            data_array_list.append(data_array)
        df = pd.DataFrame(np.stack(data_array_list))
        df.index = index_
        df = df.loc[seq_id_dict[taskname[0]][taskname[2]], :]
        result.append(df)
    return result


def get_all_task_feature(
    possum_index_dict: dict,
    path_to_json_seq_id: str,
    feature_name_list: list,
    path_to_fasta_pattern: str,
    path_to_with_pattern: str
):
    seq_id_dict = None
    with open(path_to_json_seq_id, 'r', encoding='UTF-8') as f:
        seq_id_dict = json.load(f)

    result = list()
    for taskname in ['t_p', 't_n', 'v_p', 'v_n']:
        df = pssm_tools.get_all_pssm_feature(
            possum_index_list=possum_index_dict['data'][taskname],
            feature_name_list=feature_name_list,
            path_to_fasta=path_to_fasta_pattern.format(taskname=taskname),
            path_to_with_pattern=path_to_with_pattern
        )
        df = df.loc[seq_id_dict[taskname[0]][taskname[2]], :]
        result.append(df)
    return result
