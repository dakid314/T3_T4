'''
Author: George Zhao
Date: 2021-07-25 13:01:05
LastEditors: George Zhao
LastEditTime: 2022-02-28 00:07:51
Description: 
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''
import os
import itertools
import re
import zipfile
import sys
sys.path.append('../..')

from utils import workdir
work_Dir = workdir.workdir(os.getcwd(), 3)
rdata_Dir = os.path.join(work_Dir, 'data')
lib_Dir = os.path.join(work_Dir, 'lib')
tmp_Dir = os.path.join(work_Dir, 'tmp')
out_Dir = os.path.join(work_Dir, 'out')

import numpy as np
import pandas as pd
from Bio import SeqIO

# For PSSM file.


def get_pssm_from_file(content: str):
    content = list(map(
        lambda l: [w for w in l if w != ''],
        map(
            lambda x: x.split(' '),
            content.split('\n\n')[0][179:].splitlines()
        )
    ))
    col_name = content[0][0:int(len(content[0]) / 2)]
    col_asdf = np.array([list(map(float, l[2:-2])) for l in content[1:]])
    col_asdf_1 = col_asdf[:, 0:int(col_asdf.shape[1] / 2)]
    col_asdf_2 = col_asdf[:, int(col_asdf.shape[1] / 2):]
    summury_data = np.array([list(map(float, l[-2:])) for l in content[1:]])
    return {
        "col_name": col_name,
        "form_1": col_asdf_1,
        "form_2": col_asdf_2,
        "summury_data": summury_data
    }


def get_pssm_in_order(order_list: list, path_with_pattern: str):
    to_chain = []
    for zipid in order_list:
        with zipfile.ZipFile(path_with_pattern.format(zipid=zipid), 'r') as __zip:
            filelist = [
                _file
                for _file in __zip.namelist()
                if re.findall(r'pssm/[a-z0-9]+/[a-z0-9]+_\d+.pssm', _file) != []
            ]
            to_chain.append(
                list(map(
                    lambda i: __zip.open(
                        f'pssm/{zipid}/{zipid}_{i}.pssm', 'r').read().decode('UTF-8'),
                    range(1, len(filelist) + 1)
                ))
            )
    return itertools.chain(*to_chain)

# For PSSM Feature.


def get_pssm_feature_in_order(
        order_list: list,
        feature_name: str,
        path_to_fasta: str,
        path_with_pattern: str):
    to_chain = []

    for zipid in order_list:
        with zipfile.ZipFile(path_with_pattern.format(zipid=zipid), 'r') as __zip:
            filelist = [
                _file
                for _file in __zip.namelist()
                if re.findall('originalFeatures/[a-z0-9]+/[a-z0-9]+_' + feature_name + '.csv', _file) != []
            ][0]
            with __zip.open(f'originalFeatures/{zipid}/{zipid}_{feature_name}.csv', 'r') as f_:
                to_chain.append(
                    pd.read_csv(f_)
                )

    df = pd.concat(to_chain, ignore_index=True)
    df.index = [i.id for i in SeqIO.parse(path_to_fasta, 'fasta')]

    return df


def get_all_pssm_feature(
    possum_index_list: list,
    feature_name_list: list,
    path_to_fasta: str,
    path_to_with_pattern: str
):
    return pd.concat([
        get_pssm_feature_in_order(
            order_list=possum_index_list,
            feature_name=feature_name,
            path_to_fasta=path_to_fasta,
            path_with_pattern=path_to_with_pattern
        ) for feature_name in feature_name_list
    ], axis=1)
