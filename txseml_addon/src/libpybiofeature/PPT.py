'''
Author: George Zhao
Date: 2021-08-02 17:33:11
LastEditors: George Zhao
LastEditTime: 2022-02-27 23:20:22
Description: 
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''
import itertools
import sys
sys.path.append('/mnt/md0/Public/一站式/txseml_backend_addon/txseml_addon/src/')
import os
import re
import collections

from utils import workdir
lib_Dir = os.path.join(workdir.workdir(os.getcwd(), 3), 'lib')

import tqdm
from Bio import SeqIO

table_1_1 = {
    'A': 0,
    'R': 1,
    'N': 2,
    'D': 3,
    'E': 4,
    'Q': 5,
    'G': 6,
    'H': 7,
    'I': 8,
    'L': 9,
    'K': 10,
    'M': 11,
    'F': 12,
    'P': 13,
    'S': 14,
    'T': 15,
    'Y': 16,
    'V': 27,
}
table_1_2 = {
    'SL': 18,
    'SS': 19,
    'TL': 20,
}
table_2 = {
    "A": 1,
    "G": 1,
    "I": 1,
    "L": 1,
    "M": 1,
    "V": 1,
    "P": 2,
    "H": 2,
    "U": 2,
    "W": 0,
    "Y": 0,
    "N": 0,
    "Q": 0,
    "S": 0,
    "T": 0,
    "D": 0,
    "E": 0,
    "K": 0,
    "R": 0,
    "C": 0,
    "F": 0,
}
table_3 = {
    "A": [9, ],
    "G": [9, ],
    "I": [9, ],
    "L": [6, 9],
    "M": [9, ],
    "V": [9, ],
    "P": [8, ],
    "H": [8, ],
    "U": [8, ],
    "W": [3, 8],
    "Y": [7, 3, 8],
    "N": [4, 8],
    "Q": [4, 8],
    "S": [4, 8],
    "T": [4, 8],
    "D": [5, 8],
    "E": [5, 8],
    "K": [6, 8],
    "R": [6, 8],
    "C": [7, 8],
    "F": [3, 8],
}
# %%
paper_col_2 = None
with open(os.path.join(lib_Dir, 'EffectiveT3/2'), 'r', encoding='UTF-8') as f:
    paper_col_2 = [i[0] for i in re.findall(r'\[((\d,?)+)\]', f.read())]

paper_col_3 = None
with open(os.path.join(lib_Dir, 'EffectiveT3/3'), 'r', encoding='UTF-8') as f:
    paper_col_3 = [i[0] for i in re.findall(r'\[((\d,?)+)\]', f.read())]

# %%


def get_col1(seq: str):
    sa = collections.Counter([
        ac
        for ac in seq
        if ac in table_1_1
    ])
    da = collections.Counter([
        f'{ac1}{ac2}'
        for ac1, ac2 in zip(seq[:len(seq) - 1], seq[1:])
        if f'{ac1}{ac2}' in table_1_2
    ])

    for k in table_1_1.keys():
        if k not in sa:
            sa[k] = 0
    for k in table_1_2.keys():
        if k not in da:
            da[k] = 0

    result = dict()
    result.update(dict(sa))
    result.update(dict(da))

    return result

# %%


def get_col2(seq: str):

    da = collections.Counter([
        f'{table_2[ac1]},{table_2[ac2]}'
        for ac1, ac2 in zip(seq[:len(seq) - 1], seq[1:])
        if f'{table_2[ac1]},{table_2[ac2]}' in paper_col_2
    ])

    ta = collections.Counter([
        f'{table_2[ac1]},{table_2[ac2]},{table_2[ac3]}'
        for ac1, ac2, ac3 in zip(seq[:len(seq) - 2], seq[1:len(seq) - 1], seq[2:])
        if f'{table_2[ac1]},{table_2[ac2]},{table_2[ac3]}' in paper_col_2
    ])

    result = dict()
    result.update(dict(da))
    result.update(dict(ta))

    for k in paper_col_2:
        if k not in result:
            result[k] = 0

    return result

# %%


def get_col3(seq: str):
    sa = collections.Counter([
        f'{table_3[ac1][0]}'
        for ac1 in seq
        if f'{table_3[ac1][0]}' in paper_col_3
    ])

    da = collections.Counter([
        f'{table_3[ac1][0]},{table_3[ac2][0]}'
        for ac1, ac2 in zip(seq[:len(seq) - 1], seq[1:])
        if f'{table_3[ac1][0]},{table_3[ac2][0]}' in paper_col_3
    ])

    ta = collections.Counter([
        f'{table_3[ac1][0]},{table_3[ac2][0]},{table_3[ac3][0]}'
        for ac1, ac2, ac3 in zip(seq[:len(seq) - 2], seq[1:len(seq) - 1], seq[2:])
        if f'{table_3[ac1][0]},{table_3[ac2][0]},{table_3[ac3][0]}' in paper_col_3
    ])

    result = dict()
    result.update(dict(sa))
    result.update(dict(da))
    result.update(dict(ta))

    for k in paper_col_3:
        if k not in result:
            result[k] = 0
    return result


def seq_feature(seq: str):
    result = dict()
    result.update(get_col1(seq))
    result.update(get_col2(seq))
    result.update(get_col3(seq))
    return result


def get_feature(path_to_fasta: str, cter: bool):
    len_of_fasta = len(list(SeqIO.parse(
        path_to_fasta,
        'fasta'
    )))
    return [
        {
            'id': seq.id,
            'Feature': seq_feature(
                seq=(
                    str(seq.seq)[1:26]
                    if len(seq.seq[1:]) >= 25 else
                    str(seq.seq)[1:] + 'A' * (25 - len(seq.seq[1:]))
                )
            )
        } for seq in tqdm.tqdm(
            SeqIO.parse(
                path_to_fasta,
                'fasta'
            ), total=len_of_fasta)
    ]


default_header = list(itertools.chain(*[
    table_1_1.keys(), table_1_2.keys(), paper_col_2, paper_col_3
]))


def PPT(seq_aa: str, desc='undefine'):
    if len(seq_aa) < 3:
        raise ValueError(
            f'PPT: Input sequences:{desc} should be greater than or 3 (Get {len(seq_aa)}).')
    result = seq_feature(seq_aa)
    return [
        result[k]
        for k in sorted(result.keys())
    ]
