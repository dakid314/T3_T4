'''
Author: George Zhao
Date: 2021-08-02 14:38:13
LastEditors: George Zhao
LastEditTime: 2022-02-27 23:18:58
Description: 
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''
import sys
sys.path.append('..')
import os
import itertools
from utils import workdir

from Bio.Align import substitution_matrices


def build_substitution_matrices(name: str):
    mat = None
    with open(os.path.join(workdir.workdir(os.getcwd(), 3), f'lib/substitution_matrices/{name}')) as f:
        mat = substitution_matrices.read(f, dtype=float)
    return mat


_mat_62 = build_substitution_matrices('BLOSUM62')

blosum62_dict = {
    aa: val.tolist()
    for aa, val in zip(_mat_62.alphabet, _mat_62)
}

_size_of_aa = len(_mat_62.alphabet)


def BLOSUM62(seq_aa: str, desc='undefine', length: int = None, cter: bool = False):
    result = list(
        itertools.chain(*[
            blosum62_dict[aa] for aa in seq_aa
        ])
    )
    if cter != False:
        result = list(reversed(result))
    if length is not None:
        if len(seq_aa) > length:
            return result[:_size_of_aa * length]
        else:
            result.extend(([0, ] * _size_of_aa) * (length - len(seq_aa)))
            return result
    else:
        return result
