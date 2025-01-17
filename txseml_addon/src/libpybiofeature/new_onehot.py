'''
Author: George Zhao
Date: 2021-08-07 23:44:00
LastEditors: George Zhao
LastEditTime: 2022-02-27 23:22:02
Description: 
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''
import sys
sys.path.append('/mnt/md0/Public/T3_T4/txseml_addon/src/')
import itertools
from utils import ds_preprocess
from Bio.Seq import Seq
default_aaorder = 'ACDEFGHIKLMNPQRSTVWY'
default_header = [aa for aa in default_aaorder]


def get_key_dict(aaorder: list):
    return {
        k: i
        for i, k in enumerate(aaorder)
    }


default_aa_dict = get_key_dict(default_aaorder)


def oneHot(seq_aa: str, desc='undefine', length: int = None, aaorder: str = default_aaorder, cter: bool = False):
    if cter != False:
        seq_aa = list(reversed(seq_aa))
    
    # Find the index of the first 'M' in the sequence
    
    
    seq_aa = str(seq_aa)
    seq_aa = Seq(seq_aa.replace('M', '', 1))
    
    result = list(
        itertools.chain(
            *[
                ds_preprocess.consturct_vertor(
                    default_aa_dict[aa]
                    if aa in default_aa_dict.keys()
                    else
                    None
                ).tolist()
                for aa in seq_aa
            ]
        )
    )

    if length is  None:
        length = 20
        if len(seq_aa) > length:
            return result[:20 * length]
        else:
            result.extend(ds_preprocess.consturct_vertor(
                None).tolist() * (length - len(seq_aa)))
            return result
    else:
        return result
