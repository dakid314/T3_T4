'''
Author: George Zhao
Date: 2021-08-05 12:00:50
LastEditors: George Zhao
LastEditTime: 2021-08-25 14:25:24
Description: 
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''
import itertools
import functools

physicochemical_properties_dict = {
    "charged": ["D", "E", "K", "H", "R"],
    "aliphatic": ["I", "L", "V"],
    "aromatic": ["F", "H", "W", "Y"],
    "polar": ["D", "E", "R", "K", "Q", "N"],
    "neutral": ["A", "G", "H", "P", "S", "T", "Y"],
    "hydrophobic": ["C", "F", "I", "L", "M", "V", "W"],
    "positively_charged": ["K", "R", "H"],
    "negatively_charged": ["D", "E"],
    "tiny": ["A", "C", "D", "G", "S", "T"],
    "small": ["E", "H", "I", "L", "K", "M", "N", "P", "Q", "V"],
    "large": ["F", "R", "W", "Y"],
    "transmembrane_amino_acid": ["I", "L", "V", "A"],
    "dipole_0": ["A", "G", "V", "I", "L", "F", "P"],
    "dipole_1": ["Y", "M", "T", "S"],
    "dipole_2": ["H", "N", "Q", "W"],
    "dipole_3": ["R", "K"],
    "dipole_4": ["D", "E", "C"],
}

ppvectorsize = len(physicochemical_properties_dict.keys())
vectorsize = ppvectorsize + 1

_key_index_dict = dict(
    zip(physicochemical_properties_dict.keys(), range(ppvectorsize))
)


def _get_support_aa():
    aa_set = set()
    for pp_key in physicochemical_properties_dict.keys():
        aa_set.update(
            physicochemical_properties_dict[pp_key]
        )
    return aa_set


support_aa = _get_support_aa()


def aa_encode(aa: str):
    _vector = [0, ] * ppvectorsize
    for pp_key in physicochemical_properties_dict.keys():
        if aa in physicochemical_properties_dict[pp_key]:
            _vector[_key_index_dict[pp_key]] = 1
    _vector.append(sum(_vector) / ppvectorsize)
    return _vector


aa_to_vector = {
    aa: aa_encode(aa) for aa in support_aa
}


def eighteen_pp_aaencode(seq_aa: str, desc: str = '', mat_or_vector: str = 'vector'):
    # For Seq aa position encode.
    result_mat = list()
    for aa in seq_aa:
        if aa not in support_aa:
            result_mat.append([0, ] * vectorsize)
            continue
        else:
            result_mat.append(aa_to_vector[aa])
    if mat_or_vector == 'vector':
        return list(itertools.chain(*result_mat))

    # mat_or_vector == 'mat'
    return result_mat


def _vector_add(l1: list, l2: list):
    if len(l1) != len(l2):
        raise ValueError(
            f'_vector_add: Found len(l1) != len(l2) (Get {len(l1)},{len(l2)}).')
    return [
        e1 + e2
        for e1, e2 in zip(l1, l2)
    ]


etpp_header = [k for k in physicochemical_properties_dict.keys()] + \
    ['etpp_percent', ]


def eighteen_pp_seqencode(seq_aa: str, desc: str = 'undefine', cter: bool = False):
    if cter != False:
        seq_aa = list(reversed(seq_aa))
    # For Seq aa composition encode.
    result_mat = list()
    for aa in seq_aa:
        if aa not in support_aa:
            result_mat.append([0, ] * vectorsize)
            continue
        else:
            result_mat.append(aa_to_vector[aa])
    result = functools.reduce(
        _vector_add, result_mat
    )
    result = list(map(lambda x: x / len(seq_aa), result))
    result[vectorsize - 1] = sum(result[:ppvectorsize]) / \
        (ppvectorsize * len(seq_aa))
    return result
