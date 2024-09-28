'''
Author: George Zhao
Date: 2021-08-07 14:54:03
LastEditors: George Zhao
LastEditTime: 2021-08-07 15:18:01
Description: 
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''
import collections

sac_default_aaorder = 'ACDEFGHIKLMNPQRSTVWY'

sac_default_header = [aa for aa in sac_default_aaorder]


def _get_dac_order(aaorder: str = sac_default_aaorder):
    return [f'{aa1},{aa2}' for aa1 in aaorder for aa2 in aaorder]


dac_default_order = _get_dac_order()
dac_default_header = dac_default_order


def _get_tac_order(aaorder: str = sac_default_aaorder):
    return [f'{aa1},{aa2},{aa3}' for aa1 in aaorder for aa2 in aaorder for aa3 in aaorder]


tac_default_order = _get_tac_order()
tac_default_header = tac_default_order


def AAC(seq_aa: str, desc='undefine', aaorder: str = sac_default_aaorder):
    code = list()
    count = collections.Counter(seq_aa)
    for key in count:
        count[key] = count[key] / len(seq_aa)
    for aa in aaorder:
        code.append(count[aa])
    return code


def DAC(seq_aa: str, interval: int = 0, desc='undefine', dacorder: list = dac_default_order):
    if len(seq_aa) < interval + 2:
        raise ValueError(
            f'DAC: Input sequences:{desc} should be greater than or equal to interval({interval}) + 2 (Get {len(seq_aa)}).')
    count = collections.Counter([
        f'{aa1},{aa2}' for aa1, aa2 in zip(
            seq_aa[:len(seq_aa) - interval - 1], seq_aa[interval + 1:]
        )
    ])
    for key in count:
        count[key] = count[key] / (len(seq_aa) - interval - 1)
    code = list()
    for dac in dacorder:
        code.append(count[dac])
    return code


def TAC(seq_aa: str, desc='undefine', tacorder: list = tac_default_header):
    if len(seq_aa) < 3:
        raise ValueError(
            f'TAC: Input sequences:{desc} should be greater than or equal to 3 (Get {len(seq_aa)}).')
    count = collections.Counter([
        f'{aa1},{aa2},{aa3}' for aa1, aa2, aa3 in zip(
            seq_aa[:len(seq_aa) - 2],
            seq_aa[1:len(seq_aa) - 1],
            seq_aa[2:]
        )
    ])
    for key in count:
        count[key] = count[key] / (len(seq_aa) - 2)
    code = list()
    for tac in tacorder:
        code.append(count[tac])
    return code
