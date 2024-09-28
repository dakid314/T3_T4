'''
Author: George Zhao
Date: 2022-06-22 10:47:15
LastEditors: George Zhao
LastEditTime: 2022-08-14 13:50:19
Description: 
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''


def valver(
    seq: str,
    terlength_min: int,
    terlength_max: int,
    cter: bool,
    padding_ac='A',
    remove_first: bool = True
):
    seq = str(seq)
    if cter == False and remove_first == True:
        seq = seq[1:]
    if cter == False:
        if len(seq) > terlength_max:
            seq = seq[0:terlength_max]
        if len(seq) < terlength_min:
            seq = seq + padding_ac * (terlength_min - len(seq))
    else:
        if len(seq) > terlength_max:
            seq = seq[-terlength_max:]
        if len(seq) < terlength_min:
            seq = padding_ac * (terlength_min - len(seq)) + seq
    return seq


def trimer(
    seq: str,
    terlength: int,
    cter: bool,
    padding_ac='A',
    remove_first: bool = True
):
    seq = str(seq)
    if cter == False and remove_first == True:
        seq = seq[1:]
    if cter == False:
        if len(seq) >= terlength:
            seq = seq[0:terlength]
        else:
            seq = seq + padding_ac * (terlength - len(seq))
    else:
        if len(seq) >= terlength:
            seq = seq[-terlength:]
        else:
            seq = padding_ac * (terlength - len(seq)) + seq
    return seq
