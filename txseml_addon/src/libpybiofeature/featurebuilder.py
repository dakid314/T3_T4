'''
Author: George Zhao
Date: 2021-08-08 21:28:24
LastEditors: George Zhao
LastEditTime: 2022-06-27 23:45:51
Description: 
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''
from re import A
import sys 
sys.path.append("/mnt/md0/Public/T3_T4/txseml_addon/src/libpybiofeature/")
import pandas as pd
import AC
import PPT
import CTD
import oneHot
import CTriad
import QSO
import BLOSUM62
import CKSAAP
import eighteen_physicochemical_properties as etpp
import tqdm
from Bio import SeqIO


def build_acc_feature(
    path_to_fasta: str,
    seq_id_list: list,
    desc: str = 'undefine',
    NCF='F',
    terlength: int = None
):

    seq_list = list(SeqIO.parse(path_to_fasta, 'fasta'))
    df = None
    if NCF == 'N':
        df = pd.DataFrame([
            AC.AAC(str(seq.seq)[1:terlength + 1])
            for seq in tqdm.tqdm(seq_list, desc=f'{desc}_AAC')
        ])
    elif NCF == 'C':
        df = pd.DataFrame([
            AC.AAC(str(seq.seq)[-1 * terlength:])
            for seq in tqdm.tqdm(seq_list, desc=f'{desc}_AAC')
        ])
    else:
        df = pd.DataFrame([
            AC.AAC(str(seq.seq)[1:])
            for seq in tqdm.tqdm(seq_list, desc=f'{desc}_AAC')
        ])

    df.columns = AC.sac_default_header
    df.index = [seq.id for seq in seq_list]

    if seq_id_list is not None:
        return df.loc[:, :]
    else:
        return df.loc[seq_id_list, :]


def build_dac_feature(
    path_to_fasta: str,
    seq_id_list: list,
    desc: str = 'undefine',
    interval: int = 0,
    NCF=' F',
    terlength: int = None
):
    seq_list = list(SeqIO.parse(path_to_fasta, 'fasta'))
    df = None
    if NCF == 'N':
        df = pd.DataFrame([
            AC.DAC(str(seq.seq)[1:terlength + 1], interval=interval)
            for seq in tqdm.tqdm(seq_list, desc=f'{desc}_DAC')
        ])
    elif NCF == 'C':
        df = pd.DataFrame([
            AC.DAC(str(seq.seq)[-1 * terlength:], interval=interval)
            for seq in tqdm.tqdm(seq_list, desc=f'{desc}_DAC')
        ])
    else:
        df = pd.DataFrame([
            AC.DAC(str(seq.seq)[1:], interval=interval)
            for seq in tqdm.tqdm(seq_list, desc=f'{desc}_DAC')
        ])

    df.columns = [
        f"{item[0]},{'*'*interval},{item[-1]}" if interval > 0 else f"{item[0]},{item[-1]}" for item in AC.dac_default_header
    ]
    df.index = [seq.id for seq in seq_list]

    if seq_id_list is not None:
        return df.loc[seq_id_list, :]
    else:
        return df.loc[seq_id_list, :]


def build_tac_feature(
    path_to_fasta: str,
    seq_id_list: list,
    desc: str = 'undefine',
    NCF=' F',
    terlength: int = None
):
    seq_list = list(SeqIO.parse(path_to_fasta, 'fasta'))
    df = None
    if NCF == 'N':
        df = pd.DataFrame([
            AC.TAC(str(seq.seq)[1:terlength + 1])
            for seq in tqdm.tqdm(seq_list, desc=f'{desc}_TAC')
        ])
    elif NCF == 'C':
        df = pd.DataFrame([
            AC.TAC(str(seq.seq)[-1 * terlength:])
            for seq in tqdm.tqdm(seq_list, desc=f'{desc}_TAC')
        ])
    else:
        df = pd.DataFrame([
            AC.TAC(str(seq.seq)[1:])
            for seq in tqdm.tqdm(seq_list, desc=f'{desc}_TAC')
        ])

    df.columns = AC.tac_default_header
    df.index = [seq.id for seq in seq_list]

    if seq_id_list is not None:
        return df.loc[seq_id_list, :]
    else:
        return df


def build_etpp_feature(
    path_to_fasta: str,
    seq_id_list: list,
    desc: str = 'undefine',
    cter: bool = False
):
    seq_list = list(SeqIO.parse(path_to_fasta, 'fasta'))

    df = pd.DataFrame([
        etpp.eighteen_pp_seqencode(str(seq.seq)[1:], cter=cter)
        for seq in tqdm.tqdm(seq_list, desc=f'{desc}_etpp')
    ])

    df.columns = etpp.etpp_header
    df.index = [seq.id for seq in seq_list]

    return df.loc[seq_id_list, :]


def build_conjoint_td_feature(
    path_to_fasta: str,
    seq_id_list: list,
    desc: str = 'undefine'
):
    seq_list = list(SeqIO.parse(path_to_fasta, 'fasta'))

    df = pd.DataFrame([
        CTriad.CTriad(str(seq.seq)[1:])
        for seq in tqdm.tqdm(seq_list, desc=f'{desc}_CTriad')
    ])

    df.columns = CTriad.features
    df.index = [seq.id for seq in seq_list]

    return df.loc[seq_id_list, :]


def build_CKSAAP_feature(
    path_to_fasta: str,
    seq_id_list: list,
    desc: str = 'undefine'
):
    seq_list = list(SeqIO.parse(path_to_fasta, 'fasta'))

    df = pd.DataFrame([
        CKSAAP.CKSAAP(str(seq.seq)[1:])
        for seq in tqdm.tqdm(seq_list, desc=f'{desc}_CKSAAP')
    ])
    df.columns = CKSAAP.defalut_param['header']
    df.index = [seq.id for seq in seq_list]

    return df.loc[seq_id_list, :]


def _get_ppt_pre25_aa(seq: str, cter: bool):
    if len(seq) >= 25:
        return seq[:25]
    if cter == True:
        return 'A' * (25 - len(seq)) + seq
    else:
        return seq + 'A' * (25 - len(seq))
def _get_ppt_pre_aa(seq: str, cter: bool):
    if cter:
        return seq[::-1]
    else:
        return seq

def build_PPT25_feature(
    path_to_fasta: str,
    seq_id_list: list,
    desc: str = 'undefine',
    fulllong: bool = False,
    cter: bool = False
):
    def _fulllong_func(x, cter): return x
    fulllong_func = _fulllong_func
    if fulllong == False:
        # For EffectiveT3.
        fulllong_func = _get_ppt_pre25_aa

    seq_list = list(SeqIO.parse(path_to_fasta, 'fasta'))

    df = pd.DataFrame([
        PPT.PPT(
            fulllong_func(
                str(seq.seq)[1:]
                if cter == False else
                ''.join(
                    reversed(
                        str(seq.seq)[1:]
                    )
                ),
                cter
            )
        )
        for seq in tqdm.tqdm(seq_list, desc=f'{desc}_PPT')
    ])
    df.columns = PPT.default_header
    df.index = [seq.id for seq in seq_list]

    return df
def build_PPT_feature(
    path_to_fasta: str,
    seq_id_list: list,
    desc: str = 'undefine',
    fulllong: bool = True,
    cter: bool = False
):
    def _fulllong_func(x, cter): return x
    fulllong_func = _fulllong_func
    if fulllong == True:
        # For EffectiveT3.
        fulllong_func = _get_ppt_pre_aa

    seq_list = list(SeqIO.parse(path_to_fasta, 'fasta'))

    df = pd.DataFrame([
        PPT.PPT(
            fulllong_func(
                str(seq.seq)[1:]
                if cter == False else
                ''.join(
                    reversed(
                        str(seq.seq)[1:]
                    )
                ),
                cter
            )
        )
        for seq in tqdm.tqdm(seq_list, desc=f'{desc}_PPT')
    ])
    df.columns = PPT.default_header
    df.index = [seq.id for seq in seq_list]

    return df

def build_oneHot_feature(
    seq_list : list,
    
    length: int,
    desc: str = 'undefine',
    cter: bool = False
):
    df = pd.DataFrame([
        
        oneHot.oneHot(str(seq.seq)[1:], length=length, cter=cter)
        for seq in tqdm.tqdm(seq_list, desc=f'{desc}_oneHot')
    ])
    df.index = [seq.id for seq in seq_list]
    
    return df


def build_CTDC_feature(
    path_to_fasta: str,
    seq_id_list: list,
    desc: str = 'undefine'
):

    seq_list = list(SeqIO.parse(path_to_fasta, 'fasta'))

    df = pd.DataFrame([
        CTD.CTDC(str(seq.seq))
        for seq in tqdm.tqdm(seq_list, desc=f'{desc}_CTDC')
    ])

    df.columns = CTD.CTDC_localobj.header
    df.index = [seq.id for seq in seq_list]

    if seq_id_list is not None:
        return df.loc[seq_id_list, :]
    else:
        return df


def build_CTDT_feature(
    path_to_fasta: str,
    seq_id_list: list,
    desc: str = 'undefine'
):

    seq_list = list(SeqIO.parse(path_to_fasta, 'fasta'))

    df = pd.DataFrame([
        CTD.CTDT(str(seq.seq))
        for seq in tqdm.tqdm(seq_list, desc=f'{desc}_CTDT')
    ])

    df.columns = CTD.CTDT_localobj.header
    df.index = [seq.id for seq in seq_list]

    return df.loc[seq_id_list, :]


def build_CTDD_feature(
    path_to_fasta: str,
    seq_id_list: list,
    desc: str = 'undefine'
):

    seq_list = list(SeqIO.parse(path_to_fasta, 'fasta'))

    df = pd.DataFrame([
        CTD.CTDD(str(seq.seq))
        for seq in tqdm.tqdm(seq_list, desc=f'{desc}_CTDD')
    ])

    df.columns = CTD.CTDD_localobj.header
    df.index = [seq.id for seq in seq_list]

    return df.loc[seq_id_list, :]


def build_qso_feature(
    path_to_fasta: str,
    seq_id_list: list,
    desc: str = 'undefine',
    cter: bool = False
):
    seq_list = list(SeqIO.parse(path_to_fasta, 'fasta'))

    df = pd.DataFrame([
        QSO.QSOrder(
            (str(seq.seq) + ('' if len(seq.seq) >= 31 else 'A' * (31 - len(seq.seq))))
            if cter == False else
            (('' if len(seq.seq) >= 31 else 'A' * (31 - len(seq.seq))) + str(seq.seq))
        )
        for seq in tqdm.tqdm(seq_list, desc=f'{desc}_QSO')
    ])

    df.columns = QSO.default_header
    df.index = [seq.id for seq in seq_list]

    return df


def build_b62_feature(
    path_to_fasta: str,
    seq_id_list: list,
    desc: str = 'undefine',
    cter: bool = False
):
    def _r0(x): return x
    def _r1(x): return ''.join(reversed(x))
    redef = _r0
    if cter == True:
        redef = _r1
    seq_list = list(SeqIO.parse(path_to_fasta, 'fasta'))

    df = pd.DataFrame([
        BLOSUM62.BLOSUM62(
            redef(str(seq.seq)[1:]), length=25)
        for seq in tqdm.tqdm(seq_list, desc=f'{desc}_B62')
    ])

    df.index = [seq.id for seq in seq_list]

    return df.loc[seq_id_list, :]
