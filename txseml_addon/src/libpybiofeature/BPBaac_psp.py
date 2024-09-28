'''
Author: George Zhao
Date: 2022-06-21 21:18:03
LastEditors: George Zhao
LastEditTime: 2022-06-22 15:23:42
Description: BPBaac Tools Profile Generation
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''
import typing
import sys
sys.path.append('/mnt/md0/Public/T3_T4/txseml_addon/src')
import itertools
from utils import seq_length_process
import collections
from Bio import SeqIO


def count_to_freq(cc: collections.Counter):
    cc = dict(cc)
    total_num = sum([cc[key] for key in cc.keys()])
    cc = {
        key: (cc[key] / total_num) for key in cc.keys()
    }
    return cc


def safe_map_geter(d: dict, k: str, default):
    if k in d:
        return d[k]
    return default


def mat_constructor(
    fasta_db: typing.Union[str, list],
    cter: bool,
    terlength: int = 100,
    padding_ac='A',
):
    # Read Seq_list
    if isinstance(fasta_db, str) == True:
        fasta_db = list(SeqIO.parse(fasta_db, "fasta"))

    # Trim Seq
    fasta_db = [
        seq_length_process.trimer(
            seq=str(seq.seq), terlength=terlength, cter=cter, padding_ac=padding_ac, remove_first=True
        )
        for seq in fasta_db
    ]

    # Feature Extract
    mat = [
        count_to_freq(collections.Counter([
            seq[position_index]
            for seq in fasta_db
        ]))
        for position_index in range(terlength)
    ]
    return mat


def mat_mapper(
    seq: str,
    pmat: typing.List[dict],
    nmat: typing.List[dict],
    cter: bool,
    terlength: int = 100,
    padding_ac='A',
):
    seq = seq_length_process.trimer(
        seq=seq, terlength=terlength, cter=cter, padding_ac=padding_ac, remove_first=True
    )
    return list(itertools.chain(*[
        [
            safe_map_geter(pmat[position_index], seq[position_index], 0),
            safe_map_geter(nmat[position_index], seq[position_index], 0),
        ]
        for position_index in range(terlength)
    ]))
