'''
Author: George Zhao
Date: 2022-06-24 09:52:27
LastEditors: George Zhao
LastEditTime: 2022-06-24 09:53:40
Description: 
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''
import pandas as pd
from Bio import SeqIO


def prepare_data(
    path_to_fasta: str,
    seq_id_list: list = None,
):
    # Loadfasta
    seq_list = list(SeqIO.parse(path_to_fasta, 'fasta'))

    if seq_id_list is None:
        return pd.DataFrame([seq_list, ])
    else:
        db = {
            seq.id: [seq, ] for seq in seq_list
        }
        return pd.DataFrame([db[seqid] for seqid in seq_id_list])


