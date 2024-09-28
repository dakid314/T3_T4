'''
Author: George Zhao
Date: 2021-05-29 14:00:56
LastEditors: George Zhao
LastEditTime: 2022-03-05 23:00:53
Description: 
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''
# %%
import os
import sys
sys.path.append('src')
# %%
import utils
from Bio import SeqIO, SeqUtils, Seq, SeqRecord
import functools
import itertools


work_Dir = utils.workdir.workdir(os.getcwd(), 3)
rdata_Dir = os.path.join(work_Dir, 'data')

# %%
import collections
import json
import re
out_Dir = os.path.join(work_Dir, 'out')
# %%
from Bio import Align
from Bio.Align import substitution_matrices
import Get_substitution_matrices as gsm
import typing
import gc
# %%


def build_aligner(substitution_matrices: str):
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    aligner.open_gap_score = 10
    aligner.extend_gap_score = 0.5
    aligner.substitution_matrix = gsm.Build_substitution_matrices(
        substitution_matrices)
    return aligner


def get_fasta_database(path_to_fasta_database: typing.List[str]):
    return [str(seq.seq) for seq in itertools.chain(*[SeqIO.parse(path, 'fasta') for path in path_to_fasta_database])]


def get_Aligner(substitution_matrices_name_list: list):
    return {
        substitution_matrices_name: build_aligner(substitution_matrices_name) for substitution_matrices_name in ['BLOSUM35', 'BLOSUM40', 'BLOSUM45']
    }


substitution_matrices_name_list = ['BLOSUM35', 'BLOSUM40', 'BLOSUM45']
substitution_matrices_list = get_Aligner(substitution_matrices_name_list)


def get_Score_Matrix(seq_to_align: str, seq_to_align_id: str, fasta_database: typing.List[str]):
    # logger.logger.info(f'Seq: {seq_to_align_id} Start to Align.')
    result = {
        substitution_matrices_name: [
            substitution_matrices_list[substitution_matrices_name].align(
                seq_to_align, seq).score for seq in fasta_database
        ] for substitution_matrices_name in substitution_matrices_list
    }
    return result
# %%


def get_Score_Matrix_In_fasta(path_to_fasta: str, fasta_database: typing.List[str]):
    return [
        {
            'id': seq.id,
            'Feature': get_Score_Matrix(str(seq.seq), seq.id, fasta_database)
        }
        for seq in SeqIO.parse(path_to_fasta, 'fasta')
    ]


# %%
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog='PairWisealigner')
    parser.add_argument('-f', type=str, required=True, help='Path to fasta.')
    parser.add_argument('-o', type=str, required=True, help='Path to out.')
    parser.add_argument('--db', type=str, required=True, help='Path to DB.')
    args = parser.parse_args()
    T3SEs_fasta = get_Score_Matrix_In_fasta(
        args.f,
        fasta_database=get_fasta_database([
            args.db,
        ])
    )
    final_dataset_dict = {
        'name': os.path.split(os.path.splitext(args.f)[0])[1],
        'data': T3SEs_fasta
    }
    with open(args.o, 'w+', encoding='UTF-8') as f:
        json.dump(
            final_dataset_dict,
            f
        )

# %%
