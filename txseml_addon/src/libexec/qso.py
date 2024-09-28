'''
Author: George Zhao
Date: 2021-08-03 16:15:30
LastEditors: George Zhao
LastEditTime: 2022-02-27 23:20:49
Description:
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''

import csv
from Bio import SeqIO
import sys 
sys.path.append("/mnt/md0/Public/一站式/txseml_backend_addon/txseml_addon/src/libpybiofeature")
import featurebuilder
import pandas as pd


def Q(fasta_file: str, output_file: str):
    # 调用 build_qso_feature 函数生成特征数据
    seq_list = list(SeqIO.parse(fasta_file, 'fasta'))
    seq_id_list = [seq.id for seq in seq_list]
    
    df = featurebuilder.build_qso_feature(fasta_file, seq_id_list)

    # 将特征数据保存为CSV文件
    df.to_csv(output_file)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog='psipred')
    parser.add_argument('-i', type=str, required=True, help='Path to fasta.')
    parser.add_argument('-o', type=str, required=True,
                        help='Path to Out.')
    args = parser.parse_args()
    Q(fasta_file=args.i,
        output_file = args.o
        )