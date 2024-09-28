import csv
from Bio import SeqIO
import sys
from Bio.Seq import Seq
sys.path.append("/mnt/md0/Public/一站式/txseml_backend_addon/txseml_addon/src/libpybiofeature")
import BPBaac_psp
import csv


def BPBaac(pos_file, neg_file, val_file,output_file,cter):
    # 读取FASTA文件
    pos_fasta_sequences = list(SeqIO.parse(pos_file, "fasta"))
    neg_fasta_sequences = list(SeqIO.parse(neg_file, "fasta"))
    val_fasta = SeqIO.parse(val_file, "fasta")
    # 拼接为 fasta_sequences
    fasta_sequences = val_fasta

    # 生成正向特征矩阵
    pmat = BPBaac_psp.mat_constructor(pos_fasta_sequences, cter=True)

    # 生成负向特征矩阵
    nmat = BPBaac_psp.mat_constructor(neg_fasta_sequences, cter=True)

    # 将每个序列映射到特征矩阵并写入CSV文件
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ['protein_id'] + [f'P{i+1}' for i in range(len(pmat))] + [f'N{i+1}' for i in range(len(nmat))]
        writer.writerow(header)  # 写入CSV文件的标题行
        for seq_record in fasta_sequences:
            sequence = seq_record.seq
            protein_id = seq_record.id
            
            sequence_str = str(sequence)
            processed_sequence = Seq(sequence_str.replace('M', '', 1))
            mapped_values = [protein_id] + BPBaac_psp.mat_mapper(processed_sequence, pmat, nmat, cter=False)
            writer.writerow(mapped_values)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog='psipred')
    parser.add_argument('-p', type=str, required=True, help='Path to positive sequences fasta file.')
    parser.add_argument('-n', type=str, required=True, help='Path to negative sequences fasta file.')
    parser.add_argument('-o', type=str, required=True, help='Path to output CSV file.')
    parser.add_argument('-v', type=str, required=True, help='Path to output CSV file.')
    parser.add_argument('-c', type=str, required=True, help='Path to output CSV file.')
    args = parser.parse_args()

    BPBaac(pos_file=args.p, neg_file=args.n, output_file=args.o,val_file=args.v,cter=args.c)