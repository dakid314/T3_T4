from Bio import SeqIO
import os

def count_protein_sequences(pos_fasta_file):
    pos_protein_count = 0
    for record in SeqIO.parse(pos_fasta_file, "fasta"):
        pos_protein_count += 1
    return pos_protein_count

def count_protein_sequences(neg_fasta_file):
    neg_protein_count = 0
    for record in SeqIO.parse(neg_fasta_file, "fasta"):
        neg_protein_count += 1
    return neg_protein_count

def rate(pos_file,neg_file):
    num = count_protein_sequences(neg_file)/count_protein_sequences(pos_file)
    return num

folder_path = '/mnt/md0/Public/T3_T4/data/bac_30'
files = os.listdir(folder_path)
pos_file = '/mnt/md0/Public/T3_T4/data/30_pos/T3_training_30.fasta'
for file in files:
    neg_file = f'{folder_path}/{file}'
    print(f'{file}:{rate(pos_file,neg_file)},{count_protein_sequences(neg_file)}')