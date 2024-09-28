from Bio import SeqIO
from Bio.Seq import Seq
import random
import os

def count_protein_sequences(pos_fasta_file):
    protein_count = 0
    for record in SeqIO.parse(pos_fasta_file, "fasta"):
        protein_count += 1
    return protein_count

def select_protein_sequences(pos_fasta_file, seed):
    random.seed(seed)
    protein_count = count_protein_sequences(pos_fasta_file)
    total_selected =  protein_count//5

    selected_proteins = random.sample(list(SeqIO.parse(pos_fasta_file, "fasta")), total_selected)
    return selected_proteins

def write_selected_proteins(selected_proteins, output_file):
    with open(output_file, "w") as f:
        for record in selected_proteins:
            seq = str(record.seq)
            seq = seq.replace('X', 'A')
            seq = seq.replace('U', 'A')
            record.seq = Seq(seq)
            SeqIO.write(record, f, "fasta")

# 示例调用
rate = 30
pos_fasta_file = f"data/pos/T3_training_{rate}.fasta"
a = 0
#neg_fasta_file = f"data/neg/all_nT3_{rate}.fasta"

while a < 5: 
    folder_path = f"data/{rate}_pos/{a}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        seed = 12345+a
        selected_proteins = select_protein_sequences(pos_fasta_file, seed)
        output_file = f'data/{rate}_pos/{a}/T3_training_{a}.fasta'
        write_selected_proteins(selected_proteins, output_file)
        print("完成！已将选定的蛋白质序列保存到", output_file)
        
    a += 1
