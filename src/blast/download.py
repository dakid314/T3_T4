from Bio import Entrez, SeqIO
import pandas as pd

data = pd.read_csv('data.csv')
Entrez.email = 'dakid314@163.com'
# 定义蛋白质编号列表
protein_ids = list(data['prot'])


for protein_id in protein_ids:
    handle = Entrez.efetch(db='protein', id=protein_id, rettype='fasta', retmode='text')
    protein_sequence = handle.read()
    handle.close()

        # 将蛋白质序列保存到文件
    with open("protein_sequence.fasta", "w") as file:
        file.write(protein_sequence)

# 合并 fasta 文件为一个文件
merged_records = []
for protein_id in protein_ids:
    record = SeqIO.read(f'{protein_id}.fasta', 'fasta')
    merged_records.append(record)

SeqIO.write(merged_records, 'Ralstonia_pseudosolanacearum_GMI1000.fasta', 'fasta')