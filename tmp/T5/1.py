from Bio.Blast.Applications import NcbiblastpCommandline
from Bio import SeqIO
import os


folder_path = '/mnt/md0/Public/T3_T4/data/T3/neg/10w'
files = os.listdir(folder_path)
bac_type = ['T3','T4','T1','T2','T6','T5'][5]
for fasta_file in files:
    negative_sequences = []
# 读取阴性蛋白质序列数据
    for seq_record in SeqIO.parse(f"{folder_path}/{fasta_file}", "fasta"):
        negative_sequences.append(seq_record)
        # 执行blast比对
    output_file = f"n{bac_type}.xml"
    blastp_cline = NcbiblastpCommandline(cmd='blastp', query=f"T5SS.fasta", subject=f"{folder_path}/{fasta_file}", outfmt=5, out=output_file)
    blastp_cline()

        # 解析blast结果
    from Bio.Blast import NCBIXML
    blast_results = NCBIXML.parse(open(output_file))
    filtered_sequences = []  # 存储经筛选后的阴性蛋白质序列

    for result in blast_results:
        for alignment in result.alignments:
            for hsp in alignment.hsps:
                if hsp.identities / alignment.length >= 0.3 and hsp.align_length / alignment.length >= 0.7:
                    filtered_sequences.append(alignment.title.split()[1])
    
    # 输出筛选后的阴性蛋白质序列
    with open(f"all_n{bac_type}.fasta", "w") as output_handle:
        for seq_record in negative_sequences:
            if seq_record.id not in filtered_sequences:
                SeqIO.write(seq_record, output_handle, "fasta")
    
    with open(f"{bac_type}_blast.fasta", "w") as output_handle:
        for seq_record in negative_sequences:
            if seq_record.id  in filtered_sequences:
                SeqIO.write(seq_record, output_handle, "fasta")
    