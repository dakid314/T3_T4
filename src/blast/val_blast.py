from Bio.Blast.Applications import NcbiblastpCommandline
from Bio import SeqIO
import os

bac_type = ['T3','T4','T1','T2','T6'][0]
folder_path = f'/mnt/md0/Public/T3_T4/data/new_{bac_type}/val_tofeature'
files = os.listdir(folder_path)
stand_ = ['strict','lossen']
for stand in stand_:
    for fasta_file in files:
        negative_sequences = []
    # 读取阴性蛋白质序列数据
        for seq_record in SeqIO.parse(f"{folder_path}/{fasta_file}", "fasta"):
            negative_sequences.append(seq_record)
            # 执行blast比对
        output_file = f"{stand}_{fasta_file}.xml"
        blastp_cline = NcbiblastpCommandline(cmd='blastp', query=f"/mnt/md0/Public/T3_T4/data/new_{bac_type}/{bac_type}_training.fasta", subject=f"{folder_path}/{fasta_file}", outfmt=5, out=output_file)
        blastp_cline()

            # 解析blast结果
        from Bio.Blast import NCBIXML
        blast_results = NCBIXML.parse(open(output_file))
        filtered_sequences = []  # 存储经筛选后的阴性蛋白质序列

        for result in blast_results:
            for alignment in result.alignments:
                for hsp in alignment.hsps:
                    start = hsp.query_start
                    end = hsp.query_end
                    covered_length = end - start + 1
                    identity_ratio = hsp.identities / hsp.align_length
                    align_length_ratio = covered_length / result.query_length
                    if align_length_ratio>1:
                        print(1)
                    if stand == 'strict':
                        if identity_ratio >= 0.7 and align_length_ratio >= 0.9:
                            filtered_sequences.append(alignment.title.split()[1])
                    if stand == 'lossen':
                        if identity_ratio >= 0.5 and align_length_ratio >= 0.7:
                            filtered_sequences.append(alignment.title.split()[1])
        
        # 输出筛选后的阴性蛋白质序列
        # with open(f"{fasta_file}_nonT4", "w") as output_handle:
        #     for seq_record in negative_sequences:
        #         if seq_record.id not in filtered_sequences:
        #             SeqIO.write(seq_record, output_handle, "fasta")
        
        with open(f"{stand}_{fasta_file}", "w") as output_handle:
            for seq_record in negative_sequences:
                if seq_record.id in filtered_sequences:
                    SeqIO.write(seq_record, output_handle, "fasta")
        print(0)
    