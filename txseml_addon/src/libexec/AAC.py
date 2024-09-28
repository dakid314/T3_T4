import csv
from Bio import SeqIO
import sys 
sys.path.append("/mnt/md0/Public/一站式/txseml_backend_addon/txseml_addon/src/libpybiofeature")
import AC
def aac(input_file, output_file):
    # 读取FASTA文件
    fasta_sequences = list(SeqIO.parse(input_file, "fasta"))

    # 构建CSV数据
    csv_data = []
    for seq_record in fasta_sequences:
        sequence = str(seq_record.seq)
        protein_id = str(seq_record.id)
        aac_features = [protein_id] + AC.AAC(sequence)  # 调用AAC函数计算特征
        csv_data.append(aac_features)

# 写入CSV文件
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['protein_id'] + AC.sac_default_header)  # 写入CSV文件的标题行
        writer.writerows(csv_data)  # 写入特征数据


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog='psipred')
    parser.add_argument('-i', type=str, required=True, help='Path to fasta.')
    parser.add_argument('-o', type=str, required=True,
                        help='Path to Out.')
    args = parser.parse_args()
    aac(input_file=args.i,
        output_file = args.o
        )