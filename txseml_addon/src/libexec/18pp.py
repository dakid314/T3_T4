import csv
import sys
sys.path.append("/mnt/md0/Public/一站式/txseml_backend_addon/txseml_addon/src/libpybiofeature/")
import eighteen_physicochemical_properties
import csv
from Bio import SeqIO
def _18pp(input_file, output_file):
    # 读取FASTA文件
    sequences = []
    protein_ids = []
    
    protein_ids = []
    for seq_record in SeqIO.parse(input_file, "fasta"):
        protein_id = seq_record.id
        protein_ids.append(protein_id)
        sequence = seq_record.seq
        sequences.append(sequence)

    # 执行编码转换并写入CSV文件
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['protein_id'] + eighteen_physicochemical_properties.etpp_header)  # 写入CSV文件的标题行
        for protein_id, sequence in zip(protein_ids, sequences):
            encoded_sequence = [protein_id] + eighteen_physicochemical_properties.eighteen_pp_seqencode(sequence)
            writer.writerow(encoded_sequence)
            
            
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog='psipred')
    parser.add_argument('-i', type=str, required=True, help='Path to fasta.')
    parser.add_argument('-o', type=str, required=True,
                        help='Path to Out.')
    args = parser.parse_args()
    _18pp(input_file=args.i,
        output_file=args.o
        )