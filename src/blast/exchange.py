from Bio import SeqIO
from Bio import Seq
def replace_x_with_a(fasta_file):
    replaced_sequences = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequence = str(record.seq)
        replaced_sequence = sequence.replace("X", "A").replace("Z", "A").replace("U", "A").replace("B", "A")
        replaced_record = record
        replaced_record.seq = Seq.Seq(replaced_sequence)
        replaced_sequences.append(replaced_record)

    return replaced_sequences

# 运行示例代码
if __name__ == "__main__":
    fasta_file = "data/new_T5/pos/T5_training_70.fasta"  # 替换为你的.fasta文件路径
    replaced_sequences = replace_x_with_a(fasta_file)
    
    # 保存替换后的序列到新的.fasta文件
    output_file = "data/new_T5/pos/T5_training_70.fasta"  # 替换为你想要保存的.fasta文件路径
    SeqIO.write(replaced_sequences, output_file, "fasta")
    print(0)