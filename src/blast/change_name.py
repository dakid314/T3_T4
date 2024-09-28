import os
import shutil

def replace_non_standard(sequence):
    standard_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
    replaced_sequence = ""
    for letter in sequence:
        if letter not in standard_amino_acids:
            replaced_sequence += "A"
        else:
            replaced_sequence += letter
    return replaced_sequence

def process_fasta_file(input_file, output_file):
    with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
        seq_name = ""
        sequence = ""
        for line in f_in:
            line = line.strip()
            if line.startswith(">"):
                if seq_name != "":
                    replaced_sequence = replace_non_standard(sequence)
                    f_out.write(f">{seq_name}\n{replaced_sequence}\n")
                    sequence = ""
                seq_name = line[1:]
            else:
                sequence += line
        # 处理最后一个序列
        replaced_sequence = replace_non_standard(sequence)
        f_out.write(f">{seq_name}\n{replaced_sequence}\n")

source_folder = '/mnt/md0/Public/T3_T4/T3/neg/source'
destination_folder = '/mnt/md0/Public/T3_T4/T3/neg/processed'

# 遍历源文件夹中的所有子文件夹
for subfolder in os.listdir(source_folder):
    subfolder_path = os.path.join(source_folder, subfolder)

    # 确保是文件夹，而不是文件
    if os.path.isdir(subfolder_path):
        # 获取子文件夹中的所有文件
        files = os.listdir(subfolder_path)

        # 遍历每个文件
        for file in files:
            # 确保是fasta文件，而不是文件夹
            if file.endswith(".fasta"):
                input_file = os.path.join(subfolder_path, file)
                output_file = os.path.join(destination_folder, file)

                # 处理fasta文件
                process_fasta_file(input_file, output_file)

                # 移动处理后的fasta文件到目标文件夹
                shutil.move(input_file, output_file)