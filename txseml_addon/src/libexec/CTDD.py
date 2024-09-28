from Bio import SeqIO
import pandas as pd
import sys 
sys.path.append("/mnt/md0/Public/一站式/txseml_backend_addon/txseml_addon/src/libpybiofeature")
import CTD
def CTDD_fasta(input_file, output_file):
    # 读取FASTA文件
    sequences = list(SeqIO.parse(input_file, "fasta"))
    
    # 定义存储CTDT结果的列表
    results = []
    
    # 对每个序列调用CTDT函数并将结果添加到列表中
    for sequence in sequences:
        protein_sequence = str(sequence.seq)
        ctdt_result = CTD.CTDD(protein_sequence)  # 调用CTDT函数
        results.append(ctdt_result)
    
    # 将结果转换为DataFrame对象
    df = pd.DataFrame(results)

# 添加 protein_id 列
    df['protein_id'] = [sequence.id for sequence in sequences]

    # 将 protein_id 列移动到第一列
    cols = df.columns.tolist()
    cols = ['protein_id'] + cols[:-1]
    df = df[cols]

    # 将结果保存为CSV文件
    df.to_csv(output_file, index=False)
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog='psipred')
    parser.add_argument('-i', type=str, required=True, help='Path to fasta.')
    parser.add_argument('-o', type=str, required=True,
                        help='Path to Out.')
    args = parser.parse_args()
    CTDD_fasta(input_file=args.i,
        output_file = args.o
        )