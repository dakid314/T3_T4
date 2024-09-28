from Bio import SeqIO
import pandas as pd
import sys 
sys.path.append("/mnt/md0/Public/一站式/txseml_backend_addon/txseml_addon/src/libpybiofeature")
import featurebuilder


def ppt25(input_file,output_file,cter):
    seq_id_list = []  # 指定要提取特征的序列 ID 列表
    for record in SeqIO.parse(input_file, "fasta"):
        seq_id_list.append(record.id)
        
    df = featurebuilder.build_PPT25_feature(input_file, seq_id_list, desc='protein', fulllong=False, cter=cter)
    # 将 DataFrame 保存为 CSV 文件
    df.insert(0, "protein_id", seq_id_list)
    df.to_csv(output_file, index=False)
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog='psipred')
    parser.add_argument('-i', type=str, required=True, help='Path to fasta.')
    parser.add_argument('-o', type=str, required=True,
                        help='Path to Out.')
    parser.add_argument('-c', type=str, required=True,
                        help='cter')
    args = parser.parse_args()
    ppt25(input_file=args.i,
        output_file = args.o,
        cter=args.c
        )