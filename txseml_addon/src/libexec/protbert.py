from transformers import AutoTokenizer, BertModel
import torch
import pandas as pd

def process_protein_sequence(protein_sequence):
    model_path = "Rostlab/prot_bert"
    tokenizer = AutoTokenizer.from_pretrained(model_path, do_lower_case=False)
    model = BertModel.from_pretrained(model_path)
    # 分词并添加特殊标记
    input_ids = tokenizer.encode('[CLS]' + protein_sequence + '[SEP]', add_special_tokens=False)
    
    # 转换为 torch Tensor
    input_ids = torch.tensor([input_ids], dtype=torch.long)
    
    # 使用 ProtBert_Embedding 生成嵌入向量
    with torch.no_grad():
        outputs = model(input_ids)
        embeddings = outputs[0][0]
    
    return embeddings.tolist()

def protein_sequences_to_csv(input_file, output_file):
    # 从 input_file 读取蛋白质序列数据
    protein_sequences = []
    with open(input_file, 'r') as file:
        for line in file:
            protein_sequences.append(line.strip())

    # 处理蛋白质序列，生成嵌入向量
    embeddings = []
    for seq in protein_sequences:
        embeddings.append(process_protein_sequence(seq))
    
    # 将嵌入向量和对应的序列写入 CSV 文件
    df = pd.DataFrame({'Protein Sequence': protein_sequences, 'Embeddings': embeddings})
    df.to_csv(output_file, index=False)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog='psipred')
    parser.add_argument('-i', type=str, required=True, help='Path to fasta.')
    parser.add_argument('-o', type=str, required=True, help='Path to Out.')
    args = parser.parse_args()
    protein_sequences_to_csv(input_file=args.i, output_file=args.o)