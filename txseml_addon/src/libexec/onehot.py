import sys
sys.path.append("/mnt/md0/Public/一站式/txseml_backend_addon/txseml_addon/src/libpybiofeature")
import featurebuilder
import pandas as pd
from Bio import SeqIO
def one_hot(input_file, output_file,cter):
    
    
    protein_ids = []
    seq_list = list(SeqIO.parse(input_file, 'fasta'))
    for seq_record in SeqIO.parse(input_file, "fasta"):
        protein_id = seq_record.id
        protein_ids.append(protein_id)
        
        
    df= pd.DataFrame(featurebuilder.build_oneHot_feature(
    seq_list  = seq_list,
    length = 100,
    cter = cter)
    )
    df.insert(0, "protein_id", protein_ids)
    df.to_csv(output_file,index=False)
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog='psipred')
    parser.add_argument('-i', type=str, required=True, help='Path to fasta.')
    parser.add_argument('-o', type=str, required=True,
                        help='Path to Out.')
    parser.add_argument('-c', type=str, required=True,
                        help='cter.')
    args = parser.parse_args()
    one_hot(input_file=args.i,
        output_file = args.o,
        cter=args.c
        )