import csv
from Bio import SeqIO
import sys 
sys.path.append("/mnt/md0/Public/一站式/txseml_backend_addon/txseml_addon/src/libpybiofeature")
import CTriad
def process_fasta(input_file, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['protein_id', 'CTriad'])

        for record in SeqIO.parse(input_file, 'fasta'):
            sequence = str(record.seq)
            ctriad = CTriad.CTriad(sequence)

            writer.writerow([record.description, ctriad])
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog='psipred')
    parser.add_argument('-i', type=str, required=True, help='Path to fasta.')
    parser.add_argument('-o', type=str, required=True,
                        help='Path to Out.')
    args = parser.parse_args()
    process_fasta(input_file=args.i,
        output_file = args.o
        )