'''
Author: George Zhao
Date: 2021-05-26 22:31:53
LastEditors: George Zhao
LastEditTime: 2022-06-29 13:47:21
Description: 
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''
# %%
import os
from Bio import SeqIO
import tqdm
import json
import requests
import re
from bs4 import BeautifulSoup
# %%


def Dual_with_result(html_result: str):
    soup = BeautifulSoup(html_result, 'lxml').text
    return {
        # theoretical pI
        'pI': float(re.findall(r'Theoretical pI: (\d+\.\d+)\n\n',
                               soup
                               )[0]),
        # total number of negatively charged residues of Asp and Glu (Nnc)
        'Nnc': float(re.findall(r'Total number of negatively charged residues \(Asp \+ Glu\): (-?\d+)\n',
                                soup)[0]),
        # positively charged residues of Arg and Lys (Npc)
        'Npc': float(re.findall(r'Total number of positively charged residues \(Arg \+ Lys\): (-?\d+)\n',
                                soup)[0]),
        # instability index
        'Ins': float(re.findall(r'The instability index \(II\) is computed to be (-?\d+\.\d+)\n',
                                soup)[0]),
        # aliphatic index
        'Ali': float(re.findall(r'Aliphatic index: (\d+\.\d+)\n',
                                soup)[0]),
        # grand average hydrophobicity
        'Hydro': float(re.findall(r'Grand average of hydropathicity \(GRAVY\): (-?\d+\.\d+)\n',
                                  soup)[0]),
    }


def Submit_to_server(seq: str):
    resp = requests.Response()
    while resp.status_code is None or resp.status_code != 200:
        try:
            resp = requests.post(
                'https://web.expasy.org/cgi-bin/protparam/protparam',
                {
                    'prot_id': '',
                    'sequence': seq,
                    'mandatory': ''
                },
                timeout=10
            )
        except requests.exceptions.RequestException as e:
            pass
    return (resp.text)


def Submit(seq: str, idtf: str, lenght: int, cter: bool):
    result_dict = dict()

    if lenght is None:
        # Full Length
        seq = seq[1:]
    else:
        if cter == True:
            seq = seq[-30:]
        else:
            seq = seq[1:][:30]

    html_text = Submit_to_server(seq)
    result_dict.update(Dual_with_result(html_text))
    return result_dict


def get_feature(path_to_fasta: str, lenght: int, cter: bool):
    return [
        {
            'id': seq.id,
            'seq': str(seq.seq),
            'Feature': Submit(seq=seq.seq, idtf=seq.id, lenght=lenght, cter=cter)
        }
        for seq in tqdm.tqdm(
            list(
                SeqIO.parse(path_to_fasta, 'fasta')
            )
        )
    ]


def go_on(path_to_save: str, path_to_fasta: str, lenght: int, cter: bool):
    result = get_feature(path_to_fasta=path_to_fasta, lenght=lenght, cter=cter)
    with open(path_to_save, "w+", encoding='UTF-8') as f:
        json.dump({
            "param": {"lenght": lenght, "cter": cter},
            "data": result
        }, f)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog='expasy')
    parser.add_argument('-f', type=str, required=True, help='Path to fasta.')
    parser.add_argument('-o', type=str, required=True,
                        help='Path to Out.',)
    parser.add_argument('-l', type=int, required=False,
                        help='lenght', default=None)
    parser.add_argument('-c', action='store_true', help='Cter.')
    args = parser.parse_args()

    go_on(
        path_to_save=args.o,
        path_to_fasta=args.f,
        lenght=args.l,
        cter=args.c,
    )
