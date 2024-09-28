'''
Author: George Zhao
Date: 2021-05-28 22:44:36
LastEditors: George Zhao
LastEditTime: 2022-03-05 23:02:40
Description: 
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''
# %%
import json
# %%
import os
import sys
sys.path.append('src')
# %%
import utils
import math
from Bio import SeqIO
import itertools
import functools
import tqdm

work_Dir = utils.workdir.workdir(os.getcwd(), 3)
rdata_Dir = os.path.join(work_Dir, 'data')

out_Dir = os.path.join(work_Dir, 'out')

# %%


def go_(path_to_fasta: str, path_to_db: str, path_to_tmp: str, path_to_out: str, path_to_python: str, tag: str, pre_file_num: int = 25, ):
    file_name = os.path.splitext(os.path.split(path_to_fasta)[1])[0]
    # logger.logger.info(f'Task: {file_name}')

    non_T3SEs_list = list(SeqIO.parse(
        path_to_fasta, 'fasta'))

    non_T3SEs_iter = iter(non_T3SEs_list)
    non_T3SEs_length = len(non_T3SEs_list)

    num_of_file = math.ceil(non_T3SEs_length / pre_file_num)
    num_counter = list()

    for n in tqdm.tqdm(range(num_of_file)):
        with open(f'{path_to_tmp}/fasta/{tag}_{file_name}_100_{n}.fasta', 'w+', encoding='UTF-8') as f:
            num_counter.append(SeqIO.write(
                list(itertools.islice(non_T3SEs_iter, 0, pre_file_num)),
                f,
                'fasta'
            ))
    for n in tqdm.tqdm(range(num_of_file)):
        os.system(
            f'{path_to_python} -u src/libexec/pairwisealigner.py -f {f"{path_to_tmp}/fasta/{tag}_{file_name}_100_{n}.fasta"} -o {f"{path_to_tmp}/aligned/{tag}_{file_name}_100_{n}.json"} --db {path_to_db}')

    for n in tqdm.tqdm(range(num_of_file)):
        os.remove(os.path.join(
            out_Dir, f"{path_to_tmp}/fasta/{tag}_{file_name}_100_{n}.fasta"))
    result_list = []
    for n in tqdm.tqdm(range(num_of_file)):
        with open(f"{path_to_tmp}/aligned/{tag}_{file_name}_100_{n}.json", 'r', encoding='utf-8') as f:
            result_list.append(json.loads(f.read())['data'])
    with open(f"{path_to_out}/aligned_{tag}_{file_name}.json", 'w+', encoding='utf-8') as f:
        json.dump({
            'name': file_name,
            'data': list(itertools.chain(*result_list))}, f)
    return functools.reduce(lambda x, y: x + y, num_counter)


# %%
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog='EP3_pairwisealigner_controler')
    parser.add_argument('-p', type=str, required=True, help='Path to python.')
    parser.add_argument('-t', type=str, required=True, help='Tag.')
    parser.add_argument('-f', type=str, required=True, help='Path to fasta.')
    parser.add_argument('-o', type=str, required=True, help='Path to out.')
    parser.add_argument('--tmp', type=str, required=False, help='Path to tmp.', default=os.path.join(
        work_Dir, 'tmp'
    ))
    parser.add_argument('--db', type=str, required=True, help='Path to DB.')
    args = parser.parse_args()

    if os.path.exists(args.tmp) == False:
        os.makedirs(args.tmp)
    if os.path.exists(os.path.join(args.tmp, 'fasta')) == False:
        os.makedirs(os.path.join(args.tmp, 'fasta'))
    if os.path.exists(os.path.join(args.tmp, 'aligned')) == False:
        os.makedirs(os.path.join(args.tmp, 'aligned'))

    if os.path.exists(args.o) == False:
        os.makedirs(args.o)

    go_(
        path_to_fasta=args.f,
        path_to_db=args.db,
        path_to_tmp=args.tmp,
        path_to_out=args.o,
        path_to_python=args.p,
        tag=args.t
    )
