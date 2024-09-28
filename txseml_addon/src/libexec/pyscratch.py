'''
Author: George Zhao
Date: 2021-08-17 20:37:19
LastEditors: George Zhao
LastEditTime: 2022-02-10 22:43:59
Description: 
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''

import os
import sys
import subprocess
import itertools
import math
import json
import warnings


from Bio import SeqIO, SeqRecord
import tqdm


def _get_ok_length_seq(seqaa: str, length: int = 2500, cter: bool = False):
    # ! min?
    length = max(2500, length)
    if cter == False:
        # Cter == False
        if len(seqaa) < 30:
            return seqaa + 'A' * (30 - len(seqaa))
        elif len(seqaa) > length:
            return seqaa[:length]
    else:
        # Cter == True
        if len(seqaa) < 30:
            return 'A' * (30 - len(seqaa)) + seqaa
        elif len(seqaa) > length:
            return seqaa[(-1) * length:]
    return seqaa


def get(
        path_to_fasta: str,
        path_to_out: str,
        path_to_tmpdir: str,
        tag_of_fasta: str,
        path_to_script: str,
        njob: int,
        cter: bool = False,  # Nter
        desc='unDefine',
        verbose=False):

    if os.path.exists(os.path.split(path_to_out)[0]) == False:
        os.makedirs(os.path.split(path_to_out)[0])
    if os.path.exists(path_to_out) == False:
        with open(path_to_out, 'w+', encoding='UTF-8') as f:
            json.dump({'data': {}}, f)

    seq_list = list(SeqIO.parse(path_to_fasta, 'fasta'))
    num_of_part = math.ceil(len(seq_list) / njob)
    seq_list_iter = iter(seq_list)
    def sbar(x, total=None, desc=None): return x
    if verbose != False:
        sbar = tqdm.tqdm
    for index_, part in sbar(
        enumerate(
            [
                list(itertools.islice(seq_list_iter, 0, njob))
                for _ in range(num_of_part)
            ]
        ),
        total=num_of_part,
        desc=tag_of_fasta
    ):
        k_of_iter = f'{tag_of_fasta}_{index_}'
        d_record = None
        with open(path_to_out, 'r', encoding='UTF-8') as f:
            d_record = json.load(f)
        if k_of_iter in d_record['data'].keys():
            if d_record['data'][k_of_iter] is not None:
                continue

        subprocess_list = list()
        for i, seq in enumerate(part):
            filename = os.path.join(
                path_to_tmpdir, f'{tag_of_fasta}_{index_}_{i}.fasta')
            outfilename = os.path.join(
                path_to_tmpdir, f'{tag_of_fasta}_{index_}_{i}.out')
            with open(filename, 'w+', encoding='UTF-8') as f:
                f.writelines([
                    f'>{k_of_iter}_{i}'.replace('_', ''),
                    '\n',
                    f'{_get_ok_length_seq(str(seq.seq), cter=cter)}'
                ])
            subprocess_list.append(
                subprocess.Popen(
                    f'{path_to_script} {filename} {outfilename}',
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    shell=True,
                )
            )

        for i, seq in enumerate(part):
            error_info = subprocess_list[i].communicate()[1]
            if error_info != b'' and verbose != False:
                warnings.warn(
                    f'Subprocess: {seq.id}: {error_info.decode("UTF-8")}')

        result = list()
        for i, seq in enumerate(part):
            filename = os.path.join(
                path_to_tmpdir, f'{tag_of_fasta}_{index_}_{i}.fasta')
            outfilename = os.path.join(
                path_to_tmpdir, f'{tag_of_fasta}_{index_}_{i}.out')
            with open(outfilename, 'r', encoding='UTF-8') as f:
                result.append(
                    {'id': seq.id, 'desc': seq.description, 'data': f.read()})
            os.remove(filename)
            os.remove(outfilename)

        d_record = None
        with open(path_to_out, 'r', encoding='UTF-8') as f:
            d_record = json.load(f)
        d_record['data'][k_of_iter] = result
        with open(path_to_out, 'w', encoding='UTF-8') as f:
            json.dump(d_record, f)
    return


if __name__ == '__main__':

    sys.path.append('src')
    from utils import workdir
    # from . import glogger
    work_Dir = workdir.workdir(os.getcwd(), 3)
    # rdata_Dir = os.path.join(work_Dir, 'data')
    tmp_Dir = os.path.join(work_Dir, 'tmp')

    # logger = glogger.Glogger('pyscratch', os.path.join(
    #     work_Dir, 'log'), timestamp=False, std_err=False)
    # log_wrapper = glogger.log_wrapper(logger=logger)

    import argparse
    parser = argparse.ArgumentParser(prog='psipred')
    parser.add_argument('-f', type=str, required=True, help='Path to fasta.')
    parser.add_argument('-o', type=str, required=True, help='Path to Out.')
    parser.add_argument('--tmp', type=str, required=False,
                        help='Path to tmp.', default=os.path.join(tmp_Dir, 'pyscratch'))
    parser.add_argument('-t', type=str, required=True, help='Tag fasta.')
    parser.add_argument('-e', type=str, required=True, help='Path to Script.')
    parser.add_argument('-v', action='store_true', help='verbose')
    parser.add_argument('-c', action='store_true', help='C ter.')
    parser.add_argument('-n', type=int, required=False,
                        help='njob.', default=1)
    args = parser.parse_args()

    if os.path.exists(args.tmp) == False:
        os.makedirs(args.tmp)

    get(
        path_to_fasta=args.f,  # 'fasta.fasta'
        path_to_out=args.o,  # 'out.json'
        path_to_tmpdir=args.tmp,  # './tmp/'
        tag_of_fasta=args.t,  # 'accpro_t4'
        path_to_script=args.e,  # 'predict_accpro.sh'
        njob=args.n,  # '8'
        cter=args.c,
        verbose=args.v
    )
