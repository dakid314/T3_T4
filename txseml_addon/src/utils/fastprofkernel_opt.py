'''
Author: George Zhao
Date: 2021-09-03 20:27:42
LastEditors: George Zhao
LastEditTime: 2022-06-22 14:41:10
Description: 
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''
import json
import subprocess
import os
import math
import itertools
import warnings
import sys
sys.path.append('src')

import libpybiofeature

from Bio import SeqIO
import pandas as pd


def prepare_data(
    path_to_fasta: str,
    path_to_profile_dir: str,
    tag_of_profile: str,
    seq_id_list: list = None,
):
    # Loadfasta
    seq_list = list(SeqIO.parse(path_to_fasta, 'fasta'))

    # LoadPssm
    # p
    p_possum_index_path = os.path.join(
        path_to_profile_dir, *['possum_index.json', ])
    if os.path.exists(p_possum_index_path) == False:
        raise FileExistsError(f'{p_possum_index_path}')

    p_possum_index = None
    with open(p_possum_index_path, 'r', encoding='UTF-8') as f:
        p_possum_index = json.load(f)
    p_seq_pssm_content = list(libpybiofeature.libdataloader.pssm_tools.get_pssm_in_order(
        p_possum_index['data'][tag_of_profile],
        os.path.join(path_to_profile_dir, *['{zipid}_pssm_files.zip', ])
    ))
    if seq_id_list is None:
        return [seq_list, p_seq_pssm_content]
    else:
        db = {
            seq.id: [seq, prof] for seq, prof in zip(seq_list,
                                                     p_seq_pssm_content)
        }
        return [db[seqid] for seqid in seq_id_list]


def make_inline_fasta(p_seq_list: list, start: int = 0, desc: str = 'unDef', lend: str = '\n'):
    num_len = 0
    if len(p_seq_list) % 10 == 0:
        num_len = int(math.log10(len(p_seq_list)))
    else:
        num_len = math.ceil(math.log10(len(p_seq_list)))
    return itertools.chain(*[
        '>{desc}{index}{lend}{seq}{lend}'.format(
            desc=desc, index=str(index).zfill(num_len),
            lend=lend,
            seq=seq.seq
        )
        for index, seq in enumerate(p_seq_list, start=start)
    ])


def make_inline_pssm_mat(p_mat_list: list, start: int = 0, desc: str = 'unDef', lend: str = '\n'):
    num_len = 0
    if len(p_mat_list) % 10 == 0:
        num_len = int(math.log10(len(p_mat_list)))
    else:
        num_len = math.ceil(math.log10(len(p_mat_list)))
    return itertools.chain(*[
        '>{desc}{index}{lend}{mat_cont}{lend}'.format(
            desc=desc, index=str(index).zfill(num_len),
            lend=lend,
            mat_cont=mat_cont
        )
        for index, mat_cont in enumerate(p_mat_list, start=start)
    ])


def make_label(p_seq_list: list, label: bool = True, label_tag: str = 'DefaultTrue', start: int = 0, desc: str = 'unDef', lend: str = '\n'):
    num_len = 0
    if len(p_seq_list) % 10 == 0:
        num_len = int(math.log10(len(p_seq_list)))
    else:
        num_len = math.ceil(math.log10(len(p_seq_list)))
    return itertools.chain(*[
        '>{desc}{index}{lend}{label}{lend}'.format(
            desc=desc, index=str(index).zfill(num_len),
            lend=lend,
            label=label_tag if label == True else 'other'
        )
        for index, _ in enumerate(p_seq_list, start=start)
    ])


def go_opt(
    p_seq_list: list,
    p_seq_pssm_content: list,
    n_seq_list: list,
    n_seq_pssm_content: list,
    path_to_exebin: str,
    path_to_model_dir: str,
    path_to_tmp: str,
    label_tag: str,
    verbose: bool = False,
    k: int = 4,
    th: float = 7.0
):
    # make fasta.
    path_to_inlinefasta = os.path.join(path_to_tmp, *['fasta.fasta', ])
    with open(path_to_inlinefasta, 'w+', encoding='UTF-8') as f:
        f.write(''.join(itertools.chain(
            *[
                make_inline_fasta(
                    p_seq_list=p_seq_list,
                    desc='p',
                ),
                make_inline_fasta(
                    p_seq_list=n_seq_list,
                    desc='n',
                )
            ]
        )))
    # make mat_cont.
    path_to_mat_cont = os.path.join(path_to_tmp, *['fasta.profiles', ])
    with open(path_to_mat_cont, 'w+', encoding='UTF-8') as f:
        f.write(''.join(itertools.chain(
            *[
                make_inline_pssm_mat(
                    p_mat_list=p_seq_pssm_content,
                    desc='p',
                ),
                make_inline_pssm_mat(
                    p_mat_list=n_seq_pssm_content,
                    desc='n',
                )
            ]
        )))
    # make label.
    path_to_label = os.path.join(path_to_tmp, *['fasta.label', ])
    with open(path_to_label, 'w+', encoding='UTF-8') as f:
        f.write(''.join(itertools.chain(
            *[
                make_label(
                    p_seq_list=p_seq_list,
                    label=True,
                    desc='p',
                    label_tag=label_tag
                ),
                make_label(
                    p_seq_list=n_seq_list,
                    label=False,
                    desc='n',
                    label_tag=label_tag
                )
            ]
        )))

    if os.path.exists(path_to_model_dir) == False:
        os.makedirs(path_to_model_dir)

    sp = subprocess.Popen(
        f'{path_to_exebin} -k {k} -s {th} -f {path_to_inlinefasta} -p {path_to_mat_cont} -l {path_to_label} -o {path_to_model_dir}',
        executable='bash',
        # cwd=path_to_tmp,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
    )

    std_info, error_info = sp.communicate()
    if error_info != b'' and verbose != False:
        warnings.warn(
            f'Subprocess: {error_info.decode("UTF-8")}')
    if os.path.exists(
        os.path.join(path_to_model_dir, *['weka.model', ])
    ) == False:
        raise RuntimeError(
            f'Find Nothing in {path_to_model_dir}, std_info:{std_info}; error_info:{error_info}')
    else:
        # remove file
        os.remove(
            path_to_inlinefasta
        )
        os.remove(
            path_to_mat_cont
        )
        os.remove(
            path_to_label
        )
    pass
    return


def go_pred(
    seq_list: list,
    seq_pssm_content: list,
    path_to_exebin: str,
    path_to_model_dir: str,
    path_to_tmp: str,
    label_tag: str,
    path_to_out: str = None,
    verbose: bool = False,
):
    # make fasta.
    path_to_inlinefasta = os.path.join(path_to_tmp, *['fasta.fasta', ])
    with open(path_to_inlinefasta, 'w+', encoding='UTF-8') as f:
        f.write(''.join(itertools.chain(
            *[
                make_inline_fasta(
                    p_seq_list=seq_list,
                    desc='pred',
                ),
            ]
        )))
    # make mat_cont.
    path_to_mat_cont = os.path.join(path_to_tmp, *['fasta.profiles', ])
    with open(path_to_mat_cont, 'w+', encoding='UTF-8') as f:
        f.write(''.join(itertools.chain(
            *[
                make_inline_pssm_mat(
                    p_mat_list=seq_pssm_content,
                    desc='pred',
                ),
            ]
        )))

    out_tmp_path = os.path.join(
        path_to_tmp, *['pred.txt', ]
    )
    sp = subprocess.Popen(
        f'{path_to_exebin} -f {path_to_inlinefasta} -p {path_to_mat_cont} -m {path_to_model_dir} -o {out_tmp_path}',
        executable='bash',
        # cwd=path_to_tmp,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
    )

    std_info, error_info = sp.communicate()
    if error_info != b'' and verbose != False:
        warnings.warn(
            f'Subprocess: {error_info.decode("UTF-8")}')
    if os.path.exists(
        out_tmp_path
    ) == False:
        raise RuntimeError(
            f'Find Nothing in {out_tmp_path}, std_info:{std_info.decode("UTF-8")}; error_info:{error_info.decode("UTF-8")}')

    df = pd.read_csv(out_tmp_path, sep=r'\s+',
                     header=None, comment='#', index_col=None)
    df.columns = ['Protein ID', 'Class', 'Score']
    df.index = [seq.id for seq in seq_list]
    df.loc[:, 'Score'] = df.apply(
        lambda x: x['Score'] if x['Class'] != 'other' else 1 - x['Score'], axis=1)

    # remove file
    os.remove(
        path_to_inlinefasta
    )
    os.remove(
        path_to_mat_cont
    )
    os.remove(
        out_tmp_path
    )

    if path_to_out is not None:
        df.to_csv(path_to_out, encoding='UTF-8')
    return df


if __name__ == '__main__':
    import argparse
    import argcomplete
    import workdir
    from libpybiofeature.libdataloader import pssm_tools
    work_Dir = workdir.workdir(
        os.getcwd(),
        2
    )
    parser = argparse.ArgumentParser(prog='fastprofkernel_opt')
    parser.set_defaults(func=lambda _: 'parser')

    parser.add_argument('--tmp', type=str, required=False, default=os.path.join(
        work_Dir, *['tmp', 'fastprofkernel']
    ), help='Path to tmp dir.')
    parser.add_argument('-v', action='store_true', help='verbose')
    parser.add_argument('-e', type=str, required=False,
                        default='profkernel-workflow', help='Path to profkernel-workflow')
    parser.add_argument('-l', type=str, required=False,
                        default='true', help='Label of tag.')

    argcomplete.autocomplete(parser)

    subparsers = parser.add_subparsers(dest='subparser')
    # optimite
    optimite_subparser = subparsers.add_parser('optimite')
    optimite_subparser.set_defaults(func=lambda _: 'optimite_subparser')

    optimite_subparser.add_argument(
        '--fp', type=str, required=True, help='Path to positive fasta.')
    optimite_subparser.add_argument(
        '--fn', type=str, required=True, help='Path to negative fasta.')
    optimite_subparser.add_argument(
        '--pp', type=str, required=True, help='Path to positive pssm profile dir.')
    optimite_subparser.add_argument(
        '--pn', type=str, required=True, help='Path to negative pssm profile dir.')
    optimite_subparser.add_argument(
        '--tp', type=str, required=True, help='Tag name of positive pssm profile.')
    optimite_subparser.add_argument(
        '--tn', type=str, required=True, help='Tag name of negative pssm profile.')

    optimite_subparser.add_argument('-o', type=str, required=True,
                                    help='Path to Out (Model).')
    optimite_subparser.add_argument('--kn', type=int, required=False,
                                    default=4, help='Param: K.')
    optimite_subparser.add_argument('--sigm', type=float, required=False,
                                    default=7.0, help='Param: sigm.')

    # predict
    predict_subparser = subparsers.add_parser('predict')
    predict_subparser.set_defaults(func=lambda _: 'predict_subparser')

    predict_subparser.add_argument('-f', type=str, required=True,
                                   help='Path to fasta.')
    predict_subparser.add_argument('-p', type=str, required=True,
                                   help='Path to pssm_profile dir.')
    predict_subparser.add_argument('-t', type=str, required=True,
                                   help='pssm_profile tag.')
    predict_subparser.add_argument('-m', type=str, required=True,
                                   help='Path to Model Dir.')
    predict_subparser.add_argument('-o', type=str, required=True,
                                   help='Path to Out.')

    args = parser.parse_args()

    if os.path.exists(args.tmp) == False:
        os.makedirs(args.tmp)

    if args.func(args) == 'optimite_subparser':
        # Loadfasta
        p_seq_list = list(SeqIO.parse(args.fp, 'fasta'))
        n_seq_list = list(SeqIO.parse(args.fn, 'fasta'))

        # LoadPssm
        # p
        p_possum_index_path = os.path.join(args.pp, *['possum_index.json', ])
        if os.path.exists(p_possum_index_path) == False:
            raise FileExistsError(f'{p_possum_index_path}')

        p_possum_index = None
        with open(p_possum_index_path, 'r', encoding='UTF-8') as f:
            p_possum_index = json.load(f)
        p_seq_pssm_content = list(pssm_tools.get_pssm_in_order(
            p_possum_index['data'][args.tp],
            os.path.join(args.pp, *['{zipid}_pssm_files.zip', ])
        ))
        # n
        n_possum_index_path = os.path.join(args.pn, *['possum_index.json', ])
        if os.path.exists(n_possum_index_path) == False:
            raise FileExistsError(f'{n_possum_index_path}')

        n_possum_index = None
        with open(n_possum_index_path, 'r', encoding='UTF-8') as f:
            n_possum_index = json.load(f)
        n_seq_pssm_content = list(pssm_tools.get_pssm_in_order(
            n_possum_index['data'][args.tn],
            os.path.join(args.pn, *['{zipid}_pssm_files.zip', ])
        ))

        go_opt(
            p_seq_list=p_seq_list,
            p_seq_pssm_content=p_seq_pssm_content,
            n_seq_list=n_seq_list,
            n_seq_pssm_content=n_seq_pssm_content,
            path_to_exebin=args.e,
            path_to_model_dir=args.o,
            path_to_tmp=args.tmp,
            label_tag=args.l,
            verbose=args.v,
            k=args.kn,
            th=args.sigm
        )
    elif args.func(args) == 'predict_subparser':
        # Loadfasta
        seq_list = list(SeqIO.parse(args.f, 'fasta'))
        # LoadPssm
        possum_index_path = os.path.join(args.p, *['possum_index.json', ])
        if os.path.exists(possum_index_path) == False:
            raise FileExistsError(f'{possum_index_path}')

        possum_index = None
        with open(possum_index_path, 'r', encoding='UTF-8') as f:
            possum_index = json.load(f)
        seq_pssm_content = list(pssm_tools.get_pssm_in_order(
            possum_index['data'][args.t],
            os.path.join(args.p, *['{zipid}_pssm_files.zip', ])
        ))

        go_pred(
            seq_list=seq_list,
            seq_pssm_content=seq_pssm_content,
            path_to_exebin=args.e,
            path_to_model_dir=args.m,
            path_to_out=args.o,
            label_tag=args.l,
            path_to_tmp=args.tmp,
            verbose=args.v,
        )
    else:
        print(parser.format_help(), end='')
