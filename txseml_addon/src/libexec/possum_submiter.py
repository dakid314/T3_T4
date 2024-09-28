'''
Author: George Zhao
Date: 2021-09-02 16:27:46
LastEditors: George Zhao
LastEditTime: 2022-08-26 18:10:44
Description: 
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''
import os

import string
import itertools
import math
import random
import urllib
import json
import requests
import requests_toolbelt
import subprocess

from Bio import SeqIO
import tqdm


pre_file_num = 500
MaxLength = 5000
MinLength = 50
proxies = None if True else {
    'https': 'http://127.0.0.1:8889',
    'http': 'http://127.0.0.1:8889',
}


def _get_ok_length_seq(seqaa: str, cter: bool = False, MaxLength=MaxLength):
    if cter == False:
        # Cter == False
        if len(seqaa) < MinLength:
            return seqaa + 'A' * (MinLength - len(seqaa))
        elif len(seqaa) > MaxLength:
            return seqaa[:MaxLength]
    else:
        # Cter == True
        if len(seqaa) < MinLength:
            return 'A' * (MinLength - len(seqaa)) + seqaa
        elif len(seqaa) > MaxLength:
            return seqaa[(-1) * MaxLength:]
    return seqaa


def get_fasta(path_to_fasta: str, path_to_out: str, cter: bool = False, MaxLength: int = MaxLength):
    seq_fasta = '\n'.join(
        [
            f'>{seq.id}\n{_get_ok_length_seq(seq.seq,cter=cter,MaxLength=MaxLength)}\n'
            for seq in
            SeqIO.parse(path_to_fasta, 'fasta')
        ]
    )
    with open(path_to_out, 'w+', encoding='utf-8') as f:
        f.write(seq_fasta)


def Submit_to_server(seq_list: list, cter: bool, desc: str, MaxLength=MaxLength):
    seq_fasta = '\n'.join(
        [
            f'>{seq.id}\n{_get_ok_length_seq(seq.seq,cter=cter,MaxLength=MaxLength)}\n' for seq in seq_list
        ]
    )
    mp_encoder = requests_toolbelt.MultipartEncoder(fields={
        'seqString': '',
        'fastaFile': ('fasta.fasta', seq_fasta.encode(), 'application/octet-stream'),
        'aac_pssm': 'on',
        'd_fpssm': 'on',
        's_fpssm': 'on',
        # 'ab_pssm': 'on',
        'pssm_composition': 'on',
        'rpm_pssm': 'on',
        'smoothed_pssm': 'on',
        'smoothing_window_select': '7',
        'sliding_window_input': '50',
        'dpc_pssm': 'on',
        # 'k_separated_bigrams_pssm': 'on',
        'k_input': '1',
        # 'tri_gram_pssm': 'on',
        # 'eedp': 'on',
        'tpc': 'on',
        # 'edp': 'on',
        # 'rpssm': 'on',
        'pse_pssm': 'on',
        'e_input': '1',
        'dp_pssm': 'on',
        'a_input': '5',
        'pssm_ac': 'on',
        'ac_LG_input': '10',
        # 'pssm_cc': 'on',
        'cc_LG_input': '10',
        # 'aadp_pssm': 'on',
        # 'aatp': 'on',
        # 'medp': 'on',
        'selectedDB': 'Uniref50',
        'selectedIterations': '3',
        'selectedEvalue': '0.001',
        'email': f'',
        'orgnization': f'',
    }, boundary='----WebKitFormBoundary' + ''.join(random.sample(string.ascii_letters + string.digits, 16)))

    return requests.Request(
        'POST',
        'https://possum.erc.monash.edu/submission',
        data=mp_encoder,
        headers={
            'sec-ch-ua': '"Chromium";v="92", " Not A;Brand";v="99", "Google Chrome";v="92"',
            'sec-ch-ua-mobile': '?0',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'same-origin',
            'Sec-Fetch-User': '?1',
            'Upgrade-Insecure-Requests': '1',
            'Connection': 'keep-alive',
            'Cache-Control': 'max-age=0',
            'Cookie': 'JSESSIONID=9E537CC75E0C78C34E79533424F2841F;',
            'Referer': 'https://possum.erc.monash.edu/server.jsp',
            'Origin': 'https://possum.erc.monash.edu',
            'Host': 'possum.erc.monash.edu',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'en,zh-CN;q=0.9,zh;q=0.8',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
            'Content-Type': mp_encoder.content_type,
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        }
    ).prepare()


def go_(path_to_fasta: str, tag_name: str, cter: bool, path_to_json: str = None, desc: str = 'unDefine', verbose: int = False, MaxLength=MaxLength):
    seq_list = list(SeqIO.parse(path_to_fasta, 'fasta'))
    part_num = math.ceil(len(seq_list) / pre_file_num)
    seq_list_iter = iter(seq_list)

    jobsName_list = list()
    def tqdm_er(x, desc, total): return x
    if verbose != False:
        tqdm_er = tqdm.tqdm
    for index, seq_sublist in enumerate(tqdm_er(
        [
            list(itertools.islice(
                seq_list_iter, 0, pre_file_num
            ))
            for _ in range(part_num)
        ],
        desc=tag_name,
        total=part_num
    )):
        pre_resq = Submit_to_server(
            seq_list=seq_sublist, cter=cter, desc=f'{tag_name}_{index}', MaxLength=MaxLength
        )
        resp_ = None
        while resp_ is None or resp_.status_code != 302:
            with requests.Session() as s:
                resp_ = s.send(
                    pre_resq,
                    proxies=proxies,
                    verify=False,
                    allow_redirects=False
                )
        jobsName_list.append(
            urllib.parse.parse_qs(
                urllib.parse.urlparse(
                    resp_.headers['Location']
                ).query
            )['jobName'][0]
        )
    if path_to_json is not None:
        if os.path.exists(os.path.split(path_to_json)[0]) == False:
            os.makedirs(os.path.split(path_to_json)[0])
        if os.path.exists(path_to_json) == False:
            with open(path_to_json, 'w+', encoding='UTF-8') as f:
                json.dump({'data': {}}, f)
        d_record = None
        with open(path_to_json, 'r', encoding='UTF-8') as f:
            d_record = json.load(f)
        d_record['data'][tag_name] = jobsName_list
        with open(path_to_json, 'w', encoding='UTF-8') as f:
            json.dump(d_record, f)
    return jobsName_list


def get_command(path_to_json: str, path_to_out_dir: str):
    id_list = None
    with open(path_to_json, 'r', encoding='UTF-8')as f:
        d = json.load(f)
        id_list = list(itertools.chain(
            *[d['data'][k] for k in d['data'].keys()]))

    for id_ in id_list:
        resp_ = None
        while resp_ is None or resp_.status_code != 200:
            with requests.Session() as s:
                resp_ = s.send(
                    requests.Request(
                        "GET",
                        f"https://possum.erc.monash.edu/static/result_files/{id_}/{id_}_pssm_features.zip"
                    ).prepare(),
                    proxies=proxies,
                    verify=False,
                    allow_redirects=False
                )
        with open(os.path.join(path_to_out_dir, *[f"{id_}_pssm_features.zip"]), 'wb+',) as f:
            f.write(resp_.content)
        resp_ = None
        while resp_ is None or resp_.status_code != 200:
            with requests.Session() as s:
                resp_ = s.send(
                    requests.Request(
                        "GET",
                        f"https://possum.erc.monash.edu/static/result_files/{id_}/{id_}_pssm_files.zip"
                    ).prepare(),
                    proxies=proxies,
                    verify=False,
                    allow_redirects=False
                )
        with open(os.path.join(path_to_out_dir, *[f"{id_}_pssm_files.zip"]), 'wb+',) as f:
            f.write(resp_.content)

    return id_list


if __name__ == '__main__':
    import argparse
    import argcomplete
    parser = argparse.ArgumentParser(prog='possum_submiter')
    argcomplete.autocomplete(parser)
    subparsers = parser.add_subparsers(dest='subparser')
    submiter_subparser = subparsers.add_parser('submiter')
    submiter_subparser.add_argument(
        '-f', type=str, required=True, help='Path to fasta.')
    submiter_subparser.add_argument(
        '--max', type=int, required=False, help='Path to fasta.', default=5000)
    submiter_subparser.add_argument('-o', type=str, required=True,
                                    help='Path to Out json.')
    submiter_subparser.add_argument(
        '-t', type=str, required=True, help='Tag fasta.')
    submiter_subparser.add_argument('-v', action='store_true', help='verbose')
    submiter_subparser.add_argument('-c', action='store_true', help='C ter.')
    submiter_subparser.set_defaults(func=lambda _: 'submiter_subparser')

    downloader_subparser = subparsers.add_parser('downloader')
    downloader_subparser.add_argument(
        '-j', type=str, required=True,
        help='Path to Out json.'
    )
    downloader_subparser.add_argument(
        '-o', type=str, required=False,
        default=None,
        help='Path to Out dir.'
    )
    downloader_subparser.set_defaults(func=lambda _: 'downloader_subparser')

    fasta_subparser = subparsers.add_parser('fasta')
    fasta_subparser.add_argument(
        '-f', type=str, required=True,
        help='Path to Fasta.'
    )
    fasta_subparser.add_argument(
        '-o', type=str, required=True,
        help='Path to Out json.'
    )
    fasta_subparser.add_argument(
        '--max', type=int, required=False, help='Path to fasta.', default=5000)
    fasta_subparser.add_argument('-c', action='store_true', help='C ter.')
    fasta_subparser.set_defaults(func=lambda _: 'fasta_subparser')
    args = parser.parse_args()

    if args.func(args) == 'submiter_subparser':
        go_(
            path_to_fasta=args.f,  # 'fasta.fasta'
            path_to_json=args.o,  # 'out.json'
            tag_name=args.t,
            cter=args.c,
            verbose=args.v,
            MaxLength=args.max
        )
    elif args.func(args) == 'downloader_subparser':
        get_command(
            path_to_json=args.j,
            path_to_out_dir=args.o
        )
    elif args.func(args) == 'fasta_subparser':
        get_fasta(
            path_to_fasta=args.f,
            path_to_out=args.o,
            cter=args.c,
            MaxLength=args.max
        )
