'''
Author: George Zhao
Date: 2021-07-08 12:59:55
LastEditors: George Zhao
LastEditTime: 2022-03-24 16:00:38
Description: 
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''
# %%
from gevent import monkey as curious_george
curious_george.patch_all(thread=False, select=False)
import os
import sys
sys.path.append('src')
import json
import re
import string
import random
import functools
import itertools
import math

import utils

work_Dir = utils.workdir.workdir(os.getcwd(), 3)
rdata_Dir = os.path.join(work_Dir, 'data')
out_Dir = os.path.join(work_Dir, 'out')
tmp_Dir = os.path.join(work_Dir, 'tmp')

logger = utils.glogger.Glogger('disorder3_submmit', os.path.join(
    work_Dir, 'log'), timestamp=False, std_err=False)
log_wrapper = utils.glogger.log_wrapper(logger=logger)

import grequests
import requests
import requests_toolbelt
import time
import tqdm
from bs4 import BeautifulSoup
from Bio import SeqIO

import warnings
from Bio import BiopythonDeprecationWarning
warnings.simplefilter("ignore", BiopythonDeprecationWarning)
# %%
proxies = None if True else {'http': 'http://127.0.0.1:10809',
                             'https': 'http://127.0.0.1:10809'}


def Submit_to_server(seqid: str, seq: str, job: str):
    mp_encoder = requests_toolbelt.MultipartEncoder(fields={
        'job': job,
        'submission_name': seqid,
        'email': f'{"".join(random.sample(string.ascii_letters + string.digits, 8))}@powerencry.com',
        'input_data': ('input.txt', seq.encode(), 'application/octet-stream')
    }, boundary='----WebKitFormBoundary' + ''.join(random.sample(string.ascii_letters + string.digits, 16)))

    return grequests.post(
        'http://bioinf.cs.ucl.ac.uk/psipred/api/submission/',
        data=mp_encoder,
        headers={
            'Referer': 'http://bioinf.cs.ucl.ac.uk/psipred/',
            'Accept-Encoding': 'gzip, deflate',
            'Accept-Language': 'en,zh-CN;q=0.9,zh;q=0.8',
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'Content-Type': mp_encoder.content_type,
            'X-Requested-With': 'XMLHttpRequest',
            'User-Agent': 'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        }
    )


def get_submit_req_function(seq: str, idtf: str, cter: bool, job: str):
    # Full Length
    if cter == False:
        if len(seq) > 1500:
            logger.logger.warning(
                f'idtf: {idtf} - Lenght > 1500: {len(seq)}')
            seq = seq[0:1500]
        if len(seq) < 30:
            logger.logger.error(
                f'idtf: {idtf} - Lenght < 30: {len(seq)}')
            seq = seq + 'A' * (30 - len(seq))
    else:
        if len(seq) > 1500:
            logger.logger.warning(
                f'idtf: {idtf} - Lenght > 1500: {len(seq)}')
            seq = seq[-1500:]
        if len(seq) < 30:
            logger.logger.error(
                f'idtf: {idtf} - Lenght < 30: {len(seq)}')
            seq = 'A' * (30 - len(seq)) + seq

    return functools.partial(
        Submit_to_server,
        seqid=idtf,
        seq=seq,
        job=job
    )


def Check_Complete(UUID_dict: dict):
    return grequests.get(
        f'http://bioinf.cs.ucl.ac.uk/psipred/api/submission/{UUID_dict["UUID"]}',
        headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json, text/javascript, */*; q=0.01'
        }
    )


def Check_Complete_wrapper(UUID_dict: dict):
    return functools.partial(
        Check_Complete,
        UUID_dict=UUID_dict
    )


def get_feature_from_list(seq_list: list, desc: str, record_filename: str, job: str, cter: bool, sc_: int = 10):
    if os.path.exists(os.path.split(record_filename)[0]) == False:
        os.makedirs(os.path.split(record_filename)[0])
    if os.path.exists(record_filename) == False:
        with open(record_filename, 'w+', encoding='UTF-8') as f:
            json.dump({'data': {}, 'job': job}, f)
    d_record = None
    with open(record_filename, 'r', encoding='UTF-8') as f:
        d_record = json.load(f)
    if d_record['job'] != job:
        raise RuntimeError(
            f'In Record file: {record_filename}: Found job:{d_record["job"]} != job')
    if desc in d_record['data'].keys():
        if d_record['data'][desc] is not None:
            return

    requ_list = [get_submit_req_function(seq.seq, seq.id, cter, job=job)
                 for seq in seq_list]

    list_resp = [None, ] * len(requ_list)

    # Submit.
    while True:
        index_to_post = []
        for index_, resp in enumerate(list_resp):
            if resp is None:
                index_to_post.append(index_)
        if index_to_post == []:
            break
        else:
            logger.logger.info(
                f'get_feature_from_list:Submit: {desc} left: {len(index_to_post)}/{len(seq_list)}')
        for index_posted, resp_geted in zip(
            index_to_post,
            grequests.map(
                [requ_list[i]() for i in index_to_post],
                size=sc_
            )
        ):
            if resp_geted is not None and resp_geted.status_code == 201:
                try:
                    list_resp[index_posted] = resp_geted.json()
                except json.JSONDecodeError as e:
                    logger.logger.error(f'{repr(e)}')
                finally:
                    pass

    # Check.
    check_requ_list = [Check_Complete_wrapper(dict_)
                       for dict_ in list_resp]
    fileuuid_list = [None, ] * len(check_requ_list)
    first_inpt = 0
    while True:
        index_to_post = []
        for index_, resp in enumerate(fileuuid_list):
            if resp is None:
                index_to_post.append(index_)
        if index_to_post == []:
            break
        else:
            logger.logger.info(
                f'get_feature_from_list:Check: {desc} left: {len(index_to_post)}/{len(seq_list)}')
        for index_posted, resp_geted in zip(
            index_to_post,
            grequests.map(
                [check_requ_list[i]() for i in index_to_post],
                size=sc_
            )
        ):
            if resp_geted is not None and resp_geted.status_code == 200:
                try:
                    _dict__ = resp_geted.json()
                    if _dict__['state'] == 'Complete':
                        fileuuid_list[index_posted] = {
                            'UUID': _dict__['submissions'][0]['UUID'],
                            'submission_name': _dict__['submissions'][0]['submission_name']
                        }
                except json.JSONDecodeError as e:
                    logger.logger.error(f'{repr(e)}')
                finally:
                    pass
        if first_inpt != 0:
            time.sleep(len(index_to_post) * 1 * min(10, first_inpt))
        first_inpt = first_inpt + 1

    d_record = None
    with open(record_filename, 'r', encoding='UTF-8') as f:
        d_record = json.load(f)
    d_record['data'][desc] = fileuuid_list
    with open(record_filename, 'w', encoding='UTF-8') as f:
        d_record = json.dump(d_record, f)
    return

# %%


def get_feature(record_filename: str, path_to_fasta: str, cter: bool, job: str, desc: str = '', num_of_each_part: int = 10, range_: list = [None, None]):
    seq_list = list(SeqIO.parse(path_to_fasta, 'fasta'))
    num_of_part = math.ceil(len(seq_list) / num_of_each_part)
    seq_list_iter = iter(seq_list)
    for index_, part in tqdm.tqdm(
        enumerate([list(itertools.islice(seq_list_iter, 0, num_of_each_part))
                   for _ in range(num_of_part)]),
        total=num_of_part,
        desc=desc
    ):
        if range_[0] is not None and range_[0] > index_:
            continue
        if range_[1] is not None and range_[1] <= index_:
            continue
        get_feature_from_list(
            part, desc=f'{desc}_{num_of_each_part}_{index_}', record_filename=record_filename, cter=cter, job=job)
    return


# %%
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog='psipred')
    parser.add_argument('-f', type=str, required=True, help='Path to fasta.')
    parser.add_argument('-o', type=str, required=True,
                        help='Path to Out.')
    parser.add_argument('-t', type=str, required=True, help='Tag fasta.')
    parser.add_argument('-n', type=int, required=False,
                        help='num of each task.', default=10)
    parser.add_argument('-s', type=int, required=False,
                        help='Start of the index.')
    parser.add_argument('-e', type=int, required=False,
                        help='End of the index.')
    parser.add_argument('-c', action='store_true', help='Cter.')
    parser.add_argument('-j', type=str, default='disopred',
                        choices=['disopred', 'psipred'])
    args = parser.parse_args()
    logger.logger.info(
        f'AppArgs: {args.f}; {args.t}; {args.o}; {args.n}; {args.s}; {args.e}; {args.j}; {args.c}')
    get_feature(
        record_filename=args.o,
        path_to_fasta=args.f,
        cter=args.c,
        job=args.j,
        desc=args.t,
        num_of_each_part=args.n,
        range_=[args.s, args.e]
    )
