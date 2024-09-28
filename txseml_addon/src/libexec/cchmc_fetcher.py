'''
Author: George Zhao
Date: 2021-07-08 12:59:55
LastEditors: George Zhao
LastEditTime: 2022-02-01 15:33:55
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

from utils import workdir
from utils import glogger

work_Dir = workdir.workdir(os.getcwd(), 3)
rdata_Dir = os.path.join(work_Dir, 'data')
out_Dir = os.path.join(work_Dir, 'out')
tmp_Dir = os.path.join(work_Dir, 'tmp')

logger = glogger.Glogger('cchmc_fetcher', os.path.join(
    work_Dir, 'log'), LOG_LEVEL=glogger.logging.DEBUG, timestamp=False, std_err=False)
log_wrapper = glogger.log_wrapper(logger=logger)

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

# %%


def Submit_to_server(seqid: str, seq: str):
    mp_encoder = requests_toolbelt.MultipartEncoder(fields={
        'seqName': f'{seqid}',
        'txtSeq': f'{seq}',
        'SS': ' SS',
        'SA': 'SA',
        'version': 'sable2',
        'SAaction': 'wApproximator',
        'SAvalue': 'REAL',
        'email': '',
        'fileName': ('', b'', 'application/octet-stream')
    }, boundary='----WebKitFormBoundary' + ''.join(random.sample(string.ascii_letters + string.digits, 16)))

    return grequests.post(
        'http://sable.cchmc.org/cgi-bin/sable_server_July2003.cgi',
        data=mp_encoder,
        headers={
            'Content-Type': mp_encoder.content_type,
            'User-Agent': r'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        },
        timeout=10
    )


def get_submit_req_function(seq: str, idtf: str):
    return functools.partial(
        Submit_to_server,
        seqid=idtf,
        seq=seq,
    )

# %%


def Check_Complete(url: str):
    return grequests.get(
        url,
        headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            # 'Accept': 'application/json, text/javascript, */*; q=0.01'
        }
    )


def Check_Complete_wrapper(url: str):
    return functools.partial(
        Check_Complete,
        url=url
    )

# %%


def get_feature_from_list(seq_list: list, desc: str, record_filename: str, sc_: int = 10):
    if os.path.exists(os.path.split(record_filename)[0]) == False:
        os.makedirs(os.path.split(record_filename)[0])
    if os.path.exists(record_filename) == False:
        with open(record_filename, 'w+', encoding='UTF-8') as f:
            json.dump({'data': {}, }, f)
    d_record = None
    with open(record_filename, 'r', encoding='UTF-8') as f:
        d_record = json.load(f)
    if desc in d_record['data'].keys():
        if d_record['data'][desc] is not None:
            return

    # Submit.
    requ_list = [get_submit_req_function(seq.seq, seq.id)
                 for seq in seq_list]

    list_resp = [None, ] * len(requ_list)

    # Check.
    # !
    fileuuid_list = [None, ] * len(list_resp)

    while True:
        longBreak = False
        for index_ in range(len(fileuuid_list)):
            if fileuuid_list[index_] is not None and fileuuid_list[index_][:len('ReSubmit://')] == 'ReSubmit://':
                list_resp[index_] = None
                fileuuid_list[index_] = None
        if len(list(filter(lambda x: x is None, fileuuid_list))) == 0:
            break
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
                if len(seq_list) != len(index_to_post):
                    time.sleep(10)
            for index_posted, resp_geted in zip(
                index_to_post,
                grequests.map(
                    [requ_list[i]() for i in index_to_post],
                    size=sc_
                )
            ):
                if resp_geted is not None and resp_geted.status_code == 200:
                    if resp_geted.text.find('Attention: you have reached the daily limit of requests') >= 0:
                        longBreak = True
                        continue
                    soup = BeautifulSoup(resp_geted.text, 'lxml')
                    if len(soup.find_all('a')) <= 0:
                        continue
                    url_to_visit = soup.find_all('a')[0].get('href')
                    list_resp[index_posted] = url_to_visit
            if longBreak == True:
                logger.logger.warning('Take a Long Break.')
                time.sleep(1 * 60 * 60 * 2)
                longBreak = False
        logger.logger.debug('Url to Check: ' +
                            ";".join(
                                [f'{i}:{url}' for i, url in enumerate(list_resp)])
                            )

        check_requ_list = [Check_Complete_wrapper(url)
                           for url in list_resp]
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
                logger.logger.debug(
                    'Url to Check Left: ' +
                    ";".join([f'{i}:{list_resp[i]}' for i in index_to_post])
                )
            for index_posted, resp_geted in zip(
                index_to_post,
                grequests.map(
                    [check_requ_list[i]() for i in index_to_post],
                    size=sc_
                )
            ):
                if resp_geted is not None and resp_geted.status_code == 200:
                    if resp_geted.text.find('Your request is in the queue with the following status: <i>Aborted</i>') >= 0:
                        fileuuid_list[index_posted] = f'ReSubmit://{list_resp[index_posted]}'
                    if resp_geted.text.find('Your request is not in the queue. Please, consider resubmitting it.') >= 0:
                        fileuuid_list[index_posted] = f'ReSubmit://{list_resp[index_posted]}'
                    if len(re.findall(r'content=\"0;URL=([A-Za-z0-9:/\?\.=\-\_]*)', resp_geted.text)) != 0:
                        url_to_visit = re.findall(
                            r'content=\"0;URL=([A-Za-z0-9:/\?\.=\-\_]*)', resp_geted.text)[0]
                        fileuuid_list[index_posted] = url_to_visit
                    else:
                        continue
            if first_inpt != 0:
                time.sleep(len(index_to_post) * 1 * min(10, first_inpt))
            first_inpt = first_inpt + 1

    # ! Download.
    check_requ_list = [Check_Complete_wrapper(url)
                       for url in fileuuid_list]
    result_content_list = [None, ] * len(check_requ_list)
    first_inpt = 0
    while True:
        index_to_post = []
        for index_, resp in enumerate(result_content_list):
            if resp is None:
                index_to_post.append(index_)
        if index_to_post == []:
            break
        else:
            logger.logger.info(
                f'get_feature_from_list:Download: {desc} left: {len(index_to_post)}/{len(seq_list)}')
            if len(seq_list) != len(index_to_post) or first_inpt != 0:
                logger.logger.debug(
                    'DownloadURL ' + ";".join(
                        [f'{i}:{fileuuid_list[i]}' for i in index_to_post])
                )
            # time.sleep(max(len(index_to_post) * 1 *
            #                min(10, first_inpt) * 6, 60 * 10))
            if first_inpt != 0:
                time.sleep(10)
            first_inpt += 1
        for index_posted, resp_geted in zip(
            index_to_post,
            grequests.map(
                [check_requ_list[i]() for i in index_to_post],
                size=sc_
            )
        ):
            if resp_geted is not None and resp_geted.status_code == 200:
                ss_data = BeautifulSoup(resp_geted.text, 'lxml').find_all(
                    'input', attrs={'name': 'ssSeq'})
                sea_data = BeautifulSoup(resp_geted.text, 'lxml').find_all(
                    'input', attrs={'name': 'seaSeq'})
                if len(ss_data) != 0 or len(sea_data) != 0:
                    result_content_list[index_posted] = {
                        'id': seq_list[index_posted].id,
                        'Feature': {
                            'cchmc': {
                                'ssSeq': ss_data[0].get('value'),
                                'seaSeq': sea_data[0].get('value')
                            }
                        }
                    }
                else:
                    logger.logger.error(
                        f'{index_posted}: {fileuuid_list[index_posted]} Find No ssSeq or seaSeq.'
                    )

    d_record = None
    with open(record_filename, 'r', encoding='UTF-8') as f:
        d_record = json.load(f)
    d_record['data'][desc] = result_content_list
    with open(record_filename, 'w', encoding='UTF-8') as f:
        d_record = json.dump(d_record, f)
    return

# %%


def get_feature(record_filename: str, path_to_fasta: str, desc: str = '', num_of_each_part: int = 10, range_: list = [None, None]):
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
            part, desc=f'{desc}_{num_of_each_part}_{index_}', record_filename=record_filename)
    return


# %%
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog='cchmc_fecher')
    parser.add_argument('-f', type=str, required=True, help='Path to fasta.')
    parser.add_argument('-o', type=str, required=True,
                        help='Path to Out.')
    parser.add_argument('-t', type=str, required=True, help='Tag fasta.')
    parser.add_argument('-n', type=int, required=False,
                        help='num of each task.', default=25)
    parser.add_argument('-s', type=int, required=False,
                        help='Start of the index.')
    parser.add_argument('-e', type=int, required=False,
                        help='End of the index.')
    args = parser.parse_args()
    logger.logger.info(
        f'AppArgs: {args.f}; {args.t}; {args.o}; {args.n}; {args.s}; {args.e}')
    get_feature(
        record_filename=args.o,
        path_to_fasta=args.f,
        desc=args.t,
        num_of_each_part=args.n,
        range_=[args.s, args.e]
    )
