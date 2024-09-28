'''
Author: George Zhao
Date: 2021-05-26 22:31:53
LastEditors: George Zhao
LastEditTime: 2022-02-14 18:46:38
Description: 
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''

# %%
import os
import sys

import tqdm
sys.path.append('/mnt/md0/Public/T3_T4/txseml_addon/src/')
# %%
import utils
from Bio import SeqIO



work_Dir = utils.workdir.workdir(os.getcwd(), 3)
rdata_Dir = os.path.join(work_Dir, 'data')

logger = utils.glogger.Glogger('EP3_Top_n_gram_data_submiter', os.path.join(
    work_Dir, 'log'))
log_wrapper = utils.glogger.log_wrapper(logger=logger)
# %%
import collections
import json
import requests
import requests_toolbelt
from bs4 import BeautifulSoup
import re
out_Dir = os.path.join(work_Dir, 'out')
import traceback
import random
import string
# %%


@log_wrapper
def Dual_with_result(html_result: str):
    soup = BeautifulSoup(html_result, 'lxml')

    return {
        'Top_n_gram': {
            'data': list(map(float, soup.find_all('table')[1].find_all(
                'td')[3].get_text().strip().split('\t')))
        }
    }


@log_wrapper
def Submit_to_server(seq_fastaFormat: str):
    resp = requests.Response()
    mp_encoder = requests_toolbelt.MultipartEncoder(fields={
        'testdata': seq_fastaFormat,
        'uploadFile': ('', b'', 'application/octet-stream'),
        'Construct': 'Construct'
    }, boundary='----WebKitFormBoundary' + ''.join(random.sample(string.ascii_letters + string.digits, 16)))

    while resp.status_code is None or resp.status_code != 200:
        try:
            with requests.Session() as s:
                resp = s.post(
                    'http://bliulab.net/DistanceSVM/Receive.jsp',
                    data=mp_encoder,
                    headers={
                        'Content-Type': mp_encoder.content_type,
                        'User-Agent': r'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    },
                    timeout=10,
                    # proxies={'http': 'http://127.0.0.1:8889'}
                )
        except requests.exceptions.RequestException as e:
            logger.logger.warning(f'In Submit_to_server: {repr(e)}')
    return (resp.text)


@log_wrapper
def Submit(seq: str, idtf: str):
    result_dict = dict()
    # logger.logger.info(f'idtf: {idtf} Start to Submit.')
    # Full Length
    if len(seq) < 10 or len(seq) > 10000:
        logger.logger.error(
            f'idtf: {idtf} - Lenght < 10 or > 10000: {len(seq)}')
    fl_text = Submit_to_server(f'>{idtf}\n{seq}')
    result_dict.update(Dual_with_result(fl_text))
    # logger.logger.info(f'idtf: {idtf} - FL Have Submit.')
    return result_dict


@log_wrapper
def get_feature(path_to_fasta: str):
    return [{
        'id': seq.id,
        'seq': str(seq.seq),
        'Feature': Submit(seq.seq, seq.id)
    }for seq in tqdm.tqdm(list(SeqIO.parse(
        path_to_fasta,
        'fasta')))]


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(prog='Top_n_gram_data_submiter')

    parser.add_argument("-f", '--fasta', type=str, required=True,
                        help='Input')
    parser.add_argument("-o", '--output', type=str, required=True,
                        help='Output')

    args = parser.parse_args()

    final_dataset_dict = {
        'data': {
            'submit': get_feature(
                args.fasta
            ),
        },
    }

    with open(args.output, 'w+', encoding='UTF-8') as f:
        json.dump(
            final_dataset_dict,
            f
        )
