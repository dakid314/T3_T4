'''
Author: George Zhao
Date: 2021-05-26 22:31:53
LastEditors: George Zhao
LastEditTime: 2022-02-01 14:38:35
Description: 
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''

# %%
import os
import sys
import time
sys.path.append('..')
# %%
import utils
from Bio import SeqIO
import functools


work_Dir = utils.workdir.workdir(os.getcwd(), 3)
rdata_Dir = os.path.join(work_Dir, 'data')

logger = utils.glogger.Glogger('EP3_Pse_in_One_data_submiter', os.path.join(
    work_Dir, 'log'), std_err=False, timestamp=False)
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
import tqdm
# %%


def Dual_with_result(html_result: str):
    soup = BeautifulSoup(html_result, 'lxml')
    resp = requests.Response()
    while resp.status_code is None or resp.status_code != 200:
        try:
            with requests.Session() as s:
                resp = s.get('http://bliulab.net' + soup.find_all('a', attrs={
                    'class': 'myButton',
                    'target': '_blank'
                })[0].get('href')[:-1])
        except requests.exceptions.RequestException as e:
            logger.logger.warning(f'In Dual_with_result: {repr(e)}')

    return {
        'PC_PSEACC': {
            'Origin': resp.text,
            'Data type': re.findall(r'Data type: (.+)\nMode: ', resp.text)[0],
            'Mode': re.findall(r'\nMode: (.+)\nPhysicochemical properties: ', resp.text)[0],
            'Physicochemical properties': re.findall(r'\nPhysicochemical properties: (.+)\nLambda: ', resp.text)[0],
            'Lambda': re.findall(r'\nLambda: (.+)\nW: ', resp.text)[0],
            'W': re.findall(r'\nW: (.+)\n\n', resp.text)[0],
            'data': list(map(float, re.findall(r'\nW: .+\n\n>.+\n([-0-9\.,]+)\r\n\n', resp.text)[0].split(','))),
        }
    }


def Submit_to_server(seq_fastaFormat: str):
    resp = requests.Response()
    mp_encoder = requests_toolbelt.MultipartEncoder(fields={
        'method': 'PC-PseAAC',
        'output_format': 'csv',
        'Hydrophobicity': 'on',
        'Hydrophilicity': 'on',
        'Mass': 'on',
        'lamada': '1',
        'w': '0.5',
        'e_mail': '',
        'rec_data': seq_fastaFormat,
        'upload_data': ('', b'', 'application/octet-stream')
    }, boundary='----WebKitFormBoundary' + ''.join(random.sample(string.ascii_letters + string.digits, 16)))

    while resp.status_code is None or resp.status_code != 200:
        try:
            with requests.Session() as s:
                resp = s.post(
                    'http://bliulab.net/Pse-in-One/PROTEIN/PC-PseAAC/',
                    data=mp_encoder,
                    headers={
                        'Content-Type': mp_encoder.content_type,
                        'User-Agent': r'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    },
                    timeout=10,
                    # proxies={'http': 'http://127.0.0.1:8889'}
                )
        except requests.exceptions.RequestException as e:
            logger.logger.warning(
                f'In Submit_to_server: {repr(e)} {resp.text} {seq_fastaFormat}')
    return (resp.text)


def Submit(seq: str, idtf: str):
    result_dict = dict()
    # logger.logger.info(f'idtf: {idtf} Start to Submit.')
    # Full Length
    fl_text = Submit_to_server(f'>{idtf}\n{seq}')
    result_dict.update(Dual_with_result(fl_text))
    # logger.logger.info(f'idtf: {idtf} - FL Have Submit.')
    return result_dict


def get_feature(path_to_fasta: str):
    return [{
        'id': seq.id,
        'seq': str(seq.seq),
        'Feature': Submit(seq.seq, seq.id)
    }for seq in tqdm.tqdm(list(SeqIO.parse(
        path_to_fasta,
        'fasta')))]


if __name__ == '__main__':
    pass
    # # %% T1 ###############################
    # # Traning
    # T3se_fasta = get_feature(
    #     os.path.join(
    #         work_Dir, 'data/db/T1/t_p.fasta')
    # )

    # no_T3se_fasta = get_feature(
    #     os.path.join(
    #         work_Dir, 'data/db/T1/t_n.fasta')
    # )

    # final_dataset_dict = {
    #     'data': {
    #         'T1se': T3se_fasta,
    #         'no_T1se': no_T3se_fasta
    #     },
    # }
    # with open(os.path.join(out_Dir, 'T1/data/PC_PSEACC_data_t.json'), 'w+', encoding='UTF-8') as f:
    #     json.dump(
    #         final_dataset_dict,
    #         f
    #     )
    # # Validate
    # T3se_fasta = get_feature(
    #     os.path.join(
    #         work_Dir, 'data/db/T1/v_p.fasta')
    # )

    # no_T3se_fasta = get_feature(
    #     os.path.join(
    #         work_Dir, 'data/db/T1/v_n.fasta')
    # )

    # final_dataset_dict = {
    #     'data': {
    #         'T1se': T3se_fasta,
    #         'no_T1se': no_T3se_fasta
    #     },
    # }
    # with open(os.path.join(out_Dir, 'T1/data/PC_PSEACC_data_v.json'), 'w+', encoding='UTF-8') as f:
    #     json.dump(
    #         final_dataset_dict,
    #         f
    #     )

    # # %% T2 ###############################
    # # Traning
    # T3se_fasta = get_feature(
    #     os.path.join(
    #         work_Dir, 'data/db/T2/t_p.fasta')
    # )

    # no_T3se_fasta = get_feature(
    #     os.path.join(
    #         work_Dir, 'data/db/T2/t_n.fasta')
    # )

    # final_dataset_dict = {
    #     'data': {
    #         'T2se': T3se_fasta,
    #         'no_T2se': no_T3se_fasta
    #     },
    # }
    # with open(os.path.join(out_Dir, 'T2/data/PC_PSEACC_data_t.json'), 'w+', encoding='UTF-8') as f:
    #     json.dump(
    #         final_dataset_dict,
    #         f
    #     )
    # # Validate
    # T3se_fasta = get_feature(
    #     os.path.join(
    #         work_Dir, 'data/db/T2/v_p.fasta')
    # )

    # no_T3se_fasta = get_feature(
    #     os.path.join(
    #         work_Dir, 'data/db/T2/v_n.fasta')
    # )

    # final_dataset_dict = {
    #     'data': {
    #         'T2se': T3se_fasta,
    #         'no_T2se': no_T3se_fasta
    #     },
    # }
    # with open(os.path.join(out_Dir, 'T2/data/PC_PSEACC_data_v.json'), 'w+', encoding='UTF-8') as f:
    #     json.dump(
    #         final_dataset_dict,
    #         f
    #     )

    # # %% T3 ###############################
    # # Traning
    # T3se_fasta = get_feature(
    #     os.path.join(
    #         work_Dir, 'data/db/T3/t_p.fasta')
    # )

    # no_T3se_fasta = get_feature(
    #     os.path.join(
    #         work_Dir, 'data/db/T3/t_n.fasta')
    # )

    # final_dataset_dict = {
    #     'data': {
    #         'T4se': T3se_fasta,
    #         'no_T4se_1': no_T3se_fasta
    #     },
    # }
    # with open(os.path.join(out_Dir, 'T3/data/PC_PSEACC_data_t.json'), 'w+', encoding='UTF-8') as f:
    #     json.dump(
    #         final_dataset_dict,
    #         f
    #     )
    # # Validate
    # T3se_fasta = get_feature(
    #     os.path.join(
    #         work_Dir, 'data/db/T3/v_p.fasta')
    # )

    # no_T3se_fasta = get_feature(
    #     os.path.join(
    #         work_Dir, 'data/db/T3/v_n.fasta')
    # )

    # final_dataset_dict = {
    #     'data': {
    #         'T4se': T3se_fasta,
    #         'no_T4se_1': no_T3se_fasta
    #     },
    # }
    # with open(os.path.join(out_Dir, 'T3/data/PC_PSEACC_data_v.json'), 'w+', encoding='UTF-8') as f:
    #     json.dump(
    #         final_dataset_dict,
    #         f
    #     )
    # %% T4 ###############################
    # Traning
    T3se_fasta = get_feature(
        os.path.join(
            work_Dir, 'data/db/T4/t_p.fasta')
    )

    no_T3se_fasta = get_feature(
        os.path.join(
            work_Dir, 'data/db/T4/t_n.fasta')
    )

    final_dataset_dict = {
        'data': {
            'T4se': T3se_fasta,
            'no_T4se_1': no_T3se_fasta
        },
    }
    with open(os.path.join(out_Dir, 'T4/data/PC_PSEACC_data_t.json'), 'w+', encoding='UTF-8') as f:
        json.dump(
            final_dataset_dict,
            f
        )
    # Validate
    T3se_fasta = get_feature(
        os.path.join(
            work_Dir, 'data/db/T4/v_p.fasta')
    )

    no_T3se_fasta = get_feature(
        os.path.join(
            work_Dir, 'data/db/T4/v_n.fasta')
    )

    final_dataset_dict = {
        'data': {
            'T4se': T3se_fasta,
            'no_T4se_1': no_T3se_fasta
        },
    }
    with open(os.path.join(out_Dir, 'T4/data/PC_PSEACC_data_v.json'), 'w+', encoding='UTF-8') as f:
        json.dump(
            final_dataset_dict,
            f
        )

    # %% T6 ###############################
    # Traning
    T3se_fasta = get_feature(
        os.path.join(
            work_Dir, 'data/db/T6/t_p.fasta')
    )

    no_T3se_fasta = get_feature(
        os.path.join(
            work_Dir, 'data/db/T6/t_n.fasta')
    )

    final_dataset_dict = {
        'data': {
            'T6se': T3se_fasta,
            'no_T6se_1': no_T3se_fasta
        },
    }
    with open(os.path.join(out_Dir, 'T6/data/PC_PSEACC_data_t.json'), 'w+', encoding='UTF-8') as f:
        json.dump(
            final_dataset_dict,
            f
        )
    # Validate
    T3se_fasta = get_feature(
        os.path.join(
            work_Dir, 'data/db/T6/v_p.fasta')
    )

    no_T3se_fasta = get_feature(
        os.path.join(
            work_Dir, 'data/db/T6/v_n.fasta')
    )

    final_dataset_dict = {
        'data': {
            'T6se': T3se_fasta,
            'no_T6se_1': no_T3se_fasta
        },
    }
    with open(os.path.join(out_Dir, 'T6/data/PC_PSEACC_data_v.json'), 'w+', encoding='UTF-8') as f:
        json.dump(
            final_dataset_dict,
            f
        )
