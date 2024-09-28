'''
Author: George Zhao
Date: 2021-05-26 22:31:53
LastEditors: George Zhao
LastEditTime: 2022-02-14 18:46:50
Description: 
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''

# %%
import os
import sys
sys.path.append('src')
# %%
import utils
from Bio import SeqIO
import functools


work_Dir = utils.workdir.workdir(os.getcwd(), 3)
rdata_Dir = os.path.join(work_Dir, 'data')

logger = utils.glogger.Glogger('EP3_iDNA_Prot_dis_data_submiter', os.path.join(
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
    return {
        'iDNA_Prot_dis': {
            'data': 1 if html_result.find('Non DNA-binding Protein') < 0 else 0
        }
    }


@log_wrapper
def Submit_to_server(seq_fastaFormat: str):
    resp = requests.Response()
    mp_encoder = requests_toolbelt.MultipartEncoder(fields={
        'R1': '0',
        'testdata': seq_fastaFormat,
        'uploadFile': ('', b'', 'application/octet-stream')
    }, boundary='----WebKitFormBoundary' + ''.join(random.sample(string.ascii_letters + string.digits, 16)))

    while resp.status_code is None or resp.status_code != 200:
        try:
            resp = requests.post(
                'http://bliulab.net/iDNA-Prot_dis/Receive.jsp',
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
    logger.logger.info(f'idtf: {idtf} Start to Submit.')
    # Full Length
    if len(seq) > 10000:
        logger.logger.error(
            f'idtf: {idtf} - Lenght > 10000: {len(seq)}')
    fl_text = Submit_to_server(f'>{idtf}\n{seq}')
    result_dict.update(Dual_with_result(fl_text))
    logger.logger.info(f'idtf: {idtf} - FL Have Submit.')
    return result_dict


@log_wrapper
def get_feature(path_to_fasta: str):
    return [{
        'id': seq.id,
        'seq': str(seq.seq),
        'Feature': Submit(seq.seq, seq.id)
    }for seq in SeqIO.parse(
        path_to_fasta,
        'fasta')]


if __name__ == '__main__':
    # %% T1 ####################################################################################3
    # Traning
    T3se_fasta = get_feature(
        os.path.join(
            work_Dir, 'data/db/T1/t_p.fasta')
    )

    no_T3se_fasta = get_feature(
        os.path.join(
            work_Dir, 'data/db/T1/t_n.fasta')
    )

    final_dataset_dict = {
        'data': {
            'T1se': T3se_fasta,
            'no_T1se': no_T3se_fasta
        },
    }
    with open(os.path.join(out_Dir, 'T1/data/iDNA_Prot_dis_data_t.json'), 'w+', encoding='UTF-8') as f:
        json.dump(
            final_dataset_dict,
            f
        )
    # Validate
    T3se_fasta = get_feature(
        os.path.join(
            work_Dir, 'data/db/T1/v_p.fasta')
    )

    no_T3se_fasta = get_feature(
        os.path.join(
            work_Dir, 'data/db/T1/v_n.fasta')
    )

    final_dataset_dict = {
        'data': {
            'T1se': T3se_fasta,
            'no_T1se': no_T3se_fasta
        },
    }
    with open(os.path.join(out_Dir, 'T1/data/iDNA_Prot_dis_data_v.json'), 'w+', encoding='UTF-8') as f:
        json.dump(
            final_dataset_dict,
            f
        )

    # %% T2 ####################################################################################3
    # Traning
    T3se_fasta = get_feature(
        os.path.join(
            work_Dir, 'data/db/T2/t_p.fasta')
    )

    no_T3se_fasta = get_feature(
        os.path.join(
            work_Dir, 'data/db/T2/t_n.fasta')
    )

    final_dataset_dict = {
        'data': {
            'T2se': T3se_fasta,
            'no_T2se': no_T3se_fasta
        },
    }
    with open(os.path.join(out_Dir, 'T2/data/iDNA_Prot_dis_data_t.json'), 'w+', encoding='UTF-8') as f:
        json.dump(
            final_dataset_dict,
            f
        )
    # Validate
    T3se_fasta = get_feature(
        os.path.join(
            work_Dir, 'data/db/T2/v_p.fasta')
    )

    no_T3se_fasta = get_feature(
        os.path.join(
            work_Dir, 'data/db/T2/v_n.fasta')
    )

    final_dataset_dict = {
        'data': {
            'T2se': T3se_fasta,
            'no_T2se': no_T3se_fasta
        },
    }
    with open(os.path.join(out_Dir, 'T2/data/iDNA_Prot_dis_data_v.json'), 'w+', encoding='UTF-8') as f:
        json.dump(
            final_dataset_dict,
            f
        )
    # %% T3 ####################################################################################3
    # Traning
    T3se_fasta = get_feature(
        os.path.join(
            work_Dir, 'data/db/T3/t_p.fasta')
    )

    no_T3se_fasta = get_feature(
        os.path.join(
            work_Dir, 'data/db/T3/t_n.fasta')
    )

    final_dataset_dict = {
        'data': {
            'T3se': T3se_fasta,
            'no_T3se': no_T3se_fasta
        },
    }
    with open(os.path.join(out_Dir, 'T3/data/iDNA_Prot_dis_data_t.json'), 'w+', encoding='UTF-8') as f:
        json.dump(
            final_dataset_dict,
            f
        )
    # Validate
    T3se_fasta = get_feature(
        os.path.join(
            work_Dir, 'data/db/T3/v_p.fasta')
    )

    no_T3se_fasta = get_feature(
        os.path.join(
            work_Dir, 'data/db/T3/v_n.fasta')
    )

    final_dataset_dict = {
        'data': {
            'T3se': T3se_fasta,
            'no_T3se': no_T3se_fasta
        },
    }
    with open(os.path.join(out_Dir, 'T3/data/iDNA_Prot_dis_data_v.json'), 'w+', encoding='UTF-8') as f:
        json.dump(
            final_dataset_dict,
            f
        )

    # %% T4 ####################################################################################3
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
            'no_T4se': no_T3se_fasta
        },
    }
    with open(os.path.join(out_Dir, 'T4/data/iDNA_Prot_dis_data_t.json'), 'w+', encoding='UTF-8') as f:
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
            'no_T4se': no_T3se_fasta
        },
    }
    with open(os.path.join(out_Dir, 'T4/data/iDNA_Prot_dis_data_v.json'), 'w+', encoding='UTF-8') as f:
        json.dump(
            final_dataset_dict,
            f
        )

    # %% T6 ####################################################################################3
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
            'no_T6se': no_T3se_fasta
        },
    }
    with open(os.path.join(out_Dir, 'T6/data/iDNA_Prot_dis_data_t.json'), 'w+', encoding='UTF-8') as f:
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
            'no_T6se': no_T3se_fasta
        },
    }
    with open(os.path.join(out_Dir, 'T6/data/iDNA_Prot_dis_data_v.json'), 'w+', encoding='UTF-8') as f:
        json.dump(
            final_dataset_dict,
            f
        )
