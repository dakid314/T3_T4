'''
Author: George Zhao
Date: 2021-08-14 21:02:29
LastEditors: George Zhao
LastEditTime: 2022-02-01 19:25:33
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

logger = utils.glogger.Glogger('disopred3_downloader', os.path.join(
    work_Dir, 'log'), timestamp=False, std_err=False)
log_wrapper = utils.glogger.log_wrapper(logger=logger)

import grequests
import requests
import time
import tqdm

# %%


def get_comb(UUID_dict: dict):
    return grequests.get(
        f'http://bioinf.cs.ucl.ac.uk/psipred/api/submissions/{UUID_dict["UUID"]}.comb',
        headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json, text/javascript, */*; q=0.01'
        }
    )


def get_comb_wrapper(UUID_dict: dict):
    return functools.partial(
        get_comb,
        UUID_dict=UUID_dict
    )


def get_pbdat(UUID_dict: dict):
    return grequests.get(
        f'http://bioinf.cs.ucl.ac.uk/psipred/api/submissions/{UUID_dict["UUID"]}.pbdat',
        headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json, text/javascript, */*; q=0.01'
        }
    )


def get_pbdat_wrapper(UUID_dict: dict):
    return functools.partial(
        get_pbdat,
        UUID_dict=UUID_dict
    )


def get_ss2(UUID_dict: dict):
    return grequests.get(
        f'http://bioinf.cs.ucl.ac.uk/psipred/api/submissions/{UUID_dict["UUID"]}.ss2',
        headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json, text/javascript, */*; q=0.01'
        }
    )


def get_ss2_wrapper(UUID_dict: dict):
    return functools.partial(
        get_ss2,
        UUID_dict=UUID_dict
    )
# %%


def getfile(
        path_to_json: str,
        path_to_outdir: str,
        desc: str,
        num_of_each_part: int = 10,
        sc_: int = None):
    uuid_dict = None
    with open(path_to_json, 'r', encoding='UTF-8') as f:
        uuid_dict = json.load(f)
    tag_name = f'{desc}_{num_of_each_part}'
    k_list = [k for k in uuid_dict['data'].keys() if k.find(tag_name) == 0]

    tout_dir = os.path.join(
        path_to_outdir, tag_name
    )

    if os.path.exists(tout_dir) == False:
        os.makedirs(tout_dir)

    uuid_list = list(itertools.chain(
        *[uuid_dict['data'][k] for k in k_list]))

    if uuid_dict['job'] == 'disopred':
        # pbdat
        check_requ_list = [get_pbdat_wrapper(dict_)
                           for dict_ in uuid_list]
        fileuuid_list = [None, ] * len(check_requ_list)
        while True:
            index_to_post = []
            for index_, resp in enumerate(fileuuid_list):
                if resp is None:
                    index_to_post.append(index_)
            if index_to_post == []:
                break
            else:
                logger.logger.info(
                    f'getfile:pbdat: {desc} left: {len(index_to_post)}/{len(uuid_list)}')
            for index_posted, resp_geted in zip(
                index_to_post,
                grequests.map(
                    [check_requ_list[i]() for i in index_to_post],
                    size=sc_
                )
            ):
                if resp_geted is not None and resp_geted.status_code == 200:
                    fileuuid_list[index_posted] = True
                    with open(os.path.join(tout_dir, f'{uuid_list[index_posted]["UUID"]}.pbdat'), 'w+', encoding='UTF-8') as f:
                        f.write(resp_geted.text)

        # comb
        check_requ_list = [get_comb_wrapper(dict_)
                           for dict_ in uuid_list]
        fileuuid_list = [None, ] * len(check_requ_list)
        while True:
            index_to_post = []
            for index_, resp in enumerate(fileuuid_list):
                if resp is None:
                    index_to_post.append(index_)
            if index_to_post == []:
                break
            else:
                logger.logger.info(
                    f'getfile:comb: {desc} left: {len(index_to_post)}/{len(uuid_list)}')
            for index_posted, resp_geted in zip(
                index_to_post,
                grequests.map(
                    [check_requ_list[i]() for i in index_to_post],
                    size=sc_
                )
            ):
                if resp_geted is not None and resp_geted.status_code == 200:
                    fileuuid_list[index_posted] = True
                    with open(os.path.join(tout_dir, f'{uuid_list[index_posted]["UUID"]}.comb'), 'w+', encoding='UTF-8') as f:
                        f.write(resp_geted.text)
    elif uuid_dict['job'] == 'psipred':
        # ss2
        check_requ_list = [get_ss2_wrapper(dict_)
                           for dict_ in uuid_list]
        fileuuid_list = [None, ] * len(check_requ_list)
        while True:
            index_to_post = []
            for index_, resp in enumerate(fileuuid_list):
                if resp is None:
                    index_to_post.append(index_)
            if index_to_post == []:
                break
            else:
                logger.logger.info(
                    f'getfile:ss2: {desc} left: {len(index_to_post)}/{len(uuid_list)}')
            for index_posted, resp_geted in zip(
                index_to_post,
                grequests.map(
                    [check_requ_list[i]() for i in index_to_post],
                    size=sc_
                )
            ):
                if resp_geted is not None and resp_geted.status_code == 200:
                    fileuuid_list[index_posted] = True
                    with open(os.path.join(tout_dir, f'{uuid_list[index_posted]["UUID"]}.ss2'), 'w+', encoding='UTF-8') as f:
                        f.write(resp_geted.text)
    else:
        raise RuntimeError(
            f"In file: {path_to_json}: uuid_dict['job'] = {uuid_dict['job']} ?")
    return


# %%
# T4
# getfile(
#     path_to_json=os.path.join(
#         work_Dir, *['out', 'T4SEs', 'data', 'Bastion4',
#                     'DISOPRED3', 'datarecord.json.t4']
#     ),
#     path_to_outdir=os.path.join(
#         work_Dir, *['out', 'T4SEs', 'data', 'Bastion4', 'DISOPRED3']
#     ),
#     desc='t_4_t_p'
# )
# getfile(
#     path_to_json=os.path.join(
#         work_Dir, *['out', 'T4SEs', 'data', 'Bastion4',
#                     'DISOPRED3', 'datarecord.json.t4']
#     ),
#     path_to_outdir=os.path.join(
#         work_Dir, *['out', 'T4SEs', 'data', 'Bastion4', 'DISOPRED3']
#     ),
#     desc='t_4_v_p'
# )
# getfile(
#     path_to_json=os.path.join(
#         work_Dir, *['out', 'T4SEs', 'data', 'Bastion4',
#                     'DISOPRED3', 'datarecord.json.t4']
#     ),
#     path_to_outdir=os.path.join(
#         work_Dir, *['out', 'T4SEs', 'data', 'Bastion4', 'DISOPRED3']
#     ),
#     desc='t_4_t_n_1'
# )
# getfile(
#     path_to_json=os.path.join(
#         work_Dir, *['out', 'T4SEs', 'data', 'Bastion4',
#                     'DISOPRED3', 'datarecord.json.t4']
#     ),
#     path_to_outdir=os.path.join(
#         work_Dir, *['out', 'T4SEs', 'data', 'Bastion4', 'DISOPRED3']
#     ),
#     desc='t_4_v_n_1'
# )
# T6
# getfile(
#     path_to_json=os.path.join(
#         work_Dir, *['out', 'T6SEs', 'data', 'PyPredT6',
#                     'DISOPRED3', 'datarecord.json.t6']
#     ),
#     path_to_outdir=os.path.join(
#         work_Dir, *['out', 'T6SEs', 'data', 'PyPredT6', 'DISOPRED3']
#     ),
#     desc='t_6_t_p'
# )
# getfile(
#     path_to_json=os.path.join(
#         work_Dir, *['out', 'T6SEs', 'data', 'PyPredT6',
#                     'DISOPRED3', 'datarecord.json.t6']
#     ),
#     path_to_outdir=os.path.join(
#         work_Dir, *['out', 'T6SEs', 'data', 'PyPredT6', 'DISOPRED3']
#     ),
#     desc='t_6_t_n_1'
# )
# getfile(
#     path_to_json=os.path.join(
#         work_Dir, *['out', 'T6SEs', 'data', 'PyPredT6',
#                     'DISOPRED3', 'datarecord.json.t6']
#     ),
#     path_to_outdir=os.path.join(
#         work_Dir, *['out', 'T6SEs', 'data', 'PyPredT6', 'DISOPRED3']
#     ),
#     desc='t_6_v_p'
# )
# getfile(
#     path_to_json=os.path.join(
#         work_Dir, *['out', 'T6SEs', 'data', 'PyPredT6',
#                     'DISOPRED3', 'datarecord.json.t6']
#     ),
#     path_to_outdir=os.path.join(
#         work_Dir, *['out', 'T6SEs', 'data', 'PyPredT6', 'DISOPRED3']
#     ),
#     desc='t_6_v_n_1'
# )
# %%
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog='disopred3_downloader')
    parser.add_argument('-f', type=str, required=True, help='Path to json.')
    parser.add_argument('-o', type=str, required=True,
                        help='Path to OutDir.')
    parser.add_argument('-t', type=str, required=True, help='Tag fasta.')
    parser.add_argument('-n', type=int, required=False,
                        help='num of each task.', default=10)
    args = parser.parse_args()
    logger.logger.info(
        f'AppArgs: {args.f}; {args.t}; {args.o}; {args.n}')
    getfile(
        path_to_json=args.f,
        path_to_outdir=args.o,
        num_of_each_part=args.n,
        desc=args.t
    )
