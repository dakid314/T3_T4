import os
import sys
sys.path.append('..')
sys.path.append('../..')
import itertools
# %%
import utils
from . import libdataloader

import numpy as np
import pandas as pd
from Bio import SeqIO, Seq
import tqdm
# %%


def get_HH_CKSAAP_domain(
    pssm_df: np.ndarray,  # L * 20
    seq: str,
    cter=False,  # cter=False
    k_: int = 3
):

    # Parament seq = A*n + seq, So need to get full pssm mat.
    seqlenght = len(seq)
    if cter == False:
        pssm_df = np.concatenate([
            pssm_df,
            np.zeros(
                shape=((seqlenght - pssm_df.shape[0]), 20)
            )
        ])
    else:
        # Cter == True
        pssm_df = np.concatenate([
            np.zeros(
                shape=((seqlenght - pssm_df.shape[0]), 20)
            ),
            pssm_df
        ])

    result = []
    for k in (range(k_ + 1)):  # 0 to 3
        for i in range(0, 20):
            for j in range(0, 20):
                s_ikj_m = list()
                for m in range(seqlenght - k - 1):
                    s_ikj_m.append(
                        max(
                            min(
                                pssm_df[m, i],
                                pssm_df[m + 1 + k, j],
                            ),
                            0
                        )
                    )
                result.append(sum(s_ikj_m) / (seqlenght - k - 1))
    return result
# %%


def get_HH_CKSAAP(
    pssm_dict: dict,
    seq: Seq.Seq,
    k_: int = 3
):
    result = []
    seq = str(seq)
    # 2~51 AA # len(AA) == 50 # 0 to 51 == 51
    seqlenght = len(seq)
    if seqlenght >= 51:
        result.append(get_HH_CKSAAP_domain(
            pssm_df=pssm_dict['form_1'][1:51],
            seq=seq[1:51],
            k_=k_,
        ))
    elif seqlenght < 51:
        result.append(get_HH_CKSAAP_domain(
            pssm_df=pssm_dict['form_1'][1:],
            seq=seq[1:] + 'A' * (50 - len(seq[1:])),
            k_=k_,
        ))
    else:
        pass

    # 52~121 AA # len(AA) == 70
    if seqlenght >= 121:
        result.append(get_HH_CKSAAP_domain(
            pssm_df=pssm_dict['form_1'][51:121],
            seq=seq[51:121],
            k_=k_,
        ))
    elif seqlenght >= 52 and seqlenght < 121:
        result.append(get_HH_CKSAAP_domain(
            pssm_df=pssm_dict['form_1'][51:],
            seq=seq[51:] + 'A' * (70 - len(seq[51:])),
            k_=k_,
        ))
    elif seqlenght < 52:
        result.append(get_HH_CKSAAP_domain(
            pssm_df=np.zeros(shape=(70, 20)),
            seq='A' * 70,
            k_=k_,
        ))
    else:
        pass

    # -50~ AA
    if seqlenght >= 50:
        result.append(get_HH_CKSAAP_domain(
            pssm_df=pssm_dict['form_1'][-50:],
            seq=seq[-50:],
            k_=k_,
            cter=True
        ))
    elif seqlenght < 50:
        result.append(get_HH_CKSAAP_domain(
            pssm_df=pssm_dict['form_1'],
            seq='A' * (50 - len(seq)) + seq,
            k_=k_,
            cter=True
        ))
    else:
        pass

    return list(itertools.chain(*result))

# %%


def build_HH_CKSAAP_feature(
        path_to_fasta: str,
        order_list: list,
        path_with_pattern: str,
        seq_id_list: list,
        desc: str = 'undefine',
):

    # logger.logger.info(f'{path_to_fasta} to Start.')

    pssm_file_content_list = libdataloader.pssm_tools.get_pssm_in_order(
        order_list,
        path_with_pattern
    )
    len_of_fasta = len(list(SeqIO.parse(
        path_to_fasta,
        'fasta'
    )))
    feature_json = [
        {
            'id': seq.id,
            'seq': len(seq.seq),
            'Feature': get_HH_CKSAAP(
                pssm_dict=libdataloader.pssm_tools.get_pssm_from_file(
                    pssm_content
                ),
                seq=seq.seq,
            )
        } for seq, pssm_content in tqdm.tqdm(zip(
            SeqIO.parse(
                path_to_fasta,
                'fasta'
            ),
            pssm_file_content_list
        ), total=len_of_fasta,
            desc=f'{desc}_HH_CKSAAP')
    ]
    return pd.DataFrame(
        [
            item['Feature']
            for item in feature_json
        ],
        index=[
            item['id']
            for item in feature_json
        ],
    ).loc[seq_id_list, :]
