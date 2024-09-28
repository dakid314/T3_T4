'''
Author: George Zhao
Date: 2022-06-22 11:16:24
LastEditors: George Zhao
LastEditTime: 2022-06-23 21:26:25
Description: 
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''
import typing
import os
import sys
sys.path.append('..')
sys.path.append('txseml_addon/src/')
import math
import pickle
import json
import itertools
import functools
import collections

import utils
from utils import seq_length_process
from . import model_optimite
from . import common

import numpy as np
import pandas as pd
from Bio import SeqIO


def seq_reverse(seq: str):
    return ''.join(reversed(seq))


def count_to_freq(cc: collections.Counter):
    # also Use to normalize the cc
    cc = dict(cc)
    total_num = sum([cc[key] for key in cc.keys()])
    cc = {
        key: (cc[key] / total_num) for key in cc.keys()
    }
    return cc


def make_daa(seq_aa: str):
    # Checked.
    return [
        f'{aa1}{aa2}' for aa1, aa2 in zip(
            seq_aa[:len(seq_aa) - 1], seq_aa[1:]
        )
    ]


def safe_map_geter(d: dict, k: str, mode: str, k2: str = None):
    # Checked.
    if mode == 'vector':
        if k in d:
            return d[k]
        return min([d[key] for key in d]) / 2
    elif mode == 'dict_dict':
        if k in d:
            if k2 in d[k]:
                return d[k][k2]
        return min(
            itertools.chain(*[
                [
                    d[key][key2]
                    for key2 in d[key]
                ]
                for key in d
            ])
        ) / 2
    else:
        raise f"Unsupport mode: {mode}."


def get_profile(
    fasta_db: typing.Union[str, list],
    cter: bool,
    terlength: int = 100,
    padding_ac='A',
):
    # Read Seq_list
    if isinstance(fasta_db, str) == True:
        fasta_db = list(SeqIO.parse(fasta_db, "fasta"))

    # Trim Seq
    fasta_db = [
        seq_length_process.trimer(
            seq=str(seq.seq), terlength=terlength, cter=cter, padding_ac=padding_ac, remove_first=True
        )
        for seq in fasta_db
    ]
    if cter == True:
        fasta_db = [
            seq_reverse(seq=seq)
            for seq in fasta_db
        ]

    # First Position AC frequence.
    ff_freqvector = count_to_freq(
        collections.Counter([seq[0] for seq in fasta_db])
    )

    # Other Position AC Given condiction frequence.
    of_freqmat = dict(collections.Counter(itertools.chain(*[
        make_daa(seq) for seq in fasta_db
    ])))

    of_freqmat = {
        fi: count_to_freq({
            itemkey[1]: of_freqmat[itemkey] for itemkey in of_freqmat if itemkey[0] == fi
        }) for fi in set([key[0] for key in of_freqmat.keys()])
    }

    return ff_freqvector, of_freqmat


def mat_mapper(
    seq: str,
    pprofile,
    nprofile,
    cter: bool,
    terlength: int = 100,
    padding_ac='A',
):
    seq = seq_length_process.trimer(
        seq=seq, terlength=terlength, cter=cter, padding_ac=padding_ac, remove_first=True
    )

    def _factor_multipy(seq_aa: str, profile):
        factor = [
            safe_map_geter(
                d=profile[1], k=daa[0], mode='dict_dict', k2=daa[1]
            )
            for daa in make_daa(seq_aa=seq_aa)
        ]
        factor.append(safe_map_geter(
            d=profile[0], k=seq_aa[0], mode='vector'
        ))

        return sum(
            map(math.log2, factor)
        )

    return _factor_multipy(seq, pprofile) - _factor_multipy(seq, nprofile)


class T3MM_Model(common.Model_Final):
    def __init__(self, cv, desc,
                 cter: bool,
                 terlength: int = 100,
                 padding_ac='A',
                 ):
        super().__init__(cv, desc=desc)
        self.model = None
        self.cter = cter
        self.terlength = terlength
        self.padding_ac = padding_ac
        pass

    def tranmodel(self, f, l):
        super().tranmodel(f, l)
        f = f[0].values.tolist()
        self.model = {
            'p': get_profile(
                [f[i] for i in range(len(f)) if l[i] == 1],
                cter=self.cter,
                terlength=self.terlength,
                padding_ac=self.padding_ac
            ),
            'n': get_profile(
                [f[i] for i in range(len(f)) if l[i] == 0],
                cter=self.cter,
                terlength=self.terlength,
                padding_ac=self.padding_ac
            ),
        }
        return self

    def predict(self, f):
        super().predict(f)
        f = f[0].values.tolist()
        return np.array([
            mat_mapper(
                seq=str(seq.seq),
                pprofile=self.model['p'],
                nprofile=self.model['n'],
                cter=self.cter,
                terlength=self.terlength,
                padding_ac=self.padding_ac
            )
            for seq in f
        ])


def Five_Cross_Get_model(
    model_construct_funtion,
    t_5C: list,
    v_f,
    v_l,
    path_to_model_pickle: str,
    path_to_model_result: str,
    size_of_data: str,
):
    model_set = list()
    model_result_set = list()
    for i in range(len(t_5C)):
        train_fl, test_fl = t_5C[i]

        model: T3MM_Model = model_construct_funtion()
        model.tranmodel(
            train_fl[0], train_fl[1]
        )
        model_set.append(model)
        model_result_set.append({
            "training": {
                "origin": {
                    f'pred': list(model.predict(
                        train_fl[0]
                    )),
                    f'label': list(train_fl[1])},
                "evaluation": {
                }, "option": {
                }
            },
            "testing": {
                "origin": {
                    f'pred': list(model.predict(
                        test_fl[0]
                    )),
                    f'label': list(test_fl[1])},
                "evaluation": {
                }, "option": {
                }
            },
            "validated": {
                "origin": {
                    f'pred': list(model.predict(
                        v_f
                    )),
                    f'label': list(v_l)},
                "evaluation": {
                }, "option": {
                }
            },
            "detail": {
                "model": 'T3MM',
                'desc': model.desc,
                'iteration': i,
                "size_of_data": size_of_data,
            }
        })

    model_result_set = utils.ds_preprocess.Five_Cross_Evaluation(
        model_result_set,
        pro_cutoff=0,
        mode='loo' if size_of_data == 'small' else None
    )

    if os.path.exists(os.path.split(path_to_model_pickle)[0]) == False:
        os.makedirs(os.path.split(path_to_model_pickle)[0])
    if os.path.exists(os.path.split(path_to_model_result)[0]) == False:
        os.makedirs(os.path.split(path_to_model_result)[0])

    with open(path_to_model_pickle, 'bw+') as f:
        pickle.dump(model_set, f)

    with open(path_to_model_result, 'w+', encoding='UTF-8') as f:
        json.dump(model_result_set, f, cls=utils.ds_preprocess.MyEncoder)

    return model_set, model_result_set


def prepare_data(
    path_to_fasta: str,
    seq_id_list: list = None,
):
    # Loadfasta
    seq_list = list(SeqIO.parse(path_to_fasta, 'fasta'))

    if seq_id_list is None:
        return pd.DataFrame([seq_list, ])
    else:
        db = {
            seq.id: [seq, ] for seq in seq_list
        }
        return pd.DataFrame([db[seqid] for seqid in seq_id_list])


def research(path_dict: dict):
    seq_id_dict = None
    with open(path_dict['seq_id'], 'r', encoding='UTF-8') as f:
        seq_id_dict = json.load(f)

    t_p_f = prepare_data(
        path_to_fasta=path_dict['fasta']['t']['p'],
        seq_id_list=seq_id_dict['t']['p'],
    )

    t_n_f = prepare_data(
        path_to_fasta=path_dict['fasta']['t']['n'],
        seq_id_list=seq_id_dict['t']['n'],
    )

    v_p_f = prepare_data(
        path_to_fasta=path_dict['fasta']['v']['p'],
        seq_id_list=seq_id_dict['v']['p'],
    )

    v_n_f = prepare_data(
        path_to_fasta=path_dict['fasta']['v']['n'],
        seq_id_list=seq_id_dict['v']['n'],
    )

    t_p_l = utils.ds_preprocess.make_binary_label(
        size=t_p_f.shape[0], label=True)
    t_n_l = utils.ds_preprocess.make_binary_label(
        size=t_n_f.shape[0], label=False)

    v_p_l = utils.ds_preprocess.make_binary_label(
        size=v_p_f.shape[0], label=True)
    v_n_l = utils.ds_preprocess.make_binary_label(
        size=v_n_f.shape[0], label=False)

    t_5C = utils.ds_preprocess.get_5C_data(
        shape_to_chiose=t_p_f.shape[0],
        t_p_f=t_p_f,
        t_p_l=t_p_l,
        t_n_f=t_n_f,
        t_n_l=t_n_l,
        shufflesplit={
            'shufflesplit_index_file': path_dict['shufflesplit_index_file']
        } if path_dict['model']['size'] == 'small' else None
    )

    t_f, t_l = utils.ds_preprocess.make_merge(
        t_p_f=t_p_f,
        t_p_l=t_p_l,
        t_n_f=t_n_f,
        t_n_l=t_n_l,
    )

    v_f, v_l = utils.ds_preprocess.make_merge(
        t_p_f=v_p_f,
        t_p_l=v_p_l,
        t_n_f=v_n_f,
        t_n_l=v_n_l,
    )

    Five_Cross_Get_model(
        model_construct_funtion=functools.partial(
            T3MM_Model, desc=path_dict['model']['cv']['desc'], cv=None, cter=path_dict['model']['cter']),
        t_5C=t_5C,
        v_f=v_f,
        v_l=v_l,
        path_to_model_pickle=path_dict['model']['cv']['model_pickle'],
        path_to_model_result=path_dict['model']['cv']['model_result'],
        size_of_data=path_dict['model']['size']
    )
    Five_Cross_Get_model(
        model_construct_funtion=functools.partial(
            T3MM_Model, desc=path_dict['model']['tt']['desc'], cv=None, cter=path_dict['model']['cter']),
        t_5C=(((t_f, t_l),
               (v_f, v_l)),),
        v_f=v_f,
        v_l=v_l,
        path_to_model_pickle=path_dict['model']['tt']['model_pickle'],
        path_to_model_result=path_dict['model']['tt']['model_result'],
        size_of_data=None
    )
