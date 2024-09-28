import sys
sys.path.append("src")
import os
import json
import pickle

import libpybiofeature
from libmodel import t3mm

import numpy as np
import pandas as pd


def load_from_bert_vec(
    path: str,
    id_list: list,
    len_: int = 100,
):

    result = pickle.load(
        open(path, "rb"))['value'][:, :len_, :]

    return pd.DataFrame(np.reshape(result, (result.shape[0], -1,)), index=id_list)


def load_from_bert_onehot(
    path: str,
    id_list: list,
    dim: int,
    len_: int = 100,
):

    def onehot_code_from_logist(
        dim: int,
        arr: np.ndarray
    ):
        return np.eye(dim)[np.argmax(arr, axis=-1)]

    result = onehot_code_from_logist(dim, pickle.load(
        open(path, "rb"))['value'])[:, :len_, :]

    return pd.DataFrame(np.reshape(result, (result.shape[0], -1,)), index=id_list)


def load_from_bert_aac(
    path: str,
    id_list: list,
    dim: int,
    len_: int = 100,
):
    def ac_code_from_logist(
        dim: int,
        arr: np.ndarray
    ):
        return np.array([
            "A", "B", "C"
        ][:dim])[np.argmax(arr, axis=-1)]
    result = ac_code_from_logist(dim, pickle.load(
        open(path, "rb"))['value'])[:, :len_]

    result = [
        libpybiofeature.AC.AAC(
            seq_aa="".join(item),
            aaorder=["A", "B", "C"][:dim]
        )
        for item in result
    ]

    return pd.DataFrame(result, index=id_list)


def load_from_bert_dac(
    path: str,
    id_list: list,
    dim: int,
    len_: int = 100,
):

    def ac_code_from_logist(
        dim: int,
        arr: np.ndarray
    ):
        return np.array([
            "A", "B", "C"
        ][:dim])[np.argmax(arr, axis=-1)]

    result = ac_code_from_logist(dim, pickle.load(
        open(path, "rb"))['value'])[:, :len_]

    result = [
        libpybiofeature.AC.DAC(
            seq_aa="".join(item),
            dacorder=libpybiofeature.AC._get_dac_order(
                aaorder=["A", "B", "C"][:dim]
            )
        )
        for item in result
    ]

    return pd.DataFrame(result, index=id_list)


def load_from_bert_tac(
    path: str,
    id_list: list,
    dim: int,
    len_: int = 100,
):

    def ac_code_from_logist(
        dim: int,
        arr: np.ndarray
    ):
        return np.array([
            "A", "B", "C"
        ][:dim])[np.argmax(arr, axis=-1)]

    result = ac_code_from_logist(dim, pickle.load(
        open(path, "rb"))['value'])[:, :len_]

    result = [
        libpybiofeature.AC.TAC(
            seq_aa="".join(item),
            tacorder=libpybiofeature.AC._get_tac_order(
                aaorder=["A", "B", "C"][:dim]
            )
        )
        for item in result
    ]

    return pd.DataFrame(result, index=id_list)


def load_feature(TxSE_args: dict):
    feature_data_set = []

    os.makedirs(TxSE_args['model']['path_to_save_dir'], exist_ok=True)

    # Extract Feature
    seq_id_dict = None
    with open(TxSE_args['seq_id'], 'r', encoding='UTF-8') as f:
        seq_id_dict = json.load(f)

    possum_index_dict = None
    with open(TxSE_args['possum']['index'], 'r', encoding='UTF-8') as f:
        possum_index_dict = json.load(f)

    

    

    # PPT Full Long
    feature_data_set.append({
        "name": "PPT_full",
        "t_p": libpybiofeature.featurebuilder.build_PPT_feature(
            path_to_fasta=TxSE_args['fasta']['t']['p'],
            seq_id_list=seq_id_dict['t']['p'],
            desc='t_p',
            fulllong=True
        ),
        "t_n": libpybiofeature.featurebuilder.build_PPT_feature(
            path_to_fasta=TxSE_args['fasta']['t']['n'],
            seq_id_list=seq_id_dict['t']['n'],
            desc='t_n',
            fulllong=True
        ),
        "v_p": libpybiofeature.featurebuilder.build_PPT_feature(
            path_to_fasta=TxSE_args['fasta']['v']['p'],
            seq_id_list=seq_id_dict['v']['p'],
            desc='v_p',
            fulllong=True
        ),
        "v_n": libpybiofeature.featurebuilder.build_PPT_feature(
            path_to_fasta=TxSE_args['fasta']['v']['n'],
            seq_id_list=seq_id_dict['v']['n'],
            desc='v_n',
            fulllong=True
        )
    })

    # PPT N25
    feature_data_set.append({
        "name": "PPT_25",
        "t_p": libpybiofeature.featurebuilder.build_PPT_feature(
            path_to_fasta=TxSE_args['fasta']['t']['p'],
            seq_id_list=seq_id_dict['t']['p'],
            desc='t_p'
        ),
        "t_n": libpybiofeature.featurebuilder.build_PPT_feature(
            path_to_fasta=TxSE_args['fasta']['t']['n'],
            seq_id_list=seq_id_dict['t']['n'],
            desc='t_n'
        ),
        "v_p": libpybiofeature.featurebuilder.build_PPT_feature(
            path_to_fasta=TxSE_args['fasta']['v']['p'],
            seq_id_list=seq_id_dict['v']['p'],
            desc='v_p'
        ),
        "v_n": libpybiofeature.featurebuilder.build_PPT_feature(
            path_to_fasta=TxSE_args['fasta']['v']['n'],
            seq_id_list=seq_id_dict['v']['n'],
            desc='v_n'
        )
    })

    

    # Onehot
    feature_data_set.append({
        "name": "Onehot",
        "t_p": libpybiofeature.featurebuilder.build_oneHot_feature(
            path_to_fasta=TxSE_args['fasta']['t']['p'],
            seq_id_list=seq_id_dict['t']['p'],
            length=100,
            cter=TxSE_args['model']['cter']
        ),
        "t_n": libpybiofeature.featurebuilder.build_oneHot_feature(
            path_to_fasta=TxSE_args['fasta']['t']['n'],
            seq_id_list=seq_id_dict['t']['n'],
            length=100,
            cter=TxSE_args['model']['cter']
        ),
        "v_p": libpybiofeature.featurebuilder.build_oneHot_feature(
            path_to_fasta=TxSE_args['fasta']['v']['p'],
            seq_id_list=seq_id_dict['v']['p'],
            length=100,
            cter=TxSE_args['model']['cter']
        ),
        "v_n": libpybiofeature.featurebuilder.build_oneHot_feature(
            path_to_fasta=TxSE_args['fasta']['v']['n'],
            seq_id_list=seq_id_dict['v']['n'],
            length=100,
            cter=TxSE_args['model']['cter']
        )
    })

    # Top_n_gram
    topm_set = {
        "name": "Top_n_gram",
    }
    topm_set["t_p"], _, topm_set["t_n"], _ = libpybiofeature.libdataloader.bliulab.build_form_of_data_set_Topm(
        path_to_Topm=TxSE_args['bliulab']['t']['top'],
        path_to_json_seq_id=TxSE_args['seq_id'],
        looking_key='t'
    )
    topm_set["v_p"], _, topm_set["v_n"], _ = libpybiofeature.libdataloader.bliulab.build_form_of_data_set_Topm(
        path_to_Topm=TxSE_args['bliulab']['v']['top'],
        path_to_json_seq_id=TxSE_args['seq_id'],
        looking_key='v'
    )
    feature_data_set.append(topm_set)

    # Bert Embedding
    feature_data_set.append({
        "name": "BertRawEmbedding",
        "t_p": load_from_bert_vec(
            path=TxSE_args['rawuntrain'].format(db_type="t_p"),
            id_list=seq_id_dict['t']['p'],
        ),
        "t_n": load_from_bert_vec(
            path=TxSE_args['rawuntrain'].format(db_type="t_n"),
            id_list=seq_id_dict['t']['n'],
        ),
        "v_p": load_from_bert_vec(
            path=TxSE_args['rawuntrain'].format(db_type="v_p"),
            id_list=seq_id_dict['v']['p'],
        ),
        "v_n": load_from_bert_vec(
            path=TxSE_args['rawuntrain'].format(db_type="v_n"),
            id_list=seq_id_dict['v']['n'],
        )
    })

    # CTriad
    feature_data_set.append({
        "name": "CTriad",
        "t_p": libpybiofeature.featurebuilder.build_conjoint_td_feature(
            path_to_fasta=TxSE_args['fasta']['t']['p'],
            seq_id_list=seq_id_dict['t']['p'],
            desc='t_p'
        ),
        "t_n": libpybiofeature.featurebuilder.build_conjoint_td_feature(
            path_to_fasta=TxSE_args['fasta']['t']['n'],
            seq_id_list=seq_id_dict['t']['n'],
            desc='t_n'
        ),
        "v_p": libpybiofeature.featurebuilder.build_conjoint_td_feature(
            path_to_fasta=TxSE_args['fasta']['v']['p'],
            seq_id_list=seq_id_dict['v']['p'],
            desc='v_p'
        ),
        "v_n": libpybiofeature.featurebuilder.build_conjoint_td_feature(
            path_to_fasta=TxSE_args['fasta']['v']['n'],
            seq_id_list=seq_id_dict['v']['n'],
            desc='v_n'
        )
    })


    # MM_probility
    # 加载训练阳性t_p、训练阴性t_n、测试阳性v_p、测试阴性v_n
    MM_seq_data = {
        "t_p": libpybiofeature.libdataloader.fasta_seq_loader.prepare_data(
            path_to_fasta=TxSE_args['fasta']['t']['p'],
            seq_id_list=seq_id_dict['t']['p'],
        )[0].values.tolist(),
        "t_n": libpybiofeature.libdataloader.fasta_seq_loader.prepare_data(
            path_to_fasta=TxSE_args['fasta']['t']['n'],
            seq_id_list=seq_id_dict['t']['n'],
        )[0].values.tolist(),
        "v_p": libpybiofeature.libdataloader.fasta_seq_loader.prepare_data(
            path_to_fasta=TxSE_args['fasta']['v']['p'],
            seq_id_list=seq_id_dict['v']['p'],
        )[0].values.tolist(),
        "v_n": libpybiofeature.libdataloader.fasta_seq_loader.prepare_data(
            path_to_fasta=TxSE_args['fasta']['v']['n'],
            seq_id_list=seq_id_dict['v']['n'],
        )[0].values.tolist(),
    }

    # 使用训练的阴、阳集分别制作马尔可夫谱
    MM_profile = {
        'p': t3mm.get_profile(
            fasta_db=MM_seq_data['t_p'],
            cter=TxSE_args['model']['cter'],
            terlength=100,
            padding_ac='A'
        ),
        'n': t3mm.get_profile(
            fasta_db=MM_seq_data['t_n'],
            cter=TxSE_args['model']['cter'],
            terlength=100,
            padding_ac='A'
        ),
    }

    # 保存马尔可夫谱，预测序列的时候需要加载这个保存的谱进行概率计算
    with open(f"{TxSE_args['model']['path_to_save_dir']}/MM_profile.json", "w+", encoding='UTF-8') as f:
        json.dump(MM_profile, f)

    # 对MM_seq_data中的t_p、t_n、v_p、v_n中的序列分别进行计算
    for data_type in MM_seq_data.keys():
        MM_seq_data[data_type] = pd.DataFrame(
            [
                t3mm.mat_mapper(
                    seq=str(seq.seq),
                    pprofile=MM_profile['p'],
                    nprofile=MM_profile['n'],
                    cter=TxSE_args['model']['cter'],
                    terlength=100,
                    padding_ac='A'
                # 这里就是对MM_seq_data[data_type]中的每一条序列进行计算
                ) for seq in MM_seq_data[data_type]
            ],
            index=seq_id_dict[data_type[0]][data_type[2]]
        )
    MM_seq_data['name'] = "MM_Probility"
    feature_data_set.append(MM_seq_data)

    for item in feature_data_set:
        for data_type in ["t_p", "t_n", "v_p", "v_n"]:
            item[data_type].replace([np.inf, -np.inf], 0.0, inplace=True)

    return feature_data_set
