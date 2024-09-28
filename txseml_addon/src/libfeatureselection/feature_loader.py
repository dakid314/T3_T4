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

    # AAC, DAC, TAC
    feature_data_set.append({
        "name": "AAC",
        "t_p": libpybiofeature.featurebuilder.build_acc_feature(
            path_to_fasta=TxSE_args['fasta']['t']['p'],
            seq_id_list=seq_id_dict['t']['p'],
            desc='t_p'
        ),
        "t_n": libpybiofeature.featurebuilder.build_acc_feature(
            path_to_fasta=TxSE_args['fasta']['t']['n'],
            seq_id_list=seq_id_dict['t']['n'],
            desc='t_n'
        ),
        "v_p": libpybiofeature.featurebuilder.build_acc_feature(
            path_to_fasta=TxSE_args['fasta']['v']['p'],
            seq_id_list=seq_id_dict['v']['p'],
            desc='v_p'
        ),
        "v_n": libpybiofeature.featurebuilder.build_acc_feature(
            path_to_fasta=TxSE_args['fasta']['v']['n'],
            seq_id_list=seq_id_dict['v']['n'],
            desc='v_n'
        ),
    })
    feature_data_set.append({
        "name": "DAC",
        "t_p": libpybiofeature.featurebuilder.build_dac_feature(
            path_to_fasta=TxSE_args['fasta']['t']['p'],
            seq_id_list=seq_id_dict['t']['p'],
            desc='t_p'
        ),
        "t_n": libpybiofeature.featurebuilder.build_dac_feature(
            path_to_fasta=TxSE_args['fasta']['t']['n'],
            seq_id_list=seq_id_dict['t']['n'],
            desc='t_n'
        ),
        "v_p": libpybiofeature.featurebuilder.build_dac_feature(
            path_to_fasta=TxSE_args['fasta']['v']['p'],
            seq_id_list=seq_id_dict['v']['p'],
            desc='v_p'
        ),
        "v_n": libpybiofeature.featurebuilder.build_dac_feature(
            path_to_fasta=TxSE_args['fasta']['v']['n'],
            seq_id_list=seq_id_dict['v']['n'],
            desc='v_n'
        ),
    })
    feature_data_set.append({
        "name": "TAC",
        "t_p": libpybiofeature.featurebuilder.build_tac_feature(
            path_to_fasta=TxSE_args['fasta']['t']['p'],
            seq_id_list=seq_id_dict['t']['p'],
            desc='t_p'
        ),
        "t_n": libpybiofeature.featurebuilder.build_tac_feature(
            path_to_fasta=TxSE_args['fasta']['t']['n'],
            seq_id_list=seq_id_dict['t']['n'],
            desc='t_n'
        ),
        "v_p": libpybiofeature.featurebuilder.build_tac_feature(
            path_to_fasta=TxSE_args['fasta']['v']['p'],
            seq_id_list=seq_id_dict['v']['p'],
            desc='v_p'
        ),
        "v_n": libpybiofeature.featurebuilder.build_tac_feature(
            path_to_fasta=TxSE_args['fasta']['v']['n'],
            seq_id_list=seq_id_dict['v']['n'],
            desc='v_n'
        ),
    })

    # BPBaac
    BPBaac_seq_data = {
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
    BPBaac_profile = {
        "p": libpybiofeature.BPBaac_psp.mat_constructor(
            fasta_db=BPBaac_seq_data['t_p'],
            cter=TxSE_args['model']['cter'],
            terlength=100,
            padding_ac='A'
        ),
        "n": libpybiofeature.BPBaac_psp.mat_constructor(
            fasta_db=BPBaac_seq_data['t_n'],
            cter=TxSE_args['model']['cter'],
            terlength=100,
            padding_ac='A'
        ),
    }

    with open(f"{TxSE_args['model']['path_to_save_dir']}/BPBaac_profile.json", "w+", encoding='UTF-8') as f:
        json.dump(BPBaac_profile, f)

    for data_type in BPBaac_seq_data.keys():
        BPBaac_seq_data[data_type] = pd.DataFrame(
            [
                libpybiofeature.BPBaac_psp.mat_mapper(
                    seq=str(seq.seq),
                    pmat=BPBaac_profile['p'],
                    nmat=BPBaac_profile['n'],
                    cter=TxSE_args['model']['cter'],
                    terlength=100,
                    padding_ac='A'
                ) for seq in BPBaac_seq_data[data_type]
            ],
            index=seq_id_dict[data_type[0]][data_type[2]]
        )
    BPBaac_seq_data['name'] = "BPBaac"
    feature_data_set.append(BPBaac_seq_data)

    # PSSM_mat N1000
    pssm_code_1000 = libpybiofeature.pssmcode.get_all_task_pssmcode(
        possum_index_dict=possum_index_dict,
        seq_id_dict=seq_id_dict,
        path_to_fasta_with_pattern=TxSE_args['possum']['fasta_pattern'],
        path_to_with_pattern=TxSE_args['possum']['pssm_rdb_pattern'],
        length=1000,
        cter=TxSE_args['model']['cter']
    )
    feature_data_set.append({
        "name": "PSSM-1000",
        "t_p": pssm_code_1000[0],
        "t_n": pssm_code_1000[1],
        "v_p": pssm_code_1000[2],
        "v_n": pssm_code_1000[3],
    })

    # PSSM_entropy
    feature_data_set.append({
        "name": "PSSM_Entropy",
        "t_p": libpybiofeature.PSSM_ENCROPY.build_feature_from_file(
            path_to_fasta=TxSE_args['fasta']['t']['p'],
            order_list=possum_index_dict['data']['t_p'],
            path_with_pattern=TxSE_args['possum']['pssm_rdb_pattern'],
            seq_id_list=seq_id_dict['t']['p'],
            desc='t_p',
        ),
        "t_n": libpybiofeature.PSSM_ENCROPY.build_feature_from_file(
            path_to_fasta=TxSE_args['fasta']['t']['n'],
            order_list=possum_index_dict['data']['t_n'],
            path_with_pattern=TxSE_args['possum']['pssm_rdb_pattern'],
            seq_id_list=seq_id_dict['t']['n'],
            desc='t_n',
        ),
        "v_p": libpybiofeature.PSSM_ENCROPY.build_feature_from_file(
            path_to_fasta=TxSE_args['fasta']['v']['p'],
            order_list=possum_index_dict['data']['v_p'],
            path_with_pattern=TxSE_args['possum']['pssm_rdb_pattern'],
            seq_id_list=seq_id_dict['v']['p'],
            desc='v_p',
        ),
        "v_n": libpybiofeature.PSSM_ENCROPY.build_feature_from_file(
            path_to_fasta=TxSE_args['fasta']['v']['n'],
            order_list=possum_index_dict['data']['v_n'],
            path_with_pattern=TxSE_args['possum']['pssm_rdb_pattern'],
            seq_id_list=seq_id_dict['v']['n'],
            desc='v_n',
        ),
    })

    # PSSM_composition
    pssm_composition_set = libpybiofeature.pssmcode.get_all_task_feature(
        possum_index_dict=possum_index_dict,
        path_to_json_seq_id=TxSE_args['seq_id'],
        feature_name_list=['pssm_composition', ],
        path_to_fasta_pattern=TxSE_args['possum']['fasta_pattern'],
        path_to_with_pattern=TxSE_args['possum']['pssm_db_pattern']
    )
    feature_data_set.append({
        "name": "PSSM_Composition",
        "t_p": pssm_composition_set[0],
        "t_n": pssm_composition_set[1],
        "v_p": pssm_composition_set[2],
        "v_n": pssm_composition_set[3],
    })

    # Rpm_PSSM
    rpm_pssm_set = libpybiofeature.pssmcode.get_all_task_feature(
        possum_index_dict=possum_index_dict,
        path_to_json_seq_id=TxSE_args['seq_id'],
        feature_name_list=['rpm_pssm', ],
        path_to_fasta_pattern=TxSE_args['possum']['fasta_pattern'],
        path_to_with_pattern=TxSE_args['possum']['pssm_db_pattern']
    )
    feature_data_set.append({
        "name": "Rpm_PSSM",
        "t_p": rpm_pssm_set[0],
        "t_n": rpm_pssm_set[1],
        "v_p": rpm_pssm_set[2],
        "v_n": rpm_pssm_set[3],
    })

    # d_fPSSM
    d_fpssm_set = libpybiofeature.pssmcode.get_all_task_feature(
        possum_index_dict=possum_index_dict,
        path_to_json_seq_id=TxSE_args['seq_id'],
        feature_name_list=['d_fpssm', ],
        path_to_fasta_pattern=TxSE_args['possum']['fasta_pattern'],
        path_to_with_pattern=TxSE_args['possum']['pssm_db_pattern']
    )
    feature_data_set.append({
        "name": "d_fPSSM",
        "t_p": d_fpssm_set[0],
        "t_n": d_fpssm_set[1],
        "v_p": d_fpssm_set[2],
        "v_n": d_fpssm_set[3],
    })

    # AAC_PSSM_set
    aac_pssm_set = libpybiofeature.pssmcode.get_all_task_feature(
        possum_index_dict=possum_index_dict,
        path_to_json_seq_id=TxSE_args['seq_id'],
        feature_name_list=['aac_pssm', ],
        path_to_fasta_pattern=TxSE_args['possum']['fasta_pattern'],
        path_to_with_pattern=TxSE_args['possum']['pssm_db_pattern']
    )
    feature_data_set.append({
        "name": "aac_pssm_set",
        "t_p": aac_pssm_set[0],
        "t_n": aac_pssm_set[1],
        "v_p": aac_pssm_set[2],
        "v_n": aac_pssm_set[3],
    })

    # TPC_PSSM_set
    tpc_pssm_set = libpybiofeature.pssmcode.get_all_task_feature(
        possum_index_dict=possum_index_dict,
        path_to_json_seq_id=TxSE_args['seq_id'],
        feature_name_list=['tpc', ],
        path_to_fasta_pattern=TxSE_args['possum']['fasta_pattern'],
        path_to_with_pattern=TxSE_args['possum']['pssm_db_pattern']
    )
    feature_data_set.append({
        "name": "TPC_PSSM",
        "t_p": tpc_pssm_set[0],
        "t_n": tpc_pssm_set[1],
        "v_p": tpc_pssm_set[2],
        "v_n": tpc_pssm_set[3],
    })

    # DP_PSSM_set
    dp_pssm_set = libpybiofeature.pssmcode.get_all_task_feature(
        possum_index_dict=possum_index_dict,
        path_to_json_seq_id=TxSE_args['seq_id'],
        feature_name_list=['dp_pssm', ],
        path_to_fasta_pattern=TxSE_args['possum']['fasta_pattern'],
        path_to_with_pattern=TxSE_args['possum']['pssm_db_pattern']
    )
    feature_data_set.append({
        "name": "DP_PSSM",
        "t_p": dp_pssm_set[0],
        "t_n": dp_pssm_set[1],
        "v_p": dp_pssm_set[2],
        "v_n": dp_pssm_set[3],
    })

    # PSSM_AC_set
    pssm_ac_set = libpybiofeature.pssmcode.get_all_task_feature(
        possum_index_dict=possum_index_dict,
        path_to_json_seq_id=TxSE_args['seq_id'],
        feature_name_list=['pssm_ac', ],
        path_to_fasta_pattern=TxSE_args['possum']['fasta_pattern'],
        path_to_with_pattern=TxSE_args['possum']['pssm_db_pattern']
    )
    feature_data_set.append({
        "name": "PSSM_AC",
        "t_p": pssm_ac_set[0],
        "t_n": pssm_ac_set[1],
        "v_p": pssm_ac_set[2],
        "v_n": pssm_ac_set[3],
    })

    # DPC_PSSM_set
    dpc_pssm_set = libpybiofeature.pssmcode.get_all_task_feature(
        possum_index_dict=possum_index_dict,
        path_to_json_seq_id=TxSE_args['seq_id'],
        feature_name_list=['dpc_pssm', ],
        path_to_fasta_pattern=TxSE_args['possum']['fasta_pattern'],
        path_to_with_pattern=TxSE_args['possum']['pssm_db_pattern']
    )
    feature_data_set.append({
        "name": "DPC_PSSM",
        "t_p": dpc_pssm_set[0],
        "t_n": dpc_pssm_set[1],
        "v_p": dpc_pssm_set[2],
        "v_n": dpc_pssm_set[3],
    })

    # S_FPSSM_set
    s_fpssm_set = libpybiofeature.pssmcode.get_all_task_feature(
        possum_index_dict=possum_index_dict,
        path_to_json_seq_id=TxSE_args['seq_id'],
        feature_name_list=['s_fpssm', ],
        path_to_fasta_pattern=TxSE_args['possum']['fasta_pattern'],
        path_to_with_pattern=TxSE_args['possum']['pssm_db_pattern']
    )
    feature_data_set.append({
        "name": "S_FPSSM",
        "t_p": s_fpssm_set[0],
        "t_n": s_fpssm_set[1],
        "v_p": s_fpssm_set[2],
        "v_n": s_fpssm_set[3],
    })

    # PSE_PSSM_set
    pse_pssm_set = libpybiofeature.pssmcode.get_all_task_feature(
        possum_index_dict=possum_index_dict,
        path_to_json_seq_id=TxSE_args['seq_id'],
        feature_name_list=['pse_pssm', ],
        path_to_fasta_pattern=TxSE_args['possum']['fasta_pattern'],
        path_to_with_pattern=TxSE_args['possum']['pssm_db_pattern']
    )
    feature_data_set.append({
        "name": "PSE_PSSM",
        "t_p": pse_pssm_set[0],
        "t_n": pse_pssm_set[1],
        "v_p": pse_pssm_set[2],
        "v_n": pse_pssm_set[3],
    })

    # Smoothed_PSSM_set
    smoothed_pssm_set = libpybiofeature.pssmcode.get_all_task_feature(
        possum_index_dict=possum_index_dict,
        path_to_json_seq_id=TxSE_args['seq_id'],
        feature_name_list=['smoothed_pssm', ],
        path_to_fasta_pattern=TxSE_args['possum']['fasta_pattern'],
        path_to_with_pattern=TxSE_args['possum']['pssm_db_pattern']
    )
    feature_data_set.append({
        "name": "Smoothed_PSSM",
        "t_p": smoothed_pssm_set[0],
        "t_n": smoothed_pssm_set[1],
        "v_p": smoothed_pssm_set[2],
        "v_n": smoothed_pssm_set[3],
    })

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

    # 18 PP
    feature_data_set.append({
        "name": "18-PP",
        "t_p": libpybiofeature.featurebuilder.build_etpp_feature(
            path_to_fasta=TxSE_args['fasta']['t']['p'],
            seq_id_list=seq_id_dict['t']['p'],
            desc='t_p'
        ),
        "t_n": libpybiofeature.featurebuilder.build_etpp_feature(
            path_to_fasta=TxSE_args['fasta']['t']['n'],
            seq_id_list=seq_id_dict['t']['n'],
            desc='t_n'
        ),
        "v_p": libpybiofeature.featurebuilder.build_etpp_feature(
            path_to_fasta=TxSE_args['fasta']['v']['p'],
            seq_id_list=seq_id_dict['v']['p'],
            desc='v_p'
        ),
        "v_n": libpybiofeature.featurebuilder.build_etpp_feature(
            path_to_fasta=TxSE_args['fasta']['v']['n'],
            seq_id_list=seq_id_dict['v']['n'],
            desc='v_n'
        )
    })

    # Expasy PP
    feature_data_set.append({
        "name": "Expasy_PP",
        "t_p": libpybiofeature.libdataloader.expasy.get_expasy_t3sps(
            path_to_json=TxSE_args['expasy']['t']['p'],
            seq_id_list=seq_id_dict['t']['p']
        ),
        "t_n": libpybiofeature.libdataloader.expasy.get_expasy_t3sps(
            path_to_json=TxSE_args['expasy']['t']['n'],
            seq_id_list=seq_id_dict['t']['n']
        ),
        "v_p": libpybiofeature.libdataloader.expasy.get_expasy_t3sps(
            path_to_json=TxSE_args['expasy']['v']['p'],
            seq_id_list=seq_id_dict['v']['p']
        ),
        "v_n": libpybiofeature.libdataloader.expasy.get_expasy_t3sps(
            path_to_json=TxSE_args['expasy']['v']['n'],
            seq_id_list=seq_id_dict['v']['n']
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

    # CTD include: CTDC, CTDD, CTDT
    # CTDC
    feature_data_set.append({
        "name": "CTDC",
        "t_p": libpybiofeature.featurebuilder.build_CTDC_feature(
            path_to_fasta=TxSE_args['fasta']['t']['p'],
            seq_id_list=seq_id_dict['t']['p'],
            desc='t_p'
        ),
        "t_n": libpybiofeature.featurebuilder.build_CTDC_feature(
            path_to_fasta=TxSE_args['fasta']['t']['n'],
            seq_id_list=seq_id_dict['t']['n'],
            desc='t_n'
        ),
        "v_p": libpybiofeature.featurebuilder.build_CTDC_feature(
            path_to_fasta=TxSE_args['fasta']['v']['p'],
            seq_id_list=seq_id_dict['v']['p'],
            desc='v_p'
        ),
        "v_n": libpybiofeature.featurebuilder.build_CTDC_feature(
            path_to_fasta=TxSE_args['fasta']['v']['n'],
            seq_id_list=seq_id_dict['v']['n'],
            desc='v_n'
        )
    })
    # CTDD
    feature_data_set.append({
        "name": "CTDD",
        "t_p": libpybiofeature.featurebuilder.build_CTDD_feature(
            path_to_fasta=TxSE_args['fasta']['t']['p'],
            seq_id_list=seq_id_dict['t']['p'],
            desc='t_p'
        ),
        "t_n": libpybiofeature.featurebuilder.build_CTDD_feature(
            path_to_fasta=TxSE_args['fasta']['t']['n'],
            seq_id_list=seq_id_dict['t']['n'],
            desc='t_n'
        ),
        "v_p": libpybiofeature.featurebuilder.build_CTDD_feature(
            path_to_fasta=TxSE_args['fasta']['v']['p'],
            seq_id_list=seq_id_dict['v']['p'],
            desc='v_p'
        ),
        "v_n": libpybiofeature.featurebuilder.build_CTDD_feature(
            path_to_fasta=TxSE_args['fasta']['v']['n'],
            seq_id_list=seq_id_dict['v']['n'],
            desc='v_n'
        )
    })
    # CTDT
    feature_data_set.append({
        "name": "CTDT",
        "t_p": libpybiofeature.featurebuilder.build_CTDT_feature(
            path_to_fasta=TxSE_args['fasta']['t']['p'],
            seq_id_list=seq_id_dict['t']['p'],
            desc='t_p'
        ),
        "t_n": libpybiofeature.featurebuilder.build_CTDT_feature(
            path_to_fasta=TxSE_args['fasta']['t']['n'],
            seq_id_list=seq_id_dict['t']['n'],
            desc='t_n'
        ),
        "v_p": libpybiofeature.featurebuilder.build_CTDT_feature(
            path_to_fasta=TxSE_args['fasta']['v']['p'],
            seq_id_list=seq_id_dict['v']['p'],
            desc='v_p'
        ),
        "v_n": libpybiofeature.featurebuilder.build_CTDT_feature(
            path_to_fasta=TxSE_args['fasta']['v']['n'],
            seq_id_list=seq_id_dict['v']['n'],
            desc='v_n'
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

    # PC_PseAAC
    pcpseaac_set = {
        "name": "PC_PseAAC",
    }
    pcpseaac_set["t_p"], _, pcpseaac_set["t_n"], _ = libpybiofeature.libdataloader.bliulab.build_form_of_data_set_PSE(
        path_to_PSE=TxSE_args['bliulab']['t']['PC'],
        path_to_json_seq_id=TxSE_args['seq_id'],
        looking_key='t'
    )
    pcpseaac_set["v_p"], _, pcpseaac_set["v_n"], _ = libpybiofeature.libdataloader.bliulab.build_form_of_data_set_PSE(
        path_to_PSE=TxSE_args['bliulab']['v']['PC'],
        path_to_json_seq_id=TxSE_args['seq_id'],
        looking_key='v'
    )
    feature_data_set.append(pcpseaac_set)

    # SC_PseAAC
    scpseaac_set = {
        "name": "SC_PseAAC",
    }
    scpseaac_set["t_p"], _, scpseaac_set["t_n"], _ = libpybiofeature.libdataloader.bliulab.build_form_of_data_set_SPSE(
        path_to_PSE=TxSE_args['bliulab']['t']['SC'],
        path_to_json_seq_id=TxSE_args['seq_id'],
        looking_key='t'
    )
    scpseaac_set["v_p"], _, scpseaac_set["v_n"], _ = libpybiofeature.libdataloader.bliulab.build_form_of_data_set_SPSE(
        path_to_PSE=TxSE_args['bliulab']['v']['SC'],
        path_to_json_seq_id=TxSE_args['seq_id'],
        looking_key='v'
    )
    feature_data_set.append(scpseaac_set)

    # Distance_pair
    idna_set = {
        "name": "Distance_pair",
    }
    idna_set["t_p"], _, idna_set["t_n"], _ = libpybiofeature.libdataloader.bliulab.build_form_of_data_set_iDNA(
        path_to_iDNA=TxSE_args['bliulab']['t']['idna'],
        path_to_json_seq_id=TxSE_args['seq_id'],
        looking_key='t'
    )
    idna_set["v_p"], _, idna_set["v_n"], _ = libpybiofeature.libdataloader.bliulab.build_form_of_data_set_iDNA(
        path_to_iDNA=TxSE_args['bliulab']['v']['idna'],
        path_to_json_seq_id=TxSE_args['seq_id'],
        looking_key='v'
    )
    feature_data_set.append(idna_set)

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

    # Structure in ONEHOT
    # SA
    feature_data_set.append({
        "name": "SA-100-OneHot",
        "t_p": load_from_bert_onehot(
            path=TxSE_args['sa'].format(db_type="t_p"),
            id_list=seq_id_dict['t']['p'],
            dim=2
        ),
        "t_n": load_from_bert_onehot(
            path=TxSE_args['sa'].format(db_type="t_n"),
            id_list=seq_id_dict['t']['n'],
            dim=2
        ),
        "v_p": load_from_bert_onehot(
            path=TxSE_args['sa'].format(db_type="v_p"),
            id_list=seq_id_dict['v']['p'],
            dim=2
        ),
        "v_n": load_from_bert_onehot(
            path=TxSE_args['sa'].format(db_type="v_n"),
            id_list=seq_id_dict['v']['n'],
            dim=2
        )
    })

    # SS
    feature_data_set.append({
        "name": "SS-100-OneHot",
        "t_p": load_from_bert_onehot(
            path=TxSE_args['ss'].format(db_type="t_p"),
            id_list=seq_id_dict['t']['p'],
            dim=3
        ),
        "t_n": load_from_bert_onehot(
            path=TxSE_args['ss'].format(db_type="t_n"),
            id_list=seq_id_dict['t']['n'],
            dim=3
        ),
        "v_p": load_from_bert_onehot(
            path=TxSE_args['ss'].format(db_type="v_p"),
            id_list=seq_id_dict['v']['p'],
            dim=3
        ),
        "v_n": load_from_bert_onehot(
            path=TxSE_args['ss'].format(db_type="v_n"),
            id_list=seq_id_dict['v']['n'],
            dim=3
        )
    })

    # DISO
    feature_data_set.append({
        "name": "DISO-100-OneHot",
        "t_p": load_from_bert_onehot(
            path=TxSE_args['diso'].format(db_type="t_p"),
            id_list=seq_id_dict['t']['p'],
            dim=2
        ),
        "t_n": load_from_bert_onehot(
            path=TxSE_args['diso'].format(db_type="t_n"),
            id_list=seq_id_dict['t']['n'],
            dim=2
        ),
        "v_p": load_from_bert_onehot(
            path=TxSE_args['diso'].format(db_type="v_p"),
            id_list=seq_id_dict['v']['p'],
            dim=2
        ),
        "v_n": load_from_bert_onehot(
            path=TxSE_args['diso'].format(db_type="v_n"),
            id_list=seq_id_dict['v']['n'],
            dim=2
        )
    })

    # Structure in AC
    # SA
    feature_data_set.append({
        "name": "SA-100-AC",
        "t_p": load_from_bert_aac(
            path=TxSE_args['sa'].format(db_type="t_p"),
            id_list=seq_id_dict['t']['p'],
            dim=2
        ),
        "t_n": load_from_bert_aac(
            path=TxSE_args['sa'].format(db_type="t_n"),
            id_list=seq_id_dict['t']['n'],
            dim=2
        ),
        "v_p": load_from_bert_aac(
            path=TxSE_args['sa'].format(db_type="v_p"),
            id_list=seq_id_dict['v']['p'],
            dim=2
        ),
        "v_n": load_from_bert_aac(
            path=TxSE_args['sa'].format(db_type="v_n"),
            id_list=seq_id_dict['v']['n'],
            dim=2
        )
    })

    # SS
    feature_data_set.append({
        "name": "SS-100-AC",
        "t_p": load_from_bert_aac(
            path=TxSE_args['ss'].format(db_type="t_p"),
            id_list=seq_id_dict['t']['p'],
            dim=3
        ),
        "t_n": load_from_bert_aac(
            path=TxSE_args['ss'].format(db_type="t_n"),
            id_list=seq_id_dict['t']['n'],
            dim=3
        ),
        "v_p": load_from_bert_aac(
            path=TxSE_args['ss'].format(db_type="v_p"),
            id_list=seq_id_dict['v']['p'],
            dim=3
        ),
        "v_n": load_from_bert_aac(
            path=TxSE_args['ss'].format(db_type="v_n"),
            id_list=seq_id_dict['v']['n'],
            dim=3
        )
    })

    # DISO
    feature_data_set.append({
        "name": "DISO-100-AC",
        "t_p": load_from_bert_aac(
            path=TxSE_args['diso'].format(db_type="t_p"),
            id_list=seq_id_dict['t']['p'],
            dim=2
        ),
        "t_n": load_from_bert_aac(
            path=TxSE_args['diso'].format(db_type="t_n"),
            id_list=seq_id_dict['t']['n'],
            dim=2
        ),
        "v_p": load_from_bert_aac(
            path=TxSE_args['diso'].format(db_type="v_p"),
            id_list=seq_id_dict['v']['p'],
            dim=2
        ),
        "v_n": load_from_bert_aac(
            path=TxSE_args['diso'].format(db_type="v_n"),
            id_list=seq_id_dict['v']['n'],
            dim=2
        )
    })

    # Structure in DC
    # SA
    feature_data_set.append({
        "name": "SA-100-DC",
        "t_p": load_from_bert_dac(
            path=TxSE_args['sa'].format(db_type="t_p"),
            id_list=seq_id_dict['t']['p'],
            dim=2
        ),
        "t_n": load_from_bert_dac(
            path=TxSE_args['sa'].format(db_type="t_n"),
            id_list=seq_id_dict['t']['n'],
            dim=2
        ),
        "v_p": load_from_bert_dac(
            path=TxSE_args['sa'].format(db_type="v_p"),
            id_list=seq_id_dict['v']['p'],
            dim=2
        ),
        "v_n": load_from_bert_dac(
            path=TxSE_args['sa'].format(db_type="v_n"),
            id_list=seq_id_dict['v']['n'],
            dim=2
        )
    })

    # SS
    feature_data_set.append({
        "name": "SS-100-DC",
        "t_p": load_from_bert_dac(
            path=TxSE_args['ss'].format(db_type="t_p"),
            id_list=seq_id_dict['t']['p'],
            dim=3
        ),
        "t_n": load_from_bert_dac(
            path=TxSE_args['ss'].format(db_type="t_n"),
            id_list=seq_id_dict['t']['n'],
            dim=3
        ),
        "v_p": load_from_bert_dac(
            path=TxSE_args['ss'].format(db_type="v_p"),
            id_list=seq_id_dict['v']['p'],
            dim=3
        ),
        "v_n": load_from_bert_dac(
            path=TxSE_args['ss'].format(db_type="v_n"),
            id_list=seq_id_dict['v']['n'],
            dim=3
        )
    })

    # DISO
    feature_data_set.append({
        "name": "DISO-100-DC",
        "t_p": load_from_bert_dac(
            path=TxSE_args['diso'].format(db_type="t_p"),
            id_list=seq_id_dict['t']['p'],
            dim=2
        ),
        "t_n": load_from_bert_dac(
            path=TxSE_args['diso'].format(db_type="t_n"),
            id_list=seq_id_dict['t']['n'],
            dim=2
        ),
        "v_p": load_from_bert_dac(
            path=TxSE_args['diso'].format(db_type="v_p"),
            id_list=seq_id_dict['v']['p'],
            dim=2
        ),
        "v_n": load_from_bert_dac(
            path=TxSE_args['diso'].format(db_type="v_n"),
            id_list=seq_id_dict['v']['n'],
            dim=2
        )
    })

    # Structure in TC
    # SA
    feature_data_set.append({
        "name": "SA-100-TC",
        "t_p": load_from_bert_tac(
            path=TxSE_args['sa'].format(db_type="t_p"),
            id_list=seq_id_dict['t']['p'],
            dim=2
        ),
        "t_n": load_from_bert_tac(
            path=TxSE_args['sa'].format(db_type="t_n"),
            id_list=seq_id_dict['t']['n'],
            dim=2
        ),
        "v_p": load_from_bert_tac(
            path=TxSE_args['sa'].format(db_type="v_p"),
            id_list=seq_id_dict['v']['p'],
            dim=2
        ),
        "v_n": load_from_bert_tac(
            path=TxSE_args['sa'].format(db_type="v_n"),
            id_list=seq_id_dict['v']['n'],
            dim=2
        )
    })

    # SS
    feature_data_set.append({
        "name": "SS-100-TC",
        "t_p": load_from_bert_tac(
            path=TxSE_args['ss'].format(db_type="t_p"),
            id_list=seq_id_dict['t']['p'],
            dim=3
        ),
        "t_n": load_from_bert_tac(
            path=TxSE_args['ss'].format(db_type="t_n"),
            id_list=seq_id_dict['t']['n'],
            dim=3
        ),
        "v_p": load_from_bert_tac(
            path=TxSE_args['ss'].format(db_type="v_p"),
            id_list=seq_id_dict['v']['p'],
            dim=3
        ),
        "v_n": load_from_bert_tac(
            path=TxSE_args['ss'].format(db_type="v_n"),
            id_list=seq_id_dict['v']['n'],
            dim=3
        )
    })

    # DISO
    feature_data_set.append({
        "name": "DISO-100-TC",
        "t_p": load_from_bert_tac(
            path=TxSE_args['diso'].format(db_type="t_p"),
            id_list=seq_id_dict['t']['p'],
            dim=2
        ),
        "t_n": load_from_bert_tac(
            path=TxSE_args['diso'].format(db_type="t_n"),
            id_list=seq_id_dict['t']['n'],
            dim=2
        ),
        "v_p": load_from_bert_tac(
            path=TxSE_args['diso'].format(db_type="v_p"),
            id_list=seq_id_dict['v']['p'],
            dim=2
        ),
        "v_n": load_from_bert_tac(
            path=TxSE_args['diso'].format(db_type="v_n"),
            id_list=seq_id_dict['v']['n'],
            dim=2
        )
    })

    # B62_Coding
    feature_data_set.append({
        "name": "BLOSUM62_Code",
        "t_p": libpybiofeature.featurebuilder.build_b62_feature(
            path_to_fasta=TxSE_args['fasta']['t']['p'],
            seq_id_list=seq_id_dict['t']['p'],
            desc='t_p'
        ),
        "t_n": libpybiofeature.featurebuilder.build_b62_feature(
            path_to_fasta=TxSE_args['fasta']['t']['n'],
            seq_id_list=seq_id_dict['t']['n'],
            desc='t_n'
        ),
        "v_p": libpybiofeature.featurebuilder.build_b62_feature(
            path_to_fasta=TxSE_args['fasta']['v']['p'],
            seq_id_list=seq_id_dict['v']['p'],
            desc='v_p'
        ),
        "v_n": libpybiofeature.featurebuilder.build_b62_feature(
            path_to_fasta=TxSE_args['fasta']['v']['n'],
            seq_id_list=seq_id_dict['v']['n'],
            desc='v_n'
        )
    })

    a35_set = {
        "name": "Similarity_BLOSUM35",
    }
    a40_set = {
        "name": "Similarity_BLOSUM40",
    }
    a45_set = {
        "name": "Similarity_BLOSUM45",
    }
    a35_set['t_p'], a40_set['t_p'], a45_set['t_p'] = libpybiofeature.libdataloader.bliulab.build_form_of_data_set_align(
        path_to_align_result=TxSE_args['a']['t']['p'],
        path_to_json_seq_id=TxSE_args['seq_id'],
        looking_key='t',
        looking_key2='p',
    )
    a35_set['t_n'], a40_set['t_n'], a45_set['t_n'] = libpybiofeature.libdataloader.bliulab.build_form_of_data_set_align(
        path_to_align_result=TxSE_args['a']['t']['n'],
        path_to_json_seq_id=TxSE_args['seq_id'],
        looking_key='t',
        looking_key2='n',
    )
    a35_set['v_p'], a40_set['v_p'], a45_set['v_p'] = libpybiofeature.libdataloader.bliulab.build_form_of_data_set_align(
        path_to_align_result=TxSE_args['a']['v']['p'],
        path_to_json_seq_id=TxSE_args['seq_id'],
        looking_key='v',
        looking_key2='p',
    )
    a35_set['v_n'], a40_set['v_n'], a45_set['v_n'] = libpybiofeature.libdataloader.bliulab.build_form_of_data_set_align(
        path_to_align_result=TxSE_args['a']['v']['n'],
        path_to_json_seq_id=TxSE_args['seq_id'],
        looking_key='v',
        looking_key2='n',
    )
    feature_data_set.append(a35_set)
    feature_data_set.append(a40_set)
    feature_data_set.append(a45_set)

    # HH_CKSAAP
    feature_data_set.append({
        "name": "HH_CKSAAP",
        "t_p": libpybiofeature.HH_CKSAAP.build_HH_CKSAAP_feature(
            TxSE_args['fasta']['t']['p'],
            possum_index_dict['data']['t_p'],
            TxSE_args['possum']['pssm_rdb_pattern'],
            seq_id_list=seq_id_dict['t']['p'],
            desc='t_p',
        ),
        "t_n": libpybiofeature.HH_CKSAAP.build_HH_CKSAAP_feature(
            TxSE_args['fasta']['t']['n'],
            possum_index_dict['data']['t_n'],
            TxSE_args['possum']['pssm_rdb_pattern'],
            seq_id_list=seq_id_dict['t']['n'],
            desc='t_n',
        ),
        "v_p": libpybiofeature.HH_CKSAAP.build_HH_CKSAAP_feature(
            TxSE_args['fasta']['v']['p'],
            possum_index_dict['data']['v_p'],
            TxSE_args['possum']['pssm_rdb_pattern'],
            seq_id_list=seq_id_dict['v']['p'],
            desc='v_p',
        ),
        "v_n": libpybiofeature.HH_CKSAAP.build_HH_CKSAAP_feature(
            TxSE_args['fasta']['v']['n'],
            possum_index_dict['data']['v_n'],
            TxSE_args['possum']['pssm_rdb_pattern'],
            seq_id_list=seq_id_dict['v']['n'],
            desc='v_n',
        )
    })

    # CKSAAP
    feature_data_set.append({
        "name": "CKSAAP",
        "t_p": libpybiofeature.featurebuilder.build_CKSAAP_feature(
            path_to_fasta=TxSE_args['fasta']['t']['p'],
            seq_id_list=seq_id_dict['t']['p'],
            desc='t_p'
        ),
        "t_n": libpybiofeature.featurebuilder.build_CKSAAP_feature(
            path_to_fasta=TxSE_args['fasta']['t']['n'],
            seq_id_list=seq_id_dict['t']['n'],
            desc='t_n'
        ),
        "v_p": libpybiofeature.featurebuilder.build_CKSAAP_feature(
            path_to_fasta=TxSE_args['fasta']['v']['p'],
            seq_id_list=seq_id_dict['v']['p'],
            desc='v_p'
        ),
        "v_n": libpybiofeature.featurebuilder.build_CKSAAP_feature(
            path_to_fasta=TxSE_args['fasta']['v']['n'],
            seq_id_list=seq_id_dict['v']['n'],
            desc='v_n'
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

    # QSO
    feature_data_set.append({
        "name": "QSO",
        "t_p": libpybiofeature.featurebuilder.build_qso_feature(
            path_to_fasta=TxSE_args['fasta']['t']['p'],
            seq_id_list=seq_id_dict['t']['p'],
            desc='t_p',
            cter=TxSE_args['model']['cter']
        ),
        "t_n": libpybiofeature.featurebuilder.build_qso_feature(
            path_to_fasta=TxSE_args['fasta']['t']['n'],
            seq_id_list=seq_id_dict['t']['n'],
            desc='t_n',
            cter=TxSE_args['model']['cter']
        ),
        "v_p": libpybiofeature.featurebuilder.build_qso_feature(
            path_to_fasta=TxSE_args['fasta']['v']['p'],
            seq_id_list=seq_id_dict['v']['p'],
            desc='v_p',
            cter=TxSE_args['model']['cter']
        ),
        "v_n": libpybiofeature.featurebuilder.build_qso_feature(
            path_to_fasta=TxSE_args['fasta']['v']['n'],
            seq_id_list=seq_id_dict['v']['n'],
            desc='v_n',
            cter=TxSE_args['model']['cter']
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
