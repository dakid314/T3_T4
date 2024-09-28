'''
Author: George Zhao
Date: 2022-05-20 19:48:53
LastEditors: George Zhao
LastEditTime: 2022-08-14 15:09:43
Description:
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''
# %%
import os
import sys
import itertools
import json
sys.path.append("../..")

from Bio import SeqIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
from scipy.stats import entropy
import tqdm
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import numpy as np
import utils
import umap.umap_ as umap

import libpybiofeature
from libpybiofeature.pssmcode import get_all_task_feature

from sklearn.cluster import k_means

n_jobs = 8


def ep3_map_range_Space(t_p_f, t_n_f, v_p_f, v_n_f):
    side = [
        np.min([t_p_f.min(), t_n_f.min(),
                v_p_f.min(), v_n_f.min()]),
        np.max([t_p_f.max(), t_n_f.max(),
                v_p_f.max(), v_n_f.max()]),
    ]
    def func_(df): return (df - side[0]) / (side[1] - side[0])
    return [func_(df) for df in [t_p_f, t_n_f, v_p_f, v_n_f]], side


def pd_map_range_Space(t_p_f, t_n_f, v_p_f, v_n_f):
    pd_merged = pd.concat([t_p_f, t_n_f, v_p_f, v_n_f])
    max_side = pd_merged.max()
    min_side = pd_merged.min()
    side = [
        min_side,
        max_side,
    ]

    def func_(df):
        return (df - side[0]) / (side[1] - side[0])
    return [func_(df) for df in [t_p_f, t_n_f, v_p_f, v_n_f]], side


def xPSE_map_range_Space(t_p_f, t_n_f, v_p_f, v_n_f):
    side = [
        np.min([t_p_f.min(), t_n_f.min(),
                v_p_f.min(), v_n_f.min()]),
        np.max([t_p_f.max(), t_n_f.max(),
                v_p_f.max(), v_n_f.max()]),
    ]
    def func_(df): return (df - side[0]) / (side[1] - side[0])
    return [func_(df) for df in [t_p_f, t_n_f, v_p_f, v_n_f]], side


def style_l(predl, truel):
    if truel == 1:
        if predl == 1:
            return 'TP'
        else:
            return 'FN'
    elif truel == 0:
        if predl == 1:
            return 'FP'
        else:
            return 'TN'
    else:
        print(predl)
        return "err"


def test_and_plot(data: np.ndarray, label: np.ndarray, desc: str, path_to_out_dir: str):
    # desc Need Type and Feature

    ground_true_label_list = label

    tsne = TSNE(
        n_components=2,
        verbose=0,
        n_jobs=n_jobs,
        # learning_rate='auto',
        # n_iter=10**8,
        # init='pca'
    )
    z0 = tsne.fit_transform(data)

    umaper = umap.UMAP(n_neighbors=5, n_components=2, n_epochs=10000,
                       min_dist=0.1, local_connectivity=1,
                       )
    z1 = umaper.fit_transform(data)

    df0 = pd.DataFrame()
    df0["comp-1"] = z0[:, 0]
    df0["comp-2"] = z0[:, 1]
    df0["truelabel"] = ['T' if item ==
                        1 else 'N' for item in ground_true_label_list]
    df1 = pd.DataFrame()
    df1["comp-1"] = z1[:, 0]
    df1["comp-2"] = z1[:, 1]
    df1["truelabel"] = ['T' if item ==
                        1 else 'N' for item in ground_true_label_list]
    fig, ax = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(14.4, 7.2),
    )
    sns.scatterplot(
        x="comp-1",
        y="comp-2",
        hue="truelabel",
        # style="predlable",
        hue_order=['T', 'N'],
        # style_order=['T', 'N'],
        # palette="hls",
        data=df0,
        ax=ax[0]
    ).set(title=f"{desc} T-SNE projection")
    sns.scatterplot(
        x="comp-1",
        y="comp-2",
        hue="truelabel",
        # style="predlable",
        hue_order=['T', 'N'],
        # style_order=['T', 'N'],
        # palette="hls",
        data=df1,
        ax=ax[1]
    ).set(title=f"{desc} UMAP projection")

    plt.tight_layout()
    plt.savefig(os.path.join(path_to_out_dir, f"{desc}_tsne.pdf"))
    plt.close(fig)
    return


def stack_df(df_list: list):
    return pd.concat(df_list)


def make_go(
    path_dict, make_go_desc: str, make_go_path_to_out_dir: str
):
    # Extract Feature
    seq_id_dict = None
    with open(path_dict['seq_id'], 'r', encoding='UTF-8') as f:
        seq_id_dict = json.load(f)

    possum_index_dict = None
    with open(path_dict['possum']['index'], 'r', encoding='UTF-8') as f:
        possum_index_dict = json.load(f)

    possum_pssm_ac_t_p_f, possum_pssm_ac_t_n_f, possum_pssm_ac_v_p_f, possum_pssm_ac_v_n_f = get_all_task_feature(
        possum_index_dict=possum_index_dict,
        path_to_json_seq_id=path_dict['seq_id'],
        feature_name_list=['pssm_ac', ],
        path_to_fasta_pattern=path_dict['possum']['fasta_pattern'],
        path_to_with_pattern=path_dict['possum']['pssm_fdb_pattern']
    )

    possum_pssm_ac = stack_df(
        [possum_pssm_ac_t_p_f, possum_pssm_ac_v_p_f, possum_pssm_ac_t_n_f, possum_pssm_ac_v_n_f])

    possum_smoothed_pssm_t_p_f, possum_smoothed_pssm_t_n_f, possum_smoothed_pssm_v_p_f, possum_smoothed_pssm_v_n_f = get_all_task_feature(
        possum_index_dict=possum_index_dict,
        path_to_json_seq_id=path_dict['seq_id'],
        feature_name_list=['smoothed_pssm', ],
        path_to_fasta_pattern=path_dict['possum']['fasta_pattern'],
        path_to_with_pattern=path_dict['possum']['pssm_fdb_pattern'])
    possum_smoothed_pssm = stack_df(
        [possum_smoothed_pssm_t_p_f, possum_smoothed_pssm_v_p_f, possum_smoothed_pssm_t_n_f, possum_smoothed_pssm_v_n_f])

    possum_aac_pssm_t_p_f, possum_aac_pssm_t_n_f, possum_aac_pssm_v_p_f, possum_aac_pssm_v_n_f = get_all_task_feature(
        possum_index_dict=possum_index_dict,
        path_to_json_seq_id=path_dict['seq_id'],
        feature_name_list=['aac_pssm', ],
        path_to_fasta_pattern=path_dict['possum']['fasta_pattern'],
        path_to_with_pattern=path_dict['possum']['pssm_fdb_pattern'])
    possum_aac_pssm = stack_df(
        [possum_aac_pssm_t_p_f, possum_aac_pssm_v_p_f, possum_aac_pssm_t_n_f, possum_aac_pssm_v_n_f])

    possum_rpm_pssm_t_p_f, possum_rpm_pssm_t_n_f, possum_rpm_pssm_v_p_f, possum_rpm_pssm_v_n_f = get_all_task_feature(
        possum_index_dict=possum_index_dict,
        path_to_json_seq_id=path_dict['seq_id'],
        feature_name_list=['rpm_pssm', ],
        path_to_fasta_pattern=path_dict['possum']['fasta_pattern'],
        path_to_with_pattern=path_dict['possum']['pssm_fdb_pattern'])
    possum_rpm_pssm = stack_df(
        [possum_rpm_pssm_t_p_f, possum_rpm_pssm_v_p_f, possum_rpm_pssm_t_n_f, possum_rpm_pssm_v_n_f])

    possum_pse_pssm_t_p_f, possum_pse_pssm_t_n_f, possum_pse_pssm_v_p_f, possum_pse_pssm_v_n_f = get_all_task_feature(
        possum_index_dict=possum_index_dict,
        path_to_json_seq_id=path_dict['seq_id'],
        feature_name_list=['pse_pssm', ],
        path_to_fasta_pattern=path_dict['possum']['fasta_pattern'],
        path_to_with_pattern=path_dict['possum']['pssm_fdb_pattern'])
    possum_pse_pssm = stack_df(
        [possum_pse_pssm_t_p_f, possum_pse_pssm_v_p_f, possum_pse_pssm_t_n_f, possum_pse_pssm_v_n_f])

    possum_dp_pssm_t_p_f, possum_dp_pssm_t_n_f, possum_dp_pssm_v_p_f, possum_dp_pssm_v_n_f = get_all_task_feature(
        possum_index_dict=possum_index_dict,
        path_to_json_seq_id=path_dict['seq_id'],
        feature_name_list=['dp_pssm', ],
        path_to_fasta_pattern=path_dict['possum']['fasta_pattern'],
        path_to_with_pattern=path_dict['possum']['pssm_fdb_pattern'])
    possum_dp_pssm = stack_df(
        [possum_dp_pssm_t_p_f, possum_dp_pssm_v_p_f, possum_dp_pssm_t_n_f, possum_dp_pssm_v_n_f])

    possum_dpc_pssm_t_p_f, possum_dpc_pssm_t_n_f, possum_dpc_pssm_v_p_f, possum_dpc_pssm_v_n_f = get_all_task_feature(
        possum_index_dict=possum_index_dict,
        path_to_json_seq_id=path_dict['seq_id'],
        feature_name_list=['dpc_pssm', ],
        path_to_fasta_pattern=path_dict['possum']['fasta_pattern'],
        path_to_with_pattern=path_dict['possum']['pssm_fdb_pattern'])
    possum_dpc_pssm = stack_df(
        [possum_dpc_pssm_t_p_f, possum_dpc_pssm_v_p_f, possum_dpc_pssm_t_n_f, possum_dpc_pssm_v_n_f])

    possum_s_fpssm_t_p_f, possum_s_fpssm_t_n_f, possum_s_fpssm_v_p_f, possum_s_fpssm_v_n_f = get_all_task_feature(
        possum_index_dict=possum_index_dict,
        path_to_json_seq_id=path_dict['seq_id'],
        feature_name_list=['s_fpssm', ],
        path_to_fasta_pattern=path_dict['possum']['fasta_pattern'],
        path_to_with_pattern=path_dict['possum']['pssm_fdb_pattern'])
    possum_s_fpssm = stack_df(
        [possum_s_fpssm_t_p_f, possum_s_fpssm_v_p_f, possum_s_fpssm_t_n_f, possum_s_fpssm_v_n_f])

    possum_tpc_t_p_f, possum_tpc_t_n_f, possum_tpc_v_p_f, possum_tpc_v_n_f = get_all_task_feature(
        possum_index_dict=possum_index_dict,
        path_to_json_seq_id=path_dict['seq_id'],
        feature_name_list=['tpc', ],
        path_to_fasta_pattern=path_dict['possum']['fasta_pattern'],
        path_to_with_pattern=path_dict['possum']['pssm_fdb_pattern'])
    possum_tpc = stack_df(
        [possum_tpc_t_p_f, possum_tpc_v_p_f, possum_tpc_t_n_f, possum_tpc_v_n_f])

    possum_tpc.replace([np.inf, -np.inf], 0, inplace=True)

    pssm_smth_t_p_f = libpybiofeature.PSSM_SMTH.build_PSSM_SMTH_feature(
        path_dict['fasta']['t']['p'],
        possum_index_dict['data']['t_p'],
        path_dict['possum']['pssm_rdb_pattern'],
        seq_id_list=seq_id_dict['t']['p'],
        length=50,
        cter=path_dict['model']['cter'],
        w=[i for i in range(1, 11)],
        desc='t_p',
    )
    pssm_smth_t_n_f = libpybiofeature.PSSM_SMTH.build_PSSM_SMTH_feature(
        path_dict['fasta']['t']['n'],
        possum_index_dict['data']['t_n'],
        path_dict['possum']['pssm_rdb_pattern'],
        seq_id_list=seq_id_dict['t']['n'],
        length=50,
        cter=path_dict['model']['cter'],
        w=[i for i in range(1, 11)],
        desc='t_n',
    )

    pssm_smth_v_p_f = libpybiofeature.PSSM_SMTH.build_PSSM_SMTH_feature(
        path_dict['fasta']['v']['p'],
        possum_index_dict['data']['v_p'],
        path_dict['possum']['pssm_rdb_pattern'],
        seq_id_list=seq_id_dict['v']['p'],
        length=50,
        cter=path_dict['model']['cter'],
        w=[i for i in range(1, 11)],
        desc='v_p',
    )
    pssm_smth_v_n_f = libpybiofeature.PSSM_SMTH.build_PSSM_SMTH_feature(
        path_dict['fasta']['v']['n'],
        possum_index_dict['data']['v_n'],
        path_dict['possum']['pssm_rdb_pattern'],
        seq_id_list=seq_id_dict['v']['n'],
        length=50,
        cter=path_dict['model']['cter'],
        w=[i for i in range(1, 11)],
        desc='v_n',
    )

    pssm_smth = stack_df([pssm_smth_t_p_f, pssm_smth_v_p_f,
                          pssm_smth_t_n_f, pssm_smth_v_n_f])

    aac_t_p_f = libpybiofeature.featurebuilder.build_acc_feature(
        path_to_fasta=path_dict['fasta']['t']['p'],
        seq_id_list=seq_id_dict['t']['p'],
        desc='t_p'
    )

    aac_t_n_f = libpybiofeature.featurebuilder.build_acc_feature(
        path_to_fasta=path_dict['fasta']['t']['n'],
        seq_id_list=seq_id_dict['t']['n'],
        desc='t_n'
    )

    aac_v_p_f = libpybiofeature.featurebuilder.build_acc_feature(
        path_to_fasta=path_dict['fasta']['v']['p'],
        seq_id_list=seq_id_dict['v']['p'],
        desc='v_p'
    )

    aac_v_n_f = libpybiofeature.featurebuilder.build_acc_feature(
        path_to_fasta=path_dict['fasta']['v']['n'],
        seq_id_list=seq_id_dict['v']['n'],
        desc='v_n'
    )

    aac = stack_df([aac_t_p_f, aac_v_p_f, aac_t_n_f, aac_v_n_f])

    dac_t_p_f = libpybiofeature.featurebuilder.build_dac_feature(
        path_to_fasta=path_dict['fasta']['t']['p'],
        seq_id_list=seq_id_dict['t']['p'],
        desc='t_p'
    )

    dac_t_n_f = libpybiofeature.featurebuilder.build_dac_feature(
        path_to_fasta=path_dict['fasta']['t']['n'],
        seq_id_list=seq_id_dict['t']['n'],
        desc='t_n'
    )

    dac_v_p_f = libpybiofeature.featurebuilder.build_dac_feature(
        path_to_fasta=path_dict['fasta']['v']['p'],
        seq_id_list=seq_id_dict['v']['p'],
        desc='v_p'
    )

    dac_v_n_f = libpybiofeature.featurebuilder.build_dac_feature(
        path_to_fasta=path_dict['fasta']['v']['n'],
        seq_id_list=seq_id_dict['v']['n'],
        desc='v_n'
    )

    dac = stack_df([dac_t_p_f, dac_v_p_f, dac_t_n_f, dac_v_n_f])

    tac_t_p_f = libpybiofeature.featurebuilder.build_tac_feature(
        path_to_fasta=path_dict['fasta']['t']['p'],
        seq_id_list=seq_id_dict['t']['p'],
        desc='t_p'
    )

    tac_t_n_f = libpybiofeature.featurebuilder.build_tac_feature(
        path_to_fasta=path_dict['fasta']['t']['n'],
        seq_id_list=seq_id_dict['t']['n'],
        desc='t_n'
    )

    tac_v_p_f = libpybiofeature.featurebuilder.build_tac_feature(
        path_to_fasta=path_dict['fasta']['v']['p'],
        seq_id_list=seq_id_dict['v']['p'],
        desc='v_p'
    )

    tac_v_n_f = libpybiofeature.featurebuilder.build_tac_feature(
        path_to_fasta=path_dict['fasta']['v']['n'],
        seq_id_list=seq_id_dict['v']['n'],
        desc='v_n'
    )

    tac = stack_df([tac_t_p_f, tac_v_p_f, tac_t_n_f, tac_v_n_f])

    CKSAAP_t_p_f = libpybiofeature.featurebuilder.build_CKSAAP_feature(
        path_to_fasta=path_dict['fasta']['t']['p'],
        seq_id_list=seq_id_dict['t']['p'],
        desc='t_p'
    )

    CKSAAP_t_n_f = libpybiofeature.featurebuilder.build_CKSAAP_feature(
        path_to_fasta=path_dict['fasta']['t']['n'],
        seq_id_list=seq_id_dict['t']['n'],
        desc='t_n'
    )

    CKSAAP_v_p_f = libpybiofeature.featurebuilder.build_CKSAAP_feature(
        path_to_fasta=path_dict['fasta']['v']['p'],
        seq_id_list=seq_id_dict['v']['p'],
        desc='v_p'
    )

    CKSAAP_v_n_f = libpybiofeature.featurebuilder.build_CKSAAP_feature(
        path_to_fasta=path_dict['fasta']['v']['n'],
        seq_id_list=seq_id_dict['v']['n'],
        desc='v_n'
    )

    CKSAAP = stack_df([CKSAAP_t_p_f, CKSAAP_v_p_f, CKSAAP_t_n_f, CKSAAP_v_n_f])

    PPT_t_p_f = libpybiofeature.featurebuilder.build_PPT_feature(
        path_to_fasta=path_dict['fasta']['t']['p'],
        seq_id_list=seq_id_dict['t']['p'],
        desc='t_p',
        fulllong=True
    )

    PPT_t_n_f = libpybiofeature.featurebuilder.build_PPT_feature(
        path_to_fasta=path_dict['fasta']['t']['n'],
        seq_id_list=seq_id_dict['t']['n'],
        desc='t_n',
        fulllong=True
    )

    PPT_v_p_f = libpybiofeature.featurebuilder.build_PPT_feature(
        path_to_fasta=path_dict['fasta']['v']['p'],
        seq_id_list=seq_id_dict['v']['p'],
        desc='v_p',
        fulllong=True
    )

    PPT_v_n_f = libpybiofeature.featurebuilder.build_PPT_feature(
        path_to_fasta=path_dict['fasta']['v']['n'],
        seq_id_list=seq_id_dict['v']['n'],
        desc='v_n',
        fulllong=True
    )

    PPT = stack_df([PPT_t_p_f, PPT_v_p_f, PPT_t_n_f, PPT_v_n_f])

    PPT_t_p_f = libpybiofeature.featurebuilder.build_PPT_feature(
        path_to_fasta=path_dict['fasta']['t']['p'],
        seq_id_list=seq_id_dict['t']['p'],
        desc='t_p',
        fulllong=True
    )

    PPT_t_n_f = libpybiofeature.featurebuilder.build_PPT_feature(
        path_to_fasta=path_dict['fasta']['t']['n'],
        seq_id_list=seq_id_dict['t']['n'],
        desc='t_n',
        fulllong=True
    )

    PPT_v_p_f = libpybiofeature.featurebuilder.build_PPT_feature(
        path_to_fasta=path_dict['fasta']['v']['p'],
        seq_id_list=seq_id_dict['v']['p'],
        desc='v_p',
        fulllong=True
    )

    PPT_v_n_f = libpybiofeature.featurebuilder.build_PPT_feature(
        path_to_fasta=path_dict['fasta']['v']['n'],
        seq_id_list=seq_id_dict['v']['n'],
        desc='v_n',
        fulllong=True
    )

    PPT_fullong = stack_df([PPT_t_p_f, PPT_v_p_f, PPT_t_n_f, PPT_v_n_f])

    aacpro_t_p_f = libpybiofeature.scratch_reader.accpro.get_func(
        path_to_json_db=path_dict['scratch']['db'],
        tag_name=path_dict['scratch']['acctag']['t']['p'],
        seqid_list=seq_id_dict['t']['p'],
        length=50,
        dig_code=True,
        cter=path_dict['model']['cter']
    )

    aacpro_t_n_f = libpybiofeature.scratch_reader.accpro.get_func(
        path_to_json_db=path_dict['scratch']['db'],
        tag_name=path_dict['scratch']['acctag']['t']['n'],
        seqid_list=seq_id_dict['t']['n'],
        length=50,
        dig_code=True,
        cter=path_dict['model']['cter']
    )
    aacpro_v_p_f = libpybiofeature.scratch_reader.accpro.get_func(
        path_to_json_db=path_dict['scratch']['db'],
        tag_name=path_dict['scratch']['acctag']['v']['p'],
        seqid_list=seq_id_dict['v']['p'],
        length=50,
        dig_code=True,
        cter=path_dict['model']['cter']
    )

    aacpro_v_n_f = libpybiofeature.scratch_reader.accpro.get_func(
        path_to_json_db=path_dict['scratch']['db'],
        tag_name=path_dict['scratch']['acctag']['v']['n'],
        seqid_list=seq_id_dict['v']['n'],
        length=50,
        dig_code=True,
        cter=path_dict['model']['cter']
    )

    aacpro50 = stack_df([aacpro_t_p_f, aacpro_v_p_f,
                         aacpro_t_n_f, aacpro_v_n_f])

    ssapro_t_p_f = libpybiofeature.scratch_reader.sspro.get_func(
        path_to_json_db=path_dict['scratch']['db'],
        tag_name=path_dict['scratch']['ssatag']['t']['p'],
        seqid_list=seq_id_dict['t']['p'],
        length=50,
        dig_code=True,
        cter=path_dict['model']['cter']
    )

    ssapro_t_n_f = libpybiofeature.scratch_reader.sspro.get_func(
        path_to_json_db=path_dict['scratch']['db'],
        tag_name=path_dict['scratch']['ssatag']['t']['n'],
        seqid_list=seq_id_dict['t']['n'],
        length=50,
        dig_code=True,
        cter=path_dict['model']['cter']
    )
    ssapro_v_p_f = libpybiofeature.scratch_reader.sspro.get_func(
        path_to_json_db=path_dict['scratch']['db'],
        tag_name=path_dict['scratch']['ssatag']['v']['p'],
        seqid_list=seq_id_dict['v']['p'],
        length=50,
        dig_code=True,
        cter=path_dict['model']['cter']
    )

    ssapro_v_n_f = libpybiofeature.scratch_reader.sspro.get_func(
        path_to_json_db=path_dict['scratch']['db'],
        tag_name=path_dict['scratch']['ssatag']['v']['n'],
        seqid_list=seq_id_dict['v']['n'],
        length=50,
        dig_code=True,
        cter=path_dict['model']['cter']
    )

    ssapro50 = stack_df([ssapro_t_p_f, ssapro_v_p_f,
                         ssapro_t_n_f, ssapro_v_n_f])

    disopred3_t_p_f = libpybiofeature.disopred3.get_pbdat_data(
        path_to_json=path_dict['DISOPRED3']['db']['index'],
        path_to_datadir=path_dict['DISOPRED3']['db']['file'],
        seqid_list=seq_id_dict['t']['p'],
        desc=path_dict['DISOPRED3']['desc']['t']['p'],
        length=50,
        dig_code=True,
        cter=path_dict['model']['cter']
    )

    disopred3_t_n_f = libpybiofeature.disopred3.get_pbdat_data(
        path_to_json=path_dict['DISOPRED3']['db']['index'],
        path_to_datadir=path_dict['DISOPRED3']['db']['file'],
        seqid_list=seq_id_dict['t']['n'],
        desc=path_dict['DISOPRED3']['desc']['t']['n'],
        length=50,
        dig_code=True,
        cter=path_dict['model']['cter']
    )
    disopred3_v_p_f = libpybiofeature.disopred3.get_pbdat_data(
        path_to_json=path_dict['DISOPRED3']['db']['index'],
        path_to_datadir=path_dict['DISOPRED3']['db']['file'],
        seqid_list=seq_id_dict['v']['p'],
        desc=path_dict['DISOPRED3']['desc']['v']['p'],
        length=50,
        dig_code=True,
        cter=path_dict['model']['cter']
    )

    disopred3_v_n_f = libpybiofeature.disopred3.get_pbdat_data(
        path_to_json=path_dict['DISOPRED3']['db']['index'],
        path_to_datadir=path_dict['DISOPRED3']['db']['file'],
        seqid_list=seq_id_dict['v']['n'],
        desc=path_dict['DISOPRED3']['desc']['v']['n'],
        length=50,
        dig_code=True,
        cter=path_dict['model']['cter']
    )

    disopred350 = stack_df(
        [disopred3_t_p_f, disopred3_v_p_f, disopred3_t_n_f, disopred3_v_n_f])

    disopred3_t_p_f = libpybiofeature.disopred3.get_pbdat_data(
        path_to_json=path_dict['DISOPRED3']['db']['index'],
        path_to_datadir=path_dict['DISOPRED3']['db']['file'],
        seqid_list=seq_id_dict['t']['p'],
        desc=path_dict['DISOPRED3']['desc']['t']['p'],
        length=100,
        dig_code=True,
        cter=path_dict['model']['cter']
    )

    disopred3_t_n_f = libpybiofeature.disopred3.get_pbdat_data(
        path_to_json=path_dict['DISOPRED3']['db']['index'],
        path_to_datadir=path_dict['DISOPRED3']['db']['file'],
        seqid_list=seq_id_dict['t']['n'],
        desc=path_dict['DISOPRED3']['desc']['t']['n'],
        length=100,
        dig_code=True,
        cter=path_dict['model']['cter']
    )
    disopred3_v_p_f = libpybiofeature.disopred3.get_pbdat_data(
        path_to_json=path_dict['DISOPRED3']['db']['index'],
        path_to_datadir=path_dict['DISOPRED3']['db']['file'],
        seqid_list=seq_id_dict['v']['p'],
        desc=path_dict['DISOPRED3']['desc']['v']['p'],
        length=100,
        dig_code=True,
        cter=path_dict['model']['cter']
    )

    disopred3_v_n_f = libpybiofeature.disopred3.get_pbdat_data(
        path_to_json=path_dict['DISOPRED3']['db']['index'],
        path_to_datadir=path_dict['DISOPRED3']['db']['file'],
        seqid_list=seq_id_dict['v']['n'],
        desc=path_dict['DISOPRED3']['desc']['v']['n'],
        length=100,
        dig_code=True,
        cter=path_dict['model']['cter']
    )

    disopred3100 = stack_df(
        [disopred3_t_p_f, disopred3_v_p_f, disopred3_t_n_f, disopred3_v_n_f])

    CTDC_t_p_f = libpybiofeature.featurebuilder.build_CTDC_feature(
        path_to_fasta=path_dict['fasta']['t']['p'],
        seq_id_list=seq_id_dict['t']['p'],
        desc='t_p'
    )

    CTDC_t_n_f = libpybiofeature.featurebuilder.build_CTDC_feature(
        path_to_fasta=path_dict['fasta']['t']['n'],
        seq_id_list=seq_id_dict['t']['n'],
        desc='t_n'
    )

    CTDC_v_p_f = libpybiofeature.featurebuilder.build_CTDC_feature(
        path_to_fasta=path_dict['fasta']['v']['p'],
        seq_id_list=seq_id_dict['v']['p'],
        desc='v_p'
    )

    CTDC_v_n_f = libpybiofeature.featurebuilder.build_CTDC_feature(
        path_to_fasta=path_dict['fasta']['v']['n'],
        seq_id_list=seq_id_dict['v']['n'],
        desc='v_n'
    )

    CTDC = stack_df([CTDC_t_p_f, CTDC_v_p_f, CTDC_t_n_f, CTDC_v_n_f])

    CTDT_t_p_f = libpybiofeature.featurebuilder.build_CTDT_feature(
        path_to_fasta=path_dict['fasta']['t']['p'],
        seq_id_list=seq_id_dict['t']['p'],
        desc='t_p'
    )

    CTDT_t_n_f = libpybiofeature.featurebuilder.build_CTDT_feature(
        path_to_fasta=path_dict['fasta']['t']['n'],
        seq_id_list=seq_id_dict['t']['n'],
        desc='t_n'
    )

    CTDT_v_p_f = libpybiofeature.featurebuilder.build_CTDT_feature(
        path_to_fasta=path_dict['fasta']['v']['p'],
        seq_id_list=seq_id_dict['v']['p'],
        desc='v_p'
    )

    CTDT_v_n_f = libpybiofeature.featurebuilder.build_CTDT_feature(
        path_to_fasta=path_dict['fasta']['v']['n'],
        seq_id_list=seq_id_dict['v']['n'],
        desc='v_n'
    )

    CTDT = stack_df([CTDT_t_p_f, CTDT_v_p_f, CTDT_t_n_f, CTDT_v_n_f])

    CTDD_t_p_f = libpybiofeature.featurebuilder.build_CTDD_feature(
        path_to_fasta=path_dict['fasta']['t']['p'],
        seq_id_list=seq_id_dict['t']['p'],
        desc='t_p'
    )

    CTDD_t_n_f = libpybiofeature.featurebuilder.build_CTDD_feature(
        path_to_fasta=path_dict['fasta']['t']['n'],
        seq_id_list=seq_id_dict['t']['n'],
        desc='t_n'
    )

    CTDD_v_p_f = libpybiofeature.featurebuilder.build_CTDD_feature(
        path_to_fasta=path_dict['fasta']['v']['p'],
        seq_id_list=seq_id_dict['v']['p'],
        desc='v_p'
    )

    CTDD_v_n_f = libpybiofeature.featurebuilder.build_CTDD_feature(
        path_to_fasta=path_dict['fasta']['v']['n'],
        seq_id_list=seq_id_dict['v']['n'],
        desc='v_n'
    )

    CTDD = stack_df([CTDD_t_p_f, CTDD_v_p_f, CTDD_t_n_f, CTDD_v_n_f])

    qso_t_p_f = libpybiofeature.featurebuilder.build_qso_feature(
        path_to_fasta=path_dict['fasta']['t']['p'],
        seq_id_list=seq_id_dict['t']['p'],
        desc='t_p',
        cter=path_dict['fasta']['cter']
    )

    qso_t_n_f = libpybiofeature.featurebuilder.build_qso_feature(
        path_to_fasta=path_dict['fasta']['t']['n'],
        seq_id_list=seq_id_dict['t']['n'],
        desc='t_n',
        cter=path_dict['fasta']['cter']
    )

    qso_v_p_f = libpybiofeature.featurebuilder.build_qso_feature(
        path_to_fasta=path_dict['fasta']['v']['p'],
        seq_id_list=seq_id_dict['v']['p'],
        desc='v_p',
        cter=path_dict['fasta']['cter']
    )

    qso_v_n_f = libpybiofeature.featurebuilder.build_qso_feature(
        path_to_fasta=path_dict['fasta']['v']['n'],
        seq_id_list=seq_id_dict['v']['n'],
        desc='v_n',
        cter=path_dict['fasta']['cter']
    )

    qso = stack_df([qso_t_p_f, qso_v_p_f, qso_t_n_f, qso_v_n_f])

    b62_t_p_f = libpybiofeature.featurebuilder.build_b62_feature(
        path_to_fasta=path_dict['fasta']['t']['p'],
        seq_id_list=seq_id_dict['t']['p'],
        desc='t_p',
        cter=path_dict['fasta']['cter']
    )

    b62_t_n_f = libpybiofeature.featurebuilder.build_b62_feature(
        path_to_fasta=path_dict['fasta']['t']['n'],
        seq_id_list=seq_id_dict['t']['n'],
        desc='t_n',
        cter=path_dict['fasta']['cter']
    )

    b62_v_p_f = libpybiofeature.featurebuilder.build_b62_feature(
        path_to_fasta=path_dict['fasta']['v']['p'],
        seq_id_list=seq_id_dict['v']['p'],
        desc='v_p',
        cter=path_dict['fasta']['cter']
    )

    b62_v_n_f = libpybiofeature.featurebuilder.build_b62_feature(
        path_to_fasta=path_dict['fasta']['v']['n'],
        seq_id_list=seq_id_dict['v']['n'],
        desc='v_n',
        cter=path_dict['fasta']['cter']
    )

    b62 = stack_df([b62_t_p_f, b62_v_p_f, b62_t_n_f, b62_v_n_f])

    id_dict = None
    with open(path_dict['possum']['index'], 'r', encoding='UTF-8') as f:
        id_dict = json.load(f)

    vector_t_p_f = libpybiofeature.HH_CKSAAP.build_HH_CKSAAP_feature(
        path_dict['fasta']['t']['p'],
        id_dict['data']['t_p'],
        path_dict['possum']['pssm_rdb_pattern'],
        seq_id_list=seq_id_dict['t']['p'],
        desc='t_p',
    )
    vector_t_n_f = libpybiofeature.HH_CKSAAP.build_HH_CKSAAP_feature(
        path_dict['fasta']['t']['n'],
        id_dict['data']['t_n'],
        path_dict['possum']['pssm_rdb_pattern'],
        seq_id_list=seq_id_dict['t']['n'],
        desc='t_n',
    )

    vector_v_p_f = libpybiofeature.HH_CKSAAP.build_HH_CKSAAP_feature(
        path_dict['fasta']['v']['p'],
        id_dict['data']['v_p'],
        path_dict['possum']['pssm_rdb_pattern'],
        seq_id_list=seq_id_dict['v']['p'],
        desc='v_p',
    )
    vector_v_n_f = libpybiofeature.HH_CKSAAP.build_HH_CKSAAP_feature(
        path_dict['fasta']['v']['n'],
        id_dict['data']['v_n'],
        path_dict['possum']['pssm_rdb_pattern'],
        seq_id_list=seq_id_dict['v']['n'],
        desc='v_n',
    )

    vector = stack_df([vector_t_p_f, vector_v_p_f, vector_t_n_f, vector_v_n_f])

    pssm_t_p_f, pssm_t_n_f, pssm_v_p_f, pssm_v_n_f = libpybiofeature.pssmcode.get_all_task_pssmcode(
        possum_index_dict=possum_index_dict,
        seq_id_dict=seq_id_dict,
        path_to_fasta_with_pattern=path_dict['possum']['fasta_pattern'],
        path_to_with_pattern=path_dict['possum']['pssm_rdb_pattern'],
        length=1000,
        cter=path_dict['model']['cter']
    )

    pssmcode = stack_df([pssm_t_p_f, pssm_v_p_f, pssm_t_n_f, pssm_v_n_f])

    onehot_t_p_f = libpybiofeature.featurebuilder.build_oneHot_feature(
        path_to_fasta=path_dict['fasta']['t']['p'],
        seq_id_list=seq_id_dict['t']['p'],
        desc='t_p',
        length=1000,
        cter=path_dict['model']['cter']
    )

    onehot_t_n_f = libpybiofeature.featurebuilder.build_oneHot_feature(
        path_to_fasta=path_dict['fasta']['t']['n'],
        seq_id_list=seq_id_dict['t']['n'],
        desc='t_n',
        length=1000,
        cter=path_dict['model']['cter']
    )

    onehot_v_p_f = libpybiofeature.featurebuilder.build_oneHot_feature(
        path_to_fasta=path_dict['fasta']['v']['p'],
        seq_id_list=seq_id_dict['v']['p'],
        desc='v_p',
        length=1000,
        cter=path_dict['model']['cter']
    )

    onehot_v_n_f = libpybiofeature.featurebuilder.build_oneHot_feature(
        path_to_fasta=path_dict['fasta']['v']['n'],
        seq_id_list=seq_id_dict['v']['n'],
        desc='v_n',
        length=1000,
        cter=path_dict['model']['cter']
    )

    onehot1000 = stack_df(
        [onehot_t_p_f, onehot_v_p_f, onehot_t_n_f, onehot_v_n_f])

    aacpro_t_p_f = libpybiofeature.scratch_reader.accpro.get_func(
        path_to_json_db=path_dict['scratch']['db'],
        tag_name=path_dict['scratch']['acctag']['t']['p'],
        seqid_list=seq_id_dict['t']['p'],
        length=1000,
        dig_code=True,
        cter=path_dict['model']['cter']
    )

    aacpro_t_n_f = libpybiofeature.scratch_reader.accpro.get_func(
        path_to_json_db=path_dict['scratch']['db'],
        tag_name=path_dict['scratch']['acctag']['t']['n'],
        seqid_list=seq_id_dict['t']['n'],
        length=1000,
        dig_code=True,
        cter=path_dict['model']['cter']
    )
    aacpro_v_p_f = libpybiofeature.scratch_reader.accpro.get_func(
        path_to_json_db=path_dict['scratch']['db'],
        tag_name=path_dict['scratch']['acctag']['v']['p'],
        seqid_list=seq_id_dict['v']['p'],
        length=1000,
        dig_code=True,
        cter=path_dict['model']['cter']
    )

    aacpro_v_n_f = libpybiofeature.scratch_reader.accpro.get_func(
        path_to_json_db=path_dict['scratch']['db'],
        tag_name=path_dict['scratch']['acctag']['v']['n'],
        seqid_list=seq_id_dict['v']['n'],
        length=1000,
        dig_code=True,
        cter=path_dict['model']['cter']
    )
    aacpro_t_p_f.columns = list(range(len(aacpro_t_p_f.columns)))
    aacpro_t_n_f.columns = list(range(len(aacpro_t_n_f.columns)))
    aacpro_v_p_f.columns = list(range(len(aacpro_v_p_f.columns)))
    aacpro_v_n_f.columns = list(range(len(aacpro_v_n_f.columns)))
    aacpro1000 = stack_df(
        [aacpro_t_p_f, aacpro_v_p_f, aacpro_t_n_f, aacpro_v_n_f])

    ssapro_t_p_f = libpybiofeature.scratch_reader.sspro.get_func(
        path_to_json_db=path_dict['scratch']['db'],
        tag_name=path_dict['scratch']['ssatag']['t']['p'],
        seqid_list=seq_id_dict['t']['p'],
        length=1000,
        dig_code=True,
        cter=path_dict['model']['cter']
    )

    ssapro_t_n_f = libpybiofeature.scratch_reader.sspro.get_func(
        path_to_json_db=path_dict['scratch']['db'],
        tag_name=path_dict['scratch']['ssatag']['t']['n'],
        seqid_list=seq_id_dict['t']['n'],
        length=1000,
        dig_code=True,
        cter=path_dict['model']['cter']
    )
    ssapro_v_p_f = libpybiofeature.scratch_reader.sspro.get_func(
        path_to_json_db=path_dict['scratch']['db'],
        tag_name=path_dict['scratch']['ssatag']['v']['p'],
        seqid_list=seq_id_dict['v']['p'],
        length=1000,
        dig_code=True,
        cter=path_dict['model']['cter']
    )

    ssapro_v_n_f = libpybiofeature.scratch_reader.sspro.get_func(
        path_to_json_db=path_dict['scratch']['db'],
        tag_name=path_dict['scratch']['ssatag']['v']['n'],
        seqid_list=seq_id_dict['v']['n'],
        length=1000,
        dig_code=True,
        cter=path_dict['model']['cter']
    )
    ssapro_t_p_f.columns = list(range(len(ssapro_t_p_f.columns)))
    ssapro_t_n_f.columns = list(range(len(ssapro_t_n_f.columns)))
    ssapro_v_p_f.columns = list(range(len(ssapro_v_p_f.columns)))
    ssapro_v_n_f.columns = list(range(len(ssapro_v_n_f.columns)))

    ssapro1000 = stack_df(
        [ssapro_t_p_f, ssapro_v_p_f, ssapro_t_n_f, ssapro_v_n_f])

    disopred3_t_p_f = libpybiofeature.disopred3.get_data(
        path_to_json=path_dict['DISOPRED3']['db']['index'],
        path_to_datadir=path_dict['DISOPRED3']['db']['file'],
        seqid_list=seq_id_dict['t']['p'],
        desc=path_dict['DISOPRED3']['desc']['t']['p'],
        length=1000,
        dig_code=True,
        cter=path_dict['model']['cter']
    )

    disopred3_t_n_f = libpybiofeature.disopred3.get_data(
        path_to_json=path_dict['DISOPRED3']['db']['index'],
        path_to_datadir=path_dict['DISOPRED3']['db']['file'],
        seqid_list=seq_id_dict['t']['n'],
        desc=path_dict['DISOPRED3']['desc']['t']['n'],
        length=1000,
        dig_code=True,
        cter=path_dict['model']['cter']
    )
    disopred3_v_p_f = libpybiofeature.disopred3.get_data(
        path_to_json=path_dict['DISOPRED3']['db']['index'],
        path_to_datadir=path_dict['DISOPRED3']['db']['file'],
        seqid_list=seq_id_dict['v']['p'],
        desc=path_dict['DISOPRED3']['desc']['v']['p'],
        length=1000,
        dig_code=True,
        cter=path_dict['model']['cter']
    )

    disopred3_v_n_f = libpybiofeature.disopred3.get_data(
        path_to_json=path_dict['DISOPRED3']['db']['index'],
        path_to_datadir=path_dict['DISOPRED3']['db']['file'],
        seqid_list=seq_id_dict['v']['n'],
        desc=path_dict['DISOPRED3']['desc']['v']['n'],
        length=1000,
        dig_code=True,
        cter=path_dict['model']['cter']
    )
    disopred3_t_p_f.columns = list(range(len(disopred3_t_p_f.columns)))
    disopred3_t_n_f.columns = list(range(len(disopred3_t_n_f.columns)))
    disopred3_v_p_f.columns = list(range(len(disopred3_v_p_f.columns)))
    disopred3_v_n_f.columns = list(range(len(disopred3_v_n_f.columns)))
    disopred31000 = stack_df(
        [disopred3_t_p_f, disopred3_v_p_f, disopred3_t_n_f, disopred3_v_n_f])

    psssa_t_p_f = utils.ds_preprocess.merge_pd_list([
        aacpro_t_p_f, ssapro_t_p_f
    ])
    psssa_t_n_f = utils.ds_preprocess.merge_pd_list([
        aacpro_t_n_f, ssapro_t_n_f
    ])
    psssa_v_p_f = utils.ds_preprocess.merge_pd_list([
        aacpro_v_p_f, ssapro_v_p_f
    ])
    psssa_v_n_f = utils.ds_preprocess.merge_pd_list([
        aacpro_v_n_f, ssapro_v_n_f
    ])

    psssa = stack_df([psssa_t_p_f, psssa_v_p_f, psssa_t_n_f, psssa_v_n_f])

    CTD_t_p_f = utils.ds_preprocess.merge_pd_list([
        CTDC_t_p_f, CTDT_t_p_f, CTDD_t_p_f
    ])
    CTD_t_n_f = utils.ds_preprocess.merge_pd_list([
        CTDC_t_n_f, CTDT_t_n_f, CTDD_t_n_f
    ])
    CTD_v_p_f = utils.ds_preprocess.merge_pd_list([
        CTDC_v_p_f, CTDT_v_p_f, CTDD_v_p_f
    ])
    CTD_v_n_f = utils.ds_preprocess.merge_pd_list([
        CTDC_v_n_f, CTDT_v_n_f, CTDD_v_n_f
    ])

    CTD = stack_df([CTD_t_p_f, CTD_v_p_f, CTD_t_n_f, CTD_v_n_f])

    onehot_100_t_p_f = libpybiofeature.featurebuilder.build_oneHot_feature(
        path_to_fasta=path_dict['onehot']['t']['p'],
        seq_id_list=seq_id_dict['t']['p'],
        desc='t_p',
        length=100,
        cter=path_dict['onehot']['cter']
    )

    onehot_100_t_n_f = libpybiofeature.featurebuilder.build_oneHot_feature(
        path_to_fasta=path_dict['onehot']['t']['n'],
        seq_id_list=seq_id_dict['t']['n'],
        desc='t_n',
        length=100,
        cter=path_dict['onehot']['cter']
    )

    onehot_100_v_p_f = libpybiofeature.featurebuilder.build_oneHot_feature(
        path_to_fasta=path_dict['onehot']['v']['p'],
        seq_id_list=seq_id_dict['v']['p'],
        desc='v_p',
        length=100,
        cter=path_dict['onehot']['cter']
    )

    onehot_100_v_n_f = libpybiofeature.featurebuilder.build_oneHot_feature(
        path_to_fasta=path_dict['onehot']['v']['n'],
        seq_id_list=seq_id_dict['v']['n'],
        desc='v_n',
        length=100,
        cter=path_dict['onehot']['cter']
    )

    onehot_100 = stack_df(
        [onehot_100_t_p_f, onehot_100_v_p_f, onehot_100_t_n_f, onehot_100_v_n_f])

    # Traning
    iDNA_t_p_f, iDNA_t_p_l, iDNA_t_n_f, iDNA_t_n_l = libpybiofeature.libdataloader.bliulab.build_form_of_data_set_iDNA(
        path_to_iDNA=path_dict['bliulab']['t']['idna'],
        path_to_json_seq_id=path_dict['seq_id'],
        looking_key='t'
    )
    Topm_t_p_f, Topm_t_p_l, Topm_t_n_f, Topm_t_n_l = libpybiofeature.libdataloader.bliulab.build_form_of_data_set_Topm(
        path_to_Topm=path_dict['bliulab']['t']['top'],
        path_to_json_seq_id=path_dict['seq_id'],
        looking_key='t'
    )
    PSE_t_p_f, PSE_t_p_l, PSE_t_n_f, PSE_t_n_l = libpybiofeature.libdataloader.bliulab.build_form_of_data_set_PSE(
        path_to_PSE=path_dict['bliulab']['t']['pse'],
        path_to_json_seq_id=path_dict['seq_id'],
        looking_key='t'
    )

    # Validate
    iDNA_v_p_f, iDNA_v_p_l, iDNA_v_n_f, iDNA_v_n_l = libpybiofeature.libdataloader.bliulab.build_form_of_data_set_iDNA(
        path_to_iDNA=path_dict['bliulab']['v']['idna'],
        path_to_json_seq_id=path_dict['seq_id'],
        looking_key='v'
    )
    Topm_v_p_f, Topm_v_p_l, Topm_v_n_f, Topm_v_n_l = libpybiofeature.libdataloader.bliulab.build_form_of_data_set_Topm(
        path_to_Topm=path_dict['bliulab']['v']['top'],
        path_to_json_seq_id=path_dict['seq_id'],
        looking_key='v'
    )
    PSE_v_p_f, PSE_v_p_l, PSE_v_n_f, PSE_v_n_l = libpybiofeature.libdataloader.bliulab.build_form_of_data_set_PSE(
        path_to_PSE=path_dict['bliulab']['v']['pse'],
        path_to_json_seq_id=path_dict['seq_id'],
        looking_key='v'
    )

    a35_t_p_f, a40_t_p_f, a45_t_p_f = libpybiofeature.libdataloader.bliulab.build_form_of_data_set_align(
        path_to_align_result=path_dict['a']['t']['p'],
        path_to_json_seq_id=path_dict['seq_id'],
        looking_key='t',
        looking_key2='p',
    )
    a35_t_n_f, a40_t_n_f, a45_t_n_f = libpybiofeature.libdataloader.bliulab.build_form_of_data_set_align(
        path_to_align_result=path_dict['a']['t']['n'],
        path_to_json_seq_id=path_dict['seq_id'],
        looking_key='t',
        looking_key2='n',
    )
    a35_v_p_f, a40_v_p_f, a45_v_p_f = libpybiofeature.libdataloader.bliulab.build_form_of_data_set_align(
        path_to_align_result=path_dict['a']['v']['p'],
        path_to_json_seq_id=path_dict['seq_id'],
        looking_key='v',
        looking_key2='p',
    )
    a35_v_n_f, a40_v_n_f, a45_v_n_f = libpybiofeature.libdataloader.bliulab.build_form_of_data_set_align(
        path_to_align_result=path_dict['a']['v']['n'],
        path_to_json_seq_id=path_dict['seq_id'],
        looking_key='v',
        looking_key2='n',
    )

    ssapro_m_t_p_f = libpybiofeature.scratch_reader.sspro.get_muti_func(
        path_to_json_db=path_dict['scratch']['db'],
        tag_name=path_dict['scratch']['ssatag']['t']['p'],
        seqid_list=seq_id_dict['t']['p'],
        length=100,
        cter=path_dict['model']['cter']
    )

    ssapro_m_t_n_f = libpybiofeature.scratch_reader.sspro.get_muti_func(
        path_to_json_db=path_dict['scratch']['db'],
        tag_name=path_dict['scratch']['ssatag']['t']['n'],
        seqid_list=seq_id_dict['t']['n'],
        length=100,
        cter=path_dict['model']['cter']
    )
    ssapro_m_v_p_f = libpybiofeature.scratch_reader.sspro.get_muti_func(
        path_to_json_db=path_dict['scratch']['db'],
        tag_name=path_dict['scratch']['ssatag']['v']['p'],
        seqid_list=seq_id_dict['v']['p'],
        length=100,
        cter=path_dict['model']['cter']
    )

    ssapro_m_v_n_f = libpybiofeature.scratch_reader.sspro.get_muti_func(
        path_to_json_db=path_dict['scratch']['db'],
        tag_name=path_dict['scratch']['ssatag']['v']['n'],
        seqid_list=seq_id_dict['v']['n'],
        length=100,
        cter=path_dict['model']['cter']
    )

    ssaprom100 = stack_df(
        [ssapro_m_t_p_f, ssapro_m_v_p_f, ssapro_m_t_n_f, ssapro_m_v_n_f])

    # Normalize
    iDNA_ds_4, iDNA_side = ep3_map_range_Space(
        iDNA_t_p_f, iDNA_t_n_f, iDNA_v_p_f, iDNA_v_n_f)
    iDNA_t_p_f, iDNA_t_n_f, iDNA_v_p_f, iDNA_v_n_f = iDNA_ds_4

    Topm_ds_4, Topm_side = ep3_map_range_Space(
        Topm_t_p_f, Topm_t_n_f, Topm_v_p_f, Topm_v_n_f)
    Topm_t_p_f, Topm_t_n_f, Topm_v_p_f, Topm_v_n_f = Topm_ds_4

    PSE_ds_4, PSE_side = ep3_map_range_Space(
        PSE_t_p_f, PSE_t_n_f, PSE_v_p_f, PSE_v_n_f)
    PSE_t_p_f, PSE_t_n_f, PSE_v_p_f, PSE_v_n_f = PSE_ds_4

    a35_ds_4, a35_side = ep3_map_range_Space(
        a35_t_p_f, a35_t_n_f, a35_v_p_f, a35_v_n_f)
    a35_t_p_f, a35_t_n_f, a35_v_p_f, a35_v_n_f = a35_ds_4

    a40_ds_4, a40_side = ep3_map_range_Space(
        a40_t_p_f, a40_t_n_f, a40_v_p_f, a40_v_n_f)
    a40_t_p_f, a40_t_n_f, a40_v_p_f, a40_v_n_f = a40_ds_4

    a45_ds_4, a45_side = ep3_map_range_Space(
        a45_t_p_f, a45_t_n_f, a45_v_p_f, a45_v_n_f)
    a45_t_p_f, a45_t_n_f, a45_v_p_f, a45_v_n_f = a45_ds_4

    iDNA = stack_df([iDNA_t_p_f, iDNA_v_p_f, iDNA_t_n_f, iDNA_v_n_f])
    Topm = stack_df([Topm_t_p_f, Topm_v_p_f, Topm_t_n_f, Topm_v_n_f])
    PSE = stack_df([PSE_t_p_f, PSE_v_p_f, PSE_t_n_f, PSE_v_n_f])
    a35 = stack_df([a35_t_p_f, a35_v_p_f, a35_t_n_f, a35_v_n_f])
    a40 = stack_df([a40_t_p_f, a40_v_p_f, a40_t_n_f, a40_v_n_f])
    a45 = stack_df([a45_t_p_f, a45_v_p_f, a45_t_n_f, a45_v_n_f])

    possum_t_p_f, possum_t_n_f, possum_v_p_f, possum_v_n_f = get_all_task_feature(
        possum_index_dict=possum_index_dict,
        path_to_json_seq_id=path_dict['seq_id'],
        feature_name_list=['pssm_composition', ],
        path_to_fasta_pattern=path_dict['possum']['fasta_pattern'],
        path_to_with_pattern=path_dict['possum']['pssm_db_pattern']
    )
    possum_composition = stack_df(
        [possum_t_p_f, possum_v_p_f, possum_t_n_f, possum_v_n_f])

    possum_t_p_f, possum_t_n_f, possum_v_p_f, possum_v_n_f = get_all_task_feature(
        possum_index_dict=possum_index_dict,
        path_to_json_seq_id=path_dict['seq_id'],
        feature_name_list=['d_fpssm', ],
        path_to_fasta_pattern=path_dict['possum']['fasta_pattern'],
        path_to_with_pattern=path_dict['possum']['pssm_db_pattern']
    )

    possum_d_fpssm = stack_df(
        [possum_t_p_f, possum_v_p_f, possum_t_n_f, possum_v_n_f])

    etpp_t_p_f = libpybiofeature.featurebuilder.build_etpp_feature(
        path_to_fasta=path_dict['fasta']['t']['p'],
        seq_id_list=seq_id_dict['t']['p'],
        desc='t_p'
    )

    etpp_t_n_f = libpybiofeature.featurebuilder.build_etpp_feature(
        path_to_fasta=path_dict['fasta']['t']['n'],
        seq_id_list=seq_id_dict['t']['n'],
        desc='t_n'
    )

    etpp_v_p_f = libpybiofeature.featurebuilder.build_etpp_feature(
        path_to_fasta=path_dict['fasta']['v']['p'],
        seq_id_list=seq_id_dict['v']['p'],
        desc='v_p'
    )

    etpp_v_n_f = libpybiofeature.featurebuilder.build_etpp_feature(
        path_to_fasta=path_dict['fasta']['v']['n'],
        seq_id_list=seq_id_dict['v']['n'],
        desc='v_n'
    )

    etpp = stack_df([etpp_t_p_f, etpp_v_p_f, etpp_t_n_f, etpp_v_n_f])

    cj_t_p_f = libpybiofeature.featurebuilder.build_conjoint_td_feature(
        path_to_fasta=path_dict['fasta']['t']['p'],
        seq_id_list=seq_id_dict['t']['p'],
        desc='t_p'
    )

    cj_t_n_f = libpybiofeature.featurebuilder.build_conjoint_td_feature(
        path_to_fasta=path_dict['fasta']['t']['n'],
        seq_id_list=seq_id_dict['t']['n'],
        desc='t_n'
    )

    cj_v_p_f = libpybiofeature.featurebuilder.build_conjoint_td_feature(
        path_to_fasta=path_dict['fasta']['v']['p'],
        seq_id_list=seq_id_dict['v']['p'],
        desc='v_p'
    )

    cj_v_n_f = libpybiofeature.featurebuilder.build_conjoint_td_feature(
        path_to_fasta=path_dict['fasta']['v']['n'],
        seq_id_list=seq_id_dict['v']['n'],
        desc='v_n'
    )

    cj = stack_df([cj_t_p_f, cj_v_p_f, cj_t_n_f, cj_v_n_f])

    aacpro_t_p_f = libpybiofeature.scratch_reader.accpro.get_func(
        path_to_json_db=path_dict['scratch']['db'],
        tag_name=path_dict['scratch']['acctag']['t']['p'],
        seqid_list=seq_id_dict['t']['p'],
        length=100,
        dig_code=True,
        cter=path_dict['model']['cter']
    )

    aacpro_t_n_f = libpybiofeature.scratch_reader.accpro.get_func(
        path_to_json_db=path_dict['scratch']['db'],
        tag_name=path_dict['scratch']['acctag']['t']['n'],
        seqid_list=seq_id_dict['t']['n'],
        length=100,
        dig_code=True,
        cter=path_dict['model']['cter']
    )
    aacpro_v_p_f = libpybiofeature.scratch_reader.accpro.get_func(
        path_to_json_db=path_dict['scratch']['db'],
        tag_name=path_dict['scratch']['acctag']['v']['p'],
        seqid_list=seq_id_dict['v']['p'],
        length=100,
        dig_code=True,
        cter=path_dict['model']['cter']
    )

    aacpro_v_n_f = libpybiofeature.scratch_reader.accpro.get_func(
        path_to_json_db=path_dict['scratch']['db'],
        tag_name=path_dict['scratch']['acctag']['v']['n'],
        seqid_list=seq_id_dict['v']['n'],
        length=100,
        dig_code=True,
        cter=path_dict['model']['cter']
    )

    aacpro100 = stack_df(
        [aacpro_t_p_f, aacpro_v_p_f, aacpro_t_n_f, aacpro_v_n_f])

    ssapro_t_p_f = libpybiofeature.scratch_reader.sspro.get_func(
        path_to_json_db=path_dict['scratch']['db'],
        tag_name=path_dict['scratch']['ssatag']['t']['p'],
        seqid_list=seq_id_dict['t']['p'],
        length=100,
        dig_code=True,
        cter=path_dict['model']['cter']
    )

    ssapro_t_n_f = libpybiofeature.scratch_reader.sspro.get_func(
        path_to_json_db=path_dict['scratch']['db'],
        tag_name=path_dict['scratch']['ssatag']['t']['n'],
        seqid_list=seq_id_dict['t']['n'],
        length=100,
        dig_code=True,
        cter=path_dict['model']['cter']
    )
    ssapro_v_p_f = libpybiofeature.scratch_reader.sspro.get_func(
        path_to_json_db=path_dict['scratch']['db'],
        tag_name=path_dict['scratch']['ssatag']['v']['p'],
        seqid_list=seq_id_dict['v']['p'],
        length=100,
        dig_code=True,
        cter=path_dict['model']['cter']
    )

    ssapro_v_n_f = libpybiofeature.scratch_reader.sspro.get_func(
        path_to_json_db=path_dict['scratch']['db'],
        tag_name=path_dict['scratch']['ssatag']['v']['n'],
        seqid_list=seq_id_dict['v']['n'],
        length=100,
        dig_code=True,
        cter=path_dict['model']['cter']
    )

    ssapro100 = stack_df(
        [ssapro_t_p_f, ssapro_v_p_f, ssapro_t_n_f, ssapro_v_n_f])

    onehot_t_p_f = libpybiofeature.featurebuilder.build_oneHot_feature(
        path_to_fasta=path_dict['fasta']['t']['p'],
        seq_id_list=seq_id_dict['t']['p'],
        desc='t_p',
        length=30,
        cter=path_dict['model']['cter']
    )

    onehot_t_n_f = libpybiofeature.featurebuilder.build_oneHot_feature(
        path_to_fasta=path_dict['fasta']['t']['n'],
        seq_id_list=seq_id_dict['t']['n'],
        desc='t_n',
        length=30,
        cter=path_dict['model']['cter']
    )

    onehot_v_p_f = libpybiofeature.featurebuilder.build_oneHot_feature(
        path_to_fasta=path_dict['fasta']['v']['p'],
        seq_id_list=seq_id_dict['v']['p'],
        desc='v_p',
        length=30,
        cter=path_dict['model']['cter']
    )

    onehot_v_n_f = libpybiofeature.featurebuilder.build_oneHot_feature(
        path_to_fasta=path_dict['fasta']['v']['n'],
        seq_id_list=seq_id_dict['v']['n'],
        desc='v_n',
        length=30,
        cter=path_dict['model']['cter']
    )

    onehot30 = stack_df([onehot_t_p_f, onehot_v_p_f,
                         onehot_t_n_f, onehot_v_n_f])

    pssm_ep_bit_t_p_f = libpybiofeature.PSSM_ENCROPY.build_feature_from_file(
        path_to_fasta=path_dict['fasta']['t']['p'],
        order_list=possum_index_dict['data']['t_p'],
        path_with_pattern=path_dict['possum']['pssm_rdb_pattern'],
        seq_id_list=seq_id_dict['t']['p'],
        desc='t_p',
    )
    pssm_ep_bit_t_n_f = libpybiofeature.PSSM_ENCROPY.build_feature_from_file(
        path_to_fasta=path_dict['fasta']['t']['n'],
        order_list=possum_index_dict['data']['t_n'],
        path_with_pattern=path_dict['possum']['pssm_rdb_pattern'],
        seq_id_list=seq_id_dict['t']['n'],
        desc='t_n',
    )
    pssm_ep_bit_v_p_f = libpybiofeature.PSSM_ENCROPY.build_feature_from_file(
        path_to_fasta=path_dict['fasta']['v']['p'],
        order_list=possum_index_dict['data']['v_p'],
        path_with_pattern=path_dict['possum']['pssm_rdb_pattern'],
        seq_id_list=seq_id_dict['v']['p'],
        desc='v_p',
    )
    pssm_ep_bit_v_n_f = libpybiofeature.PSSM_ENCROPY.build_feature_from_file(
        path_to_fasta=path_dict['fasta']['v']['n'],
        order_list=possum_index_dict['data']['v_n'],
        path_with_pattern=path_dict['possum']['pssm_rdb_pattern'],
        seq_id_list=seq_id_dict['v']['n'],
        desc='v_n',
    )

    pssm_ep_ds_4, pssm_ep_side = pd_map_range_Space(
        t_p_f=pssm_ep_bit_t_p_f, t_n_f=pssm_ep_bit_t_n_f, v_p_f=pssm_ep_bit_v_p_f, v_n_f=pssm_ep_bit_v_n_f)
    pssm_ep_t_p_f, pssm_ep_t_n_f, pssm_ep_v_p_f, pssm_ep_v_n_f = pssm_ep_ds_4

    pssm_ep_bit = stack_df(
        [pssm_ep_bit_t_p_f, pssm_ep_bit_v_p_f, pssm_ep_bit_t_n_f, pssm_ep_bit_v_n_f])
    pssm_ep = stack_df([pssm_ep_t_p_f, pssm_ep_v_p_f,
                        pssm_ep_t_n_f, pssm_ep_v_n_f])

    psipred_t_p_f = libpybiofeature.psipred.get_data(
        path_to_json=path_dict['psipred']['db']['index'],
        seqid_list=seq_id_dict['t']['p'],
        path_to_datadir=path_dict['psipred']['db']['file'],
        desc=path_dict['psipred']['desc']['t']['p'],
        length=100,
        cter=path_dict['model']['cter']
    )

    psipred_t_n_f = libpybiofeature.psipred.get_data(
        path_to_json=path_dict['psipred']['db']['index'],
        seqid_list=seq_id_dict['t']['n'],
        path_to_datadir=path_dict['psipred']['db']['file'],
        desc=path_dict['psipred']['desc']['t']['n'],
        length=100,
        cter=path_dict['model']['cter']
    )

    psipred_v_p_f = libpybiofeature.psipred.get_data(
        path_to_json=path_dict['psipred']['db']['index'],
        seqid_list=seq_id_dict['v']['p'],
        path_to_datadir=path_dict['psipred']['db']['file'],
        desc=path_dict['psipred']['desc']['v']['p'],
        length=100,
        cter=path_dict['model']['cter']
    )

    psipred_v_n_f = libpybiofeature.psipred.get_data(
        path_to_json=path_dict['psipred']['db']['index'],
        seqid_list=seq_id_dict['v']['n'],
        path_to_datadir=path_dict['psipred']['db']['file'],
        desc=path_dict['psipred']['desc']['v']['n'],
        length=100,
        cter=path_dict['model']['cter']
    )

    psipred = stack_df([psipred_t_p_f, psipred_v_p_f,
                        psipred_t_n_f, psipred_v_n_f])

    rsa_t_p, ss_t_p = libpybiofeature.libdataloader.cchmc.get_cchmc_data(
        path_to_json=path_dict['cchmc']['db'],
        seq_id_list=seq_id_dict['t']['p'],
        desc='t_p',
        num_of_each_part=25,
        length=100,
        cter=path_dict['model']['cter']
    )
    rsa_t_n, ss_t_n = libpybiofeature.libdataloader.cchmc.get_cchmc_data(
        path_to_json=path_dict['cchmc']['db'],
        seq_id_list=seq_id_dict['t']['n'],
        desc='t_n',
        num_of_each_part=25,
        length=100,
        cter=path_dict['model']['cter']
    )
    rsa_v_p, ss_v_p = libpybiofeature.libdataloader.cchmc.get_cchmc_data(
        path_to_json=path_dict['cchmc']['db'],
        seq_id_list=seq_id_dict['v']['p'],
        desc='v_p',
        num_of_each_part=25,
        length=100,
        cter=path_dict['model']['cter']
    )
    rsa_v_n, ss_v_n = libpybiofeature.libdataloader.cchmc.get_cchmc_data(
        path_to_json=path_dict['cchmc']['db'],
        seq_id_list=seq_id_dict['v']['n'],
        desc='v_n',
        num_of_each_part=25,
        length=100,
        cter=path_dict['model']['cter']
    )

    rsa = stack_df([rsa_t_p, rsa_v_p, rsa_t_n, rsa_v_n])
    ss = stack_df([ss_t_p, ss_v_p, ss_t_n, ss_v_n])

    expasy_t_p_f = libpybiofeature.libdataloader.expasy.get_expasy_t3sps(
        path_to_json=path_dict['expasy']['t']['p'],
        seq_id_list=seq_id_dict['t']['p']
    )
    expasy_t_n_f = libpybiofeature.libdataloader.expasy.get_expasy_t3sps(
        path_to_json=path_dict['expasy']['t']['n'],
        seq_id_list=seq_id_dict['t']['n']
    )
    expasy_v_p_f = libpybiofeature.libdataloader.expasy.get_expasy_t3sps(
        path_to_json=path_dict['expasy']['v']['p'],
        seq_id_list=seq_id_dict['v']['p']
    )
    expasy_v_n_f = libpybiofeature.libdataloader.expasy.get_expasy_t3sps(
        path_to_json=path_dict['expasy']['v']['n'],
        seq_id_list=seq_id_dict['v']['n']
    )

    expasy = stack_df([expasy_t_p_f, expasy_v_p_f, expasy_t_n_f, expasy_v_n_f])

    SPSE_t_p_f, SPSE_t_p_l, SPSE_t_n_f, SPSE_t_n_l = libpybiofeature.libdataloader.bliulab.build_form_of_data_set_SPSE(
        path_to_PSE=path_dict['bliulab']['t']['SC'],
        path_to_json_seq_id=path_dict['seq_id'],
        looking_key='t'
    )

    SPSE_v_p_f, SPSE_v_p_l, SPSE_v_n_f, SPSE_v_n_l = libpybiofeature.libdataloader.bliulab.build_form_of_data_set_SPSE(
        path_to_PSE=path_dict['bliulab']['v']['SC'],
        path_to_json_seq_id=path_dict['seq_id'],
        looking_key='v'
    )

    SPSE_ds_4, SPSE_side = xPSE_map_range_Space(
        SPSE_t_p_f, SPSE_t_n_f, SPSE_v_p_f, SPSE_v_n_f)
    SPSE_t_p_f, SPSE_t_n_f, SPSE_v_p_f, SPSE_v_n_f = SPSE_ds_4

    SPSE = stack_df([SPSE_t_p_f, SPSE_v_p_f, SPSE_t_n_f, SPSE_v_n_f])

    fasta_seq_t_p_f = libpybiofeature.libdataloader.fasta_seq_loader.prepare_data(
        path_to_fasta=path_dict['fasta']['t']['p'],
        seq_id_list=seq_id_dict['t']['p'],
    )

    fasta_seq_t_n_f = libpybiofeature.libdataloader.fasta_seq_loader.prepare_data(
        path_to_fasta=path_dict['fasta']['t']['n'],
        seq_id_list=seq_id_dict['t']['n'],
    )

    fasta_seq_v_p_f = libpybiofeature.libdataloader.fasta_seq_loader.prepare_data(
        path_to_fasta=path_dict['fasta']['v']['p'],
        seq_id_list=seq_id_dict['v']['p'],
    )

    fasta_seq_v_n_f = libpybiofeature.libdataloader.fasta_seq_loader.prepare_data(
        path_to_fasta=path_dict['fasta']['v']['n'],
        seq_id_list=seq_id_dict['v']['n'],
    )

    fasta_seq_t_p_l = utils.ds_preprocess.make_binary_label(
        size=fasta_seq_t_p_f.shape[0], label=True)
    fasta_seq_t_n_l = utils.ds_preprocess.make_binary_label(
        size=fasta_seq_t_n_f.shape[0], label=False)

    fasta_seq_v_p_l = utils.ds_preprocess.make_binary_label(
        size=fasta_seq_v_p_f.shape[0], label=True)
    fasta_seq_v_n_l = utils.ds_preprocess.make_binary_label(
        size=fasta_seq_v_n_f.shape[0], label=False)

    fasta_seq_p_f, _ = utils.ds_preprocess.make_merge(
        t_p_f=fasta_seq_t_p_f,
        t_p_l=fasta_seq_t_p_l,
        t_n_f=fasta_seq_v_p_f,
        t_n_l=fasta_seq_v_p_l,
    )

    fasta_seq_n_f, _ = utils.ds_preprocess.make_merge(
        t_p_f=fasta_seq_t_n_f,
        t_p_l=fasta_seq_t_n_l,
        t_n_f=fasta_seq_v_n_f,
        t_n_l=fasta_seq_v_n_l,
    )

    fasta_seq_p_f = fasta_seq_p_f.values.tolist()
    fasta_seq_n_f = fasta_seq_n_f.values.tolist()

    profile = {
        "p": libpybiofeature.BPBaac_psp.mat_constructor(
            fasta_db=fasta_seq_p_f,
            cter=path_dict['model']['cter'],
            terlength=100,
            padding_ac='A'
        ),
        "n": libpybiofeature.BPBaac_psp.mat_constructor(
            fasta_db=fasta_seq_n_f,
            cter=path_dict['model']['cter'],
            terlength=100,
            padding_ac='A'
        ),
    }

    bpbaac_p = pd.DataFrame([libpybiofeature.BPBaac_psp.mat_mapper(
        seq=seq,
        pmat=profile['p'],
        nmat=profile['n'],
        cter=path_dict['model']['cter'],
        terlength=100,
        padding_ac='A'
    )for seq in fasta_seq_p_f])
    bpbaac_n = pd.DataFrame([libpybiofeature.BPBaac_psp.mat_mapper(
        seq=seq,
        pmat=profile['p'],
        nmat=profile['n'],
        cter=path_dict['model']['cter'],
        terlength=100,
        padding_ac='A'
    )for seq in fasta_seq_n_f])

    BPBaac = stack_df([bpbaac_p, bpbaac_n, ])

    groundtrue_label_list = np.concatenate(
        [SPSE_t_p_l, SPSE_v_p_l, SPSE_t_n_l, SPSE_v_n_l])

    feature_list = [possum_pssm_ac, pssm_smth, aac, dac, CKSAAP, PPT, PPT_fullong, aacpro50, ssapro50, disopred350,
                    CTDC, CTDT, CTDD, qso, b62, vector, pssmcode, onehot1000, aacpro1000, ssapro1000,
                    disopred31000, psssa, CTD, onehot_100, Topm, PSE, a35, a40, a45, possum_composition, etpp,
                    cj, aacpro100, ssapro100, onehot30, pssm_ep_bit, pssm_ep, psipred, rsa, ss, expasy, SPSE, possum_smoothed_pssm,
                    possum_aac_pssm, possum_rpm_pssm, possum_pse_pssm, possum_dp_pssm, possum_dpc_pssm, possum_s_fpssm, possum_tpc, BPBaac, possum_d_fpssm, tac, ssaprom100, disopred3100]
    featurename_list = ["possum_pssm_ac", "pssm_smth", "aac", "dac", "CKSAAP", "PPT", "PPT_fullong", "aacpro50",
                        "ssapro50", "disopred350", "CTDC", "CTDT", "CTDD", "qso", "b62", "vector", "pssmcode", "onehot1000", "aacpro1000", "ssapro1000",
                        "disopred31000", "psssa", "CTD", "onehot_100", "Topm", "PSE", "a35", "a40", "a45",
                        "possum_composition", "etpp", "cj", "aacpro100", "ssapro100", "onehot30", "pssm_ep_bit", "pssm_ep",
                        "psipred", "rsa", "ss", "expasy", "SPSE", "possum_smoothed_pssm",
                        "possum_aac_pssm", "possum_rpm_pssm", "possum_pse_pssm", "possum_dp_pssm", "possum_dpc_pssm", "possum_s_fpssm", "possum_tpc", "BPBaac", "possum_d_fpssm", "tac", "ssaprom100", "disopred3100"]

    path_to_out_dir = os.path.join(
        make_go_path_to_out_dir, 'featuretsne')

    if os.path.exists(path_to_out_dir) == False:
        os.makedirs(path_to_out_dir)

    for feature_data, featurename in zip(feature_list, featurename_list):
        try:
            test_and_plot(
                data=feature_data,
                label=groundtrue_label_list,
                desc=f"{make_go_desc}_{featurename}",
                path_to_out_dir=path_to_out_dir
            )
        except ValueError as e:
            print(f"{make_go_desc}{featurename} Fail.")
            raise e
    return


if __name__ == "__main__":
    import utils
    work_Dir = utils.workdir.workdir(os.getcwd(), 4)
    for protype, cter_bool, db_size in [
        (1, True, 'small'),
        (2, False, 'small'),
        (3, False, 'big'),
        (4, True, 'big'),
        (6, False, 'small'),
    ]:
        Tx_arg = {
            "type": f'T{protype}',
            'seq_id': os.path.join(work_Dir, *['data', 'db', f'T{protype}', 'seq_id.json']),
            'shufflesplit_index_file': os.path.join(work_Dir, *['data', 'db', f'T{protype}', 'seq_id_shufflesplit.json']),
            'scratch': {
                'db': os.path.join(work_Dir, *['out', f'T{protype}', 'data', f'scratch_t{protype}.json']),
                'acctag': {
                    't': {'p': f't{protype}_t_p_acc', 'n': f't{protype}_t_n_acc'},
                    'v': {'p': f't{protype}_v_p_acc', 'n': f't{protype}_v_n_acc'},
                },
                'ssatag': {
                    't': {'p': f't{protype}_t_p_ssa', 'n': f't{protype}_t_n_ssa'},
                    'v': {'p': f't{protype}_v_p_ssa', 'n': f't{protype}_v_n_ssa'},
                }
            },
            'onehot': {
                'cter': cter_bool,
                't': {
                    'p': os.path.join(work_Dir, *['data', 'db', f'T{protype}', 't_p.fasta']),
                    'n': os.path.join(work_Dir, *['data', 'db', f'T{protype}', 't_n.fasta'])
                },
                'v': {
                    'p': os.path.join(work_Dir, *['data', 'db', f'T{protype}', 'v_p.fasta']),
                    'n': os.path.join(work_Dir, *['data', 'db', f'T{protype}', 'v_n.fasta'])
                },
            },
            'fasta': {
                'cter': cter_bool,
                't': {
                    'p': os.path.join(work_Dir, *['data', 'db', f'T{protype}', 't_p.fasta']),
                    'n': os.path.join(work_Dir, *['data', 'db', f'T{protype}', 't_n.fasta'])
                },
                'v': {
                    'p': os.path.join(work_Dir, *['data', 'db', f'T{protype}', 'v_p.fasta']),
                    'n': os.path.join(work_Dir, *['data', 'db', f'T{protype}', 'v_n.fasta'])
                },
            },
            'DISOPRED3': {
                'db': {
                    'index': os.path.join(work_Dir, *['out', f'T{protype}', 'data', f'DISOPRED3_t{protype}.json']),
                    'file': os.path.join(work_Dir, *['out', f'T{protype}', 'data', f'DISOPRED3_t{protype}_file'])
                },
                'desc': {
                    't': {'p': f't_{protype}_t_p', 'n': f't_{protype}_t_n'},
                    'v': {'p': f't_{protype}_v_p', 'n': f't_{protype}_v_n'},
                }
            },
            'expasy': {
                "t": {
                    'p': os.path.join(work_Dir, *['out', f'T{protype}', 'data', f'expasy.t{protype}.t_p.json']),
                    "n": os.path.join(work_Dir, *['out', f'T{protype}', 'data', f'expasy.t{protype}.t_n.json'])
                },
                "v": {
                    'p': os.path.join(work_Dir, *['out', f'T{protype}', 'data', f'expasy.t{protype}.v_p.json']),
                    "n": os.path.join(work_Dir, *['out', f'T{protype}', 'data', f'expasy.t{protype}.v_n.json'])
                }
            },
            'cchmc': {
                'db': os.path.join(work_Dir, *['out', f'T{protype}', 'data', f'cchmc.t{protype}.json']),
            },
            'bliulab': {
                't': {'idna': os.path.join(work_Dir, *['out', f'T{protype}', 'data', 'iDNA_Prot_dis_data_t.json', ]),
                      'top': os.path.join(work_Dir, *['out', f'T{protype}', 'data', 'Top_n_gram_data_t.json', ]),
                      'pse': os.path.join(work_Dir, *['out', f'T{protype}', 'data', 'PC_PSEACC_data_t.json', ]),
                      'SC': os.path.join(work_Dir, *['out', f'T{protype}', 'data', 'SC_PSEACC_data_t.json', ]),
                      'PC': os.path.join(work_Dir, *['out', f'T{protype}', 'data', 'PC_PSEACC_data_t.json', ]),
                      },
                'v': {
                    'idna': os.path.join(work_Dir, *['out', f'T{protype}', 'data', 'iDNA_Prot_dis_data_v.json', ]),
                    'top': os.path.join(work_Dir, *['out', f'T{protype}', 'data', 'Top_n_gram_data_v.json', ]),
                    'pse': os.path.join(work_Dir, *['out', f'T{protype}', 'data', 'PC_PSEACC_data_v.json', ]),
                    'SC': os.path.join(work_Dir, *['out', f'T{protype}', 'data', 'SC_PSEACC_data_v.json', ]),
                    'PC': os.path.join(work_Dir, *['out', f'T{protype}', 'data', 'PC_PSEACC_data_v.json', ]),
                },
            },
            'possum': {
                'index': os.path.join(work_Dir, *['out', f'T{protype}', 'data', 'possum', 'possum_index.json']),
                'fasta_pattern': os.path.join(work_Dir, *['data', 'db', f'T{protype}', '{taskname}.fasta']),
                'pssm_db_pattern': os.path.join(work_Dir, *['out', f'T{protype}', 'data', 'possum', '{zipid}_pssm_features.zip']),
                'pssm_fdb_pattern': os.path.join(work_Dir, *['out', f'T{protype}', 'data', 'possum', '{zipid}_pssm_features.zip']),
                'pssm_rdb_pattern': os.path.join(work_Dir, *['out', f'T{protype}', 'data', 'possum', '{zipid}_pssm_files.zip'])
            },
            'a': {
                't': {
                    "p": os.path.join(work_Dir, *['out', f'T{protype}', 'data', f'aligned_t{protype}_t_p.json', ]),
                    "n": os.path.join(work_Dir, *['out', f'T{protype}', 'data', f'aligned_t{protype}_t_n.json', ]),
                },
                'v': {
                    "p": os.path.join(work_Dir, *['out', f'T{protype}', 'data', f'aligned_t{protype}_v_p.json', ]),
                    "n": os.path.join(work_Dir, *['out', f'T{protype}', 'data', f'aligned_t{protype}_v_n.json', ]),
                }
            },
            'psipred': {
                'db': {
                    'index': os.path.join(work_Dir, *['out', f'T{protype}', 'data', f'psipred_t{protype}.json']),
                    'file': os.path.join(work_Dir, *['out', f'T{protype}', 'data', f'psipred_t{protype}_file'])
                },
                'desc': {
                    't': {'p': f't_{protype}_t_p', 'n': f't_{protype}_t_n'},
                    'v': {'p': f't_{protype}_v_p', 'n': f't_{protype}_v_n'},
                }
            },
            'model': {
                'size': db_size,
                'cter': cter_bool,
                'cv': {
                    'desc': f'T{protype}_CV',
                    'model_pickle': os.path.join(work_Dir, *['out', f'T{protype}', 'model', 'T4SEXGB', 'cv_model.pkl']),
                    'model_result': os.path.join(work_Dir, *['out', f'T{protype}', 'model', 'T4SEXGB', 'cv_model.json'])
                },
                'tt': {
                    'desc': f'T{protype}_TT',
                    'model_pickle': os.path.join(work_Dir, *['out', f'T{protype}', 'model', 'T4SEXGB', 'tt_model.pkl']),
                    'model_result': os.path.join(work_Dir, *['out', f'T{protype}', 'model', 'T4SEXGB', 'tt_model.json'])
                },
            }
        }

        make_go(
            path_dict=Tx_arg,
            make_go_desc=Tx_arg['type'],
            make_go_path_to_out_dir=os.path.join(
                work_Dir, *['tmp', 'data_out_md_docs',
                            'research', f'T{protype}', ]
            )
        )

# %%
