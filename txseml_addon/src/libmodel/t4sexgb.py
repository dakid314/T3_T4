'''
Author: George Zhao
Date: 2022-03-16 16:50:26
LastEditors: George Zhao
LastEditTime: 2022-08-15 22:24:02
Description: 
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''
import os
import json
import pickle
import math
import functools

from . import model_optimite
from . import common
import utils
import libpybiofeature
from libpybiofeature.pssmcode import get_all_task_feature

import numpy as np

from sklearn.model_selection import ShuffleSplit
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

n_jobs = 1
allow_load_from_disk = False
submodel_choise = None
submodel_desc = None
only_tt_model = False
only_cv_model = False


def xPSE_map_range_Space(t_p_f, t_n_f, v_p_f, v_n_f):
    side = [
        np.min([t_p_f.min(), t_n_f.min(),
                v_p_f.min(), v_n_f.min()]),
        np.max([t_p_f.max(), t_n_f.max(),
                v_p_f.max(), v_n_f.max()]),
    ]
    def func_(df): return (df - side[0]) / (side[1] - side[0])
    return [func_(df) for df in [t_p_f, t_n_f, v_p_f, v_n_f]], side


class T4SEXGB_Model(common.Model_Final):

    def __init__(self, cv, desc):
        super().__init__(cv, desc=desc)
        self.model = None

        self.side_store = None
        pass

    def tranmodel(self, f, l):
        super().tranmodel(f, l)
        self.model = [
            RandomForestClassifier(
                n_estimators=300, max_features='sqrt', n_jobs=n_jobs),
            GaussianNB(),
            XGBClassifier(
                n_estimators=700, learning_rate=0.1, n_jobs=n_jobs),
            LogisticRegression(
                multi_class='auto', solver='liblinear', n_jobs=n_jobs),
            GradientBoostingClassifier(
                n_estimators=300, learning_rate=0.2), SVC(gamma=0.03125, C=4, probability=True),
            MLPClassifier(
                hidden_layer_sizes=(48, 16), max_iter=1000),
            ExtraTreesClassifier(
                n_estimators=900, max_features='sqrt', n_jobs=n_jobs),
            KNeighborsClassifier(n_neighbors=2, n_jobs=n_jobs)
        ]

        for i in range(len(self.model)):
            self.model[i].fit(f, l)

        return self

    def predict(self, f):
        super().predict(f)
        result = np.stack(
            [
                self.model[i].predict_proba(f)[:, 1]
                for i in range(len(self.model))
            ]
        ).T
        if submodel_choise is None:
            result = (result >= 0.5)
            return result.sum(axis=1) / len(self.model)
        if isinstance(submodel_choise, int) == False or submodel_choise < 0 or submodel_choise >= 8:
            raise ValueError(f"Wrong submodel_choise: {submodel_choise}")
        return result[:, submodel_choise]


def Five_Cross_Get_model(
    model_construct_funtion,
    t_5C: list,
    v_f,
    v_l,
    path_to_model_pickle: str,
    path_to_model_result: str,
    size_of_data: str,
    data_to_store,
):
    model_set = list()
    model_result_set = list()
    loaded_from_disk = False
    if allow_load_from_disk == True:
        if os.path.exists(path_to_model_pickle) == True:
            with open(path_to_model_pickle, 'br') as f:
                model_set = pickle.load(f)
            loaded_from_disk = True
    submodel_desc_ = submodel_desc
    if submodel_choise is not None:
        if submodel_desc_ is None:
            submodel_desc_ = submodel_choise
        path_tmp = os.path.split(path_to_model_result)
        path_to_model_result = os.path.join(
            path_tmp[0], submodel_desc_ + '_' + path_tmp[1])
    for i in range(len(t_5C)):
        train_fl, test_fl = t_5C[i]

        model: T4SEXGB_Model = None
        if loaded_from_disk == True:
            model = model_set[i]
        else:
            model: T4SEXGB_Model = model_construct_funtion()
            model.tranmodel(
                train_fl[0], train_fl[1]
            )
            model.side_store = data_to_store
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
                "model": 'T4SEXGB',
                'desc': model.desc,
                'iteration': i,
                "size_of_data": size_of_data,
            }
        })
        if submodel_choise is not None:
            model_result_set[-1]['detail']['model'] += f'_{submodel_desc_}'

    model_result_set = utils.ds_preprocess.Five_Cross_Evaluation(
        model_result_set,
        pro_cutoff=0.5,
        mode='loo' if size_of_data == 'small' else None
    )

    if os.path.exists(os.path.split(path_to_model_pickle)[0]) == False:
        os.makedirs(os.path.split(path_to_model_pickle)[0])
    if os.path.exists(os.path.split(path_to_model_result)[0]) == False:
        os.makedirs(os.path.split(path_to_model_result)[0])

    if loaded_from_disk == False:
        with open(path_to_model_pickle, 'bw+') as f:
            pickle.dump(model_set, f)

    with open(path_to_model_result, 'w+', encoding='UTF-8') as f:
        json.dump(model_result_set, f, cls=utils.ds_preprocess.MyEncoder)

    return model_set, model_result_set


def research(path_dict: dict):
    seq_id_dict = None
    with open(path_dict['seq_id'], 'r', encoding='UTF-8') as f:
        seq_id_dict = json.load(f)

    possum_index_dict = None
    with open(path_dict['possum']['index'], 'r', encoding='UTF-8') as f:
        possum_index_dict = json.load(f)

    possum_t_p_f, possum_t_n_f, possum_v_p_f, possum_v_n_f = get_all_task_feature(
        possum_index_dict=possum_index_dict,
        path_to_json_seq_id=path_dict['seq_id'],
        feature_name_list=['smoothed_pssm', 'aac_pssm',
                           'rpm_pssm', 'pse_pssm', 'dp_pssm', ],
        path_to_fasta_pattern=path_dict['possum']['fasta_pattern'],
        path_to_with_pattern=path_dict['possum']['pssm_db_pattern']
    )

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

    ssapro_t_p_f = libpybiofeature.scratch_reader.sspro.get_muti_func(
        path_to_json_db=path_dict['scratch']['db'],
        tag_name=path_dict['scratch']['ssatag']['t']['p'],
        seqid_list=seq_id_dict['t']['p'],
        length=100,
        cter=path_dict['model']['cter']
    )

    ssapro_t_n_f = libpybiofeature.scratch_reader.sspro.get_muti_func(
        path_to_json_db=path_dict['scratch']['db'],
        tag_name=path_dict['scratch']['ssatag']['t']['n'],
        seqid_list=seq_id_dict['t']['n'],
        length=100,
        cter=path_dict['model']['cter']
    )
    ssapro_v_p_f = libpybiofeature.scratch_reader.sspro.get_muti_func(
        path_to_json_db=path_dict['scratch']['db'],
        tag_name=path_dict['scratch']['ssatag']['v']['p'],
        seqid_list=seq_id_dict['v']['p'],
        length=100,
        cter=path_dict['model']['cter']
    )

    ssapro_v_n_f = libpybiofeature.scratch_reader.sspro.get_muti_func(
        path_to_json_db=path_dict['scratch']['db'],
        tag_name=path_dict['scratch']['ssatag']['v']['n'],
        seqid_list=seq_id_dict['v']['n'],
        length=100,
        cter=path_dict['model']['cter']
    )

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

    PSE_t_p_f, PSE_t_p_l, PSE_t_n_f, PSE_t_n_l = libpybiofeature.libdataloader.bliulab.build_form_of_data_set_PSE(
        path_to_PSE=path_dict['bliulab']['t']['PC'],
        path_to_json_seq_id=path_dict['seq_id'],
        looking_key='t'
    )

    PSE_v_p_f, PSE_v_p_l, PSE_v_n_f, PSE_v_n_l = libpybiofeature.libdataloader.bliulab.build_form_of_data_set_PSE(
        path_to_PSE=path_dict['bliulab']['v']['PC'],
        path_to_json_seq_id=path_dict['seq_id'],
        looking_key='v'
    )

    PSE_ds_4, PSE_side = xPSE_map_range_Space(
        PSE_t_p_f, PSE_t_n_f, PSE_v_p_f, PSE_v_n_f)
    PSE_t_p_f, PSE_t_n_f, PSE_v_p_f, PSE_v_n_f = PSE_ds_4

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

    t_p_f = utils.ds_preprocess.merge_pd_list([
        aac_t_p_f, dac_t_p_f, tac_t_p_f, CKSAAP_t_p_f, cj_t_p_f, CTDC_t_p_f, CTDT_t_p_f, CTDD_t_p_f, possum_t_p_f, aacpro_t_p_f, ssapro_t_p_f, disopred3_t_p_f, PSE_t_p_f, SPSE_t_p_f
    ])
    t_n_f = utils.ds_preprocess.merge_pd_list([
        aac_t_n_f, dac_t_n_f, tac_t_n_f, CKSAAP_t_n_f, cj_t_n_f, CTDC_t_n_f, CTDT_t_n_f, CTDD_t_n_f, possum_t_n_f, aacpro_t_n_f, ssapro_t_n_f, disopred3_t_n_f, PSE_t_n_f, SPSE_t_n_f
    ])
    v_p_f = utils.ds_preprocess.merge_pd_list([
        aac_v_p_f, dac_v_p_f, tac_v_p_f, CKSAAP_v_p_f, cj_v_p_f, CTDC_v_p_f, CTDT_v_p_f, CTDD_v_p_f, possum_v_p_f, aacpro_v_p_f, ssapro_v_p_f, disopred3_v_p_f, PSE_v_p_f, SPSE_v_p_f
    ])
    v_n_f = utils.ds_preprocess.merge_pd_list([
        aac_v_n_f, dac_v_n_f, tac_v_n_f, CKSAAP_v_n_f, cj_v_n_f, CTDC_v_n_f, CTDT_v_n_f, CTDD_v_n_f, possum_v_n_f, aacpro_v_n_f, ssapro_v_n_f, disopred3_v_n_f, PSE_v_n_f, SPSE_v_n_f
    ])

    t_p_f.columns = list(range(len(t_p_f.columns)))
    t_n_f.columns = list(range(len(t_n_f.columns)))
    v_p_f.columns = list(range(len(v_p_f.columns)))
    v_n_f.columns = list(range(len(v_n_f.columns)))

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

    p_f, p_l = utils.ds_preprocess.make_merge(
        t_p_f=t_p_f,
        t_p_l=t_p_l,
        t_n_f=v_p_f,
        t_n_l=v_p_l,
    )

    n_f, n_l = utils.ds_preprocess.make_merge(
        t_p_f=t_n_f,
        t_p_l=t_n_l,
        t_n_f=v_n_f,
        t_n_l=v_n_l,
    )

    all_f, all_l = utils.ds_preprocess.make_merge(
        t_p_f=p_f,
        t_p_l=p_l,
        t_n_f=n_f,
        t_n_l=n_l,
    )

    if only_tt_model == True and only_cv_model == True:
        if os.path.exists(path_dict['csv_out_dir']) == False:
            os.makedirs(path_dict['csv_out_dir'])
        all_f.to_csv(os.path.join(
            path_dict['csv_out_dir'], *['feature.csv', ]))

    side_to_save = [
        PSE_side, SPSE_side
    ]

    model_cv = 10
    if path_dict['model']['size'] == 'small':
        model_cv = ShuffleSplit(n_splits=10, test_size=0.5, random_state=0)

    if only_tt_model == False:
        Five_Cross_Get_model(
            model_construct_funtion=functools.partial(
                T4SEXGB_Model, desc=path_dict['model']['cv']['desc'], cv=model_cv),
            t_5C=t_5C,
            v_f=v_f,
            v_l=v_l,
            path_to_model_pickle=path_dict['model']['cv']['model_pickle'],
            path_to_model_result=path_dict['model']['cv']['model_result'],
            size_of_data=path_dict['model']['size'],
            data_to_store=side_to_save
        )
    if only_cv_model == False:
        Five_Cross_Get_model(
            model_construct_funtion=functools.partial(
                T4SEXGB_Model, desc=path_dict['model']['tt']['desc'], cv=model_cv),
            t_5C=(((t_f, t_l),
                   (v_f, v_l)),),
            v_f=v_f,
            v_l=v_l,
            path_to_model_pickle=path_dict['model']['tt']['model_pickle'],
            path_to_model_result=path_dict['model']['tt']['model_result'],
            size_of_data=None,
            data_to_store=side_to_save
        )
