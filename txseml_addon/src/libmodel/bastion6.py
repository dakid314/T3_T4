'''
Author: George Zhao
Date: 2022-03-08 11:35:43
LastEditors: George Zhao
LastEditTime: 2022-08-19 16:25:14
Description: 
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''
import os
import json
import pickle
import math
import itertools
import functools

import utils
from . import common
from . import model_optimite
import libpybiofeature
from libpybiofeature.pssmcode import get_all_task_feature

from sklearn.model_selection import ShuffleSplit
import pandas as pd
import numpy as np

n_jobs = 16
allow_load_from_disk = False
submodel_choise = None
submodel_desc = None
only_tt_model = False
only_cv_model = False

svm_param_o = {
    "gamma": [2**i for i in range(-6, 6 + 1)],
    "C": [2**i for i in range(-6, 6 + 1)],
}


class svm_optim_chiosed(model_optimite.svm_optim):
    svm_choised_scale = 0.75

    def __init__(self, param_o, cv, default_param={
        'verbose': False,
        'kernel': 'rbf',
        'probability': True
    }) -> None:
        super().__init__(param_o, cv, default_param=default_param)
        self.col_name = None
        self.origin_col = None
        pass

    def _chiose_col_name(self, X: pd.DataFrame, scale: float = svm_choised_scale):
        self.origin_col = X.columns
        self.col_name = np.random.choice(
            X.columns, size=max(1, math.floor(len(X.columns) * scale)), replace=False)
        return self

    def _get_col_named(self, X: pd.DataFrame):
        if self.origin_col.tolist() != X.columns.tolist():
            raise RuntimeError('self.origin_col != X.columns')
        return X.loc[:, self.col_name]

    def best_fit(self, X, y, verbose, n_jobs):
        self._chiose_col_name(X)
        return super().best_fit(
            X=self._get_col_named(X),
            y=y,
            verbose=verbose,
            n_jobs=n_jobs
        )

    def fit(self, X, y):
        self._chiose_col_name(X)
        return super().fit(self._get_col_named(X), y)

    def predict_proba(self, X):
        return super().predict_proba(
            X=self._get_col_named(X)
        )

    def find_parm(self, X, y, n_jobs, verbose):
        return super().find_parm(X, y, n_jobs, verbose=verbose)


# %%
class Bastion6_Model(common.Model_Final):

    def __init__(self, N, cv, desc):
        super().__init__(cv, desc=desc)
        self.num_of_clsif = N
        self.model_svm_2d_list = None

        self.feature_dividend_list = None
        pass

    def tranmodel(self, f, l, feature_dividend_list=[0, 520, 1198, None]):
        super().tranmodel(f, l)

        self.feature_dividend_list = feature_dividend_list

        numofcls_for_group = math.floor(
            self.num_of_clsif / (
                len(self.feature_dividend_list) - 1
            )
        )

        self.model_svm_2d_list = list()
        for feature_group_index in range(
            1,
            len(self.feature_dividend_list)
        ):
            model_svm_1d_list = list()
            for _ in range(numofcls_for_group):
                model_svm_1d_list.append(
                    svm_optim_chiosed(param_o=svm_param_o, cv=self.cv).best_fit(
                        f.iloc[
                            :,
                            self.feature_dividend_list[
                                feature_group_index - 1
                            ]:
                            self.feature_dividend_list[
                                feature_group_index
                            ]
                        ],
                        l,
                        verbose=-1,
                        n_jobs=n_jobs
                    )
                )
            self.model_svm_2d_list.append(model_svm_1d_list)

        return self

    def predict(self, f):
        super().predict(f)
        numofcls_for_group = math.floor(
            self.num_of_clsif / (
                len(self.feature_dividend_list) - 1
            )
        )
        result = np.stack(
            list(
                itertools.chain(*[
                    [
                        m.predict_proba(
                            f.iloc[
                                :,
                                self.feature_dividend_list[
                                    feature_group_index - 1
                                ]:
                                self.feature_dividend_list[
                                    feature_group_index
                                ]
                            ]
                        )
                        for m in self.model_svm_2d_list[feature_group_index - 1]
                    ]
                    for feature_group_index in range(
                        1,
                        len(self.feature_dividend_list)
                    )
                ])
            )
        ).T
        if submodel_choise is None:
            return result.sum(axis=1) / self.num_of_clsif
        if isinstance(submodel_choise, int) == False or submodel_choise < 0 or submodel_choise >= (len(self.feature_dividend_list) - 1):
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

        model: Bastion6_Model = None
        if loaded_from_disk == True:
            model = model_set[i]
        else:
            model: Bastion6_Model = model_construct_funtion()
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
                    f'label': list(train_fl[1])
                },
                "evaluation": {
                },
                "option": {
                }
            },
            "testing": {
                "origin": {
                    f'pred': list(model.predict(
                        test_fl[0]
                    )),
                    f'label': list(test_fl[1])
                },
                "evaluation": {
                },
                "option": {
                }
            },
            "validated": {
                "origin": {
                    f'pred': list(model.predict(
                        v_f
                    )),
                    f'label': list(v_l)},
                "evaluation": {
                },
                "option": {
                }
            },
            "detail": {
                "model": 'Bastion6',
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

    possum_index_dict = None
    with open(path_dict['possum']['index'], 'r', encoding='UTF-8') as f:
        possum_index_dict = json.load(f)

    possum_t_p_f, possum_t_n_f, possum_v_p_f, possum_v_n_f = get_all_task_feature(
        possum_index_dict=possum_index_dict,
        path_to_json_seq_id=path_dict['seq_id'],
        feature_name_list=['dpc_pssm', 's_fpssm', 'pse_pssm', ],
        path_to_fasta_pattern=path_dict['possum']['fasta_pattern'],
        path_to_with_pattern=path_dict['possum']['pssm_fdb_pattern']
    )

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

    # %%
    t_p_f = utils.ds_preprocess.merge_pd_list([
        aac_t_p_f, dac_t_p_f, qso_t_p_f, CTDC_t_p_f, CTDT_t_p_f, b62_t_p_f, possum_t_p_f
    ])
    t_n_f = utils.ds_preprocess.merge_pd_list([
        aac_t_n_f, dac_t_n_f, qso_t_n_f, CTDC_t_n_f, CTDT_t_n_f, b62_t_n_f, possum_t_n_f
    ])
    v_p_f = utils.ds_preprocess.merge_pd_list([
        aac_v_p_f, dac_v_p_f, qso_v_p_f, CTDC_v_p_f, CTDT_v_p_f, b62_v_p_f, possum_v_p_f
    ])
    v_n_f = utils.ds_preprocess.merge_pd_list([
        aac_v_n_f, dac_v_n_f, qso_v_n_f, CTDC_v_n_f, CTDT_v_n_f, b62_v_n_f, possum_v_n_f
    ])

    t_p_l = utils.ds_preprocess.make_binary_label(
        size=t_p_f.shape[0], label=True)
    t_n_l = utils.ds_preprocess.make_binary_label(
        size=t_n_f.shape[0], label=False)

    v_p_l = utils.ds_preprocess.make_binary_label(
        size=v_p_f.shape[0], label=True)
    v_n_l = utils.ds_preprocess.make_binary_label(
        size=v_n_f.shape[0], label=False)

    # %%
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

    seqbase_t_p_f = utils.ds_preprocess.merge_pd_list([
        aac_t_p_f, dac_t_p_f, qso_t_p_f,
    ])
    seqbase_t_n_f = utils.ds_preprocess.merge_pd_list([
        aac_t_n_f, dac_t_n_f, qso_t_n_f,
    ])
    seqbase_v_p_f = utils.ds_preprocess.merge_pd_list([
        aac_v_p_f, dac_v_p_f, qso_v_p_f,
    ])
    seqbase_v_n_f = utils.ds_preprocess.merge_pd_list([
        aac_v_n_f, dac_v_n_f, qso_v_n_f,
    ])

    seqbase_p_f, seqbase_p_l = utils.ds_preprocess.make_merge(
        t_p_f=seqbase_t_p_f,
        t_p_l=t_p_l,
        t_n_f=seqbase_v_p_f,
        t_n_l=v_p_l,
    )

    seqbase_n_f, seqbase_n_l = utils.ds_preprocess.make_merge(
        t_p_f=seqbase_t_n_f,
        t_p_l=t_n_l,
        t_n_f=seqbase_v_n_f,
        t_n_l=v_n_l,
    )

    seqbase_all_f, _ = utils.ds_preprocess.make_merge(
        t_p_f=seqbase_p_f,
        t_p_l=seqbase_p_l,
        t_n_f=seqbase_n_f,
        t_n_l=seqbase_n_l,
    )

    # if only_tt_model == True and only_cv_model == True:
    #     if os.path.exists(path_dict['csv_out_dir']) == False:
    #         os.makedirs(path_dict['csv_out_dir'])
    #     seqbase_all_f.to_csv(os.path.join(
    #         path_dict['csv_out_dir'], *['bastion6_seqbase.csv', ]))

    model_cv = 10
    if path_dict['model']['size'] == 'small':
        model_cv = ShuffleSplit(n_splits=10, test_size=0.5, random_state=0)

    if only_tt_model == False:
        Five_Cross_Get_model(
            model_construct_funtion=functools.partial(
                Bastion6_Model, desc=path_dict['model']['cv']['desc'], cv=model_cv, N=100),
            t_5C=t_5C,
            v_f=v_f,
            v_l=v_l,
            path_to_model_pickle=path_dict['model']['cv']['model_pickle'],
            path_to_model_result=path_dict['model']['cv']['model_result'],
            size_of_data=path_dict['model']['size']
        )
    if only_cv_model == False:
        Five_Cross_Get_model(
            model_construct_funtion=functools.partial(
                Bastion6_Model, desc=path_dict['model']['tt']['desc'], cv=model_cv, N=100),
            t_5C=(((t_f, t_l), (v_f, v_l)),),
            v_f=v_f,
            v_l=v_l,
            path_to_model_pickle=path_dict['model']['tt']['model_pickle'],
            path_to_model_result=path_dict['model']['tt']['model_result'],
            size_of_data=None
        )
