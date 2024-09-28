'''
Author: George Zhao
Date: 2022-03-14 14:27:54
LastEditors: George Zhao
LastEditTime: 2022-06-25 13:32:21
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

n_jobs = 16
allow_load_from_disk = False
submodel_choise = None
submodel_desc = None
only_tt_model = False

svm_param_o = {
    "C": [2**i for i in range(-6, 7)],
    "gamma": [2**i for i in range(-6, 7)],
}

svm_optim: model_optimite.svm_optim = functools.partial(
    model_optimite.svm_optim,
    default_param={
        'verbose': False,
        'probability': True,
        "kernel": 'rbf'
    }
)

mlp_param_o = {
    "solver": ['adam', 'sgd', 'lbfgs'],
    "activation": ['relu', 'tanh', 'logistic', 'identity'],
    "learning_rate": ['adaptive', 'invscaling', 'constant']
}

mlp_optim: model_optimite.mlp_optim = functools.partial(
    model_optimite.mlp_optim,
    default_param={
        'verbose': False,
        'early_stopping': True,
        'hidden_layer_sizes': (64, 32),
        'max_iter': 1000
    }
)

lr_param_o = [{
    "penalty": ['none'],
    "solver": ['newton-cg', 'lbfgs', 'sag', 'saga']
}, {
    "penalty": ['l1', ],
    "C": [10**i for i in range(-10, 10)],
    "solver": ['liblinear', 'saga']
}, {
    "penalty": ['l2', ],
    "C": [10**i for i in range(-10, 10)],
    "solver": ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
}, {
    "penalty": ['elasticnet', ],
    'l1_ratio': [0.1 * i for i in range(0, 11, 1)],
    "C": [10**i for i in range(-10, 10)],
    "solver": ['saga']
}, ]


def rf_param_o(num_of_feature: int):
    return {
        "n_estimators": [1000, ],
        "criterion": ["gini", "entropy"],
        "max_features": [
            i for i in range(
                1,
                math.floor(
                    max(
                        np.sqrt(num_of_feature), num_of_feature / 2
                    )
                ) + 1
            )
        ]
    }


def knn_optim_o(num_of_feature: int):
    return {
        'n_neighbors': [
            i for i in range(
                1,
                math.floor(
                    max(
                        np.sqrt(num_of_feature), num_of_feature / 2
                    )
                ) + 1
            )
        ]
    }


class Bastion4_Model(common.Model_Final):

    def __init__(self, cv, desc):
        super().__init__(cv, desc=desc)
        self.model_list = None

        self.feature_dividend_list = None
        pass

    def tranmodel(self, f, l, feature_dividend_list):
        super().tranmodel(f, l)
        self.feature_dividend_list = feature_dividend_list

        self.model_list = list()
        for feature_group_index in range(
            1,
            len(self.feature_dividend_list)
        ):
            self.model_list.append(
                self._tran_per_feature(
                    f.iloc[
                        :,
                        self.feature_dividend_list[feature_group_index - 1]:
                        self.feature_dividend_list[feature_group_index]
                    ],
                    l
                )
            )
        return self

    def predict(self, f):
        super().predict(f)
        # Each element with same index in self.model_list
        # len(self.model_list) == len(self.feature_dividend_list)
        if submodel_choise is not None:
            if isinstance(submodel_choise, int) == False or submodel_choise < 0 or submodel_choise >= 6:
                raise ValueError(f"Wrong submodel_choise: {submodel_choise}")
            return self._pred_per_model(f=f, model_type_index=submodel_choise)

        return np.stack([
            # Predict with this model in same type, Return the result voted vector.
            self._pred_per_model(f=f, model_type_index=model_index) >= 0.5
            for model_index in range(len(self.model_list[0]))
        ]).T.sum(axis=1) / len(self.model_list[0])

    def _pred_per_model(self, f, model_type_index):

        return np.stack([
            self.model_list[feature_group_index - 1][model_type_index].predict_proba(
                f.iloc[
                    :,
                    self.feature_dividend_list[feature_group_index - 1]:
                    self.feature_dividend_list[feature_group_index]
                ],
            ) >= 0.5
            for feature_group_index in range(
                1,
                len(self.feature_dividend_list)
            )
        ]).T.sum(axis=1) / len(self.model_list)

    def _tran_per_feature(
        self, f, l
    ):
        result = list()
        result.append(
            svm_optim(
                param_o=svm_param_o, cv=self.cv
            ).best_fit(
                f, l, verbose=-1, n_jobs=n_jobs
            )
        )
        result.append(
            model_optimite.nb_optim().best_fit(
                f, l, verbose=-1, n_jobs=n_jobs
            )
        )
        result.append(
            model_optimite.knn_optim(
                param_o=knn_optim_o(len(self.feature_dividend_list) - 1), cv=self.cv
            ).best_fit(
                f, l, verbose=-1, n_jobs=n_jobs
            )
        )
        result.append(
            model_optimite.rf_optim(
                param_o=rf_param_o(len(self.feature_dividend_list) - 1), cv=self.cv
            ).best_fit(
                f, l, verbose=-1, n_jobs=n_jobs
            )
        )
        result.append(
            mlp_optim(
                param_o=mlp_param_o, cv=self.cv
            ).best_fit(
                f, l, verbose=-1, n_jobs=n_jobs
            )
        )
        result.append(
            model_optimite.lr_optim(
                param_o=lr_param_o, cv=self.cv
            ).best_fit(
                f, l, verbose=-1, n_jobs=n_jobs
            )
        )
        return result

    def _predict(self, f):
        # Wrong Method: no voting and wrong pipline.
        super().predict(f)
        return np.stack([
            self._pred_per_feature(
                f.iloc[
                        :,
                        self.feature_dividend_list[feature_group_index - 1]:
                        self.feature_dividend_list[feature_group_index]
                        ],
                self.model_list[feature_group_index - 1]
            )
            for feature_group_index in range(
                1,
                len(self.feature_dividend_list)
            )
        ]).T.sum(axis=1) / len(self.model_list)

    def _pred_per_feature(
        self, f, model_result_list
    ):
        return np.stack([
            model_element.predict_proba(f)
            for model_element in model_result_list
        ]).T.sum(axis=1) / len(model_result_list)


def Five_Cross_Get_model(
    feature_dividend_list,
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

        model: common.Model_Final = None
        if loaded_from_disk == True:
            model = model_set[i]
        else:
            model: Bastion4_Model = model_construct_funtion()
            model.tranmodel(
                train_fl[0], train_fl[1], feature_dividend_list=feature_dividend_list
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
                "model": 'Bastion4',
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
        feature_name_list=['pssm_ac', ],
        path_to_fasta_pattern=path_dict['possum']['fasta_pattern'],
        path_to_with_pattern=path_dict['possum']['pssm_fdb_pattern']
    )

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

    feature_dividend_list = utils.ds_preprocess.make_feature_dividend_list(
        f_length_list=[f.shape[1] for f in [
            aac_t_p_f, dac_t_p_f, CKSAAP_t_p_f, PPT_t_p_f, possum_t_p_f, pssm_smth_t_p_f, disopred3_t_p_f, aacpro_t_p_f, ssapro_t_p_f
        ]]
    )
    t_p_f = utils.ds_preprocess.merge_pd_list([
        aac_t_p_f, dac_t_p_f, CKSAAP_t_p_f, PPT_t_p_f, possum_t_p_f, pssm_smth_t_p_f, disopred3_t_p_f, aacpro_t_p_f, ssapro_t_p_f
    ])
    t_n_f = utils.ds_preprocess.merge_pd_list([
        aac_t_n_f, dac_t_n_f, CKSAAP_t_n_f, PPT_t_n_f, possum_t_n_f, pssm_smth_t_n_f, disopred3_t_n_f, aacpro_t_n_f, ssapro_t_n_f
    ])
    v_p_f = utils.ds_preprocess.merge_pd_list([
        aac_v_p_f, dac_v_p_f, CKSAAP_v_p_f, PPT_v_p_f, possum_v_p_f, pssm_smth_v_p_f, disopred3_v_p_f, aacpro_v_p_f, ssapro_v_p_f
    ])
    v_n_f = utils.ds_preprocess.merge_pd_list([
        aac_v_n_f, dac_v_n_f, CKSAAP_v_n_f, PPT_v_n_f, possum_v_n_f, pssm_smth_v_n_f, disopred3_v_n_f, aacpro_v_n_f, ssapro_v_n_f
    ])

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

    model_cv = 10
    if path_dict['model']['size'] == 'small':
        model_cv = ShuffleSplit(n_splits=10, test_size=0.5, random_state=0)

    if only_tt_model == False:
        Five_Cross_Get_model(
            feature_dividend_list=feature_dividend_list,
            model_construct_funtion=functools.partial(
                Bastion4_Model, desc=path_dict['model']['cv']['desc'], cv=model_cv),
            t_5C=t_5C,
            v_f=v_f,
            v_l=v_l,
            path_to_model_pickle=path_dict['model']['cv']['model_pickle'],
            path_to_model_result=path_dict['model']['cv']['model_result'],
            size_of_data=path_dict['model']['size']
        )
    Five_Cross_Get_model(
        feature_dividend_list=feature_dividend_list,
        model_construct_funtion=functools.partial(
            Bastion4_Model, desc=path_dict['model']['tt']['desc'], cv=model_cv),
        t_5C=(((t_f, t_l),
               (v_f, v_l)),),
        v_f=v_f,
        v_l=v_l,
        path_to_model_pickle=path_dict['model']['tt']['model_pickle'],
        path_to_model_result=path_dict['model']['tt']['model_result'],
        size_of_data=None
    )
