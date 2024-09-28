'''
Author: George Zhao
Date: 2022-02-24 19:57:12
LastEditors: George Zhao
LastEditTime: 2022-06-24 22:14:27
Description: 
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''
# %%
import os
import sys
import pickle
import functools
import json
sys.path.append('src')
# %%
import utils
work_Dir = utils.workdir.workdir(os.getcwd(), 3)
rdata_Dir = os.path.join(work_Dir, 'data')
out_Dir = os.path.join(work_Dir, 'out')
# %%
import numpy as np
from sklearn.model_selection import ShuffleSplit
# %%
from libpybiofeature.libdataloader.protr_loader import build_form_of_data_set_protr
from libpybiofeature.pssmcode import get_all_task_feature
from libmodel import common
from libmodel import model_optimite
# %%
lgb_param_o = {
    "weight_column": [i for i in range(1, 11)],
    "learning_rate": [2**i for i in range(-10, 0)],
    "num_leaves": [i for i in range(50, 801, 50)],
    "max_depth": [i for i in range(5, 11)],
    "min_data_in_leaf": [2**i for i in range(1, 7)],
    "max_bin": [2**i for i in range(5, 11)],
    "feature_fraction": [(0.5 + i * 0.02) for i in range(26)],
    "min_sum_hessian": [(0 + 0.001 * i)for i in range(1, 21)],
    "lambda_l1": [(i * 0.002)for i in range(6)],
    "lambda_l2": [(i * 0.002)for i in range(6)],
    "drop_rate": [i * 0.1 for i in range(11)],
    "max_drop": [(1 + 2 * i) for i in range(15)]
}
sc_ = 1
n_jobs = 16
step_param = {
    1: {
        'param_size_': 3,
        'n_Individual': 25,
        'sc_': 1,
        'append': False
    },
    2: {
        "n_iter": 10,
        "p_of_mutation": 0.5,
        "p_of_recombination": 0.5,
        "n_Population_scale": 10,
        'early_stop_wait_iter': 5,
        "replace_": True
    }}
# %%
allow_load_from_disk = False
submodel_choise = None
submodel_desc = None
only_tt_model = False
# %%


class Bastion3_Model(common.Model_Final):

    def __init__(self, desc, cv):
        super().__init__(cv=cv, desc=desc)
        self.model_aac = None
        self.model_dc = None
        self.model_qso = None
        self.model_ctdc = None
        self.model_ctdt = None
        self.model_pssm1 = None
        self.model_pssm2 = None
        self.model_pssm3 = None
        self.model_pssm4 = None
        self.model_pssm5 = None
        pass

    def tranmodel(
        self,
        Sequence_based_features_f,
        Physicochemical_properties_f,
        possum_f,
        l,
        n_jobs=n_jobs
    ):
        super().tranmodel(None, l)

        self.model_aac = model_optimite.lgbm_optim(lgb_param_o=lgb_param_o, cv=self.cv).best_fit(
            Sequence_based_features_f.iloc[:,
                                           0:20], l, n_jobs=n_jobs, step_param=step_param
        )

        self.model_dc = model_optimite.lgbm_optim(lgb_param_o=lgb_param_o, cv=self.cv).best_fit(
            Sequence_based_features_f.iloc[:,
                                           20:], l, n_jobs=n_jobs, step_param=step_param
        )

        self.model_qso = model_optimite.lgbm_optim(lgb_param_o=lgb_param_o, cv=self.cv).best_fit(
            Physicochemical_properties_f.iloc[:,
                                              0:100], l, n_jobs=n_jobs, step_param=step_param
        )

        self.model_ctdc = model_optimite.lgbm_optim(lgb_param_o=lgb_param_o, cv=self.cv).best_fit(
            Physicochemical_properties_f.iloc[:,
                                              100:121], l, n_jobs=n_jobs, step_param=step_param
        )

        self.model_ctdt = model_optimite.lgbm_optim(lgb_param_o=lgb_param_o, cv=self.cv).best_fit(
            Physicochemical_properties_f.iloc[:,
                                              121:], l, n_jobs=n_jobs, step_param=step_param
        )

        self.model_pssm1 = model_optimite.lgbm_optim(lgb_param_o=lgb_param_o, cv=self.cv).best_fit(
            possum_f.iloc[:,
                          0:400], l, n_jobs=n_jobs, step_param=step_param
        )

        self.model_pssm2 = model_optimite.lgbm_optim(lgb_param_o=lgb_param_o, cv=self.cv).best_fit(
            possum_f.iloc[:,
                          400:800], l, n_jobs=n_jobs, step_param=step_param
        )

        self.model_pssm3 = model_optimite.lgbm_optim(lgb_param_o=lgb_param_o, cv=self.cv).best_fit(
            possum_f.iloc[:,
                          800:820], l, n_jobs=n_jobs, step_param=step_param
        )

        self.model_pssm4 = model_optimite.lgbm_optim(lgb_param_o=lgb_param_o, cv=self.cv).best_fit(
            possum_f.iloc[:,
                          820:1220], l, n_jobs=n_jobs, step_param=step_param
        )

        self.model_pssm5 = model_optimite.lgbm_optim(lgb_param_o=lgb_param_o, cv=self.cv).best_fit(
            possum_f.iloc[:,
                          1220:], l, n_jobs=n_jobs, step_param=step_param
        )

        return self

    def predict(self,
                Sequence_based_features_f,
                Physicochemical_properties_f,
                possum_f):
        super().predict(None)

        pred_acc = self.model_aac.predict_proba(
            Sequence_based_features_f.iloc[:, 0:20])
        pred_dc = self.model_dc.predict_proba(
            Sequence_based_features_f.iloc[:, 20:])
        pred_qso = self.model_qso.predict_proba(
            Physicochemical_properties_f.iloc[:, 0:100])
        pred_ctdc = self.model_ctdc.predict_proba(
            Physicochemical_properties_f.iloc[:, 100:121])
        pred_ctdt = self.model_ctdt.predict_proba(
            Physicochemical_properties_f.iloc[:, 121:])
        pred_pssm1 = self.model_pssm1.predict_proba(
            possum_f.iloc[:, 0:400])
        pred_pssm2 = self.model_pssm2.predict_proba(
            possum_f.iloc[:, 400:800])
        pred_pssm3 = self.model_pssm3.predict_proba(
            possum_f.iloc[:, 800:820])
        pred_pssm4 = self.model_pssm4.predict_proba(
            possum_f.iloc[:, 820:1220])
        pred_pssm5 = self.model_pssm5.predict_proba(
            possum_f.iloc[:, 1220:])

        seq_base = (pred_acc + pred_dc + pred_qso) / 3

        pp_base = (pred_ctdc + pred_ctdt) / 2

        pssm_base = (pred_pssm1 + pred_pssm2 + pred_pssm3 +
                     pred_pssm4 + pred_pssm5) / 5

        final_out = (seq_base + pp_base + pssm_base * 2) / 4

        if submodel_choise is not None:
            if submodel_choise == 0:
                return seq_base
            elif submodel_choise == 1:
                return pp_base
            elif submodel_choise == 2:
                return pssm_base
            else:
                raise ValueError(f"Wrong submodel_choise: {submodel_choise}")
        # Default return
        return final_out

    def save_to_file(self, path_to_dir):
        return super().save_to_file(path_to_dir)

    def load_model(self, path_to_dir):
        return super().load_model(path_to_dir)

    def clean_model(self):
        return super().clean_model()


def Five_Cross_Get_model(
    model_construct_funtion,
    Sequence_based_features_t_5C: list,
    Physicochemical_properties_t_5C: list,
    possum_t_5C: list,
    Sequence_based_features_v_f,
    Physicochemical_properties_v_f,
    possum_v_f,
    possum_v_l,
    path_to_model_pickle: str,
    path_to_model_result: str,
    size_of_data: str,
):
    # Training
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
    for i in range(len(Sequence_based_features_t_5C)):
        Sequence_based_features_train, Sequence_based_features_test = Sequence_based_features_t_5C[
            i]
        Physicochemical_properties_train, Physicochemical_properties_test = Physicochemical_properties_t_5C[
            i]
        possum_train, possum_test = possum_t_5C[i]
        model: common.Model_Final = None
        if loaded_from_disk == True:
            model = model_set[i]
        else:
            model: common.Model_Final = model_construct_funtion()
            model.tranmodel(
                Sequence_based_features_train[0],
                Physicochemical_properties_train[0],
                possum_train[0],
                Sequence_based_features_train[1]
            )
            model_set.append(model)

        model_result_set.append(
            {
                "training": {
                    "origin": {
                        f'pred': list(model.predict(
                            Sequence_based_features_train[0],
                            Physicochemical_properties_train[0],
                            possum_train[0]
                        )),
                        f'label': list(possum_train[1])},
                    "evaluation": {
                    },
                    "option": {
                    }
                },
                "testing": {
                    "origin": {
                        f'pred': list(model.predict(
                            Sequence_based_features_test[0],
                            Physicochemical_properties_test[0],
                            possum_test[0]
                        )),
                        f'label': list(possum_test[1])},
                    "evaluation": {
                    },
                    "option": {
                    }
                },
                "validated": {
                    "origin": {
                        f'pred': list(model.predict(
                            Sequence_based_features_v_f,
                            Physicochemical_properties_v_f,
                            possum_v_f
                        )),
                        f'label': list(possum_v_l)},
                    "evaluation": {
                    },
                    "option": {
                    }
                },
                "detail": {
                    "model": 'Bastion3',
                    'desc': model.desc,
                    'iteration': i,
                    "size_of_data": size_of_data,
                }
            }
        )

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
    protr_t_p, protr_t_n = build_form_of_data_set_protr(
        path_to_p_protr=path_dict['protr']['t']['p'],
        path_to_n_protr=path_dict['protr']['t']['n'],
        path_to_seq_id=path_dict['seq_id'],
        looking_key='t',
        path_to_json_seq_id=path_dict['seq_id']
    )
    protr_v_p, protr_v_n = build_form_of_data_set_protr(
        path_to_p_protr=path_dict['protr']['v']['p'],
        path_to_n_protr=path_dict['protr']['v']['n'],
        path_to_seq_id=path_dict['seq_id'],
        looking_key='v',
        path_to_json_seq_id=path_dict['seq_id']
    )

    Sequence_based_features_columns = list(protr_t_p.columns)[0:420]
    Physicochemical_properties_columns = list(protr_t_p.columns)[420:]

    possum_index_dict = None
    with open(path_dict['possum']['index'], 'r', encoding='UTF-8') as f:
        possum_index_dict = json.load(f)

    possum_t_p, possum_t_n, possum_v_p, possum_v_n = get_all_task_feature(
        possum_index_dict=possum_index_dict,
        path_to_json_seq_id=path_dict['seq_id'],
        feature_name_list=[
            'pssm_composition', 'rpm_pssm', 'd_fpssm', 'tpc', 'dp_pssm'
        ],
        path_to_fasta_pattern=path_dict['possum']['fasta_pattern'],
        path_to_with_pattern=path_dict['possum']['pssm_fdb_pattern']
    )

    # Sequence_based_features_columns
    Sequence_based_features_t_5C = utils.ds_preprocess.get_5C_data(
        shape_to_chiose=protr_t_p.shape[0],
        t_p_f=protr_t_p.loc[:, Sequence_based_features_columns],
        t_p_l=np.ones(shape=protr_t_p.shape[0]),
        t_n_f=protr_t_n.loc[:, Sequence_based_features_columns],
        t_n_l=np.zeros(shape=protr_t_n.shape[0]),
        shufflesplit={
            'shufflesplit_index_file': path_dict['shufflesplit_index_file']
        } if path_dict['model']['size'] == 'small' else None
    )
    Sequence_based_features_t_f, Sequence_based_features_t_l = utils.ds_preprocess.make_merge(
        t_p_f=protr_t_p.loc[:, Sequence_based_features_columns],
        t_p_l=np.ones(shape=protr_t_p.shape[0]),
        t_n_f=protr_t_n.loc[:, Sequence_based_features_columns],
        t_n_l=np.zeros(shape=protr_t_n.shape[0])
    )
    Sequence_based_features_v_f, Sequence_based_features_v_l = utils.ds_preprocess.make_merge(
        t_p_f=protr_v_p.loc[:, Sequence_based_features_columns],
        t_p_l=np.ones(shape=protr_v_p.shape[0]),
        t_n_f=protr_v_n.loc[:, Sequence_based_features_columns],
        t_n_l=np.zeros(shape=protr_v_n.shape[0])
    )

    # Physicochemical_properties_columns
    Physicochemical_properties_t_5C = utils.ds_preprocess.get_5C_data(
        shape_to_chiose=protr_t_p.shape[0],
        t_p_f=protr_t_p.loc[:, Physicochemical_properties_columns],
        t_p_l=np.ones(shape=protr_t_p.shape[0]),
        t_n_f=protr_t_n.loc[:, Physicochemical_properties_columns],
        t_n_l=np.zeros(shape=protr_t_n.shape[0]),
        shufflesplit={
            'shufflesplit_index_file': path_dict['shufflesplit_index_file']
        } if path_dict['model']['size'] == 'small' else None
    )
    Physicochemical_properties_t_f, Physicochemical_properties_t_l = utils.ds_preprocess.make_merge(
        t_p_f=protr_t_p.loc[:, Physicochemical_properties_columns],
        t_p_l=np.ones(shape=protr_t_p.shape[0]),
        t_n_f=protr_t_n.loc[:, Physicochemical_properties_columns],
        t_n_l=np.zeros(shape=protr_t_n.shape[0])
    )
    Physicochemical_properties_v_f, Physicochemical_properties_v_l = utils.ds_preprocess.make_merge(
        t_p_f=protr_v_p.loc[:, Physicochemical_properties_columns],
        t_p_l=np.ones(shape=protr_v_p.shape[0]),
        t_n_f=protr_v_n.loc[:, Physicochemical_properties_columns],
        t_n_l=np.zeros(shape=protr_v_n.shape[0])
    )

    # possum Feature
    possum_t_5C = utils.ds_preprocess.get_5C_data(
        shape_to_chiose=possum_t_p.shape[0],
        t_p_f=possum_t_p,
        t_p_l=np.ones(shape=possum_t_p.shape[0]),
        t_n_f=possum_t_n,
        t_n_l=np.zeros(shape=possum_t_n.shape[0]),
        shufflesplit={
            'shufflesplit_index_file': path_dict['shufflesplit_index_file']
        } if path_dict['model']['size'] == 'small' else None
    )
    possum_t_f, possum_t_l = utils.ds_preprocess.make_merge(
        t_p_f=possum_t_p,
        t_p_l=np.ones(shape=possum_t_p.shape[0]),
        t_n_f=possum_t_n,
        t_n_l=np.zeros(shape=possum_t_n.shape[0])
    )
    possum_v_f, possum_v_l = utils.ds_preprocess.make_merge(
        t_p_f=possum_v_p,
        t_p_l=np.ones(shape=possum_v_p.shape[0]),
        t_n_f=possum_v_n,
        t_n_l=np.zeros(shape=possum_v_n.shape[0])
    )

    model_cv = 10
    if path_dict['model']['size'] == 'small':
        model_cv = ShuffleSplit(n_splits=10, test_size=0.5, random_state=0)

    if only_tt_model == False:
        Five_Cross_Get_model(
            model_construct_funtion=functools.partial(
                Bastion3_Model, desc=path_dict['model']['cv']['desc'], cv=model_cv),
            Sequence_based_features_t_5C=Sequence_based_features_t_5C,
            Physicochemical_properties_t_5C=Physicochemical_properties_t_5C,
            possum_t_5C=possum_t_5C,
            Sequence_based_features_v_f=Sequence_based_features_v_f,
            Physicochemical_properties_v_f=Physicochemical_properties_v_f,
            possum_v_f=possum_v_f,
            possum_v_l=possum_v_l,
            path_to_model_pickle=path_dict['model']['cv']['model_pickle'],
            path_to_model_result=path_dict['model']['cv']['model_result'],
            size_of_data=path_dict['model']['size']
        )

    Five_Cross_Get_model(
        model_construct_funtion=functools.partial(
            Bastion3_Model, desc=path_dict['model']['tt']['desc'], cv=model_cv),
        Sequence_based_features_t_5C=(((Sequence_based_features_t_f, Sequence_based_features_t_l), (
            Sequence_based_features_v_f, Sequence_based_features_v_l)),),
        Physicochemical_properties_t_5C=(((Physicochemical_properties_t_f, Physicochemical_properties_t_l), (
            Physicochemical_properties_v_f, Physicochemical_properties_v_l)),),
        possum_t_5C=(((possum_t_f, possum_t_l), (possum_v_f, possum_v_l)),),
        Sequence_based_features_v_f=Sequence_based_features_v_f,
        Physicochemical_properties_v_f=Physicochemical_properties_v_f,
        possum_v_f=possum_v_f,
        possum_v_l=possum_v_l,
        path_to_model_pickle=path_dict['model']['tt']['model_pickle'],
        path_to_model_result=path_dict['model']['tt']['model_result'],
        size_of_data=None
    )
