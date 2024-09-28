'''
Author: George Zhao
Date: 2022-03-05 22:46:29
LastEditors: George Zhao
LastEditTime: 2022-08-19 17:00:23
Description: 
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''
import os
import json
import pickle
import functools
import sys
sys.path.append('src')

import utils
from . import common
from . import model_optimite
import libpybiofeature

import pandas as pd
import numpy as np
from sklearn.model_selection import ShuffleSplit

n_jobs = 16
allow_load_from_disk = False
submodel_choise = None
submodel_desc = None
only_tt_model = False
only_cv_model = False


def merge_pd_list(pd_list: list):
    return pd.concat(pd_list, axis=1)


def ep3_map_range_Space(t_p_f, t_n_f, v_p_f, v_n_f):
    side = [
        np.min([t_p_f.min(), t_n_f.min(),
                v_p_f.min(), v_n_f.min()]),
        np.max([t_p_f.max(), t_n_f.max(),
                v_p_f.max(), v_n_f.max()]),
    ]
    def func_(df): return (df - side[0]) / (side[1] - side[0])
    return [func_(df) for df in [t_p_f, t_n_f, v_p_f, v_n_f]], side


svm_param_o = [{
    "kernel": ["linear", ],
    "C": [10**i for i in range(-6, 6 + 1)],
}, {
    "kernel": ["rbf", "sigmoid"],
    "gamma": [10**i for i in range(-6, 6 + 1)],
    "C": [10**i for i in range(-6, 6 + 1)],
}, {
    "kernel": ["poly", ],
    "gamma": [10**i for i in range(-6, 0)],
    "C": [10**i for i in range(-6, 0)],
}]

lpa_param_o = [{
    'max_iter': [1000, 100, 10000, 10],
    'kernel': ['rbf', ],
    'gamma': [20 * 2**i for i in range(-5, 5)]
},
    {
    'max_iter': [1000, 100, 10000, 10],
        'kernel': ['knn', ],
        'n_neighbors':[i for i in range(3, 30)],
},
]


class EP3_Model(common.Model_Final):
    def __init__(self, cv, desc):
        super().__init__(cv, desc=desc)
        self.model_lpa1 = None
        self.model_lpa2 = None
        self.model_lpa3 = None
        self.model_svm1 = None
        self.model_svm2 = None
        self.model_svm3 = None

        self.side_store = None

    def tranmodel(self, idna_f, top_f, pse_f, a35, a40, a45, l):
        super().tranmodel(f=None, l=None)
        self.model_lpa1 = model_optimite.lpa_model(param_o=lpa_param_o, cv=self.cv).best_fit(
            idna_f, l, verbose=-1, n_jobs=n_jobs
        )
        self.model_lpa2 = model_optimite.lpa_model(param_o=lpa_param_o, cv=self.cv).best_fit(
            top_f, l, verbose=-1, n_jobs=n_jobs
        )
        self.model_lpa3 = model_optimite.lpa_model(param_o=lpa_param_o, cv=self.cv).best_fit(
            pse_f, l, verbose=-1, n_jobs=n_jobs
        )
        self.model_svm1 = model_optimite.svm_optim(param_o=svm_param_o, cv=self.cv).best_fit(
            a35, l, verbose=-1, n_jobs=n_jobs
        )
        self.model_svm2 = model_optimite.svm_optim(param_o=svm_param_o, cv=self.cv).best_fit(
            a40, l, verbose=-1, n_jobs=n_jobs
        )
        self.model_svm3 = model_optimite.svm_optim(param_o=svm_param_o, cv=self.cv).best_fit(
            a45, l, verbose=-1, n_jobs=n_jobs
        )
        return self

    def predict(self, idna_f, top_f, pse_f, a35, a40, a45):
        super().predict(f=None)
        result = np.stack(
            [
                self.model_lpa1.predict_proba(idna_f),
                self.model_lpa2.predict_proba(top_f),
                self.model_lpa3.predict_proba(pse_f),
                self.model_svm1.predict_proba(a35),
                self.model_svm2.predict_proba(a40),
                self.model_svm3.predict_proba(a45),
            ]
        ).T
        if submodel_choise is None:
            result = (result > 0.5)
            return result.sum(axis=1) / 6
        if isinstance(submodel_choise, int) == False or submodel_choise < 0 or submodel_choise >= 6:
            raise ValueError(f"Wrong submodel_choise: {submodel_choise}")
        return result[:, submodel_choise]


def Five_Cross_Get_model(
    model_construct_funtion,
    idna_f_Five_Cross_set: list,
    top_f_Five_Cross_set: list,
    pse_f_Five_Cross_set: list,
    a35_f_Five_Cross_set: list,
    a40_f_Five_Cross_set: list,
    a45_f_Five_Cross_set: list,
    idna_v_f,
    top_v_f,
    pse_v_f,
    a35_v_f,
    a40_v_f,
    a45_v_f,
    a45_v_l,
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
    for i in range(len(idna_f_Five_Cross_set)):

        model: EP3_Model = None
        if loaded_from_disk == True:
            model = model_set[i]
        else:
            model: EP3_Model = model_construct_funtion()
            model.tranmodel(
                idna_f=idna_f_Five_Cross_set[i][0][0],
                top_f=top_f_Five_Cross_set[i][0][0],
                pse_f=pse_f_Five_Cross_set[i][0][0],
                a35=a35_f_Five_Cross_set[i][0][0],
                a40=a40_f_Five_Cross_set[i][0][0],
                a45=a45_f_Five_Cross_set[i][0][0],
                l=a45_f_Five_Cross_set[i][0][1],
            )
            model.side_store = data_to_store

            model_set.append(model)

        model_result_set.append(
            {
                "training": {
                    "origin": {
                        f'pred': list(model.predict(
                            idna_f=idna_f_Five_Cross_set[i][0][0],
                            top_f=top_f_Five_Cross_set[i][0][0],
                            pse_f=pse_f_Five_Cross_set[i][0][0],
                            a35=a35_f_Five_Cross_set[i][0][0],
                            a40=a40_f_Five_Cross_set[i][0][0],
                            a45=a45_f_Five_Cross_set[i][0][0],
                        )),
                        f'label': list(a45_f_Five_Cross_set[i][0][1])},
                    "evaluation": {
                    },
                    "option": {
                    }
                },
                "testing": {
                    "origin": {
                        f'pred': list(model.predict(
                            idna_f=idna_f_Five_Cross_set[i][1][0],
                            top_f=top_f_Five_Cross_set[i][1][0],
                            pse_f=pse_f_Five_Cross_set[i][1][0],
                            a35=a35_f_Five_Cross_set[i][1][0],
                            a40=a40_f_Five_Cross_set[i][1][0],
                            a45=a45_f_Five_Cross_set[i][1][0],
                        )),
                        f'label': list(a45_f_Five_Cross_set[i][1][1])},
                    "evaluation": {
                    },
                    "option": {
                    }
                },
                "validated": {
                    "origin": {
                        f'pred': list(model.predict(
                            idna_f=idna_v_f,
                            top_f=top_v_f,
                            pse_f=pse_v_f,
                            a35=a35_v_f,
                            a40=a40_v_f,
                            a45=a45_v_f,
                        )),
                        f'label': list(a45_v_l)},
                    "evaluation": {
                    },
                    "option": {
                    }
                },
                "detail": {
                    "model": 'EP3',
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

    iDNA_t_5C = utils.ds_preprocess.get_5C_data(
        shape_to_chiose=iDNA_t_p_f.shape[0],
        t_p_f=iDNA_t_p_f,
        t_p_l=iDNA_t_p_l,
        t_n_f=iDNA_t_n_f,
        t_n_l=iDNA_t_n_l,
        shufflesplit={
            'shufflesplit_index_file': path_dict['shufflesplit_index_file']
        } if path_dict['model']['size'] == 'small' else None
    )
    Topm_t_5C = utils.ds_preprocess.get_5C_data(
        shape_to_chiose=Topm_t_p_f.shape[0],
        t_p_f=Topm_t_p_f,
        t_p_l=Topm_t_p_l,
        t_n_f=Topm_t_n_f,
        t_n_l=Topm_t_n_l,
        shufflesplit={
            'shufflesplit_index_file': path_dict['shufflesplit_index_file']
        } if path_dict['model']['size'] == 'small' else None
    )
    PSE_t_5C = utils.ds_preprocess.get_5C_data(
        shape_to_chiose=PSE_t_p_f.shape[0],
        t_p_f=PSE_t_p_f,
        t_p_l=PSE_t_p_l,
        t_n_f=PSE_t_n_f,
        t_n_l=PSE_t_n_l,
        shufflesplit={
            'shufflesplit_index_file': path_dict['shufflesplit_index_file']
        } if path_dict['model']['size'] == 'small' else None
    )
    a35_t_5C = utils.ds_preprocess.get_5C_data(
        shape_to_chiose=a35_t_p_f.shape[0],
        t_p_f=a35_t_p_f,
        t_p_l=PSE_t_p_l,
        t_n_f=a35_t_n_f,
        t_n_l=PSE_t_n_l,
        shufflesplit={
            'shufflesplit_index_file': path_dict['shufflesplit_index_file']
        } if path_dict['model']['size'] == 'small' else None
    )
    a40_t_5C = utils.ds_preprocess.get_5C_data(
        shape_to_chiose=a40_t_p_f.shape[0],
        t_p_f=a40_t_p_f,
        t_p_l=PSE_t_p_l,
        t_n_f=a40_t_n_f,
        t_n_l=PSE_t_n_l,
        shufflesplit={
            'shufflesplit_index_file': path_dict['shufflesplit_index_file']
        } if path_dict['model']['size'] == 'small' else None
    )
    a45_t_5C = utils.ds_preprocess.get_5C_data(
        shape_to_chiose=a45_t_p_f.shape[0],
        t_p_f=a45_t_p_f,
        t_p_l=PSE_t_p_l,
        t_n_f=a45_t_n_f,
        t_n_l=PSE_t_n_l,
        shufflesplit={
            'shufflesplit_index_file': path_dict['shufflesplit_index_file']
        } if path_dict['model']['size'] == 'small' else None
    )

    iDNA_t_f, iDNA_t_l = utils.ds_preprocess.make_merge(
        t_p_f=iDNA_t_p_f,
        t_p_l=iDNA_t_p_l,
        t_n_f=iDNA_t_n_f,
        t_n_l=iDNA_t_n_l,
    )
    Topm_t_f, Topm_t_l = utils.ds_preprocess.make_merge(
        t_p_f=Topm_t_p_f,
        t_p_l=Topm_t_p_l,
        t_n_f=Topm_t_n_f,
        t_n_l=Topm_t_n_l,
    )
    PSE_t_f, PSE_t_l = utils.ds_preprocess.make_merge(
        t_p_f=PSE_t_p_f,
        t_p_l=PSE_t_p_l,
        t_n_f=PSE_t_n_f,
        t_n_l=PSE_t_n_l,
    )
    a35_t_f, a35_t_l = utils.ds_preprocess.make_merge(
        t_p_f=a35_t_p_f,
        t_p_l=PSE_t_p_l,
        t_n_f=a35_t_n_f,
        t_n_l=PSE_t_n_l,
    )
    a40_t_f, a40_t_l = utils.ds_preprocess.make_merge(
        t_p_f=a40_t_p_f,
        t_p_l=PSE_t_p_l,
        t_n_f=a40_t_n_f,
        t_n_l=PSE_t_n_l,
    )
    a45_t_f, a45_t_l = utils.ds_preprocess.make_merge(
        t_p_f=a45_t_p_f,
        t_p_l=PSE_t_p_l,
        t_n_f=a45_t_n_f,
        t_n_l=PSE_t_n_l,
    )
    iDNA_v_f, iDNA_v_l = utils.ds_preprocess.make_merge(
        t_p_f=iDNA_v_p_f,
        t_p_l=iDNA_v_p_l,
        t_n_f=iDNA_v_n_f,
        t_n_l=iDNA_v_n_l,
    )
    Topm_v_f, Topm_v_l = utils.ds_preprocess.make_merge(
        t_p_f=Topm_v_p_f,
        t_p_l=Topm_v_p_l,
        t_n_f=Topm_v_n_f,
        t_n_l=Topm_v_n_l,
    )
    PSE_v_f, PSE_v_l = utils.ds_preprocess.make_merge(
        t_p_f=PSE_v_p_f,
        t_p_l=PSE_v_p_l,
        t_n_f=PSE_v_n_f,
        t_n_l=PSE_v_n_l,
    )
    a35_v_f, a35_v_l = utils.ds_preprocess.make_merge(
        t_p_f=a35_v_p_f,
        t_p_l=PSE_v_p_l,
        t_n_f=a35_v_n_f,
        t_n_l=PSE_v_n_l,
    )
    a40_v_f, a40_v_l = utils.ds_preprocess.make_merge(
        t_p_f=a40_v_p_f,
        t_p_l=PSE_v_p_l,
        t_n_f=a40_v_n_f,
        t_n_l=PSE_v_n_l,
    )
    a45_v_f, a45_v_l = utils.ds_preprocess.make_merge(
        t_p_f=a45_v_p_f,
        t_p_l=PSE_v_p_l,
        t_n_f=a45_v_n_f,
        t_n_l=PSE_v_n_l,
    )

    PSE_p_f, PSE_p_l = utils.ds_preprocess.make_merge(
        t_p_f=PSE_t_p_f,
        t_p_l=PSE_t_p_l,
        t_n_f=PSE_v_p_f,
        t_n_l=PSE_v_p_l,
    )
    PSE_n_f, PSE_n_l = utils.ds_preprocess.make_merge(
        t_p_f=PSE_t_n_f,
        t_p_l=PSE_t_n_l,
        t_n_f=PSE_v_n_f,
        t_n_l=PSE_v_n_l,
    )

    PSE_all_f, _ = utils.ds_preprocess.make_merge(
        t_p_f=PSE_p_f,
        t_p_l=PSE_p_l,
        t_n_f=PSE_n_f,
        t_n_l=PSE_n_l,
    )

    if only_tt_model == True and only_cv_model == True:
        if os.path.exists(path_dict['csv_out_dir']) == False:
            os.makedirs(path_dict['csv_out_dir'])
        PSE_all_f.to_csv(os.path.join(
            path_dict['csv_out_dir'], *['ep3_pse.csv', ]))

    model_cv = 10
    if path_dict['model']['size'] == 'small':
        model_cv = ShuffleSplit(n_splits=10, test_size=0.5, random_state=0)

    side_to_save = [
        iDNA_side, Topm_side, PSE_side, a35_side, a40_side, a45_side
    ]

    if only_tt_model == False:
        Five_Cross_Get_model(
            model_construct_funtion=functools.partial(
                EP3_Model, desc=path_dict['model']['cv']['desc'], cv=model_cv),
            idna_f_Five_Cross_set=iDNA_t_5C,
            top_f_Five_Cross_set=Topm_t_5C,
            pse_f_Five_Cross_set=PSE_t_5C,
            a35_f_Five_Cross_set=a35_t_5C,
            a40_f_Five_Cross_set=a40_t_5C,
            a45_f_Five_Cross_set=a45_t_5C,
            idna_v_f=iDNA_v_f,
            top_v_f=Topm_v_f,
            pse_v_f=PSE_v_f,
            a35_v_f=a35_v_f,
            a40_v_f=a40_v_f,
            a45_v_f=a45_v_f,
            a45_v_l=a45_v_l,
            path_to_model_pickle=path_dict['model']['cv']['model_pickle'],
            path_to_model_result=path_dict['model']['cv']['model_result'],
            size_of_data=path_dict['model']['size'],
            data_to_store=side_to_save,
        )
    if only_cv_model == False:
        Five_Cross_Get_model(
            model_construct_funtion=functools.partial(
                EP3_Model, desc=path_dict['model']['tt']['desc'], cv=model_cv),
            idna_f_Five_Cross_set=(
                ([iDNA_t_f, iDNA_t_l], [iDNA_v_f, iDNA_v_l]),),
            top_f_Five_Cross_set=(
                ([Topm_t_f, Topm_t_l], [Topm_v_f, Topm_v_l]),),
            pse_f_Five_Cross_set=(([PSE_t_f, PSE_t_l], [PSE_v_f, PSE_v_l]),),
            a35_f_Five_Cross_set=(([a35_t_f, a35_t_l], [a35_v_f, a35_v_l]),),
            a40_f_Five_Cross_set=(([a40_t_f, a40_t_l], [a40_v_f, a40_v_l]),),
            a45_f_Five_Cross_set=(([a45_t_f, a45_t_l], [a45_v_f, a45_v_l]),),
            idna_v_f=iDNA_v_f,
            top_v_f=Topm_v_f,
            pse_v_f=PSE_v_f,
            a35_v_f=a35_v_f,
            a40_v_f=a40_v_f,
            a45_v_f=a45_v_f,
            a45_v_l=a45_v_l,
            path_to_model_pickle=path_dict['model']['tt']['model_pickle'],
            path_to_model_result=path_dict['model']['tt']['model_result'],
            size_of_data=None,
            data_to_store=side_to_save,
        )
